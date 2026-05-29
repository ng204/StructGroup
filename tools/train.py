import argparse
import datetime
import os
import os.path as osp
import shutil
import time
import sys
sys.path.append("/home/ng204/xueshuang/SoftGroup")
import torch

# 允许 TF32 / 提高 matmul 精度设置（对 Amp / 大量 matmul 的 MLP/Conv 有加速效果）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    # PyTorch 1.12+ 支持该接口；老版本忽略即可
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass
import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import (PanopticEval, ScanNetEval, evaluate_offset_mae,
                                  evaluate_semantic_acc, evaluate_semantic_miou)
from softgroup.model import SoftGroup
from softgroup.util import (AverageMeter, SummaryWriter, build_optimizer, checkpoint_save,
                            collect_results_cpu, cosine_lr_after_step, get_dist_info,
                            get_max_memory, get_root_logger, init_dist, is_main_process,
                            is_multiple, is_power2, load_checkpoint)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    # 重新训练主干网络backbone参数设置
    #parser.add_argument('--config', default='/StructGroup/configs/StructGroup/StructGroup_Panax_backbone_fold5.yaml', type=str, help='path to config file')     
    
    # 从冻结的主干中训练模型
    parser.add_argument('--config', default='/StructGroup/configs/StructGroup/StructGroup_Panax_fold5.yaml', type=str, help='path to config file')

    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--resume', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    parser.add_argument('--skip_validate', action='store_true', help='skip validation')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id(s) to use. Single GPU: "0" or "1". Multi-GPU: "0,1" or use --dist for distributed training')
    args = parser.parse_args()
    return args


def train(epoch, model, optimizer, scaler, train_loader, cfg, logger, writer):
    model.train()
    iter_time = AverageMeter(True)
    data_time = AverageMeter(True)
    meter_dict = {}
    end = time.time()

    if train_loader.sampler is not None and cfg.dist:
        train_loader.sampler.set_epoch(epoch)

    for i, batch in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)
        cosine_lr_after_step(optimizer, cfg.optimizer.lr, epoch - 1, cfg.step_epoch, cfg.epochs)
        # 传递epoch和iter信息给模型（用于低频约束loss和medoid计算）
        if isinstance(batch, dict):
            batch['epoch'] = epoch
            batch['iter'] = (epoch - 1) * len(train_loader) + i
        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            loss, log_vars = model(batch, return_loss=True)

        # meter_dict
        for k, v in log_vars.items():
            if k not in meter_dict.keys():
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if cfg.get('clip_grad_norm', None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # time and print
        remain_iter = len(train_loader) * (cfg.epochs - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        lr = optimizer.param_groups[0]['lr']

        if is_multiple(i, 10):
            log_str = f'Epoch [{epoch}/{cfg.epochs}][{i}/{len(train_loader)}]  '
            log_str += f'lr: {lr:.2g}, eta: {remain_time}, mem: {get_max_memory()}, '\
                f'data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}'
            for k, v in meter_dict.items():
                log_str += f', {k}: {v.val:.4f}'
            logger.info(log_str)
    writer.add_scalar('train/learning_rate', lr, epoch)
    for k, v in meter_dict.items():
        writer.add_scalar(f'train/{k}', v.avg, epoch)
    log_dir = writer.logdir
    checkpoint_save(epoch, model, optimizer, log_dir, cfg.save_freq)


def validate(epoch, model, val_loader, cfg, logger, writer):
    logger.info('Validation')
    results = []
    all_sem_preds, all_sem_labels, all_offset_preds, all_offset_labels = [], [], [], []
    all_inst_labels, all_pred_insts, all_gt_insts = [], [], []
    all_panoptic_preds = []
    _, world_size = get_dist_info()
    progress_bar = tqdm(total=len(val_loader) * world_size, disable=not is_main_process())
    val_set = val_loader.dataset
    eval_tasks = cfg.model.test_cfg.eval_tasks
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            result = model(batch)
            results.append(result)
            progress_bar.update(world_size)
        progress_bar.close()
        results = collect_results_cpu(results, len(val_set))
    if is_main_process():
        for res in results:
            if 'semantic' in eval_tasks or 'panoptic' in eval_tasks:
                all_sem_labels.append(res['semantic_labels'])
                all_inst_labels.append(res['instance_labels'])
            if 'semantic' in eval_tasks:
                all_sem_preds.append(res['semantic_preds'])
                all_offset_preds.append(res['offset_preds'])
                all_offset_labels.append(res['offset_labels'])
            if 'instance' in eval_tasks:
                all_pred_insts.append(res['pred_instances'])
                all_gt_insts.append(res['gt_instances'])
            if 'panoptic' in eval_tasks:
                all_panoptic_preds.append(res['panoptic_preds'])
        if 'instance' in eval_tasks:
            logger.info('=' * 80)
            logger.info('Evaluate instance segmentation (实例分割评估)')
            logger.info('指标说明:')
            logger.info('  AP (Average Precision): 平均精度，衡量检测质量')
            logger.info('  AP_50/AP_25: IoU阈值为0.5/0.25时的平均精度')
            logger.info('  RC/AR (Recall): 召回率，衡量检测的完整性')
            logger.info('-' * 80)
            eval_min_npoint = getattr(cfg, 'eval_min_npoint', None)
            scannet_eval = ScanNetEval(val_set.CLASSES, eval_min_npoint)
            eval_res = scannet_eval.evaluate(all_pred_insts, all_gt_insts)
            
            # 记录到 tensorboard
            writer.add_scalar('val/AP', eval_res['all_ap'], epoch)
            writer.add_scalar('val/AP_50', eval_res['all_ap_50%'], epoch)
            writer.add_scalar('val/AP_25', eval_res['all_ap_25%'], epoch)
            writer.add_scalar('val/AR', eval_res['all_rc'], epoch)
            writer.add_scalar('val/AR_50', eval_res['all_rc_50%'], epoch)
            writer.add_scalar('val/AR_25', eval_res['all_rc_25%'], epoch)
            
            # 打印每个类别的详细指标
            logger.info('各类别详细指标:')
            class_names_map = {'stem': '茎秆', 'leaf': '叶片', 'branch': '分枝'}  # 中文映射，便于理解
            for class_name in val_set.CLASSES:
                if class_name in eval_res['classes']:
                    class_metrics = eval_res['classes'][class_name]
                    display_name = class_names_map.get(class_name, class_name)
                    logger.info('  {:<8} ({:<4}): AP={:.3f}, AP_50={:.3f}, AP_25={:.3f}, '
                              'AR={:.3f}, AR_50={:.3f}, AR_25={:.3f}'.format(
                        class_name, display_name,
                        class_metrics['ap'], class_metrics['ap50%'], class_metrics['ap25%'],
                        class_metrics['rc'], class_metrics['rc50%'], class_metrics['rc25%']))
                    # 同时记录到 tensorboard
                    writer.add_scalar(f'val/AP_{class_name}', class_metrics['ap'], epoch)
                    writer.add_scalar(f'val/AP_50_{class_name}', class_metrics['ap50%'], epoch)
                    writer.add_scalar(f'val/AP_25_{class_name}', class_metrics['ap25%'], epoch)
                    writer.add_scalar(f'val/AR_{class_name}', class_metrics['rc'], epoch)
            
            # 打印平均指标
            logger.info('-' * 80)
            logger.info('平均指标 (Average):')
            logger.info('  AP={:.3f}, AP_50={:.3f}, AP_25={:.3f}'.format(
                eval_res['all_ap'], eval_res['all_ap_50%'], eval_res['all_ap_25%']))
            logger.info('  AR={:.3f}, AR_50={:.3f}, AR_25={:.3f}'.format(
                eval_res['all_rc'], eval_res['all_rc_50%'], eval_res['all_rc_25%']))
            logger.info('=' * 80)
        if 'panoptic' in eval_tasks:
            logger.info('Evaluate panoptic segmentation')
            eval_min_npoint = getattr(cfg, 'eval_min_npoint', None)
            panoptic_eval = PanopticEval(val_set.THING, val_set.STUFF, min_points=eval_min_npoint)
            eval_res = panoptic_eval.evaluate(all_panoptic_preds, all_sem_labels, all_inst_labels)
            writer.add_scalar('val/PQ', eval_res[0], epoch)
            logger.info('PQ: {:.1f}'.format(eval_res[0]))
        if 'semantic' in eval_tasks:
            logger.info('=' * 80)
            logger.info('Evaluate semantic segmentation (语义分割评估)')
            logger.info('指标说明:')
            logger.info('  mIoU (mean IoU): 平均交并比，衡量语义分割精度')
            logger.info('  Acc (Accuracy): 像素准确率')
            logger.info('  Offset MAE: 偏移量的平均绝对误差，衡量实例边界预测精度')
            logger.info('-' * 80)
            miou = evaluate_semantic_miou(all_sem_preds, all_sem_labels, cfg.model.ignore_label,
                                          logger)
            acc = evaluate_semantic_acc(all_sem_preds, all_sem_labels, cfg.model.ignore_label,
                                        logger)
            mae = evaluate_offset_mae(all_offset_preds, all_offset_labels, all_inst_labels,
                                      cfg.model.ignore_label, logger)
            logger.info(f'  mIoU: {miou:.4f}, Acc: {acc:.4f}, Offset MAE: {mae:.4f}')
            logger.info('=' * 80)
            writer.add_scalar('val/mIoU', miou, epoch)
            writer.add_scalar('val/Acc', acc, epoch)
            writer.add_scalar('val/Offset MAE', mae, epoch)


def main():
    args = get_args()
    
    # 设置GPU（支持多GPU）
    if args.gpu is not None:
        if isinstance(args.gpu, (list, tuple)):
            # 多GPU：使用逗号分隔的GPU列表
            gpu_str = ','.join(map(str, args.gpu))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            print(f"Using GPUs: {args.gpu}")
        else:
            # 单GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            print(f"Using GPU: {args.gpu}")
    else:
        # 如果没有指定，使用所有可用GPU
        print("Using all available GPUs")

    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    if args.dist:
        init_dist()
    cfg.dist = args.dist

    # work_dir & logger
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    log_dir = osp.join(cfg.work_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_file = osp.join(cfg.work_dir, f'{timestamp}/train.log')
    logger = get_root_logger(log_file=log_file)
    logger.info(f'Config:\n{cfg_txt}')
    logger.info(f'Distributed: {args.dist}')
    logger.info(f'Mix precision training: {cfg.fp16}')
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(osp.join(cfg.work_dir, timestamp))

    # model
    # 警告：spconv库与DataParallel存在兼容性问题，可能导致CUDA错误
    # spconv使用稀疏卷积，在多GPU环境下容易出现内存访问错误
    # 建议：使用单GPU训练（--gpu 0 或 --gpu 1）
    if not args.dist and args.gpu is not None:
        gpu_list = [int(x.strip()) for x in str(args.gpu).split(',')]
        if len(gpu_list) > 1:
            logger.error('=' * 80)
            logger.error('ERROR: DataParallel with spconv causes CUDA errors!')
            logger.error('spconv (sparse convolution) is NOT compatible with DataParallel.')
            logger.error('')
            logger.error('SOLUTION: Use single GPU training instead:')
            logger.error(f'  python tools/train.py --config {args.config} --gpu 0')
            logger.error('  or')
            logger.error(f'  python tools/train.py --config {args.config} --gpu 1')
            logger.error('')
            logger.error('For faster training, optimize the code (already done) or use distributed training (--dist).')
            logger.error('=' * 80)
            # 强制使用单GPU，避免CUDA错误
            logger.warning(f'Forcing single GPU mode (GPU {gpu_list[0]}) to avoid CUDA errors.')
            model = SoftGroup(**cfg.model).cuda(gpu_list[0])
        else:
            model = SoftGroup(**cfg.model).cuda(gpu_list[0] if len(gpu_list) > 0 else 0)
    elif args.dist:
        model = SoftGroup(**cfg.model).cuda()
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    else:
        model = SoftGroup(**cfg.model).cuda()
    
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # data
    train_set = build_dataset(cfg.data.train, logger)
    val_set = build_dataset(cfg.data.test, logger)
    train_loader = build_dataloader(
        train_set, training=True, dist=args.dist, **cfg.dataloader.train)
    val_loader = build_dataloader(val_set, training=False, dist=args.dist, **cfg.dataloader.test)

    # optim
    optimizer = build_optimizer(model, cfg.optimizer)

    # pretrain, resume
    start_epoch = 1
    if args.resume:
        logger.info(f'Resume from {args.resume}')
        start_epoch = load_checkpoint(args.resume, logger, model, optimizer=optimizer)
    elif cfg.pretrain:
        logger.info(f'Load pretrain from {cfg.pretrain}')
        load_checkpoint(cfg.pretrain, logger, model)

    # train and val
    logger.info('Training')
    for epoch in range(start_epoch, cfg.epochs + 1):
        logger.info(f'##################CURRENT EPOCH : {epoch}########################')
        train(epoch, model, optimizer, scaler, train_loader, cfg, logger, writer)
        if not args.skip_validate and (is_multiple(epoch, cfg.save_freq) or is_power2(epoch)):
            validate(epoch, model, val_loader, cfg, logger, writer)
        writer.flush()


if __name__ == '__main__':
    main()
