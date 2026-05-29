import argparse
import multiprocessing as mp
import os
import os.path as osp

import numpy as np
import torch
import yaml
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import (PanopticEval, ScanNetEval, evaluate_offset_mae,
                                  evaluate_semantic_acc, evaluate_semantic_miou)
from softgroup.model import SoftGroup
from softgroup.util import (collect_results_cpu, get_dist_info, get_root_logger, init_dist,
                            is_main_process, load_checkpoint, rle_decode)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import time
def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    # parser.add_argument('config', type=str, help='path to config file')
    
    
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--out', type=str, help='directory for output results')
    parser.add_argument('--work_dir', type=str, help='working directory for inference')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (0 or 1, default: 0)')
    args = parser.parse_args()
    return args


def save_npy(root, name, scan_ids, arrs):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.npy') for i in scan_ids]
    pool = mp.Pool()
    pool.starmap(np.save, zip(paths, arrs))
    pool.close()
    pool.join()


def save_single_instance(root, scan_id, insts, nyu_id=None):
    f = open(osp.join(root, f'{scan_id}.txt'), 'w')
    os.makedirs(osp.join(root, 'predicted_masks'), exist_ok=True)
    for i, inst in enumerate(insts):
        assert scan_id == inst['scan_id']
        label_id = inst['label_id']
        # scannet dataset use nyu_id for evaluation
        if nyu_id is not None:
            label_id = nyu_id[label_id - 1]
        conf = inst['conf']
        f.write(f'predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f}\n')
        mask_path = osp.join(root, 'predicted_masks', f'{scan_id}_{i:03d}.txt')
        mask = rle_decode(inst['pred_mask'])
        np.savetxt(mask_path, mask, fmt='%d')
    f.close()


def save_pred_instances(root, name, scan_ids, pred_insts, nyu_id=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    nyu_ids = [nyu_id] * len(scan_ids)
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts, nyu_ids))
    pool.close()
    pool.join()


def save_gt_instance(path, gt_inst, nyu_id=None):
    if nyu_id is not None:
        sem = gt_inst // 1000
        ignore = sem == 0
        ins = gt_inst % 1000
        nyu_id = np.array(nyu_id)
        sem = nyu_id[sem - 1]
        sem[ignore] = 0
        gt_inst = sem * 1000 + ins
    np.savetxt(path, gt_inst, fmt='%d')


def save_gt_instances(root, name, scan_ids, gt_insts, nyu_id=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.txt') for i in scan_ids]
    pool = mp.Pool()
    nyu_ids = [nyu_id] * len(scan_ids)
    pool.starmap(save_gt_instance, zip(paths, gt_insts, nyu_ids))
    pool.close()
    pool.join()


def save_panoptic_single(path, panoptic_pred, learning_map_inv, num_classes):
    # convert cls to kitti format
    panoptic_ids = panoptic_pred >> 16
    panoptic_cls = panoptic_pred & 0xFFFF
    new_learning_map_inv = {num_classes: 0}
    for k, v in learning_map_inv.items():
        if k == 0:
            continue
        if k < 9:
            new_k = k + 10
        else:
            new_k = k - 9
        new_learning_map_inv[new_k] = v
    panoptic_cls = np.vectorize(new_learning_map_inv.__getitem__)(panoptic_cls).astype(
        panoptic_pred.dtype)
    panoptic_pred = (panoptic_cls & 0xFFFF) | (panoptic_ids << 16)
    panoptic_pred.tofile(path)


def save_panoptic(root, name, scan_ids, arrs, learning_map_inv, num_classes):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.label'.replace('velodyne', 'predictions')) for i in scan_ids]
    learning_map_invs = [learning_map_inv] * len(scan_ids)
    num_classes_list = [num_classes] * len(scan_ids)
    for p in paths:
        os.makedirs(osp.dirname(p), exist_ok=True)
    pool = mp.Pool()
    pool.starmap(save_panoptic_single, zip(paths, arrs, learning_map_invs, num_classes_list))


def main():
    args = get_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(f"Using GPU: {args.gpu}")
    
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    # 设置工作目录
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif cfg.work_dir:
        pass  # 使用配置文件中的work_dir
    else:
        cfg.work_dir = './work_dirs/inference'

    if args.dist:
        init_dist()

    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = osp.join(cfg.work_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # 设置输出目录：优先使用用户指定的--out，否则使用默认路径
    if args.out:
        output_dir = args.out
    else:
        output_dir = osp.join(log_dir, 'Inference')

    args.out = output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_file = osp.join(cfg.work_dir, f'{timestamp}/inference.log')
    logger = get_root_logger(log_file=log_file)
    # logger = get_root_logger()

    model = SoftGroup(**cfg.model).cuda()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    logger.info(f'Load state dict from {args.checkpoint}')
    load_checkpoint(args.checkpoint, logger, model)

    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, dist=args.dist, **cfg.dataloader.test)
    results = []
    scan_ids, coords, colors, sem_preds, sem_labels = [], [], [], [], []
    offset_preds, offset_labels, inst_labels, pred_insts, gt_insts = [], [], [], [], []
    boundary_scores_list = []
    panoptic_preds = []
    _, world_size = get_dist_info()
    progress_bar = tqdm(total=len(dataloader) * world_size, disable=not is_main_process())
    eval_tasks = cfg.model.test_cfg.eval_tasks
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            result = model(batch)
            results.append(result)
            progress_bar.update(world_size)
        progress_bar.close()
        results = collect_results_cpu(results, len(dataset))
    if is_main_process():
        for res in results:
            scan_ids.append(res['scan_id'])
            if 'semantic' in eval_tasks or 'panoptic' in eval_tasks:
                sem_labels.append(res['semantic_labels'])
                inst_labels.append(res['instance_labels'])
            if 'semantic' in eval_tasks:
                coords.append(res['coords_float'])
                colors.append(res['color_feats'])
                sem_preds.append(res['semantic_preds'])
                offset_preds.append(res['offset_preds'])
                offset_labels.append(res['offset_labels'])
                # 可选导出 boundary_scores（仅在启用边界注意力时存在）
                if 'boundary_scores' in res:
                    boundary_scores_list.append(res['boundary_scores'])
            if 'instance' in eval_tasks:
                pred_insts.append(res['pred_instances'])
                gt_insts.append(res['gt_instances'])
            if 'panoptic' in eval_tasks:
                panoptic_preds.append(res['panoptic_preds'])
        if 'instance' in eval_tasks:
            logger.info('=' * 80)
            logger.info('Evaluate instance segmentation (实例分割评估)')
            logger.info('指标说明:')
            logger.info('  AP (Average Precision): 平均精度，衡量检测质量')
            logger.info('  AP_50/AP_25: IoU阈值为0.5/0.25时的平均精度')
            logger.info('  RC/AR (Recall): 召回率，衡量检测的完整性')
            logger.info('-' * 80)
            eval_min_npoint = getattr(cfg, 'eval_min_npoint', None)
            scannet_eval = ScanNetEval(dataset.CLASSES, eval_min_npoint)
            eval_res = scannet_eval.evaluate(pred_insts, gt_insts)
            
            # 打印每个类别的详细指标
            logger.info('各类别详细指标:')
            class_names_map = {'stem': '茎秆', 'leaf': '叶片', 'branch': '分枝'}  # 中文映射，便于理解
            for class_name in dataset.CLASSES:
                if class_name in eval_res['classes']:
                    class_metrics = eval_res['classes'][class_name]
                    display_name = class_names_map.get(class_name, class_name)
                    logger.info('  {:<8} ({:<4}): AP={:.3f}, AP_50={:.3f}, AP_25={:.3f}, '
                              'AR={:.3f}, AR_50={:.3f}, AR_25={:.3f}'.format(
                        class_name, display_name,
                        class_metrics['ap'], class_metrics['ap50%'], class_metrics['ap25%'],
                        class_metrics['rc'], class_metrics['rc50%'], class_metrics['rc25%']))
            
            # 打印平均指标
            logger.info('-' * 80)
            logger.info('平均指标 (Average):')
            logger.info('  AP={:.3f}, AP_50={:.3f}, AP_25={:.3f}'.format(
                eval_res['all_ap'], eval_res['all_ap_50%'], eval_res['all_ap_25%']))
            logger.info('  AR={:.3f}, AR_50={:.3f}, AR_25={:.3f}'.format(
                eval_res['all_rc'], eval_res['all_rc_50%'], eval_res['all_rc_25%']))
            logger.info('=' * 80)
            
        if 'panoptic' in eval_tasks:
            logger.info('=' * 80)
            logger.info('Evaluate panoptic segmentation (全景分割评估)')
            logger.info('指标说明:')
            logger.info('  PQ (Panoptic Quality): 全景质量，综合衡量语义和实例分割性能')
            logger.info('-' * 80)
            eval_min_npoint = getattr(cfg, 'eval_min_npoint', None)
            panoptic_eval = PanopticEval(dataset.THING, dataset.STUFF, min_points=eval_min_npoint)
            eval_res = panoptic_eval.evaluate(panoptic_preds, sem_labels, inst_labels)
            logger.info(f'  PQ: {eval_res[0]:.1f}')
            logger.info('=' * 80)
            
        if 'semantic' in eval_tasks:
            logger.info('=' * 80)
            logger.info('Evaluate semantic segmentation (语义分割评估)')
            logger.info('指标说明:')
            logger.info('  mIoU (mean IoU): 平均交并比，衡量语义分割精度')
            logger.info('  Acc (Accuracy): 像素准确率')
            logger.info('  Offset MAE: 偏移量的平均绝对误差，衡量实例边界预测精度')
            logger.info('-' * 80)
            ignore_label = cfg.model.ignore_label
            miou = evaluate_semantic_miou(sem_preds, sem_labels, ignore_label, logger)
            acc = evaluate_semantic_acc(sem_preds, sem_labels, ignore_label, logger)
            mae = evaluate_offset_mae(offset_preds, offset_labels, inst_labels, ignore_label, logger)
            logger.info(f'  mIoU: {miou:.4f}, Acc: {acc:.4f}, Offset MAE: {mae:.4f}')
            logger.info('=' * 80)

        # save output
        if not args.out:
            return
        logger.info('Save results')
        # 修复coords和colors维度不匹配的问题
        # coords和colors直接保存，不转换为array（因为每个场景大小不同）
        if 'semantic' in eval_tasks:
            save_npy(args.out, 'coords', scan_ids, coords)
            save_npy(args.out, 'colors', scan_ids, colors)
            # 保存语义相关文件，保持与visualization.py兼容
            save_npy(args.out, 'semantic_pred', scan_ids, sem_preds)
            save_npy(args.out, 'semantic_label', scan_ids, sem_labels)
            save_npy(args.out, 'offset_pred', scan_ids, offset_preds)
            save_npy(args.out, 'offset_label', scan_ids, offset_labels)
            # 保存 boundary_scores（如果存在）
            if len(boundary_scores_list) == len(scan_ids):
                save_npy(args.out, 'boundary_scores', scan_ids, boundary_scores_list)
        if 'instance' in eval_tasks:
            nyu_id = dataset.NYU_ID
            save_pred_instances(args.out, 'pred_instance', scan_ids, pred_insts, nyu_id)
            save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts, nyu_id)
        if 'panoptic' in eval_tasks:
            save_panoptic(args.out, 'panoptic', scan_ids, panoptic_preds, dataset.learning_map_inv,
                          cfg.model.semantic_classes)


if __name__ == '__main__':
    main()
