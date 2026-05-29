import functools
import os
import os.path as osp
from collections import OrderedDict
from math import cos, pi

import torch
from torch import distributed as dist

from .dist import get_dist_info, master_only


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, apply_dist_reduce=False):
        self.apply_dist_reduce = apply_dist_reduce
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def dist_reduce(self, val):
        rank, world_size = get_dist_info()
        if world_size == 1:
            return val
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val, device='cuda')
        dist.all_reduce(val)
        return val.item() / world_size

    def get_val(self):
        if self.apply_dist_reduce:
            return self.dist_reduce(self.val)
        else:
            return self.val

    def get_avg(self):
        if self.apply_dist_reduce:
            return self.dist_reduce(self.avg)
        else:
            return self.avg

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Epoch counts from 0 to N-1
def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * \
            (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    return num != 0 and num % multiple == 0


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


@master_only
def checkpoint_save(epoch, model, optimizer, work_dir, save_freq=16):
    if hasattr(model, 'module'):
        model = model.module
    f = os.path.join(work_dir, f'epoch_{epoch}.pth')
    checkpoint = {
        'net': weights_to_cpu(model.state_dict()),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, f)

    # 更新 latest.pth 软链接
    latest_path = os.path.join(work_dir, 'latest.pth')
    if os.path.islink(latest_path) or os.path.exists(latest_path):
        os.remove(latest_path)
    os.system(f'cd {work_dir}; ln -s {osp.basename(f)} latest.pth')

    # 只保留最近的 5 个 checkpoint，其余删除以节省磁盘空间
    # 匹配形如 epoch_XX.pth 的文件
    ckpt_files = [
        x for x in os.listdir(work_dir)
        if x.startswith('epoch_') and x.endswith('.pth')
    ]
    # 提取 epoch 数字并排序
    def _get_epoch_id(name):
        try:
            base = osp.splitext(name)[0]  # epoch_XX
            return int(base.split('_')[-1])
        except Exception:
            return -1

    ckpt_files = sorted(ckpt_files, key=_get_epoch_id)

    # 保留最后 5 个，其余删除
    max_keep = 5
    if len(ckpt_files) > max_keep:
        to_remove = ckpt_files[0:len(ckpt_files) - max_keep]
        for rm_name in to_remove:
            rm_path = os.path.join(work_dir, rm_name)
            if os.path.isfile(rm_path):
                os.remove(rm_path)


def load_checkpoint(checkpoint, logger, model, optimizer=None, strict=False):
    if hasattr(model, 'module'):
        model = model.module
    device = torch.cuda.current_device()
    state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(device))
    src_state_dict = state_dict['net']
    target_state_dict = model.state_dict()
    skip_keys = []
    # skip mismatch size tensors in case of pretraining
    for k in src_state_dict.keys():
        if k not in target_state_dict:
            continue
        if src_state_dict[k].size() != target_state_dict[k].size():
            skip_keys.append(k)
    for k in skip_keys:
        del src_state_dict[k]
    missing_keys, unexpected_keys = model.load_state_dict(src_state_dict, strict=strict)
    if skip_keys:
        logger.info(
            f'removed keys in source state_dict due to size mismatch: {", ".join(skip_keys)}')
    if missing_keys:
        logger.info(f'missing keys in source state_dict: {", ".join(missing_keys)}')
    if unexpected_keys:
        logger.info(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}')

    # load optimizer
    if optimizer is not None:
        assert 'optimizer' in state_dict
        optimizer.load_state_dict(state_dict['optimizer'])

    if 'epoch' in state_dict:
        epoch = state_dict['epoch']
    else:
        epoch = 0
    return epoch + 1


def get_max_memory():
    mem = torch.cuda.max_memory_allocated()
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)], dtype=torch.int, device='cuda')
    _, world_size = get_dist_info()
    if world_size > 1:
        dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
    return mem_mb.item()


def cuda_cast(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for x in args:
            if isinstance(x, torch.Tensor):
                x = x.cuda()
            new_args.append(x)
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.cuda()
            new_kwargs[k] = v
        return func(*new_args, **new_kwargs)

    return wrapper
