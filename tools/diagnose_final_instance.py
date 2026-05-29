"""
诊断脚本：统计 GT leaf 的最终实例-recall（包含后处理）
对每个 GT leaf 实例，找所有最终预测实例的最大 IoU（maxIoU）
统计 maxIoU > 0.1/0.3/0.5 的比例，以及 size ratio 和连通块数
用于评估连通性后处理的效果
"""
import argparse
import os
import os.path as osp
import numpy as np
import torch
import yaml
from munch import Munch
from tqdm import tqdm
from softgroup.data import build_dataloader, build_dataset
from softgroup.model import SoftGroup
from softgroup.util import get_root_logger, load_checkpoint, rle_decode
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def analyze_connectivity(coords, radius=0.015):
    """
    分析点云的连通分量数
    
    Args:
        coords: (N, 3) 点坐标
        radius: 连通性判断的半径阈值
    
    Returns:
        num_components: 连通分量数
    """
    if len(coords) == 0:
        return 0
    if len(coords) == 1:
        return 1
    
    try:
        tree = cKDTree(coords)
        pairs = tree.query_pairs(radius)
        pairs_list = list(pairs)
        
        if len(pairs_list) == 0:
            return len(coords)
        
        N = len(coords)
        row = np.array([p[0] for p in pairs_list], dtype=np.int32)
        col = np.array([p[1] for p in pairs_list], dtype=np.int32)
        data = np.ones(len(pairs_list), dtype=np.float32)
        
        row_sym = np.concatenate([row, col])
        col_sym = np.concatenate([col, row])
        data_sym = np.concatenate([data, data])
        
        assert len(row_sym) == len(col_sym) == len(data_sym), \
            f"Array length mismatch: row={len(row_sym)}, col={len(col_sym)}, data={len(data_sym)}"
        
        adj_matrix = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
        
        n_components, _ = connected_components(
            adj_matrix, 
            directed=False, 
            return_labels=True
        )
        
        return n_components
    except Exception as e:
        print(f"Warning: connectivity analysis failed: {e}, returning {len(coords)} components")
        return len(coords)


def calculate_iou(mask1, mask2):
    """计算两个mask的IoU"""
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def get_args():
    parser = argparse.ArgumentParser('Diagnose Final Instance Recall')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--out', type=str, default='leaf_final_instance_stats.txt', help='output file')
    return parser.parse_args()


def main():
    args = get_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(f"Using GPU: {args.gpu}")
    
    # 加载配置
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    
    # 创建logger
    logger = get_root_logger()
    
    # 加载模型
    model = SoftGroup(**cfg.model).cuda()
    logger.info(f'Load state dict from {args.checkpoint}')
    load_checkpoint(args.checkpoint, logger, model)
    model.eval()
    
    # 加载验证集
    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, dist=False, **cfg.dataloader.test)
    
    # 统计变量
    leaf_class_id = 1  # leaf 类别ID（stem=0, leaf=1, branch=2）
    leaf_label_id = leaf_class_id + 1  # 输出中是 1-based (1, 2, 3)
    all_max_ious = []
    stats = {
        'total_gt_leaves': 0,
        'recall_0.1': 0,
        'recall_0.3': 0,
        'recall_0.5': 0,
    }
    
    # 细粒度诊断统计
    detailed_stats = []
    
    print(f"\n开始诊断，验证集大小: {len(dataloader)}")
    print("=" * 80)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
            # 获取 GT labels
            instance_labels = batch['instance_labels'].cuda()  # (N,)
            semantic_labels = batch['semantic_labels'].cuda()  # (N,)
            coords_float = batch['coords_float'].cuda()  # (N, 3)
            
            # 前向传播获取最终实例（包含后处理）
            result = model(batch)
            pred_instances = result['pred_instances']  # List of dicts
            
            # 获取 GT instance 信息
            # 找到所有 leaf 的 GT instances
            unique_instances = torch.unique(instance_labels)
            leaf_gt_instances = []
            
            for inst_id in unique_instances:
                if inst_id == -100:  # ignore label
                    continue
                inst_mask = (instance_labels == inst_id)
                if inst_mask.sum() == 0:
                    continue
                first_point_idx = inst_mask.nonzero(as_tuple=False)[0, 0]
                sem_label = semantic_labels[first_point_idx].item()
                
                if sem_label == leaf_class_id:
                    leaf_gt_instances.append(inst_id.item())
            
            if len(leaf_gt_instances) == 0:
                continue
            
            # 提取预测的 leaf 实例
            pred_leaf_instances = []
            for inst in pred_instances:
                if inst['label_id'] == leaf_label_id:
                    pred_leaf_instances.append(inst)
            
            if len(pred_leaf_instances) == 0:
                # 没有预测的 leaf 实例，所有 GT leaf 的 maxIoU = 0
                for gt_inst_id in leaf_gt_instances:
                    stats['total_gt_leaves'] += 1
                    all_max_ious.append(0.0)
                    detailed_stats.append({
                        'best_iou': 0.0,
                        'size_ratio': 0.0,
                        'num_components': 0,
                        'predicted_size': 0,
                        'gt_size': (instance_labels == gt_inst_id).sum().item()
                    })
                continue
            
            # 解码所有预测的 leaf mask
            num_points = coords_float.size(0)
            pred_masks = []
            for inst in pred_leaf_instances:
                mask = rle_decode(inst['pred_mask']).astype(bool)
                if len(mask) != num_points:
                    # mask 长度不匹配，跳过
                    continue
                pred_masks.append(mask)
            
            if len(pred_masks) == 0:
                # 没有有效的预测 mask
                for gt_inst_id in leaf_gt_instances:
                    stats['total_gt_leaves'] += 1
                    all_max_ious.append(0.0)
                    detailed_stats.append({
                        'best_iou': 0.0,
                        'size_ratio': 0.0,
                        'num_components': 0,
                        'predicted_size': 0,
                        'gt_size': (instance_labels == gt_inst_id).sum().item()
                    })
                continue
            
            # 对每个 GT leaf instance，计算与所有预测实例的 IoU
            coords_float_np = coords_float.cpu().numpy()
            
            for gt_inst_id in leaf_gt_instances:
                gt_mask = (instance_labels == gt_inst_id).cpu().numpy().astype(bool)
                gt_size = gt_mask.sum()
                
                if gt_size == 0:
                    continue
                
                # 计算与所有预测实例的 IoU
                max_iou = 0.0
                best_pred_idx = -1
                best_pred_mask = None
                
                for pred_idx, pred_mask in enumerate(pred_masks):
                    iou = calculate_iou(gt_mask, pred_mask)
                    if iou > max_iou:
                        max_iou = iou
                        best_pred_idx = pred_idx
                        best_pred_mask = pred_mask
                
                # 更新统计
                stats['total_gt_leaves'] += 1
                stats['recall_0.1'] += (max_iou > 0.1)
                stats['recall_0.3'] += (max_iou > 0.3)
                stats['recall_0.5'] += (max_iou > 0.5)
                all_max_ious.append(max_iou)
                
                # 细粒度诊断：分析 best prediction
                if best_pred_mask is not None and best_pred_mask.sum() > 0:
                    predicted_size = best_pred_mask.sum()
                    size_ratio = predicted_size / gt_size if gt_size > 0 else 0.0
                    
                    # 分析连通性
                    pred_point_indices = np.nonzero(best_pred_mask)[0]
                    pred_coords = coords_float_np[pred_point_indices]
                    
                    # 获取连通半径（与后处理中使用的相同）
                    leaf_grouping_radius = cfg.model.grouping_cfg.class_specific_radius[1] if hasattr(cfg.model.grouping_cfg, 'class_specific_radius') and len(cfg.model.grouping_cfg.class_specific_radius) > 1 else 0.018
                    connectivity_radius_ratio = getattr(cfg.model.test_cfg, 'connectivity_radius_ratio', 0.6)
                    connectivity_radius = leaf_grouping_radius * connectivity_radius_ratio
                    if hasattr(cfg.model.test_cfg, 'connectivity_radius'):
                        connectivity_radius = float(cfg.model.test_cfg.connectivity_radius)
                    
                    num_components = analyze_connectivity(pred_coords, radius=connectivity_radius)
                else:
                    predicted_size = 0
                    size_ratio = 0.0
                    num_components = 0
                
                detailed_stats.append({
                    'best_iou': max_iou,
                    'size_ratio': size_ratio,
                    'num_components': num_components,
                    'predicted_size': predicted_size,
                    'gt_size': gt_size
                })
    
    # 计算比例
    if stats['total_gt_leaves'] > 0:
        recall_01_ratio = stats['recall_0.1'] / stats['total_gt_leaves']
        recall_03_ratio = stats['recall_0.3'] / stats['total_gt_leaves']
        recall_05_ratio = stats['recall_0.5'] / stats['total_gt_leaves']
    else:
        recall_01_ratio = recall_03_ratio = recall_05_ratio = 0.0
    
    # 计算统计信息
    all_max_ious = np.array(all_max_ious)
    if len(all_max_ious) > 0:
        mean_max_iou = all_max_ious.mean()
        median_max_iou = np.median(all_max_ious)
        min_max_iou = all_max_ious.min()
        max_max_iou = all_max_ious.max()
    else:
        mean_max_iou = median_max_iou = min_max_iou = max_max_iou = 0.0
    
    # 分析细粒度统计
    if len(detailed_stats) > 0:
        size_ratios = np.array([s['size_ratio'] for s in detailed_stats])
        num_components_list = np.array([s['num_components'] for s in detailed_stats])
        best_ious = np.array([s['best_iou'] for s in detailed_stats])
        
        # Size ratio 分布
        size_ratio_mean = size_ratios.mean()
        size_ratio_median = np.median(size_ratios)
        size_ratio_std = size_ratios.std()
        
        # 过分割/欠分割统计
        severe_underseg = (size_ratios < 0.5).sum()
        mild_underseg = ((size_ratios >= 0.5) & (size_ratios < 0.8)).sum()
        good_seg = ((size_ratios >= 0.8) & (size_ratios < 1.2)).sum()
        mild_overseg = ((size_ratios >= 1.2) & (size_ratios < 2.0)).sum()
        severe_overseg = (size_ratios >= 2.0).sum()
        
        # 连通性统计
        connected = (num_components_list == 1).sum()
        mild_split = ((num_components_list >= 2) & (num_components_list <= 3)).sum()
        severe_split = (num_components_list > 3).sum()
        
        # 相关性分析
        size_ratio_iou_corr = np.corrcoef(size_ratios, best_ious)[0, 1] if len(size_ratios) > 1 else 0.0
        components_iou_corr = np.corrcoef(num_components_list, best_ious)[0, 1] if len(num_components_list) > 1 else 0.0
        
        # 按 maxIoU 区间分组统计
        mask_01_03 = (best_ious >= 0.1) & (best_ious < 0.3)
        if mask_01_03.sum() > 0:
            size_ratio_01_03 = size_ratios[mask_01_03]
            num_components_01_03 = num_components_list[mask_01_03]
            median_size_ratio_01_03 = np.median(size_ratio_01_03)
            median_components_01_03 = np.median(num_components_01_03)
            count_01_03 = mask_01_03.sum()
        else:
            median_size_ratio_01_03 = 0.0
            median_components_01_03 = 0.0
            count_01_03 = 0
    else:
        size_ratio_mean = size_ratio_median = size_ratio_std = 0.0
        severe_underseg = mild_underseg = good_seg = mild_overseg = severe_overseg = 0
        connected = mild_split = severe_split = 0
        size_ratio_iou_corr = components_iou_corr = 0.0
        median_size_ratio_01_03 = 0.0
        median_components_01_03 = 0.0
        count_01_03 = 0
    
    # 输出结果
    print("\n" + "=" * 80)
    print("GT Leaf Final Instance-Recall 诊断结果（包含后处理）")
    print("=" * 80)
    print(f"总 GT leaf 数量: {stats['total_gt_leaves']}")
    print(f"\n召回率统计:")
    print(f"  触达召回 (maxIoU > 0.1): {stats['recall_0.1']}/{stats['total_gt_leaves']} = {recall_01_ratio:.4f} ({recall_01_ratio*100:.2f}%)")
    print(f"  粗召回 (maxIoU > 0.3):   {stats['recall_0.3']}/{stats['total_gt_leaves']} = {recall_03_ratio:.4f} ({recall_03_ratio*100:.2f}%)")
    print(f"  强召回 (maxIoU > 0.5):   {stats['recall_0.5']}/{stats['total_gt_leaves']} = {recall_05_ratio:.4f} ({recall_05_ratio*100:.2f}%)")
    print(f"\nmaxIoU 统计:")
    print(f"  平均值: {mean_max_iou:.4f}")
    print(f"  中位数: {median_max_iou:.4f}")
    print(f"  最小值: {min_max_iou:.4f}")
    print(f"  最大值: {max_max_iou:.4f}")
    
    if len(detailed_stats) > 0:
        print(f"\n过分割/欠分割分析 (predicted_size / GT_size):")
        print(f"  平均值: {size_ratio_mean:.4f}")
        print(f"  中位数: {size_ratio_median:.4f}")
        print(f"  标准差: {size_ratio_std:.4f}")
        print(f"  分布:")
        print(f"    严重欠分割 (<0.5):  {severe_underseg:3d} ({severe_underseg/len(detailed_stats)*100:.2f}%)")
        print(f"    轻度欠分割 (0.5-0.8): {mild_underseg:3d} ({mild_underseg/len(detailed_stats)*100:.2f}%)")
        print(f"    分割合适 (0.8-1.2):   {good_seg:3d} ({good_seg/len(detailed_stats)*100:.2f}%)")
        print(f"    轻度过分割 (1.2-2.0): {mild_overseg:3d} ({mild_overseg/len(detailed_stats)*100:.2f}%)")
        print(f"    严重过分割 (>2.0):   {severe_overseg:3d} ({severe_overseg/len(detailed_stats)*100:.2f}%)")
        
        print(f"\n连通性分析 (best prediction 的连通块数):")
        print(f"  连通 (=1):     {connected:3d} ({connected/len(detailed_stats)*100:.2f}%)")
        print(f"  轻度分裂 (2-3): {mild_split:3d} ({mild_split/len(detailed_stats)*100:.2f}%)")
        print(f"  严重分裂 (>3): {severe_split:3d} ({severe_split/len(detailed_stats)*100:.2f}%)")
        
        print(f"\n相关性分析:")
        print(f"  Size Ratio 与 IoU 的相关性: {size_ratio_iou_corr:.4f}")
        print(f"  连通块数与 IoU 的相关性: {components_iou_corr:.4f}")
        
        if count_01_03 > 0:
            print(f"\n按 maxIoU 区间分组统计:")
            print(f"  maxIoU ∈ [0.1, 0.3) 区间（触达但不够好）:")
            print(f"    数量: {count_01_03}")
            print(f"    pred_size/GT_size 中位数: {median_size_ratio_01_03:.4f}")
            print(f"    连通块数中位数: {median_components_01_03:.4f}")
    
    print("=" * 80)
    
    # 保存到文件
    with open(args.out, 'w') as f:
        f.write("GT Leaf Final Instance-Recall 诊断结果（包含后处理）\n")
        f.write("=" * 80 + "\n")
        f.write(f"总 GT leaf 数量: {stats['total_gt_leaves']}\n")
        f.write(f"\n召回率统计:\n")
        f.write(f"  触达召回 (maxIoU > 0.1): {stats['recall_0.1']}/{stats['total_gt_leaves']} = {recall_01_ratio:.4f} ({recall_01_ratio*100:.2f}%)\n")
        f.write(f"  粗召回 (maxIoU > 0.3):   {stats['recall_0.3']}/{stats['total_gt_leaves']} = {recall_03_ratio:.4f} ({recall_03_ratio*100:.2f}%)\n")
        f.write(f"  强召回 (maxIoU > 0.5):   {stats['recall_0.5']}/{stats['total_gt_leaves']} = {recall_05_ratio:.4f} ({recall_05_ratio*100:.2f}%)\n")
        f.write(f"\nmaxIoU 统计:\n")
        f.write(f"  平均值: {mean_max_iou:.4f}\n")
        f.write(f"  中位数: {median_max_iou:.4f}\n")
        f.write(f"  最小值: {min_max_iou:.4f}\n")
        f.write(f"  最大值: {max_max_iou:.4f}\n")
        
        if len(detailed_stats) > 0:
            f.write(f"\n过分割/欠分割分析 (predicted_size / GT_size):\n")
            f.write(f"  平均值: {size_ratio_mean:.4f}\n")
            f.write(f"  中位数: {size_ratio_median:.4f}\n")
            f.write(f"  标准差: {size_ratio_std:.4f}\n")
            f.write(f"  分布:\n")
            f.write(f"    严重欠分割 (<0.5):  {severe_underseg:3d} ({severe_underseg/len(detailed_stats)*100:.2f}%)\n")
            f.write(f"    轻度欠分割 (0.5-0.8): {mild_underseg:3d} ({mild_underseg/len(detailed_stats)*100:.2f}%)\n")
            f.write(f"    分割合适 (0.8-1.2):   {good_seg:3d} ({good_seg/len(detailed_stats)*100:.2f}%)\n")
            f.write(f"    轻度过分割 (1.2-2.0): {mild_overseg:3d} ({mild_overseg/len(detailed_stats)*100:.2f}%)\n")
            f.write(f"    严重过分割 (>2.0):   {severe_overseg:3d} ({severe_overseg/len(detailed_stats)*100:.2f}%)\n")
            
            f.write(f"\n连通性分析 (best prediction 的连通块数):\n")
            f.write(f"  连通 (=1):     {connected:3d} ({connected/len(detailed_stats)*100:.2f}%)\n")
            f.write(f"  轻度分裂 (2-3): {mild_split:3d} ({mild_split/len(detailed_stats)*100:.2f}%)\n")
            f.write(f"  严重分裂 (>3): {severe_split:3d} ({severe_split/len(detailed_stats)*100:.2f}%)\n")
            
            f.write(f"\n相关性分析:\n")
            f.write(f"  Size Ratio 与 IoU 的相关性: {size_ratio_iou_corr:.4f}\n")
            f.write(f"  连通块数与 IoU 的相关性: {components_iou_corr:.4f}\n")
            
            if count_01_03 > 0:
                f.write(f"\n按 maxIoU 区间分组统计:\n")
                f.write(f"  maxIoU ∈ [0.1, 0.3) 区间（触达但不够好）:\n")
                f.write(f"    数量: {count_01_03}\n")
                f.write(f"    pred_size/GT_size 中位数: {median_size_ratio_01_03:.4f}\n")
                f.write(f"    连通块数中位数: {median_components_01_03:.4f}\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\n结果已保存到: {args.out}")


if __name__ == '__main__':
    main()

