"""
诊断脚本：统计 GT leaf 的 proposal-recall
对每个 GT leaf 实例，找所有 proposal 的最大 IoU（maxIoU）
统计 maxIoU > 0.1/0.3/0.5 的比例（分别代表：是否触达/粗召回/强召回）
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
from softgroup.util import get_root_logger, load_checkpoint
from softgroup.ops.functions import get_mask_iou_on_cluster
from softgroup.ops import voxelization
import spconv.pytorch as spconv
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
    
    # 使用 KDTree 找邻居
    try:
        tree = cKDTree(coords)
        pairs = tree.query_pairs(radius)
        
        # 转换为列表以确保可以索引
        pairs_list = list(pairs)
        
        if len(pairs_list) == 0:
            # 没有连接，每个点都是独立的连通分量
            return len(coords)
        
        # 构建邻接矩阵
        N = len(coords)
        row = np.array([p[0] for p in pairs_list], dtype=np.int32)
        col = np.array([p[1] for p in pairs_list], dtype=np.int32)
        data = np.ones(len(pairs_list), dtype=np.float32)
        
        # 构建对称矩阵（无向图）
        row_sym = np.concatenate([row, col])
        col_sym = np.concatenate([col, row])
        data_sym = np.concatenate([data, data])
        
        # 确保长度一致
        assert len(row_sym) == len(col_sym) == len(data_sym), \
            f"Array length mismatch: row={len(row_sym)}, col={len(col_sym)}, data={len(data_sym)}"
        
        adj_matrix = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
        
        # 计算连通分量
        n_components, _ = connected_components(
            adj_matrix, 
            directed=False, 
            return_labels=True
        )
        
        return n_components
    except Exception as e:
        # 如果出错，返回点数（最坏情况：每个点都是独立的）
        print(f"Warning: connectivity analysis failed: {e}, returning {len(coords)} components")
        return len(coords)


def get_args():
    parser = argparse.ArgumentParser('Diagnose Leaf Proposal Recall')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--out', type=str, default='leaf_proposal_recall_stats.txt', help='output file')
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
    all_max_ious = []  # 所有 GT leaf 的 maxIoU
    stats = {
        'total_gt_leaves': 0,
        'recall_0.1': 0,  # maxIoU > 0.1 的数量
        'recall_0.3': 0,  # maxIoU > 0.3 的数量
        'recall_0.5': 0,  # maxIoU > 0.5 的数量
    }
    
    # 细粒度诊断统计
    detailed_stats = []  # 存储每个 GT leaf 的详细信息
    
    print(f"\n开始诊断，验证集大小: {len(dataloader)}")
    print("=" * 80)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
            # 获取 GT labels
            instance_labels = batch['instance_labels'].cuda()  # (N,)
            semantic_labels = batch['semantic_labels'].cuda()  # (N,)
            batch_idxs = batch['batch_idxs'].cuda()  # (N,)
            
            # 前向传播获取 proposals
            # 需要手动调用 forward_grouping 来获取 proposals
            batch_idxs = batch['batch_idxs'].cuda()
            voxel_coords = batch['voxel_coords'].cuda()
            p2v_map = batch['p2v_map'].cuda()
            v2p_map = batch['v2p_map'].cuda()
            coords_float = batch['coords_float'].cuda()
            feats = batch['feats'].cuda()
            spatial_shape = batch['spatial_shape']
            batch_size = batch['batch_size']
            
            # 构建 input（SparseConvTensor）
            if model.with_coords:
                feats_with_coords = torch.cat((feats, coords_float), 1)
            else:
                feats_with_coords = feats
            voxel_feats = voxelization(feats_with_coords, p2v_map)
            input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
            
            # 获取语义和offset预测
            semantic_scores, pt_offsets, output_feats = model.forward_backbone(input, v2p_map)
            
            # 获取 proposals
            proposals_idx, proposals_offset = model.forward_grouping(
                semantic_scores,
                pt_offsets,
                batch_idxs,
                coords_float,
                model.grouping_cfg,
                lvl_fusion=False
            )
            
            if proposals_offset.numel() <= 1:
                # 没有 proposal，跳过
                continue
            
            # 保存原始 proposals_idx 用于后续分析（包含 proposal_id）
            proposals_idx_full = proposals_idx.clone()  # (sumNPoint, 2)
            
            # proposals_idx 是 (sumNPoint, 2) 格式，需要提取第二列（point indices）
            # 并确保在 CUDA 上、连续、int32 类型
            proposals_idx = proposals_idx[:, 1].int().cuda().contiguous()
            proposals_offset = proposals_offset.int().cuda().contiguous()
            
            # 获取 GT instance 信息
            # 找到所有 leaf 的 GT instances
            unique_instances = torch.unique(instance_labels)
            leaf_gt_instances = []
            
            for inst_id in unique_instances:
                if inst_id == -100:  # ignore label
                    continue
                # 检查这个 instance 的语义标签（取第一个点的语义标签）
                inst_mask = (instance_labels == inst_id)
                if inst_mask.sum() == 0:
                    continue
                first_point_idx = inst_mask.nonzero(as_tuple=False)[0, 0]
                sem_label = semantic_labels[first_point_idx].item()
                
                if sem_label == leaf_class_id:
                    leaf_gt_instances.append(inst_id.item())
            
            if len(leaf_gt_instances) == 0:
                continue
            
            # 计算每个 GT leaf instance 的 pointnum
            instance_pointnum = []
            instance_labels_for_iou = []
            for inst_id in leaf_gt_instances:
                inst_mask = (instance_labels == inst_id)
                pointnum = inst_mask.sum().item()
                if pointnum > 0:
                    instance_pointnum.append(pointnum)
                    instance_labels_for_iou.append(inst_id)
            
            if len(instance_pointnum) == 0:
                continue
            
            # 转换为 tensor
            instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int32).cuda().contiguous()
            # 重新映射 instance_labels 为 0, 1, 2, ... (用于 IoU 计算)
            inst_id_to_idx = {inst_id: idx for idx, inst_id in enumerate(instance_labels_for_iou)}
            instance_labels_mapped = instance_labels.clone()
            for inst_id, idx in inst_id_to_idx.items():
                instance_labels_mapped[instance_labels == inst_id] = idx
            # 将不在 leaf_gt_instances 中的 instance 设为 -100
            instance_labels_mapped[~torch.isin(instance_labels, torch.tensor(instance_labels_for_iou).cuda())] = -100
            # 确保是连续的
            instance_labels_mapped = instance_labels_mapped.contiguous()
            
            # 计算 IoU: (nProposal, nGTLeaf)
            proposals_iou = get_mask_iou_on_cluster(
                proposals_idx,
                proposals_offset,
                instance_labels_mapped,
                instance_pointnum
            )  # (nProposal, nGTLeaf)
            
            # 对每个 GT leaf，找最大 IoU 和对应的 best proposal
            max_ious, best_proposal_indices = proposals_iou.max(dim=0)  # (nGTLeaf,)
            max_ious_np = max_ious.cpu().numpy()
            best_proposal_indices_np = best_proposal_indices.cpu().numpy()
            
            # 更新统计
            stats['total_gt_leaves'] += len(max_ious_np)
            stats['recall_0.1'] += (max_ious_np > 0.1).sum()
            stats['recall_0.3'] += (max_ious_np > 0.3).sum()
            stats['recall_0.5'] += (max_ious_np > 0.5).sum()
            
            all_max_ious.extend(max_ious_np.tolist())
            
            # 细粒度诊断：分析每个 GT leaf 的 best proposal
            proposals_offset_cpu = proposals_offset.cpu().numpy()
            coords_float_cpu = coords_float.cpu().numpy()
            proposals_idx_full_cpu = proposals_idx_full.cpu().numpy()
            
            for gt_idx in range(len(instance_labels_for_iou)):
                best_proposal_idx = best_proposal_indices_np[gt_idx]
                best_iou = max_ious_np[gt_idx]
                gt_inst_id = instance_labels_for_iou[gt_idx]
                
                # 计算 GT size
                gt_size = instance_pointnum[gt_idx].item()
                
                # 计算 predicted size（best proposal 的点数）
                proposal_start = proposals_offset_cpu[best_proposal_idx]
                proposal_end = proposals_offset_cpu[best_proposal_idx + 1]
                predicted_size = proposal_end - proposal_start
                
                # 计算 size ratio
                size_ratio = predicted_size / gt_size if gt_size > 0 else 0.0
                
                # 分析连通性：提取 best proposal 的所有点
                if predicted_size > 0:
                    proposal_point_indices = proposals_idx_full_cpu[proposal_start:proposal_end, 1]
                    proposal_coords = coords_float_cpu[proposal_point_indices]
                    
                    # 使用 KNN 构建连通图，然后分析连通分量
                    num_components = analyze_connectivity(proposal_coords, radius=0.015)
                else:
                    num_components = 0
                
                detailed_stats.append({
                    'best_iou': best_iou,
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
        size_ratio_iou_corr = np.corrcoef(size_ratios, best_ious)[0, 1]
        components_iou_corr = np.corrcoef(num_components_list, best_ious)[0, 1]
        
        # 按 maxIoU 区间分组统计
        # [0.1, 0.3) 区间：触达但不够好
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
    print("GT Leaf Proposal-Recall 诊断结果")
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
        
        print(f"\n连通性分析 (best proposal 的连通块数):")
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
        f.write("GT Leaf Proposal-Recall 诊断结果\n")
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
            
            f.write(f"\n连通性分析 (best proposal 的连通块数):\n")
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

