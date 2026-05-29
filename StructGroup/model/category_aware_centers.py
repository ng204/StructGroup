"""
按类别定义中心与offset监督目标
- Leaf/Branch: 使用鲁棒中心（medoid或剔除极端点后的中心）
- Stem: 使用多锚点（沿主轴均匀采样）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist


def compute_robust_center(coords, method='medoid', percentile=95.0):
    """
    计算鲁棒中心（用于leaf/branch）
    
    Args:
        coords: (N, 3) 点坐标
        method: 'medoid' 或 'trimmed_mean'
        percentile: 如果使用trimmed_mean，保留的百分位数
    
    Returns:
        center: (3,) 鲁棒中心坐标
    """
    if coords.size(0) == 0:
        return coords.new_zeros(3)
    
    if method == 'medoid':
        # 使用medoid（距离所有点距离之和最小的点）
        if coords.size(0) == 1:
            return coords[0]
        
        # 优化：对于大实例，使用采样近似medoid
        max_points_for_exact = 1000  # 超过1000点使用近似方法
        if coords.size(0) > max_points_for_exact:
            # 采样方法：随机采样或均匀采样
            sample_size = min(max_points_for_exact, coords.size(0))
            if coords.size(0) > sample_size:
                # 均匀采样
                indices = torch.linspace(0, coords.size(0) - 1, sample_size).long()
                coords_sampled = coords[indices]
            else:
                coords_sampled = coords
            
            # 在采样点上计算medoid
            coords_np = coords_sampled.cpu().numpy()
            dist_matrix = cdist(coords_np, coords_np)
            dist_sum = dist_matrix.sum(axis=1)
            medoid_idx_sampled = np.argmin(dist_sum)
            
            # 找到采样medoid在原始坐标中的最近点
            medoid_coord = coords_sampled[medoid_idx_sampled]
            dists_to_medoid = torch.norm(coords - medoid_coord.unsqueeze(0), dim=1)
            medoid_idx = dists_to_medoid.argmin()
            return coords[medoid_idx]
        else:
            # 小实例：精确计算
            coords_np = coords.cpu().numpy()
            dist_matrix = cdist(coords_np, coords_np)
            dist_sum = dist_matrix.sum(axis=1)
            medoid_idx = np.argmin(dist_sum)
            return coords[medoid_idx]
    
    elif method == 'trimmed_mean':
        # 剔除极端点后的均值
        if coords.size(0) <= 2:
            return coords.mean(dim=0)
        
        # 计算每个点到质心的距离
        centroid = coords.mean(dim=0)
        dists = torch.norm(coords - centroid.unsqueeze(0), dim=1)
        
        # 保留距离小于percentile的点
        threshold = torch.quantile(dists, percentile / 100.0)
        valid_mask = dists <= threshold
        if valid_mask.sum() == 0:
            return centroid
        return coords[valid_mask].mean(dim=0)
    
    else:
        # 默认使用均值
        return coords.mean(dim=0)


def compute_stem_anchors(coords, num_anchors=None, method='uniform'):
    """
    为stem计算多锚点（沿主轴均匀采样）
    
    Args:
        coords: (N, 3) stem点坐标
        num_anchors: 锚点数量，如果None则自动计算
        method: 'uniform' 或 'pca'
    
    Returns:
        anchors: (M, 3) 锚点坐标
        point_to_anchor: (N,) 每个点对应的最近锚点索引
    """
    if coords.size(0) == 0:
        return coords.new_zeros(0, 3), torch.zeros(0, dtype=torch.long, device=coords.device)
    
    if coords.size(0) == 1:
        return coords, torch.zeros(1, dtype=torch.long, device=coords.device)
    
    # 自动计算锚点数量：根据点数和stem长度
    if num_anchors is None:
        # 计算stem的"长度"（沿主轴的投影范围）
        coords_np = coords.cpu().numpy()
        centroid = coords_np.mean(axis=0)
        centered = coords_np - centroid
        
        # PCA找到主轴
        if centered.shape[0] >= 3:
            cov = np.cov(centered.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            main_axis = eigenvecs[:, -1]  # 最大特征值对应的方向
            
            # 计算沿主轴的范围
            projections = centered @ main_axis
            axis_length = projections.max() - projections.min()
            
            # 根据长度确定锚点数量：每0.05单位一个锚点，最少3个，最多20个
            num_anchors = max(3, min(20, int(axis_length / 0.05) + 1))
        else:
            num_anchors = min(3, coords.size(0))
    
    num_anchors = min(num_anchors, coords.size(0))
    
    if method == 'pca':
        # 使用PCA找到主轴，沿主轴均匀采样
        coords_np = coords.cpu().numpy()
        centroid = coords_np.mean(axis=0)
        centered = coords_np - centroid
        
        if centered.shape[0] >= 3:
            cov = np.cov(centered.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            main_axis = eigenvecs[:, -1]  # 最大特征值对应的方向
            
            # 沿主轴投影
            projections = centered @ main_axis
            proj_min, proj_max = projections.min(), projections.max()
            
            # 沿主轴均匀采样锚点
            anchor_projections = np.linspace(proj_min, proj_max, num_anchors)
            anchors = centroid + anchor_projections[:, None] * main_axis[None, :]
        else:
            # 点数太少，直接均匀采样
            indices = np.linspace(0, coords_np.shape[0] - 1, num_anchors).astype(int)
            anchors = coords_np[indices]
        
        anchors = torch.from_numpy(anchors).float().to(coords.device)
    
    elif method == 'uniform':
        # 简单均匀采样点
        indices = torch.linspace(0, coords.size(0) - 1, num_anchors).long()
        anchors = coords[indices]
    
    else:
        # 默认使用k-means聚类（简化版：使用均匀采样）
        indices = torch.linspace(0, coords.size(0) - 1, num_anchors).long()
        anchors = coords[indices]
    
    # 计算每个点对应的最近锚点
    # anchors: (M, 3), coords: (N, 3)
    dists = torch.cdist(coords, anchors)  # (N, M)
    point_to_anchor = dists.argmin(dim=1)  # (N,)
    
    return anchors, point_to_anchor


def compute_category_aware_offset_targets(coords, instance_labels, instance_pointnum, 
                                         instance_cls, ignore_label=-100,
                                         leaf_class_id=1, branch_class_id=2, stem_class_id=0,
                                         robust_center_method='medoid',
                                         stem_num_anchors=None):
    """
    按类别计算offset监督目标
    
    Args:
        coords: (N, 3) 点坐标
        instance_labels: (N,) 实例标签（0-based，-1表示背景）
        instance_pointnum: (num_instances,) 每个实例的点数
        instance_cls: (num_instances,) 每个实例的类别（0=stem, 1=leaf, 2=branch）
        ignore_label: 忽略标签
        leaf_class_id: leaf类别ID（语义分割中的ID，0-based）
        branch_class_id: branch类别ID
        stem_class_id: stem类别ID
        robust_center_method: leaf/branch使用的鲁棒中心方法
        stem_num_anchors: stem锚点数量（None则自动计算）
    
    Returns:
        offset_targets: (N, 3) offset监督目标
        center_info: dict，包含每个实例的中心信息
    """
    N = coords.size(0)
    num_instances = instance_pointnum.size(0)
    offset_targets = torch.zeros_like(coords)
    center_info = {}
    
    # 构建实例到点的映射
    instance_offsets = torch.cumsum(instance_pointnum, dim=0)
    instance_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=coords.device), 
                                  instance_offsets])
    
    for inst_id in range(num_instances):
        if instance_cls[inst_id] == ignore_label:
            continue
        
        # 获取该实例的所有点
        start_idx = instance_offsets[inst_id]
        end_idx = instance_offsets[inst_id + 1]
        inst_mask = (instance_labels == inst_id)
        inst_coords = coords[inst_mask]
        
        if inst_coords.size(0) == 0:
            continue
        
        inst_class = instance_cls[inst_id].item()
        
        if inst_class == stem_class_id:
            # Stem: 使用多锚点
            anchors, point_to_anchor = compute_stem_anchors(
                inst_coords, num_anchors=stem_num_anchors, method='pca')
            
            # 每个点指向其最近的锚点
            for local_idx, global_idx in enumerate(torch.nonzero(inst_mask).squeeze(1)):
                anchor_idx = point_to_anchor[local_idx].item()
                target_anchor = anchors[anchor_idx]
                offset_targets[global_idx] = target_anchor - coords[global_idx]
            
            center_info[inst_id] = {
                'type': 'multi_anchor',
                'anchors': anchors,
                'point_to_anchor': point_to_anchor
            }
        
        else:
            # Leaf/Branch: 使用鲁棒中心
            robust_center = compute_robust_center(inst_coords, method=robust_center_method)
            
            # 所有点指向鲁棒中心
            for global_idx in torch.nonzero(inst_mask).squeeze(1):
                offset_targets[global_idx] = robust_center - coords[global_idx]
            
            center_info[inst_id] = {
                'type': 'robust_center',
                'center': robust_center
            }
    
    return offset_targets, center_info

