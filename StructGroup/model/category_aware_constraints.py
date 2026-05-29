"""
按类别设计约束项
- Leaf/Branch: 分离性约束（实例间最小间隔）
- Stem: 连续性约束（offset局部平滑）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree


class SeparationConstraintLoss(nn.Module):
    """
    Leaf/Branch的分离性约束：同一类别不同实例的endpoint/中心之间设定最小间隔
    优化版本：只对少量实例计算，使用GPU加速
    """
    def __init__(self, min_separation=0.02, margin=0.01, max_instances=32):
        """
        Args:
            min_separation: 最小间隔（米）
            margin: 惩罚的margin（小于min_separation-margin时开始惩罚）
            max_instances: 每个batch最多参与约束的实例数量（默认32）
        """
        super().__init__()
        self.min_separation = min_separation
        self.margin = margin
        self.max_instances = max_instances
    
    def forward(self, endpoints, instance_labels, instance_cls, 
                target_class_ids=[1, 2], ignore_label=-100):
        """
        Args:
            endpoints: (N, 3) 每个点的endpoint坐标（coords + offset）
            instance_labels: (N,) 实例标签
            instance_cls: (num_instances,) 每个实例的类别
            target_class_ids: 需要应用分离性约束的类别ID列表
            ignore_label: 忽略标签
        
        Returns:
            loss: 分离性约束loss
        """
        if endpoints.size(0) == 0:
            return endpoints.new_tensor(0.0)
        
        # 收集每个目标类别的实例中心
        class_centers = {}  # {class_id: [(inst_id, center), ...]}
        
        for class_id in target_class_ids:
            class_centers[class_id] = []
            
            # 找到所有属于该类别的实例
            for inst_id in range(instance_cls.size(0)):
                if instance_cls[inst_id].item() == class_id:
                    inst_mask = (instance_labels == inst_id)
                    if inst_mask.sum() > 0:
                        inst_endpoints = endpoints[inst_mask]
                        # 计算该实例的endpoint中心（使用均值）
                        center = inst_endpoints.mean(dim=0)
                        class_centers[class_id].append((inst_id, center))
        
        # 计算同一类别不同实例之间的分离性loss
        total_loss = 0.0
        total_pairs = 0
        
        for class_id, centers in class_centers.items():
            if len(centers) < 2:
                continue
            
            # 优化：只对少量实例计算（随机采样或选择前max_instances个）
            if len(centers) > self.max_instances:
                # 随机采样max_instances个实例
                indices = torch.randperm(len(centers), device=endpoints.device)[:self.max_instances]
                centers = [centers[i] for i in indices]
            
            # 计算所有实例对之间的距离（GPU加速）
            centers_list = [c[1] for c in centers]
            centers_tensor = torch.stack(centers_list)  # (M, 3)
            
            # 计算所有对之间的距离（使用torch.cdist，GPU加速）
            dists = torch.cdist(centers_tensor, centers_tensor)  # (M, M)
            
            # 只考虑上三角（避免重复）
            mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
            pair_dists = dists[mask]  # (M*(M-1)/2,)
            
            # 计算loss：如果距离 < min_separation，则惩罚
            # 使用hinge loss: max(0, min_separation - dist - margin)
            violations = F.relu(self.min_separation - pair_dists - self.margin)
            total_loss += violations.sum()
            total_pairs += violations.numel()
        
        if total_pairs > 0:
            return total_loss / total_pairs
        else:
            return endpoints.new_tensor(0.0)


class ContinuityConstraintLoss(nn.Module):
    """
    Stem的连续性约束：offset局部平滑/一致性约束
    优化版本：只对少量点计算，使用GPU加速的kNN
    """
    def __init__(self, k=16, smooth_weight=1.0, max_points=2048):
        """
        Args:
            k: kNN的k值
            smooth_weight: 平滑权重
            max_points: 最多参与计算的点数（默认2048）
        """
        super().__init__()
        self.k = k
        self.smooth_weight = smooth_weight
        self.max_points = max_points
    
    def forward(self, coords, offsets, instance_labels, instance_cls,
                stem_class_id=0, ignore_label=-100):
        """
        Args:
            coords: (N, 3) 点坐标
            offsets: (N, 3) offset预测
            instance_labels: (N,) 实例标签
            instance_cls: (num_instances,) 每个实例的类别
            stem_class_id: stem类别ID
            ignore_label: 忽略标签
        
        Returns:
            loss: 连续性约束loss
        """
        if coords.size(0) == 0:
            return coords.new_tensor(0.0)
        
        # 只对stem实例应用连续性约束
        stem_mask = torch.zeros(coords.size(0), dtype=torch.bool, device=coords.device)
        for inst_id in range(instance_cls.size(0)):
            if instance_cls[inst_id].item() == stem_class_id:
                stem_mask |= (instance_labels == inst_id)
        
        if stem_mask.sum() == 0:
            return coords.new_tensor(0.0)
        
        stem_coords = coords[stem_mask]
        stem_offsets = offsets[stem_mask]
        
        if stem_coords.size(0) < self.k + 1:
            return coords.new_tensor(0.0)
        
        # 优化：只对少量点计算（采样）
        if stem_coords.size(0) > self.max_points:
            # 均匀采样
            indices = torch.linspace(0, stem_coords.size(0) - 1, self.max_points).long()
            stem_coords = stem_coords[indices]
            stem_offsets = stem_offsets[indices]
        
        # 使用GPU加速的kNN（优先使用torch_cluster）
        try:
            from torch_cluster import knn
            batch = torch.zeros(stem_coords.size(0), dtype=torch.long, device=stem_coords.device)
            row, col = knn(stem_coords, stem_coords, self.k + 1, batch, batch)  # (num_edges,)
            
            # 排除自己（第一个邻居是自己）
            mask = row != col
            row = row[mask]
            col = col[mask]
            
            # 计算offset差异（向量化操作，GPU加速）
            center_offsets = stem_offsets[row]  # (num_edges, 3)
            neighbor_offsets = stem_offsets[col]  # (num_edges, 3)
            offset_diff = torch.norm(neighbor_offsets - center_offsets, dim=1)  # (num_edges,)
            
            if offset_diff.numel() > 0:
                return self.smooth_weight * offset_diff.mean()
            else:
                return coords.new_tensor(0.0)
        except ImportError:
            # 回退：使用torch的cdist计算kNN（GPU加速，但较慢）
            # 计算所有点对距离
            dists = torch.cdist(stem_coords, stem_coords)  # (M, M)
            # 找到每个点的k+1个最近邻（包括自己）
            _, knn_indices = torch.topk(dists, k=min(self.k + 1, stem_coords.size(0)), dim=1, largest=False)  # (M, k+1)
            
            # 排除自己，只保留k个邻居
            knn_indices = knn_indices[:, 1:]  # (M, k)
            
            # 向量化计算offset差异
            center_offsets = stem_offsets.unsqueeze(1)  # (M, 1, 3)
            neighbor_offsets = stem_offsets[knn_indices]  # (M, k, 3)
            offset_diff = torch.norm(neighbor_offsets - center_offsets, dim=2)  # (M, k)
            
            if offset_diff.numel() > 0:
                return self.smooth_weight * offset_diff.mean()
            else:
                return coords.new_tensor(0.0)


class OffsetDirectionConsistencyLoss(nn.Module):
    """
    Leaf的offset方向一致性约束：使整片叶内部投票方向更统一
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, offsets, instance_labels, instance_cls,
                leaf_class_id=1, ignore_label=-100):
        """
        Args:
            offsets: (N, 3) offset预测
            instance_labels: (N,) 实例标签
            instance_cls: (num_instances,) 每个实例的类别
            leaf_class_id: leaf类别ID
            ignore_label: 忽略标签
        
        Returns:
            loss: 方向一致性loss
        """
        if offsets.size(0) == 0:
            return offsets.new_tensor(0.0)
        
        # 只对leaf实例应用方向一致性约束
        leaf_mask = torch.zeros(offsets.size(0), dtype=torch.bool, device=offsets.device)
        for inst_id in range(instance_cls.size(0)):
            if instance_cls[inst_id].item() == leaf_class_id:
                leaf_mask |= (instance_labels == inst_id)
        
        if leaf_mask.sum() == 0:
            return offsets.new_tensor(0.0)
        
        leaf_offsets = offsets[leaf_mask]
        
        # 归一化offset方向
        offset_norms = torch.norm(leaf_offsets, dim=1, keepdim=True)
        offset_directions = leaf_offsets / (offset_norms + 1e-8)  # (M, 3)
        
        # 计算每个实例内部的方向一致性
        # 对于每个leaf实例，计算其内部所有点的方向向量的方差
        total_loss = 0.0
        total_instances = 0
        
        for inst_id in range(instance_cls.size(0)):
            if instance_cls[inst_id].item() != leaf_class_id:
                continue
            
            inst_mask = (instance_labels == inst_id)
            if inst_mask.sum() < 2:
                continue
            
            # 获取该实例的offset方向
            inst_global_indices = torch.nonzero(inst_mask).squeeze(1)
            inst_leaf_local_indices = []
            for gidx in inst_global_indices:
                # 找到gidx在leaf_mask中的位置
                leaf_indices = torch.nonzero(leaf_mask).squeeze(1)
                local_pos = (leaf_indices == gidx).nonzero()
                if local_pos.numel() > 0:
                    inst_leaf_local_indices.append(local_pos.item())
            
            if len(inst_leaf_local_indices) < 2:
                continue
            
            inst_leaf_local_indices = torch.tensor(inst_leaf_local_indices, dtype=torch.long, device=offsets.device)
            inst_directions = offset_directions[inst_leaf_local_indices]  # (P, 3)
            
            # 计算平均方向
            mean_direction = inst_directions.mean(dim=0)  # (3,)
            mean_direction = mean_direction / (torch.norm(mean_direction) + 1e-8)
            
            # 计算每个方向与平均方向的差异（1 - 点积）
            dot_products = (inst_directions * mean_direction.unsqueeze(0)).sum(dim=1)  # (P,)
            consistency = dot_products.mean()  # 越接近1越好
            
            # loss = 1 - consistency（方向越不一致，loss越大）
            total_loss += (1.0 - consistency)
            total_instances += 1
        
        if total_instances > 0:
            return total_loss / total_instances
        else:
            return offsets.new_tensor(0.0)

