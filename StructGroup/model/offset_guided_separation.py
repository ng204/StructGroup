"""
基于Offset方向的连接处分离
这是一个理论可靠的方法：利用offset预测的方向信息来分离粘连的器官
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OffsetGuidedSeparation(nn.Module):
    """
    基于Offset方向的智能分离
    
    核心思想：
    1. 叶片的offset指向叶片中心
    2. 茎秆的offset沿茎秆延伸
    3. 连接处的offset方向会突变
    4. 利用这个突变来分离
    """
    def __init__(self, channels):
        super().__init__()
        
        # Offset方向一致性网络
        self.direction_consistency = nn.Sequential(
            nn.Linear(channels + 6, 64),  # 特征 + offset方向 + 邻域方向
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出一致性分数
        )
        
        # 连接处检测器（基于offset方向变化）
        self.junction_detector = nn.Sequential(
            nn.Linear(3 + 3, 32),  # 当前offset + 邻域平均offset
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def compute_offset_divergence(self, offsets, coords, k=8, radius=0.02):
        """
        计算每个点的offset与其邻域offset的差异
        差异大 = 可能是连接处
        
        Args:
            offsets: (N, 3) offset向量
            coords: (N, 3) 坐标
            k: 邻域大小
            radius: 邻域半径
        Returns:
            divergence: (N,) offset divergence分数
        """
        N = offsets.size(0)
        
        # 归一化offset到单位向量
        offset_directions = F.normalize(offsets + 1e-6, dim=-1)  # (N, 3)
        
        # 计算每个点邻域的平均offset方向
        neighbor_offsets = []
        
        # 使用分块避免内存溢出
        batch_size = 2000
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            batch_coords = coords[i:end_i]
            
            # 找邻域
            dists = torch.cdist(batch_coords, coords)  # (batch, N)
            neighbors_mask = (dists < radius) & (dists > 0)  # 排除自己
            
            # 计算邻域平均方向
            batch_neighbor_dirs = []
            for j in range(len(batch_coords)):
                if neighbors_mask[j].sum() > 0:
                    neighbor_dirs = offset_directions[neighbors_mask[j]]
                    avg_dir = neighbor_dirs.mean(dim=0)
                    avg_dir = F.normalize(avg_dir.unsqueeze(0), dim=-1).squeeze(0)
                else:
                    avg_dir = offset_directions[i + j]
                batch_neighbor_dirs.append(avg_dir)
            
            neighbor_offsets.append(torch.stack(batch_neighbor_dirs))
        
        neighbor_offset_directions = torch.cat(neighbor_offsets, dim=0)  # (N, 3)
        
        # 计算方向差异（cosine distance）
        similarity = (offset_directions * neighbor_offset_directions).sum(dim=-1)  # (N,)
        divergence = 1 - similarity  # (N,) 范围[0, 2]，越大越不一致
        
        return divergence, neighbor_offset_directions
    
    def forward(self, point_features, coords, offsets, semantic_scores):
        """
        检测offset方向不一致的区域（连接处）
        
        Returns:
            enhanced_features: (N, C) 增强的特征
            junction_mask: (N,) 连接处分数(0-1)
            separation_mask: (N,) 应该被重点分离的点
        """
        # 1. 计算offset divergence
        divergence, neighbor_directions = self.compute_offset_divergence(
            offsets, coords, k=8, radius=0.02
        )  # (N,)
        
        # 2. 结合语义信息检测连接处
        # 连接处特征：offset方向突变 + 语义模糊（stem和leaf分数都不低）
        stem_score = semantic_scores[:, 0]
        leaf_score = semantic_scores[:, 1]
        branch_score = semantic_scores[:, 2]
        
        # 语义模糊度：多个类别分数都较高
        semantic_ambiguity = torch.min(
            stem_score + leaf_score,
            stem_score + branch_score
        )  # (N,)
        
        # 综合判断连接处
        offset_dir = F.normalize(offsets + 1e-6, dim=-1)
        combined_input = torch.cat([offset_dir, neighbor_directions], dim=-1)
        junction_score_raw = self.junction_detector(combined_input).squeeze(-1)  # (N,)
        
        # 结合offset divergence和语义模糊度
        junction_mask = junction_score_raw * divergence * semantic_ambiguity
        junction_mask = torch.clamp(junction_mask, 0, 1)
        
        # 3. 标记应该被重点分离的点
        # Offset divergence高 + 在leaf或branch区域 = 应该分离
        separation_mask = (divergence > 0.3) * ((leaf_score > 0.3) | (branch_score > 0.3))
        
        return junction_mask, separation_mask


class OffsetDirectionLoss(nn.Module):
    """
    Offset方向一致性损失
    鼓励同一实例内的offset方向一致，不同实例间的offset方向不同
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, offsets, instance_labels, coords):
        """
        Args:
            offsets: (N, 3) 预测的offset
            instance_labels: (N,) 实例标签
            coords: (N, 3) 坐标
        Returns:
            direction_loss: 方向一致性损失
        """
        offset_dirs = F.normalize(offsets + 1e-6, dim=-1)  # (N, 3)
        
        loss = 0.0
        unique_instances = torch.unique(instance_labels)
        
        for inst_id in unique_instances:
            if inst_id <= 0:  # 忽略背景
                continue
            
            inst_mask = instance_labels == inst_id
            inst_offsets = offset_dirs[inst_mask]
            
            if len(inst_offsets) < 2:
                continue
            
            # 同一实例内的offset方向应该相似
            # 计算方向的标准差（越小越一致）
            mean_dir = inst_offsets.mean(dim=0)
            mean_dir = F.normalize(mean_dir.unsqueeze(0), dim=-1)
            
            # 每个点与平均方向的差异
            similarities = (inst_offsets * mean_dir).sum(dim=-1)  # (M,)
            direction_variance = 1 - similarities.mean()
            
            loss += direction_variance
        
        loss = loss / (len(unique_instances) + 1e-6)
        
        return loss

