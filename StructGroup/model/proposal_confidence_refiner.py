"""
Proposal置信度细化器
为每个proposal预测一个细化的置信度，特别关注叶柄区域

核心思想：
1. 叶柄proposal的几何特征特殊（细长、小）
2. 这些proposal通常置信度较低
3. 通过几何特征识别并提升它们的置信度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProposalConfidenceRefiner(nn.Module):
    """
    基于几何特征细化proposal置信度
    特别针对叶柄等细长结构
    """
    def __init__(self, feature_dim):
        super().__init__()
        
        # 几何特征提取
        # 输入：proposal的统计特征
        self.geometry_encoder = nn.Sequential(
            nn.Linear(12, 64),  # 质心(3) + bbox(3) + 点数(1) + elongation(1) + 4个统计
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 语义特征编码
        self.semantic_encoder = nn.Sequential(
            nn.Linear(3, 16),  # 3个类别的平均语义分数
            nn.ReLU()
        )
        
        # 置信度调整预测器
        self.confidence_adjuster = nn.Sequential(
            nn.Linear(32 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # 输出[-1, 1]，作为调整因子
        )
        
        # 叶柄检测器（判断是否是叶柄proposal）
        self.petiole_detector = nn.Sequential(
            nn.Linear(32 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1，叶柄概率
        )
    
    def extract_proposal_geometry(self, coords_list):
        """
        为每个proposal提取几何特征
        
        Args:
            coords_list: list of (N_i, 3)，每个proposal的坐标
        Returns:
            geom_features: (K, 12) 几何特征
        """
        batch_geom = []
        
        for coords in coords_list:
            if len(coords) == 0:
                # 空proposal
                feat = torch.zeros(12, device=coords.device)
                batch_geom.append(feat)
                continue
            
            # 基础统计
            centroid = coords.mean(dim=0)  # (3,)
            bbox = coords.max(dim=0)[0] - coords.min(dim=0)[0]  # (3,)
            num_points = torch.tensor([len(coords)], dtype=torch.float32, device=coords.device)
            
            # PCA计算elongation
            centered = coords - centroid
            if len(coords) > 3:
                cov = torch.mm(centered.T, centered) / len(coords)
                try:
                    eigenvalues, _ = torch.linalg.eigh(cov)
                    eigenvalues = torch.abs(eigenvalues)
                    eigenvalues, _ = torch.sort(eigenvalues, descending=True)
                    
                    elongation = eigenvalues[0] / (eigenvalues[1] + 1e-6)
                    flatness = eigenvalues[1] / (eigenvalues[2] + 1e-6)
                    compactness = eigenvalues[2] / (eigenvalues[0] + 1e-6)
                except:
                    elongation = torch.tensor([1.0], device=coords.device)
                    flatness = torch.tensor([1.0], device=coords.device)
                    compactness = torch.tensor([1.0], device=coords.device)
            else:
                elongation = torch.tensor([1.0], device=coords.device)
                flatness = torch.tensor([1.0], device=coords.device)
                compactness = torch.tensor([1.0], device=coords.device)
            
            # 空间延展
            extent = bbox.norm()  # 总体大小
            
            # 组合特征
            feat = torch.cat([
                centroid,           # 3
                bbox,               # 3
                num_points,         # 1
                elongation.unsqueeze(0) if elongation.dim()==0 else elongation,  # 1
                flatness.unsqueeze(0) if flatness.dim()==0 else flatness,        # 1
                compactness.unsqueeze(0) if compactness.dim()==0 else compactness, # 1
                extent.unsqueeze(0),  # 1
                torch.tensor([len(coords) / (extent + 1e-6)], device=coords.device)  # 密度, 1
            ])  # 总共12维
            
            batch_geom.append(feat)
        
        geom_features = torch.stack(batch_geom)  # (K, 12)
        return geom_features
    
    def forward(self, proposals_idx, proposals_offset, coords_float, semantic_scores, cls_scores):
        """
        细化proposal的置信度
        
        Args:
            proposals_idx: proposal点索引
            proposals_offset: proposal偏移
            coords_float: 所有点坐标
            semantic_scores: (N, num_classes) 语义分数
            cls_scores: (K, num_classes+1) 原始分类分数
        Returns:
            refined_scores: (K, num_classes+1) 细化后的分数
            petiole_probs: (K,) 叶柄概率
        """
        K = proposals_offset.size(0) - 1
        
        # 为每个proposal提取坐标和语义统计
        coords_list = []
        semantic_stats = []
        
        for i in range(K):
            start = 0 if i == 0 else proposals_offset[i-1]
            end = proposals_offset[i]
            
            point_indices = proposals_idx[start:end, 1].long()
            prop_coords = coords_float[point_indices]
            prop_sem_scores = semantic_scores[point_indices]
            
            coords_list.append(prop_coords)
            
            # 语义统计：该proposal中各类别的平均分数
            if len(prop_sem_scores) > 0:
                sem_stat = prop_sem_scores.mean(dim=0)  # (num_classes,)
            else:
                sem_stat = torch.zeros(semantic_scores.size(1), device=semantic_scores.device)
            semantic_stats.append(sem_stat)
        
        # 提取几何特征
        geom_features = self.extract_proposal_geometry(coords_list)  # (K, 12)
        
        # 编码
        geom_encoded = self.geometry_encoder(geom_features)  # (K, 32)
        sem_encoded = self.semantic_encoder(torch.stack(semantic_stats))  # (K, 16)
        
        # 组合
        combined = torch.cat([geom_encoded, sem_encoded], dim=-1)  # (K, 48)
        
        # 预测置信度调整
        conf_adjustment = self.confidence_adjuster(combined)  # (K, 1)
        
        # 检测叶柄
        petiole_probs = self.petiole_detector(combined).squeeze(-1)  # (K,)
        
        # 细化分类分数
        # 对于几何特征像叶柄的proposal（细长、小），提升其leaf类别的分数
        refined_scores = cls_scores.clone()
        
        # 叶柄提升策略：
        # 如果petiole_prob高 且 点数少 且 elongation高 → 提升leaf分数
        for i in range(K):
            if petiole_probs[i] > 0.5:  # 判定为叶柄
                # 提升leaf类别(索引1)的分数
                refined_scores[i, 1] = refined_scores[i, 1] * (1 + 0.3 * petiole_probs[i])
        
        # 应用通用调整
        refined_scores = refined_scores * (1 + 0.2 * conf_adjustment)
        
        return refined_scores, petiole_probs

