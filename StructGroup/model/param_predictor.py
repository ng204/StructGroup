"""
自适应参数预测网络
自动预测每个类别的最优聚类参数（radius, score_thr, npoint_thr）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveParamPredictor(nn.Module):
    """
    轻量级参数预测网络
    根据场景特征自动预测每个类别的最优聚类参数
    """
    def __init__(self, feature_dim, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        
        # 场景级特征编码器（不使用BatchNorm，因为是1D输入）
        self.scene_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 每个类别的参数预测头
        # 每个类别预测3个参数偏移: radius_offset, score_thr_offset, npoint_offset
        self.param_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3)  # 3个参数
            ) for _ in range(num_classes)
        ])
        
        # 基础参数（可学习）- 调整为更激进的检测
        self.base_radius = nn.Parameter(torch.tensor([0.02, 0.006, 0.012]))  # leaf和branch更小
        self.base_score_thr = nn.Parameter(torch.tensor([0.01, 0.003, 0.005]))  # 更低
        self.base_npoint = nn.Parameter(torch.tensor([40.0, 2.0, 5.0]))  # leaf和branch更低
        
        # 参数范围 - 扩大让网络有更多调整空间
        self.radius_range = 0.015  # ±0.015（更大范围）
        self.score_range = 0.015   # ±0.015
        self.npoint_range = 30.0   # ±30
    
    def extract_scene_features(self, point_features, semantic_scores):
        """
        提取场景级别的全局特征
        
        Args:
            point_features: (N, C) 点特征
            semantic_scores: (N, num_classes) 语义分数
        Returns:
            scene_feature: (C,) 场景特征
        """
        # 方法1: 简单平均
        global_feat = point_features.mean(dim=0)
        
        # 方法2: 加入语义信息的加权平均（可选）
        # weights = semantic_scores.max(dim=-1)[0]  # (N,)
        # global_feat = (point_features * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
        
        return global_feat
    
    def forward(self, point_features, semantic_scores):
        """
        预测每个类别的最优参数
        
        Args:
            point_features: (N, C) 点特征
            semantic_scores: (N, num_classes) 语义分数
        Returns:
            predicted_params: dict with keys 'radius', 'score_thr', 'npoint_thr'
                              each is a list of length num_classes
        """
        # 1. 提取场景特征
        scene_feat = self.extract_scene_features(point_features, semantic_scores)  # (C,)
        
        # 2. 编码
        encoded_feat = self.scene_encoder(scene_feat)  # (64,)
        
        # 3. 为每个类别预测参数偏移
        radius_list = []
        score_thr_list = []
        npoint_list = []
        
        for class_id in range(self.num_classes):
            # 预测偏移量
            param_offsets = self.param_predictors[class_id](encoded_feat)  # (3,)
            
            # radius: base + tanh(offset) * range
            # tanh限制在[-1, 1]，确保参数在合理范围
            radius = self.base_radius[class_id] + torch.tanh(param_offsets[0]) * self.radius_range
            radius = torch.clamp(radius, 0.005, 0.05)  # 硬限制范围
            
            # score_thr: base + sigmoid(offset) * range - range/2
            # sigmoid(x)在[0,1]，通过-range/2使其在[-range/2, range/2]
            score_thr = self.base_score_thr[class_id] + (torch.sigmoid(param_offsets[1]) - 0.5) * self.score_range
            score_thr = torch.clamp(score_thr, 0.001, 0.05)
            
            # npoint_thr: base * exp(tanh(offset) * 0.5)
            # exp(tanh(offset)*0.5)在[0.6, 1.6]范围
            npoint = self.base_npoint[class_id] * torch.exp(torch.tanh(param_offsets[2]) * 0.5)
            npoint = torch.clamp(npoint, 1.0, 100.0)
            
            radius_list.append(radius)
            score_thr_list.append(score_thr)
            npoint_list.append(npoint)
        
        predicted_params = {
            'radius': radius_list,
            'score_thr': score_thr_list,
            'npoint_thr': npoint_list
        }
        
        return predicted_params
    
    def get_param_regularization_loss(self, predicted_params):
        """
        参数正则化损失
        鼓励参数在合理范围内
        """
        loss = 0.0
        
        # Radius正则化
        radius_tensor = torch.stack(predicted_params['radius'])
        # 惩罚过大或过小的radius
        loss += F.relu(radius_tensor - 0.03).sum() * 10  # 不要>0.03
        loss += F.relu(0.005 - radius_tensor).sum() * 10  # 不要<0.005
        
        # Score threshold正则化
        score_tensor = torch.stack(predicted_params['score_thr'])
        loss += F.relu(score_tensor - 0.03).sum() * 10
        loss += F.relu(0.001 - score_tensor).sum() * 10
        
        # Npoint正则化
        npoint_tensor = torch.stack(predicted_params['npoint_thr'])
        loss += F.relu(npoint_tensor - 100).sum() * 0.01
        loss += F.relu(1 - npoint_tensor).sum() * 0.01
        
        return loss


class BoundaryAwareLoss(nn.Module):
    """
    边界感知损失
    对边界/连接处的offset预测给予更高权重
    """
    def __init__(self, boundary_weight=2.0):
        super().__init__()
        self.boundary_weight = boundary_weight
    
    def forward(self, offset_pred, offset_gt, boundary_scores, valid_mask=None):
        """
        Args:
            offset_pred: (N, 3) 预测的offset
            offset_gt: (N, 3) GT offset
            boundary_scores: (N,) 边界分数(0-1)
            valid_mask: (N,) 有效点mask
        Returns:
            weighted_loss: 边界加权的offset loss
        """
        # 计算point-wise L1 loss
        point_wise_loss = F.l1_loss(offset_pred, offset_gt, reduction='none').mean(dim=-1)  # (N,)
        
        # 边界处加权（边界分数越高，权重越大）
        weights = 1.0 + (self.boundary_weight - 1.0) * boundary_scores  # (N,)
        
        # 应用valid mask
        if valid_mask is not None:
            weighted_loss = (point_wise_loss * weights * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        else:
            weighted_loss = (point_wise_loss * weights).mean()
        
        return weighted_loss


def compute_simple_curvature(coords, radius=0.02):
    """
    简化的曲率计算（不需要KNN库）
    使用局部点分布的方差作为曲率近似
    
    Args:
        coords: (N, 3) 点坐标
        radius: 邻域半径
    Returns:
        curvature_approx: (N, 3) 曲率近似（局部方差）
    """
    N = coords.size(0)
    curvature_features = []
    
    # 分批处理
    batch_size = 2000
    
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        batch_coords = coords[i:end_i]
        
        # 计算到所有点的距离
        dists = torch.cdist(batch_coords, coords)  # (batch, N)
        
        # 找邻域内的点
        neighbors_mask = dists < radius
        
        # 计算局部方差
        batch_curvatures = []
        for j in range(len(batch_coords)):
            neighbor_coords = coords[neighbors_mask[j]]
            if len(neighbor_coords) > 3:
                # 方差作为曲率近似
                local_var = neighbor_coords.var(dim=0)
            else:
                local_var = torch.zeros(3, device=coords.device)
            batch_curvatures.append(local_var)
        
        curvature_features.append(torch.stack(batch_curvatures))
    
    curvature_approx = torch.cat(curvature_features, dim=0)
    return curvature_approx


