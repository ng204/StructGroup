"""
叶柄-分支连接处分离模块
Petiole-Branch Junction Separation

针对Panax植物的特定结构：
- 茎秆 → 分支 → 叶柄 → 叶片
- 难点：叶柄末端与分支的连接处
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import MLP


class PetioleBranchSeparator(nn.Module):
    """
    叶柄-分支连接处智能分离
    
    关键思想:
    1. 叶柄是细长的（高elongation）
    2. 叶片主体是扁平的（高flatness）  
    3. 分支也是细长的
    4. 连接处：从叶片扁平→叶柄圆柱→分支圆柱
    """
    def __init__(self, channels):
        super().__init__()
        
        # 局部形状分类器：判断是叶片主体/叶柄/分支
        self.shape_classifier = nn.Sequential(
            nn.Linear(channels + 9, 128),  # 特征 + 局部几何
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3类：leaf_body, petiole, branch
            nn.Softmax(dim=-1)
        )
        
        # 连接强度预测：该点作为连接点的概率
        self.connection_predictor = nn.Sequential(
            nn.Linear(channels + 6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 特征增强（针对连接处）
        self.feature_enhance = MLP(channels, channels, norm_fn=None, num_layers=2)
    
    def compute_local_shape_features(self, coords, neighborhood_size=0.015):
        """
        计算局部形状特征
        
        关键特征:
        - Elongation: 细长度（主轴/次轴）
        - Flatness: 扁平度（次轴/第三轴）
        - Thickness: 最小尺寸
        
        Args:
            coords: (N, 3)
            neighborhood_size: 邻域大小（1.5cm）
        Returns:
            shape_features: (N, 9) [elongation, flatness, thickness, + bbox(6)]
        """
        N = coords.size(0)
        
        # 分批计算避免内存溢出
        batch_size = 3000
        all_shape_feats = []
        
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            batch_coords = coords[i:end_i]
            
            # 计算到所有点的距离
            dists = torch.cdist(batch_coords, coords)  # (batch, N)
            
            # 找邻域
            neighbors_mask = dists < neighborhood_size
            
            batch_feats = []
            for j in range(len(batch_coords)):
                neighbor_coords = coords[neighbors_mask[j]]
                
                if len(neighbor_coords) < 4:
                    # 邻域太小，用默认值
                    feat = torch.tensor([1.0, 1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], 
                                       device=coords.device)
                else:
                    # 计算PCA
                    centered = neighbor_coords - neighbor_coords.mean(dim=0)
                    cov = torch.mm(centered.T, centered) / len(neighbor_coords)
                    
                    try:
                        eigenvalues, _ = torch.linalg.eigh(cov)
                        eigenvalues = torch.abs(eigenvalues)
                        eigenvalues, _ = torch.sort(eigenvalues, descending=True)
                        
                        # 形状指标
                        elongation = eigenvalues[0] / (eigenvalues[1] + 1e-6)
                        flatness = eigenvalues[1] / (eigenvalues[2] + 1e-6)
                        thickness = torch.sqrt(eigenvalues[2])
                        
                        # Bbox
                        bbox = neighbor_coords.max(dim=0)[0] - neighbor_coords.min(dim=0)[0]
                        
                        feat = torch.cat([
                            elongation.unsqueeze(0),
                            flatness.unsqueeze(0), 
                            thickness.unsqueeze(0),
                            bbox,
                            bbox  # 重复以凑够9维
                        ])[:9]
                    except:
                        feat = torch.ones(9, device=coords.device) * 0.01
                
                batch_feats.append(feat)
            
            all_shape_feats.append(torch.stack(batch_feats))
        
        shape_features = torch.cat(all_shape_feats, dim=0)
        return shape_features
    
    def forward(self, point_features, coords, semantic_scores):
        """
        检测并处理叶柄-分支连接处
        
        Returns:
            enhanced_features: 增强的特征
            petiole_scores: 叶柄区域分数
            connection_scores: 连接处分数
        """
        # 1. 计算局部形状特征
        shape_features = self.compute_local_shape_features(coords)  # (N, 9)
        
        # 2. 分类：叶片主体/叶柄/分支
        combined = torch.cat([point_features, shape_features], dim=-1)
        shape_probs = self.shape_classifier(combined)  # (N, 3)
        # [0]: leaf_body概率, [1]: petiole概率, [2]: branch概率
        
        # 3. 预测连接强度
        # 连接处特征：叶柄概率高 + 靠近branch语义
        petiole_prob = shape_probs[:, 1]
        branch_semantic = semantic_scores[:, 2]  # branch的语义分数
        
        connection_input = torch.cat([
            point_features,
            petiole_prob.unsqueeze(-1),
            branch_semantic.unsqueeze(-1),
            coords,
            shape_features[:, :3]  # elongation, flatness, thickness
        ], dim=-1)
        
        connection_scores = self.connection_predictor(connection_input).squeeze(-1)  # (N,)
        
        # 4. 增强连接处的特征（更容易被分离）
        feature_delta = self.feature_enhance(point_features)
        # 连接处特征增强
        enhanced_features = point_features + connection_scores.unsqueeze(-1) * feature_delta
        
        return enhanced_features, petiole_prob, connection_scores


class PetioleAwareLoss(nn.Module):
    """
    叶柄感知损失
    对叶柄区域（特别是连接处）的offset预测加强约束
    """
    def __init__(self, petiole_weight=3.0):
        super().__init__()
        self.petiole_weight = petiole_weight
    
    def forward(self, offset_pred, offset_gt, petiole_scores, connection_scores, valid_mask):
        """
        Args:
            offset_pred: (N, 3)
            offset_gt: (N, 3)
            petiole_scores: (N,) 叶柄区域分数
            connection_scores: (N,) 连接处分数
            valid_mask: (N,)
        """
        # Offset loss
        offset_loss_pw = F.l1_loss(offset_pred, offset_gt, reduction='none').mean(dim=-1)  # (N,)
        
        # 叶柄区域加权（特别是连接处）
        # 连接处权重最高
        weights = 1.0 + (self.petiole_weight - 1.0) * petiole_scores * connection_scores
        
        weighted_loss = (offset_loss_pw * weights * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        
        return weighted_loss

