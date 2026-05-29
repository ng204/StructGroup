"""
轻量级边界注意力模块
专门用于增强叶柄-茎秆连接处的特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .blocks import MLP


def ball_query_voxel(coords, radius, max_neighbors):
    """
    近似的 ball-query 邻域搜索，使用体素哈希避免全局 cdist
    仅依赖 CPU 上的轻量循环，适用于 N<=5e4 的点云。

    Args:
        coords: (N, 3) float tensor, 任意 device
        radius: float, 邻域半径
        max_neighbors: int, 每个点最多邻居数（用于后续统计）
    Returns:
        knn_idx: (N, K) long tensor, 每行是邻居索引（包含自身，K<=max_neighbors）
        neighbors: (N, K, 3) float tensor, 对应邻居坐标
    """
    device = coords.device
    coords_cpu = coords.detach().cpu()  # 在 CPU 上构建体素哈希，避免 GPU 上 Python 循环
    N = coords_cpu.size(0)
    if N == 0:
        return torch.empty(0, 0, dtype=torch.long, device=device), torch.empty(0, 0, 3, device=device)

    radius = float(radius)
    radius2 = radius * radius

    # 体素大小取 radius，保证一个体素内点之间距离不超过 ~sqrt(3)*radius
    voxel_size = radius
    voxel_coords = torch.floor(coords_cpu / voxel_size).long()  # (N, 3)

    # 构建体素 -> 点索引 列表
    vox2points = {}
    for i in range(N):
        vx, vy, vz = int(voxel_coords[i, 0]), int(voxel_coords[i, 1]), int(voxel_coords[i, 2])
        key = (vx, vy, vz)
        if key not in vox2points:
            vox2points[key] = []
        vox2points[key].append(i)

    neighbor_lists = []
    max_found = 1

    max_candidates = max_neighbors * 8  # 给 topk 留一定冗余

    for i in range(N):
        vx, vy, vz = int(voxel_coords[i, 0]), int(voxel_coords[i, 1]), int(voxel_coords[i, 2])
        candidate_idx = []
        # 自身体素及其 26 个邻居体素
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (vx + dx, vy + dy, vz + dz)
                    if key in vox2points:
                        candidate_idx.extend(vox2points[key])

        if not candidate_idx:
            neighbor_lists.append([i])
            continue

        # 截断候选数量，避免极端情况下体素过密
        if len(candidate_idx) > max_candidates:
            candidate_idx = candidate_idx[:max_candidates]
        cand_tensor = torch.tensor(candidate_idx, dtype=torch.long)

        # 只在局部候选上计算距离，避免全局 cdist
        cand_xyz = coords_cpu[cand_tensor]  # (M, 3)
        diff = cand_xyz - coords_cpu[i].unsqueeze(0)  # (M, 3)
        dist2 = (diff * diff).sum(dim=-1)  # (M,)

        # 使用最近邻近似 KNN，不过度依赖 radius 过滤，避免邻域过小
        if dist2.numel() == 0:
            neighbor_lists.append([i])
            continue
        k_sel = min(max_neighbors, dist2.numel())
        _, topk_idx = torch.topk(dist2, k_sel, largest=False, sorted=False)
        cand_tensor = cand_tensor[topk_idx]

        neighbor_lists.append(cand_tensor.tolist())
        if len(cand_tensor) > max_found:
            max_found = len(cand_tensor)

    K = max_neighbors
    knn_idx = torch.empty(N, K, dtype=torch.long)
    for i, neigh in enumerate(neighbor_lists):
        if len(neigh) == 0:
            knn_idx[i].fill_(i)
        else:
            # 填充 / 截断到 K
            if len(neigh) >= K:
                row = neigh[:K]
            else:
                # 不重复邻居点，不足部分用自身索引填充
                row = neigh + [i] * (K - len(neigh))
            knn_idx[i] = torch.tensor(row, dtype=torch.long)

    knn_idx = knn_idx.to(device)
    neighbors = coords[knn_idx]  # (N, K, 3)
    return knn_idx, neighbors


class SimpleBoundaryAttention(nn.Module):
    """
    轻量级边界注意力模块
    通过局部几何特征检测边界/连接处，并增强这些区域的特征
    """
    def __init__(self,
                 in_channels,
                 k_list=[8, 16, 32],
                 use_semantic_fusion=True,
                 max_geo_points_train=4096,
                 max_geo_points_eval=20000,
                 geo_voxel_size=0.05):
        super().__init__()
        self.k_list = k_list  # 多尺度邻域
        self.use_semantic_fusion = use_semantic_fusion
        # 几何采样与传播相关参数
        self.max_geo_points_train = max_geo_points_train
        self.max_geo_points_eval = max_geo_points_eval
        self.geo_voxel_size = geo_voxel_size
        
        # 几何特征维度: 每个尺度 (local_var:3 + curvature:1 + linearity:1 + planarity:1 + sphericity:1 + boundary_indicator:1) = 8
        # 多尺度: 8 * 3 = 24
        geo_dim = 8 * len(k_list)
        
        # 边界检测网络 - 更深更宽，加入残差连接
        self.boundary_detector = nn.Sequential(
            nn.Linear(in_channels + geo_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),  # 增加一层
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 特征增强网络 - 使用残差连接
        self.feature_refine = MLP(in_channels, in_channels, norm_fn=nn.BatchNorm1d, num_layers=3)
        
        # 语义特征融合（如果启用）
        if use_semantic_fusion:
            self.semantic_fusion = nn.Sequential(
                nn.Linear(in_channels + 2, in_channels),  # +2 for [boundary_score, entropy]
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, in_channels)
            )
    
    def compute_knn(self, coords, k, radius):
        """
        使用体素哈希 + ball-query 的近似 KNN
        完全去掉全局 cdist，避免 O(N^2) 内存/时间开销。
        """
        # ball_query_voxel 直接返回固定 K= max_neighbors 的索引
        knn_idx, _ = ball_query_voxel(coords, radius=radius, max_neighbors=k)
        neighbors = coords[knn_idx]
        return knn_idx, neighbors
    
    def compute_local_geometry_single_scale(self, coords, k):
        """
        计算单尺度局部几何特征
        
        Returns:
            geo_features: (N, 8) 包含多种几何描述子
        """
        # 该函数在 AMP 下需要强制使用 fp32，否则 torch.linalg.eigh 不支持 half
        with torch.cuda.amp.autocast(enabled=False):
        # 1. 获取KNN邻域（通过 ball-query 近似）
        # 半径与 k 相关：k 越大，允许的半径略大
         base_radius = 0.02
        radius = base_radius * (k / 16.0) ** (1.0 / 3.0)
        # 确保几何计算使用 fp32
        coords_f32 = coords.to(torch.float32)
        knn_idx, neighbors = self.compute_knn(coords_f32, k, radius)
        
        # 2. 局部方差（fp32）
        local_var = neighbors.var(dim=1, unbiased=False)  # (N, 3)
        
        # 3. PCA分析（fp32, 无偏校正）
        centroid = neighbors.mean(dim=1, keepdim=True)
        centered = neighbors - centroid
        cov = torch.bmm(centered.transpose(1, 2), centered) / max(k - 1, 1)
        
            # 特征值分解（必须在 fp32 下）
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # sorted ascending: λ0 < λ1 < λ2
        eigenvalues = eigenvalues.clamp(min=1e-8)  # 避免除零
        
        lambda_sum = eigenvalues.sum(dim=1, keepdim=True) + 1e-8
        
        # 曲率: 最小特征值占比 (高=边界/角点)
        curvature = eigenvalues[:, 0:1] / lambda_sum
        
        # 线性度: (λ2 - λ1) / λ2 (高=线状结构，如茎秆)
        linearity = (eigenvalues[:, 2:3] - eigenvalues[:, 1:2]) / (eigenvalues[:, 2:3] + 1e-8)
        
        # 平面度: (λ1 - λ0) / λ2 (高=平面结构)
        planarity = (eigenvalues[:, 1:2] - eigenvalues[:, 0:1]) / (eigenvalues[:, 2:3] + 1e-8)
        
        # 球形度: λ0 / λ2 (高=各向同性，如连接点)
        sphericity = eigenvalues[:, 0:1] / (eigenvalues[:, 2:3] + 1e-8)
        
        # 4. 法向量一致性
        normals = eigenvectors[:, :, 0]  # 最小特征值对应的特征向量
        neighbor_normals = normals[knn_idx]
        consistency = torch.abs(torch.sum(normals.unsqueeze(1) * neighbor_normals, dim=2))
        normal_consistency = consistency.mean(dim=1, keepdim=True)
        
        # 组合特征 (N, 8)
        geo_features = torch.cat([
            local_var,           # (N, 3) 局部方差
            curvature,           # (N, 1) 曲率
            linearity,           # (N, 1) 线性度
            planarity,           # (N, 1) 平面度  
            sphericity,          # (N, 1) 球形度 - 连接处通常较高
        ], dim=-1)
        
        return geo_features, normal_consistency
    
    def compute_multiscale_geometry(self, coords):
        """
        计算多尺度几何特征 - 不同尺度捕获不同结构
        小尺度: 局部细节、噪声
        大尺度: 整体结构、连接处
        """
        all_features = []
        
        for k in self.k_list:
            geo_feat, normal_cons = self.compute_local_geometry_single_scale(coords, k)
            # 将法向量一致性转换为不一致性（边界处高）
            boundary_indicator = 1.0 - normal_cons
            all_features.append(torch.cat([geo_feat, boundary_indicator], dim=-1))
        
        # 拼接多尺度特征 (N, 8 * num_scales)
        multi_geo = torch.cat(all_features, dim=-1)
        
        # 可选归一化：使用无偏差估计，避免小 batch 下方差偏差
        std = multi_geo.std(dim=0, keepdim=True, unbiased=False)
        multi_geo = (multi_geo - multi_geo.mean(dim=0, keepdim=True)) / (std + 1e-8)
        
        return multi_geo
    
    @torch.no_grad()
    def _select_geo_samples(self, coords, semantic_probs=None, boundary_hint=None):
        """
        从全体点中选择一个子集用于几何计算（训练期更小，测试期可更大）。
        优先选择语义不确定/疑似边界点，避免纯随机采不到关键区域。
        """
        N = coords.size(0)
        if self.training:
            max_geo = self.max_geo_points_train
        else:
            max_geo = self.max_geo_points_eval

        if N <= max_geo:
            return None  # 全点计算几何

        score = None
        if boundary_hint is not None:
            score = boundary_hint.squeeze(-1)
        elif semantic_probs is not None:
            # 使用语义熵作为不确定性度量
            p = semantic_probs.clamp_min(1e-8)
            entropy = -(p * p.log()).sum(dim=1)  # (N,)
            score = entropy

        if score is None:
            # 退化情况：没有语义信息，用随机采样
            return torch.randperm(N, device=coords.device)[:max_geo]
        else:
            Ns = min(N, max_geo)
            return torch.topk(score, k=Ns, largest=True, sorted=False).indices

    @torch.no_grad()
    def _propagate_by_voxel(self, coords, sample_idx, sample_feat):
        """
        将采样点上的几何特征通过粗 voxel 映射传播到全体点。
        coords: (N,3), sample_idx: (Ns,), sample_feat: (Ns,D)
        返回: (N,D)
        """
        coords_f32 = coords.to(torch.float32)
        Ns = sample_idx.numel()
        N = coords.size(0)
        D = sample_feat.size(1)

        vs = float(self.geo_voxel_size)

        sample_vox = torch.floor(coords_f32[sample_idx] / vs).long()  # (Ns,3)
        all_vox = torch.floor(coords_f32 / vs).long()                  # (N,3)

        # 拼接后做unique，获取统一的voxel索引空间
        both = torch.cat([sample_vox, all_vox], dim=0)                 # (Ns+N,3)
        uniq_vox, inv = torch.unique(both, dim=0, return_inverse=True)
        inv_s = inv[:Ns]   # 采样点 -> voxel索引
        inv_a = inv[Ns:]   # 全体点 -> voxel索引

        V = uniq_vox.size(0)
        voxel_feat = sample_feat.new_zeros((V, D))
        counts = sample_feat.new_zeros((V,))

        # 对每个voxel聚合采样点特征
        voxel_feat.index_add_(0, inv_s, sample_feat)
        counts.index_add_(0, inv_s, sample_feat.new_ones((Ns,)))
        voxel_feat = voxel_feat / (counts.unsqueeze(-1) + 1e-6)

        # 默认特征：所有采样点几何的整体均值
        default = sample_feat.mean(dim=0, keepdim=True)  # (1,D)
        out = default.expand(N, -1).clone()              # (N,D)

        valid = counts[inv_a] > 0
        if valid.any():
            out[valid] = voxel_feat[inv_a[valid]]

        return out
    
    def forward(self, point_features, coords, semantic_probs=None):
        """
        前向传播
        
        Args:
            point_features: (N, C) 点特征
            coords: (N, 3) 坐标
            semantic_probs: (N, num_classes) 语义概率（可选，用于融合）
        Returns:
            enhanced_features: (N, C) 边界增强后的特征
            boundary_scores: (N,) 边界/连接处分数（0-1）
        """
        # 1. 选择用于几何计算的子集（训练期更少点，加速）
        with torch.no_grad():
            sample_idx = self._select_geo_samples(coords, semantic_probs=semantic_probs)
            if sample_idx is None:
                geo_features = self.compute_multiscale_geometry(coords)  # (N, geo_dim)
            else:
                geo_sample = self.compute_multiscale_geometry(coords[sample_idx])  # (Ns, geo_dim)
                geo_features = self._propagate_by_voxel(coords, sample_idx, geo_sample)  # (N, geo_dim)
        
        # 2. 组合特征预测边界分数
        combined_features = torch.cat([point_features, geo_features], dim=-1)
        boundary_scores = self.boundary_detector(combined_features).squeeze(-1)  # (N,)
        
        # 3. 基于边界分数增强特征
        feature_delta = self.feature_refine(point_features)
        attention_weights = boundary_scores.unsqueeze(-1)  # (N, 1)
        enhanced_features = point_features + attention_weights * feature_delta
        
        # 4. 语义特征融合（如果提供语义概率）
        if self.use_semantic_fusion and semantic_probs is not None:
            # 计算语义不确定性（高=边界/连接处）
            semantic_entropy = -(semantic_probs * torch.log(semantic_probs + 1e-8)).sum(dim=1, keepdim=True)  # (N, 1)
            num_classes = semantic_probs.size(1)
            semantic_entropy = semantic_entropy / torch.log(
                torch.tensor(float(num_classes), device=semantic_probs.device, dtype=semantic_probs.dtype)
            )  # 归一化到[0,1]

            # 使用 2 维融合特征: [boundary_score, semantic_entropy]
            boundary_semantic_2d = torch.cat(
                [boundary_scores.unsqueeze(-1), semantic_entropy], dim=-1
            )  # (N, 2)

            # 特征融合
            fused_input = torch.cat([enhanced_features, boundary_semantic_2d], dim=-1)
            enhanced_features = enhanced_features + self.semantic_fusion(fused_input)
        
        return enhanced_features, boundary_scores


class BoundaryAwareLoss(nn.Module):
    """
    边界感知损失
    对边界处的offset预测给予更高权重，并添加边界正则化
    """
    def __init__(self, boundary_weight=2.0, smooth_weight=0.1):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.smooth_weight = smooth_weight  # 边界平滑正则化权重
    
    def forward(self, offset_pred, offset_gt, boundary_scores, valid_mask=None, coords=None):
        """
        Args:
            offset_pred: (N, 3) 预测的offset
            offset_gt: (N, 3) GT offset
            boundary_scores: (N,) 边界分数
            valid_mask: (N,) 有效点mask（可选）
            coords: (N, 3) 坐标（可选，用于平滑正则化）
        Returns:
            weighted_loss: 边界加权的offset loss
        """
        # 基础L1 loss（不依赖边界分数）
        point_wise_loss = F.l1_loss(offset_pred, offset_gt, reduction='none').mean(dim=-1)  # (N,)
        
        # 边界分数在 loss 中不反向传播，避免退化到全 0
        boundary_detached = boundary_scores.detach()
        
        # 边界处加权（使用 detach 后的分数）
        weights = 1.0 + (self.boundary_weight - 1.0) * boundary_detached  # (N,)
        
        # 应用valid mask
        if valid_mask is not None:
            weighted_loss = (point_wise_loss * weights * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        else:
            weighted_loss = (point_wise_loss * weights).mean()
        
        # 边界平滑正则化：边界处的offset应该更平滑
        if coords is not None and self.smooth_weight > 0:
            N = coords.size(0)
            if N < 10000:  # 只对小点云计算，避免过慢
                # 只在高分边界点上采样做平滑（减少计算量）
                with torch.no_grad():
                    boundary_mask = boundary_detached > 0.5
                    boundary_idx = torch.nonzero(boundary_mask, as_tuple=False).view(-1)
                if boundary_idx.numel() > 0:
                    # 一次 ball-query，复用邻域索引
                    knn_idx, _ = ball_query_voxel(coords, radius=0.015, max_neighbors=8)  # (N, k)
                    boundary_knn_idx = knn_idx[boundary_idx]  # (Nb, k)
                    neighbor_offsets = offset_pred[boundary_knn_idx]  # (Nb, k, 3)
                    neighbor_scores = boundary_detached[boundary_knn_idx]  # (Nb, k)

                    # 非边界邻居提供平滑参考
                    weights_knn = (1.0 - neighbor_scores).unsqueeze(-1)  # (Nb, k, 1)
                    local_mean = (neighbor_offsets * weights_knn).sum(dim=1) / (weights_knn.sum(dim=1) + 1e-8)  # (Nb, 3)

                    target_offsets = offset_pred[boundary_idx]  # (Nb, 3)
                    smooth_loss = F.mse_loss(target_offsets, local_mean, reduction='mean')

                    weighted_loss = weighted_loss + self.smooth_weight * smooth_loss
        
        return weighted_loss


