"""
实例级自适应模块 - 革命性改进
每个proposal预测自己的最优参数，而不是整个类别共享
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import MLP
from .boundary_attention import ball_query_voxel


class ProposalFeatureExtractor(nn.Module):
    """为每个proposal提取丰富的特征，用于预测其最优参数"""
    def __init__(self, channels):
        super().__init__()
        
        # 几何特征编码器（现在输入是11维）
        self.geom_encoder = nn.Sequential(
            nn.Linear(11, 32),  # 质心(3) + bbox(3) + 主轴方向(3) + 细长比(1) + log点数(1)
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # 特征统计编码器（现在输入是3C+1维）
        feat_input_dim = channels * 3 + 1  # mean, std, max, correlation
        self.feat_encoder = nn.Sequential(
            nn.Linear(feat_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(16 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
    
    def forward(self, proposal_features, proposal_coords, proposal_point_features):
        """
        Args:
            proposal_features: (K, C) K个proposal的特征
            proposal_coords: list of (N_i, 3), 每个proposal的坐标
            proposal_point_features: list of (N_i, C), 每个proposal中所有点的特征
        Returns:
            enriched_features: (K, 16) 增强的proposal特征
        """
        batch_geom = []
        batch_feat = []
        
        for i, coords in enumerate(proposal_coords):
            N_i = coords.size(0)
            
            # === 几何统计：不让梯度回流到coords，稳定主干 ===
            with torch.no_grad():
                centroid = coords.mean(dim=0)  # (3,)
                bbox = coords.max(dim=0)[0] - coords.min(dim=0)[0]  # (3,)
                bbox = bbox.clamp(min=1e-6)  # 避免除零
            
                # 主方向（PCA）- 强制float32以保证AMP稳定性
                centered = coords - centroid.unsqueeze(0)
                
            if N_i > 1:
                centered_f32 = centered.float()
                # 添加数值稳定性：确保分母不为0，并检查 centered 是否全为0
                if centered_f32.abs().max() < 1e-8:
                    # 所有点重合，无法计算主方向
                    main_direction = torch.zeros(3, device=coords.device, dtype=coords.dtype)
                    main_direction[0] = 1.0
                    elongation = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
                else:
                    cov = torch.mm(centered_f32.T, centered_f32) / max(N_i - 1, 1.0)
                    # 添加小的正则化项避免奇异矩阵
                    cov = cov + torch.eye(3, device=cov.device, dtype=cov.dtype) * 1e-6
                    try:
                        # torch.linalg.eigh 本身返回升序特征值/向量：λ0 ≤ λ1 ≤ λ2
                        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                        # 检查是否有 NaN/Inf
                        if torch.any(torch.isnan(eigenvalues)) or torch.any(torch.isinf(eigenvalues)):
                            raise ValueError("NaN/Inf in eigenvalues")
                        main_direction = eigenvectors[:, -1].to(coords.dtype)  # 最大特征向量

                        # 方向符号规范化：绝对值最大的分量为正
                        max_idx = torch.argmax(main_direction.abs())
                        sign = torch.sign(main_direction[max_idx])
                        sign = torch.where(sign == 0, main_direction.new_tensor(1.0), sign)
                        main_direction = main_direction * sign

                        # 细长比：log(λ_max / λ_mid)，添加更严格的数值保护
                        lam = eigenvalues.clamp_min(1e-6)  # 提高最小值，避免比值过大
                        ratio = lam[-1] / lam[-2]
                        ratio = ratio.clamp(min=1.0, max=1e6)  # 限制比值范围，避免 log 溢出
                        elongation = torch.log(ratio).to(coords.dtype)
                    except Exception:
                        main_direction = torch.zeros(3, device=coords.device, dtype=coords.dtype)
                        main_direction[0] = 1.0
                        elongation = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
            else:
                # N_i<=1 时无几何形状信息，统一设为"无信息=0"
                main_direction = torch.zeros(3, device=coords.device, dtype=coords.dtype)
                main_direction[0] = 1.0
                elongation = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
            
            # 点的数量（归一化到合理范围）
            num_points = torch.tensor(N_i, dtype=coords.dtype, device=coords.device)
            log_num_points = torch.log(num_points + 1.0)  # log(点数+1)，范围更稳定
            
            # 组合几何特征：质心(3) + bbox(3) + 主方向(3) + 细长比(1) + log点数(1) = 11维
            geom_feat = torch.cat([
                centroid,
                bbox,
                main_direction,
                elongation.unsqueeze(0),
                log_num_points.unsqueeze(0)
            ])  # (11,)
            batch_geom.append(geom_feat)
            
            # === 点特征统计：从proposal_point_features中detach，避免干扰backbone ===
            feat = proposal_features[i].detach()                 # (C,) 已聚合特征
            pts_feat = proposal_point_features[i].detach()       # (N_i, C)

            with torch.no_grad():
                if pts_feat.numel() > 0 and N_i > 0:
                    feat_mean = pts_feat.mean(dim=0)        # (C,)
                    feat_std = pts_feat.std(dim=0, unbiased=False)  # (C,)
                    feat_max = pts_feat.max(dim=0)[0]       # (C,)
                    # 添加与原始proposal特征的相关性
                    norm_mean = torch.norm(feat_mean)
                    norm_feat = torch.norm(feat)
                    denom = norm_mean * norm_feat + 1e-6
                    # 额外检查：如果分母仍然太小，使用默认值
                    if denom < 1e-5:
                        feat_correlation = torch.tensor(0.0, device=feat.device, dtype=feat.dtype)
                    else:
                        feat_correlation = (feat_mean * feat).sum() / denom
                        feat_correlation = feat_correlation.clamp(-1.0, 1.0)  # 限制在合理范围
                else:
                    # 没有点时退化为原始feat
                    feat_mean = feat
                    feat_std = torch.zeros_like(feat)
                    feat_max = feat
                    feat_correlation = torch.tensor(1.0, device=feat.device, dtype=feat.dtype)

            # 使用更紧凑的特征表示：mean, std, max, correlation
            feat_stats = torch.cat([
                feat_mean,
                feat_std,
                feat_max,
                feat_correlation.unsqueeze(0)
            ], dim=0)  # (3C + 1,)
            batch_feat.append(feat_stats)
        
        geom_features = torch.stack(batch_geom)  # (K, 11)
        feat_features = torch.stack(batch_feat)  # (K, 3C+1)
        
        # 编码
        geom_encoded = self.geom_encoder(geom_features)  # (K, 16)
        feat_encoded = self.feat_encoder(feat_features)  # (K, 32)
        
        # 融合
        combined = torch.cat([geom_encoded, feat_encoded], dim=-1)  # (K, 48)
        enriched = self.fusion(combined)  # (K, 16)
        
        return enriched


class InstanceAdaptiveModule(nn.Module):
    """
    实例级自适应模块
    为每个proposal预测其最优的分类阈值和细化参数
    """
    def __init__(self, channels, instance_classes, score_scale=0.1, reliability_floor=0.2):
        super().__init__()
        self.instance_classes = instance_classes
        # 对应实例分类head的logits维度：instance_classes + 1（含背景）
        self.num_classes_with_bg = instance_classes + 1
        self.score_scale = score_scale  # 调整幅度上限（0~1），默认0.1更温和
        self.reliability_floor = reliability_floor  # 可靠性最小权重
        
        # Proposal特征提取
        self.proposal_extractor = ProposalFeatureExtractor(channels)
        
        # 为每个proposal预测调整因子（控制整体温度）
        self.score_adjuster = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # 输出0-1，用于映射到[-1,1]的可调节比例
        )

        # 为每个proposal预测每个类别的logit bias，实现真正“实例级阈值化”
        self.logit_bias = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes_with_bg),
            nn.Tanh()  # 输出[-1,1]，后续乘以score_scale
        )

        # 可学习的类别先验logits，用于reliability插值时的“回到先验分布”
        # 初始化为0，相当于均匀分布的logits；训练过程中可自适应调节
        self.prior_logits = nn.Parameter(torch.zeros(self.num_classes_with_bg), requires_grad=True)
        
        # 预测该proposal的可靠性（用于NMS/置信度加权）
        self.reliability_predictor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, inst_feats, proposals_idx, proposals_offset, coords_float, cls_scores, point_features):
        """
        为每个proposal预测自适应参数
        
        Args:
            inst_feats: (K, C) proposal特征
            proposals_idx: proposal点索引
            proposals_offset: (K+1,) proposal点数量的前缀和
            coords_float: 所有点的坐标
            cls_scores: (K, num_classes+1) 分类分数
            point_features: (N, C) 每个点的特征（与inst_feats聚合前对应）
        Returns:
            adjusted_scores: (K, num_classes+1) 调整后的分类分数
            reliability: (K,) 可靠性分数
        """
        K = inst_feats.size(0)
        
        # 边界情况：没有proposal时直接返回
        if K == 0:
            return cls_scores, torch.tensor([], device=cls_scores.device, dtype=cls_scores.dtype)
        
        # 调试期检查：cls_scores 应为 logits 而非 softmax 概率
        if self.training:
            with torch.no_grad():
                row_sum = cls_scores.sum(dim=1)
                if (
                    cls_scores.min() >= 0.0
                    and cls_scores.max() <= 1.0
                    and torch.allclose(row_sum, row_sum.new_ones(row_sum.shape), atol=1e-2)
                ):
                    raise ValueError(
                        "[InstanceAdaptiveModule] cls_scores looks like probabilities (softmax outputs); "
                        "expected raw logits. Please ensure you pass pre-softmax scores into InstanceAdaptiveModule."
                    )
        
        # 为每个proposal提取坐标
        proposal_coords_list = []
        proposal_point_features_list = []

        # 预计算质心作为空proposal的占位
        global_centroid = coords_float.mean(dim=0, keepdim=True)  # (1, 3)
        global_feat_mean = point_features.mean(dim=0, keepdim=True)  # (1, C)

        # 一次性根据offset计算每个proposal的长度，减少CPU/GPU同步
        # 显式detach+cpu，避免隐式行为差异
        sizes_raw = (proposals_offset[1:] - proposals_offset[:-1]).to(torch.int64).detach().cpu().tolist()
        # 负数 -> 0，避免 torch.split 崩溃；同时统计异常数量
        sizes = [int(s) if int(s) > 0 else 0 for s in sizes_raw]  # 长度为K的list[int]

        num_empty = sum(1 for s in sizes if s == 0)
        num_neg = sum(1 for s in sizes_raw if int(s) < 0)

        if K > 0 and self.training:
            if num_neg > 0:
                print(f"[InstanceAdaptiveModule] warning: {num_neg} negative sizes in proposals_offset, "
                      f"clamped to 0.")
            empty_ratio = num_empty / float(K)
            if empty_ratio > 1e-3:
                # 出现较多空proposal时给出提示，方便用户回查grouping逻辑
                print(f"[InstanceAdaptiveModule] warning: empty proposal ratio={empty_ratio:.4f} (>0.1%), "
                      f"please check grouping / proposals_offset.")

        # debug期一致性断言：offset长度之和必须与idx长度一致（在clamp之后检查）
        if self.training:
            total_size = sum(int(s) for s in sizes)
            assert total_size == proposals_idx.size(0), \
                f"[InstanceAdaptiveModule] offset mismatch after clamp: sum(sizes)={total_size} vs idx={proposals_idx.size(0)}"
        all_point_indices = proposals_idx[:, 1].long()
            
        # 训练期：已通过上面的断言保证 sum(sizes)==len(idx)，这里直接使用torch.split
        # 推理期：一旦出现不一致，使用安全fallback逻辑，避免崩溃
        total = sum(int(s) for s in sizes)
        L = all_point_indices.numel()

        if self.training or total == L:
            # 使用torch.split避免手写游标，逻辑更简洁可靠
            splits = torch.split(all_point_indices, sizes)
            for idx in splits:
                if idx.numel() == 0:
                    # 空proposal - 使用全局质心和全局特征均值
                    proposal_coords_list.append(global_centroid)
                    proposal_point_features_list.append(global_feat_mean)
                else:
                    idx = idx.clamp_(0, coords_float.size(0) - 1)
                    proposal_coords_list.append(coords_float[idx])
                    proposal_point_features_list.append(point_features[idx])
        else:
            # eval fallback：offset/idx不一致时，避免直接崩溃
            print(
                f"[InstanceAdaptiveModule] warning: sum(sizes)={total} != num_indices={L} at eval; "
                f"using safe fallback slicing for proposals."
            )
            cur = 0
            for s in sizes:
                s = max(int(s), 0)
                idx = all_point_indices[cur:cur + s]
                cur += s
                if idx.numel() == 0:
                    proposal_coords_list.append(global_centroid)
                    proposal_point_features_list.append(global_feat_mean)
                else:
                    idx = idx.clamp_(0, coords_float.size(0) - 1)
                    proposal_coords_list.append(coords_float[idx])
                    proposal_point_features_list.append(point_features[idx])

            # 如果仍有剩余索引（total < L），尽量分配给最后一个proposal，避免点完全丢失
            if cur < L:
                idx = all_point_indices[cur:].clamp_(0, coords_float.size(0) - 1)
                if len(proposal_coords_list) > 0:
                    # 追加到最后一个proposal
                    proposal_coords_list[-1] = torch.cat(
                        [proposal_coords_list[-1], coords_float[idx]], dim=0
                    )
                    proposal_point_features_list[-1] = torch.cat(
                        [proposal_point_features_list[-1], point_features[idx]], dim=0
                    )
                else:
                    # 极端情况：没有有效proposal，创建一个新的
                    proposal_coords_list.append(coords_float[idx])
                    proposal_point_features_list.append(point_features[idx])
        
        # 提取增强特征
        enriched_features = self.proposal_extractor(
            inst_feats, proposal_coords_list, proposal_point_features_list
        )  # (K, 16)
        
        # 预测调整因子
        score_adjustment = self.score_adjuster(enriched_features)  # (K, 1)
        reliability = self.reliability_predictor(enriched_features).squeeze(-1)  # (K,)
        
        # 调整分类分数 - 使用温和的温度缩放 + 每类logit bias
        
        # 1. 温度缩放（对所有类别，保持softmax分布）
        temp_delta = (score_adjustment - 0.5) * self.score_scale * 0.5  # (K, 1), 范围[-0.025, 0.025]
        temperature = 1.0 + temp_delta  # (K, 1), 范围[0.975, 1.025]
        temperature = temperature.clamp(min=0.95, max=1.05)  # 限制在很小的范围
        
        # 对logits应用轻微的温度缩放
        adjusted_scores = cls_scores / temperature  # (K, num_classes+1)
        
        # 2. 为每个proposal、每个类别预测logit bias（在logits层面，softmax前）
        #    这样stem/leaf/branch都能各自学习到不同的“实例级阈值”趋势
        class_bias = self.logit_bias(enriched_features) * self.score_scale  # (K, num_classes+1)

        if self.training:
            # 训练期强校验：维度必须一致，否则说明head约定不一致，应尽早暴露问题
            assert adjusted_scores.size(1) == class_bias.size(1), \
                f"[InstanceAdaptiveModule] class dim mismatch: cls_scores={adjusted_scores.size(1)} vs bias={class_bias.size(1)}"
            adjusted_scores = adjusted_scores + class_bias
        else:
            # 推理期做容错处理，防止尺寸不匹配导致崩溃（但仍打印提示）
            if adjusted_scores.size(1) != class_bias.size(1):
                print(
                    f"[InstanceAdaptiveModule] warning: class dim mismatch at eval: "
                    f"cls_scores={adjusted_scores.size(1)} vs bias={class_bias.size(1)}; "
                    f"applying safe truncation."
                )
                min_C = min(adjusted_scores.size(1), class_bias.size(1))
                adjusted_scores[:, :min_C] = adjusted_scores[:, :min_C] + class_bias[:, :min_C]
            else:
                adjusted_scores = adjusted_scores + class_bias
        
        # 3. 使用reliability在logit空间与“类别先验”插值（更像“回到先验分布”而非简单缩放）
        reliability_weight = self.reliability_floor + (1.0 - self.reliability_floor) * reliability  # (K,)
        rw = reliability_weight.unsqueeze(-1)  # (K,1)

        # 将可学习的prior_logits对齐到当前logits的通道数（防御性截断）
        # 对prior_logits做零均值约束，避免整体漂移（softmax对整体平移不敏感）
        prior = self.prior_logits - self.prior_logits.mean()
        prior_logits_full = prior.view(1, -1).to(adjusted_scores.device, adjusted_scores.dtype)
        if prior_logits_full.size(1) < adjusted_scores.size(1):
            # 若prior通道比当前少，用0填充剩余通道
            pad_C = adjusted_scores.size(1) - prior_logits_full.size(1)
            prior_logits_full = torch.cat(
                [prior_logits_full, prior_logits_full.new_zeros(1, pad_C)],
                dim=1
            )
        elif prior_logits_full.size(1) > adjusted_scores.size(1):
            prior_logits_full = prior_logits_full[:, :adjusted_scores.size(1)]

        adjusted_scores = rw * adjusted_scores + (1.0 - rw) * prior_logits_full

        # 对于空proposal，强制其logits退化为背景，避免产生噪声实例
        if num_empty > 0:
            empty_mask = torch.tensor([s <= 0 for s in sizes],
                                      device=adjusted_scores.device,
                                      dtype=torch.bool)
            if empty_mask.any():
                reliability = reliability.masked_fill(empty_mask, 0.0)
                # 将空proposal全部推向背景：非背景logit设为极小，背景logit为0
                # 这里背景索引按 SoftGroup 实例头约定：0..instance_classes-1 为前景，instance_classes 为背景
                bg_idx = self.instance_classes
                assert bg_idx < adjusted_scores.size(1), \
                    f"[InstanceAdaptiveModule] bg_idx={bg_idx} out of range for logits dim={adjusted_scores.size(1)}"
                adjusted_scores[empty_mask] = -1e4
                adjusted_scores[empty_mask, bg_idx] = 0.0
        
        return adjusted_scores, reliability


class ReliabilityLoss(nn.Module):
    """Reliability监督损失 - 让reliability预测与实际IoU相关"""
    def __init__(self):
        super().__init__()
    
    def forward(self, reliability, iou_scores):
        """
        Args:
            reliability: (K,) 预测的可靠性
            iou_scores: (K,) 真实IoU（需要外部计算）
        Returns:
            loss: scalar
        """
        if reliability.numel() == 0:
            return torch.tensor(0.0, device=reliability.device)
        # 清理 NaN/Inf
        reliability_clean = torch.where(torch.isnan(reliability) | torch.isinf(reliability),
                                       torch.zeros_like(reliability), reliability)
        iou_scores_clean = iou_scores.detach()
        iou_scores_clean = torch.where(torch.isnan(iou_scores_clean) | torch.isinf(iou_scores_clean),
                                      torch.zeros_like(iou_scores_clean), iou_scores_clean)
        # 限制在合理范围
        reliability_clean = reliability_clean.clamp(0.0, 1.0)
        iou_scores_clean = iou_scores_clean.clamp(0.0, 1.0)
        # 使用smooth L1 loss让reliability预测IoU
        loss = F.smooth_l1_loss(reliability_clean, iou_scores_clean)
        # 最终检查
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=reliability.device, dtype=reliability.dtype)
        return loss


class JunctionAwareAttention(nn.Module):
    """
    连接处感知注意力
    专门检测和处理叶柄-茎秆连接处
    """
    def __init__(self, channels, geo_base_radius=0.02):
        super().__init__()
        self.channels = channels
        # 几何邻域基础半径（真实世界单位），建议与点云真实尺度/voxel_size 对齐
        # 例如 coords 已在预处理阶段被校准为米，则 geo_base_radius=0.02 表示约 2cm 邻域
        self.geo_base_radius = geo_base_radius
        
        # 连接处检测网络: 特征(C) + 几何(9) + 语义歧义(1) + 坐标(3) = C+13
        self.junction_detector = nn.Sequential(
            nn.Linear(channels + 13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 方向预测网络（预测offset应该指向的方向）
        self.direction_predictor = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3D方向
        )
        
        # 特征细化
        self.feature_refine = MLP(channels, channels, norm_fn=None, num_layers=2)
    
    def compute_rich_geometry(self, coords, k=8, semantic_ambiguity=None):
        """
        计算较为真实的局部几何特征，而不是全局常数：
        - 局部方差
        - 曲率 / 线性度 / 平面度 / 球形度
        - 法向量一致性
        - 平均KNN距离（稀疏性）
        """
        N = coords.size(0)
        if N == 0:
            return torch.zeros(0, 9, device=coords.device, dtype=coords.dtype)

        with torch.no_grad():
            # 整个几何计算在 AMP 下强制使用 fp32，避免 linalg.eigh 的 half 不支持问题
            with torch.cuda.amp.autocast(enabled=False):
                coords_f32 = coords.to(torch.float32)

                # 对大场景做子采样以减轻 CPU Python 循环的开销
                # 优先按语义歧义 semantic_ambiguity 选点，其次才随机
                max_geo_points = 20000
                if N > max_geo_points:
                    if semantic_ambiguity is not None:
                        score = semantic_ambiguity.squeeze(-1)
                        Ns = min(N, max_geo_points)
                        sample_idx = torch.topk(score, k=Ns, largest=True, sorted=False).indices
                    else:
                        sample_idx = torch.randperm(N, device=coords.device)[:max_geo_points]
                else:
                    sample_idx = torch.arange(N, device=coords.device)
        
                coords_used = coords_f32[sample_idx]  # (Ns, 3)
                Ns = coords_used.size(0)

                # 1. 基于体素加速的近似 KNN（复用 boundary_attention 的 ball_query_voxel）
                #    基础半径使用 geo_base_radius（真实世界尺度），再按 k 做轻微缩放
                base_radius = float(self.geo_base_radius)
                radius = base_radius * (k / 8.0) ** (1.0 / 3.0)
                knn_idx, neighbors = ball_query_voxel(coords_used, radius=radius, max_neighbors=k)  # (Ns,k),(Ns,k,3)

                # 1.5 处理无效邻居：将 knn_idx<0 的邻居替换为中心点，使其对方差/协方差贡献为0
                center_pts = coords_used.unsqueeze(1).expand(-1, k, -1)   # (Ns,k,3)
                valid_nbr = knn_idx >= 0                                  # (Ns,k)
                neighbors_fill = torch.where(valid_nbr.unsqueeze(-1), neighbors, center_pts)  # (Ns,k,3)

                # 2. 局部方差（基于 neighbors_fill）
                centroid = neighbors_fill.mean(dim=1, keepdim=True)           # (Ns,1,3)
                centered = neighbors_fill - centroid                           # (Ns,k,3)
                local_var = centered.var(dim=1, unbiased=False)               # (Ns,3)

                # 3. 局部协方差 + PCA
                k_eff = max(k - 1, 1.0)
                cov = torch.matmul(centered.transpose(1, 2), centered) / k_eff  # (Ns, 3, 3)
                # 添加正则化避免奇异矩阵
                eye_reg = torch.eye(3, device=cov.device, dtype=cov.dtype).unsqueeze(0) * 1e-6
                cov = cov + eye_reg

                # 必须在 fp32 下调用 linalg.eigh
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # (Ns, 3), (Ns, 3, 3) 升序
                # 检查并清理 NaN/Inf
                eigenvalues = torch.where(torch.isnan(eigenvalues) | torch.isinf(eigenvalues),
                                         torch.ones_like(eigenvalues) * 1e-6, eigenvalues)
                lam = eigenvalues.clamp_min(1e-6)  # 提高最小值
                lam_sum = lam.sum(dim=1, keepdim=True).clamp_min(1e-6)

                # 曲率: 最小特征值占比
                curvature = (lam[:, 0:1] / lam_sum).clamp(0.0, 1.0)  # (Ns,1)
                # 线性度: (λ2 - λ1) / λ2
                linearity = ((lam[:, 2:3] - lam[:, 1:2]) / (lam[:, 2:3] + 1e-6)).clamp(0.0, 1.0)
                # 平面度: (λ1 - λ0) / λ2
                planarity = ((lam[:, 1:2] - lam[:, 0:1]) / (lam[:, 2:3] + 1e-6)).clamp(0.0, 1.0)
                # 球形度: λ0 / λ2
                sphericity = (lam[:, 0:1] / (lam[:, 2:3] + 1e-6)).clamp(0.0, 1.0)
        
                # 4. 法向量一致性（使用最小特征值对应的特征向量作为法向量）
                normals = eigenvectors[:, :, 0]                 # (Ns,3)
            
                # 防御性处理：使用 valid_nbr 掩码，避免无效邻居污染法向一致性
                knn_idx_safe = knn_idx.clamp(min=0)             # -1 -> 0，防止索引越界
                neighbor_normals = normals[knn_idx_safe]        # (Ns,k,3)
                center_normals = normals.unsqueeze(1)           # (Ns,1,3)
                cos_sim = torch.abs((neighbor_normals * center_normals).sum(dim=-1))  # (Ns,k)

                # 只对有效邻居求均值，避免无效邻居污染几何
                valid_nbr_f = valid_nbr.to(cos_sim.dtype)
                den_n = valid_nbr_f.sum(dim=1, keepdim=True).clamp_min(1.0)
                normal_consistency = (cos_sim * valid_nbr_f).sum(dim=1, keepdim=True) / den_n  # (Ns,1)
            
                # 5. 平均KNN距离（反映局部稀疏性），同样只在有效邻域上计算
                knn_dist = torch.norm(neighbors_fill - center_pts, dim=-1)  # (Ns,k)
                den_d = valid_nbr_f.sum(dim=1, keepdim=True).clamp_min(1.0)
                mean_knn_dist = (knn_dist * valid_nbr_f).sum(dim=1, keepdim=True) / den_d      # (Ns,1)

                # 组合成9维几何特征：3(var) + 1(curv) + 1(linear) + 1(plane) + 1(sphere) + 1(consistency) + 1(dist)
                geom_features_sample = torch.cat([
                    local_var,           # (Ns,3)
                    curvature,           # (Ns,1)
                    linearity,           # (Ns,1)
                    planarity,           # (Ns,1)
                    sphericity,          # (Ns,1)
                    normal_consistency,  # (Ns,1)
                    mean_knn_dist        # (Ns,1)
                ], dim=-1)  # (Ns,9)
            
                # 简单归一化，避免尺度差距过大（在采样子集上统计）
                mean = geom_features_sample.mean(dim=0, keepdim=True)
                std = geom_features_sample.std(dim=0, keepdim=True, unbiased=False) + 1e-6
                geom_features_sample = (geom_features_sample - mean) / std

                # === 将采样点几何传播到全体点：按粗 voxel 分组，使用全GPU张量运算（基于unique(dim=0)） ===
                voxel_size = 0.05
                sample_vox = torch.floor(coords_used / voxel_size).long()   # (Ns,3) on GPU
                all_vox = torch.floor(coords_f32 / voxel_size).long()       # (N,3)  on GPU

                # 将采样点和全体点的voxel坐标拼接，一次unique获取统一的voxel索引空间
                both_vox = torch.cat([sample_vox, all_vox], dim=0)          # (Ns+N,3)
                uniq_vox, inv_all = torch.unique(both_vox, dim=0, return_inverse=True)
                inv_sample = inv_all[:Ns]   # 采样点对应的voxel索引
                inv_points = inv_all[Ns:]   # 全体点对应的voxel索引

                V = uniq_vox.size(0)
                voxel_feats = torch.zeros(V, 9, device=coords.device, dtype=torch.float32)
                counts = torch.zeros(V, device=coords.device, dtype=torch.float32)

                # 在采样点上按 voxel 聚合几何特征（index_add 实现加和，再除以计数）
                voxel_feats.index_add_(0, inv_sample, geom_features_sample)
                ones = torch.ones(Ns, device=coords.device, dtype=torch.float32)
                counts.index_add_(0, inv_sample, ones)
                voxel_feats = voxel_feats / (counts.unsqueeze(-1) + 1e-6)  # (V,9)

                # 为全体点分配几何特征：如果所在voxel有采样点则用voxel_feats，否则用整体均值兜底
                default_feat = geom_features_sample.mean(dim=0, keepdim=True)  # (1,9)
                geom_features = default_feat.expand(N, -1).clone()             # (N,9)

                valid = counts[inv_points] > 0
                if valid.any():
                    geom_features[valid] = voxel_feats[inv_points[valid]]
        
            return geom_features.to(coords.dtype)
    
    def forward(self, point_features, coords, semantic_scores):
        """
        检测连接处并增强特征
        
        Args:
            point_features: (N, C)
            coords: (N, 3)
            semantic_scores: (N, num_classes) softmax后的
        Returns:
            enhanced_features: (N, C)
            junction_scores: (N,) 连接处分数
            predicted_directions: (N, 3) 预测的offset方向
        """
        num_classes = semantic_scores.size(1)
        
        # 1. 先根据语义分数计算歧义度，用于采样/辅助监督
        # 连接处特征：stem和leaf/branch的语义分数都较高的边界点
        # 安全地获取语义分数，避免越界
        stem_score = semantic_scores[:, 0] if num_classes > 0 else torch.zeros(coords.size(0), device=coords.device)
        leaf_score = semantic_scores[:, 1] if num_classes > 1 else torch.zeros(coords.size(0), device=coords.device)
        branch_score = semantic_scores[:, 2] if num_classes > 2 else torch.zeros(coords.size(0), device=coords.device)
        
        stem_leaf_product = stem_score * leaf_score  # (N,)
        stem_branch_product = stem_score * branch_score  # (N,)
        semantic_ambiguity = torch.max(stem_leaf_product, stem_branch_product).unsqueeze(-1)  # (N, 1)
        # 语义歧义只作为几何/连接处检测的提示信号，不反向修改语义头
        semantic_ambiguity_detached = semantic_ambiguity.detach()

        # 2. 计算几何特征（按歧义度优先采样）
        geom_features = self.compute_rich_geometry(coords, semantic_ambiguity=semantic_ambiguity_detached)  # (N, 9)
        
        # 3. 检测连接处：几何 + 语义歧义 + 原始特征（歧义使用detach后的版本）
        combined = torch.cat([point_features, geom_features, semantic_ambiguity_detached, coords], dim=-1)  # (N, C+13)
        junction_scores_raw = self.junction_detector(combined).squeeze(-1)  # (N,)
        
        # 3. 防止junction_scores退化：添加下限并与语义歧义对齐
        # 语义歧义高的点应该有更高的junction_score
        ambiguity_aligned = semantic_ambiguity_detached.squeeze(-1) * 0.3  # 缩放到合理范围（使用detach防止梯度泄露到语义头）
        junction_scores = torch.clamp(junction_scores_raw + ambiguity_aligned, min=0.01, max=1.0)
        
        # 4. 预测offset方向
        predicted_directions = self.direction_predictor(point_features)  # (N, 3)
        predicted_directions = F.normalize(predicted_directions, dim=-1)
        
        # 5. 基于连接处分数增强特征
        feature_delta = self.feature_refine(point_features)
        enhanced_features = point_features + junction_scores.unsqueeze(-1) * feature_delta
        
        return enhanced_features, junction_scores, predicted_directions


class JunctionAwareLoss(nn.Module):
    """连接处感知损失"""
    def __init__(self, junction_weight=3.0, direction_weight=0.5, junction_reg_weight=0.1):
        super().__init__()
        self.junction_weight = junction_weight
        self.direction_weight = direction_weight
        self.junction_reg_weight = junction_reg_weight  # junction_scores正则化权重
    
    def forward(self, offset_pred, offset_gt, junction_scores, predicted_directions, valid_mask, 
                semantic_ambiguity=None):
        """
        Args:
            offset_pred: (N, 3)
            offset_gt: (N, 3)
            junction_scores: (N,) 连接处分数
            predicted_directions: (N, 3) 预测的方向
            valid_mask: (N,)
            semantic_ambiguity: (N,) 可选，语义歧义分数用于辅助监督
        """
        # 1. 基础offset loss，连接处加权（权重使用detach，避免通过权重"投机"降低loss）
        offset_loss_pw = F.l1_loss(offset_pred, offset_gt, reduction='none').mean(dim=-1)  # (N,)
        # 检查并清理 NaN/Inf
        offset_loss_pw = torch.where(torch.isnan(offset_loss_pw) | torch.isinf(offset_loss_pw),
                                     torch.zeros_like(offset_loss_pw), offset_loss_pw)
        
        w = junction_scores.detach().clamp(0.0, 1.0)  # 确保权重在合理范围
        weights = 1.0 + (self.junction_weight - 1.0) * w
        valid_sum = valid_mask.sum()
        if valid_sum > 0:
            weighted_offset_loss = (offset_loss_pw * weights * valid_mask).sum() / (valid_sum + 1e-8)
        else:
            weighted_offset_loss = torch.tensor(0.0, device=offset_pred.device, dtype=offset_pred.dtype)
        
        # 检查 NaN/Inf
        if torch.isnan(weighted_offset_loss) or torch.isinf(weighted_offset_loss):
            weighted_offset_loss = torch.tensor(0.0, device=offset_pred.device, dtype=offset_pred.dtype)
        
        # 2. 方向一致性loss（预测方向应该与GT offset方向一致）
        # 使用eps避免offset为0时产生NaN
        gt_directions = F.normalize(offset_gt, dim=-1, eps=1e-6)  # (N, 3)
        pred_dirs_norm = F.normalize(predicted_directions, dim=-1, eps=1e-6)  # (N, 3)
        direction_similarity = (pred_dirs_norm * gt_directions).sum(dim=-1)  # (N,) cosine similarity
        direction_similarity = direction_similarity.clamp(-1.0, 1.0)  # 限制在有效范围
        direction_loss = (1 - direction_similarity).clamp(min=0)  # (N,)
        # 检查并清理 NaN/Inf
        direction_loss = torch.where(torch.isnan(direction_loss) | torch.isinf(direction_loss),
                                    torch.zeros_like(direction_loss), direction_loss)
        
        # 连接处的方向一致性更重要（同样使用detach的权重，避免梯度直接鼓励分数压低）
        den = (w * valid_mask).sum() + 1e-6
        if den > 1e-5:
            weighted_direction_loss = (direction_loss * w * valid_mask).sum() / den
        else:
            weighted_direction_loss = torch.tensor(0.0, device=offset_pred.device, dtype=offset_pred.dtype)
        
        # 检查 NaN/Inf
        if torch.isnan(weighted_direction_loss) or torch.isinf(weighted_direction_loss):
            weighted_direction_loss = torch.tensor(0.0, device=offset_pred.device, dtype=offset_pred.dtype)
        
        # 3. junction_scores辅助监督：与语义歧义对齐
        junction_reg_loss = torch.tensor(0.0, device=offset_pred.device, dtype=offset_pred.dtype)
        if semantic_ambiguity is not None:
            # 鼓励junction_scores与语义歧义正相关
            junction_reg_loss = F.mse_loss(junction_scores, semantic_ambiguity.detach())
            # 检查并清理 NaN/Inf
            if torch.isnan(junction_reg_loss) or torch.isinf(junction_reg_loss):
                junction_reg_loss = torch.tensor(0.0, device=offset_pred.device, dtype=offset_pred.dtype)
        
        total_loss = weighted_offset_loss + self.direction_weight * weighted_direction_loss + \
                     self.junction_reg_weight * junction_reg_loss
        
        # 最终检查：确保总损失不是 NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=offset_pred.device, dtype=offset_pred.dtype)
        
        return total_loss, {
            'offset_loss_base': weighted_offset_loss.item(),
            'direction_loss': weighted_direction_loss.item(),
            'junction_reg_loss': junction_reg_loss.item() if isinstance(junction_reg_loss, torch.Tensor) else junction_reg_loss
        }
