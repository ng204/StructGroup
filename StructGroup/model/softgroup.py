import functools
from collections import OrderedDict

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..ops import (ball_query, bfs_cluster, get_mask_iou_on_cluster, get_mask_iou_on_pred,
                   get_mask_label, global_avg_pool, sec_max, sec_min, voxelization,
                   voxelization_idx)
from ..util import cuda_cast, force_fp32, rle_decode, rle_encode
from .blocks import MLP, ResidualBlock, UBlock
from .boundary_attention import SimpleBoundaryAttention, BoundaryAwareLoss
from .param_predictor import AdaptiveParamPredictor
from .offset_guided_separation import OffsetGuidedSeparation, OffsetDirectionLoss
from .proposal_confidence_refiner import ProposalConfidenceRefiner
from .query_mask_head import QueryMaskHead
from .instance_adaptive import InstanceAdaptiveModule, ReliabilityLoss, JunctionAwareAttention, JunctionAwareLoss
from .category_aware_centers import compute_category_aware_offset_targets
from .category_aware_constraints import (SeparationConstraintLoss, ContinuityConstraintLoss,
                                       OffsetDirectionConsistencyLoss)
from .category_aware_proposal_retention import retain_proposals_by_category


class SoftGroup(nn.Module):

    def __init__(self,
                 in_channels=3,
                 channels=32,
                 num_blocks=7,
                 semantic_only=False,
                 semantic_classes=20,
                 instance_classes=18,
                 semantic_weight=None,
                 sem2ins_classes=[],
                 ignore_label=-100,
                 with_coords=True,
                 grouping_cfg=None,
                 instance_voxel_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 fixed_modules=[]):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_blocks = num_blocks
        self.semantic_only = semantic_only
        self.semantic_classes = semantic_classes
        self.instance_classes = instance_classes
        self.semantic_weight = semantic_weight
        self.sem2ins_classes = sem2ins_classes
        self.ignore_label = ignore_label
        self.with_coords = with_coords
        self.grouping_cfg = grouping_cfg
        self.instance_voxel_cfg = instance_voxel_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fixed_modules = fixed_modules

        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        # backbone
        if with_coords:
            in_channels += 3
            self.in_channels += 3
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, channels, kernel_size=3, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())

        # point-wise prediction
        self.semantic_linear = MLP(channels, semantic_classes, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2)

        # topdown refinement path
        if not semantic_only:
            self.tiny_unet = UBlock([channels, 2 * channels], norm_fn, 2, block, indice_key_id=11)
            self.tiny_unet_outputlayer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())
            self.cls_linear = nn.Linear(channels, instance_classes + 1)
            self.mask_linear = MLP(channels, instance_classes + 1, norm_fn=None, num_layers=3)  # 从2层增加到3层，提升mask预测能力
            self.iou_score_linear = nn.Linear(channels, instance_classes + 1)

            # 每个proposal的自适应mask阈值预测头（B方案）
            self.mask_thr_head = nn.Sequential(
                nn.Linear(channels, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
            
            # Mask Quality Head：预测proposal与匹配GT的mask IoU（用于重排序）
            # 专门用于提升高IoU下的AP，只对leaf类别使用
            self.mask_quality_head = nn.Sequential(
                nn.Linear(channels, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # 输出 q ∈ [0, 1]，表示proposal的mask IoU质量
            )
            # 将阈值限制在一个合理区间内，避免过高/过低
            self.mask_t_min = 0.2
            self.mask_t_max = 0.8
        
        # 边界注意力模块（可选）
        self.use_boundary_attention = getattr(grouping_cfg, 'use_boundary_attention', False) if grouping_cfg else False
        if self.use_boundary_attention and not semantic_only:
            boundary_cfg = getattr(grouping_cfg, 'boundary_attention_cfg', {})
            use_semantic_fusion = boundary_cfg.get('use_semantic_fusion', True) if isinstance(boundary_cfg, dict) else getattr(boundary_cfg, 'use_semantic_fusion', True)
            self.boundary_attention = SimpleBoundaryAttention(
                channels, 
                k_list=getattr(boundary_cfg, 'k_list', [8, 16, 32]) if not isinstance(boundary_cfg, dict) else boundary_cfg.get('k_list', [8, 16, 32]),
                use_semantic_fusion=use_semantic_fusion
            )
            boundary_weight = getattr(boundary_cfg, 'boundary_weight', 2.0) if not isinstance(boundary_cfg, dict) else boundary_cfg.get('boundary_weight', 2.0)
            smooth_weight = getattr(boundary_cfg, 'smooth_weight', 0.1) if not isinstance(boundary_cfg, dict) else boundary_cfg.get('smooth_weight', 0.1)
            self.boundary_loss_fn = BoundaryAwareLoss(boundary_weight=boundary_weight, smooth_weight=smooth_weight)
        
        # 参数预测网络（可选）
        self.use_param_predictor = getattr(grouping_cfg, 'use_param_predictor', False) if grouping_cfg else False
        if self.use_param_predictor and not semantic_only:
            self.param_predictor = AdaptiveParamPredictor(channels, semantic_classes)
        
        # Offset方向一致性（可选）
        self.use_offset_direction = getattr(grouping_cfg, 'use_offset_direction_loss', False) if grouping_cfg else False
        if self.use_offset_direction:
            self.offset_direction_loss_fn = OffsetDirectionLoss()
            self.offset_guided_sep = OffsetGuidedSeparation(channels)
        
        # Proposal置信度细化器（专门提升叶柄等细长proposal的置信度）
        self.use_proposal_refiner = getattr(grouping_cfg, 'use_proposal_refiner', False) if grouping_cfg else False
        if self.use_proposal_refiner and not semantic_only:
            self.proposal_refiner = ProposalConfidenceRefiner(channels)
        
        # 实例级自适应模块（可选）
        self.use_instance_adaptive = getattr(grouping_cfg, 'use_instance_adaptive', False) if grouping_cfg else False
        if self.use_instance_adaptive and not semantic_only:
            ia_cfg = getattr(grouping_cfg, 'instance_adaptive_cfg', None)
            ia_kwargs = {}
            if ia_cfg:
                if isinstance(ia_cfg, dict):
                    ia_kwargs = ia_cfg
                else:
                    for key in ['score_scale', 'reliability_floor']:
                        if hasattr(ia_cfg, key):
                            ia_kwargs[key] = getattr(ia_cfg, key)
            self.instance_adaptive = InstanceAdaptiveModule(
                channels=channels,
                instance_classes=instance_classes,
                **ia_kwargs)
            self.reliability_loss_fn = ReliabilityLoss()
        
        # 连接处感知注意力（可选）
        self.use_junction_attention = getattr(grouping_cfg, 'use_junction_attention', False) if grouping_cfg else False
        if self.use_junction_attention and not semantic_only:
            # 几何基础半径（真实世界单位），建议与数据尺度/voxel_size 对齐
            geo_base_radius = getattr(grouping_cfg, 'junction_geo_radius', 0.02)
            self.junction_attention = JunctionAwareAttention(channels, geo_base_radius=geo_base_radius)
            self.junction_loss_fn = JunctionAwareLoss(
                junction_weight=getattr(grouping_cfg, 'junction_weight', 3.0),
                direction_weight=getattr(grouping_cfg, 'junction_direction_weight', 0.5),
                junction_reg_weight=getattr(grouping_cfg, 'junction_reg_weight', 0.1)
            )
        
        # ============ 按类别约束loss（新增） ============
        self.use_category_aware_constraints = getattr(train_cfg, 'use_category_aware_constraints', False) if train_cfg else False
        if self.use_category_aware_constraints and not semantic_only:
            # Leaf/Branch分离性约束
            min_separation = getattr(train_cfg, 'separation_min_distance', 0.02)
            separation_margin = getattr(train_cfg, 'separation_margin', 0.01)
            # 分离性约束：支持max_instances参数（GPU加速优化）
            separation_max_instances = getattr(train_cfg, 'separation_max_instances', 32)
            self.separation_loss_fn = SeparationConstraintLoss(
                min_separation=min_separation, margin=separation_margin,
                max_instances=separation_max_instances)
            
            # Stem连续性约束：支持max_points参数（GPU加速优化）
            continuity_k = getattr(train_cfg, 'continuity_knn', 16)
            continuity_weight = getattr(train_cfg, 'continuity_smooth_weight', 1.0)
            continuity_max_points = getattr(train_cfg, 'continuity_max_points', 2048)
            self.continuity_loss_fn = ContinuityConstraintLoss(
                k=continuity_k, smooth_weight=continuity_weight,
                max_points=continuity_max_points)
            
            # Leaf方向一致性约束
            self.direction_consistency_loss_fn = OffsetDirectionConsistencyLoss()
        
        # 按类别中心计算配置
        self.use_category_aware_centers = getattr(train_cfg, 'use_category_aware_centers', False) if train_cfg else False
        if self.use_category_aware_centers:
            self.robust_center_method = getattr(train_cfg, 'robust_center_method', 'medoid')
            self.stem_num_anchors = getattr(train_cfg, 'stem_num_anchors', None)

        # Query-based mask head（可选，替代聚类）
        self.use_query_mask_head = getattr(grouping_cfg, 'use_query_mask_head', False) if grouping_cfg else False
        if self.use_query_mask_head and not semantic_only:
            self.query_mask_head = QueryMaskHead(
                in_channels=channels,
                num_classes=semantic_classes,
                num_queries=getattr(grouping_cfg, 'query_num', 100),
                embed_dim=getattr(grouping_cfg, 'query_embed_dim', 128),
                num_heads=getattr(grouping_cfg, 'query_num_heads', 4),
                num_attn_layers=getattr(grouping_cfg, 'query_attn_layers', 2),
                no_object_weight=getattr(grouping_cfg, 'query_no_object_weight', 0.2),
            )

        self.init_weights()

        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()
        if not self.semantic_only:
            for m in [self.cls_linear, self.iou_score_linear]:
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, return_loss=False):
        if return_loss:
            return self.forward_train(**batch)
        else:
            return self.forward_test(**batch)

    @cuda_cast
    def forward_train(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                      semantic_labels, instance_labels, instance_pointnum, instance_cls,
                      pt_offset_labels, spatial_shape, batch_size, **kwargs):
        losses = {}
        if self.with_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, output_feats = self.forward_backbone(input, v2p_map)
        
        # 边界注意力增强（如果启用）
        boundary_scores = None
        if self.use_boundary_attention:
            semantic_probs = semantic_scores.softmax(dim=-1)
            output_feats, boundary_scores = self.boundary_attention(
                output_feats, coords_float, semantic_probs=semantic_probs)
        
        # 连接处感知注意力（如果启用）
        junction_scores = None
        predicted_directions = None
        semantic_ambiguity = None
        if self.use_junction_attention:
            semantic_probs = semantic_scores.softmax(dim=-1)
            output_feats, junction_scores, predicted_directions = self.junction_attention(
                output_feats, coords_float, semantic_probs)
            # 计算语义歧义用于辅助监督
            num_classes = semantic_probs.size(1)
            if num_classes >= 3:
                stem_leaf = semantic_probs[:, 0] * semantic_probs[:, 1]
                stem_branch = semantic_probs[:, 0] * semantic_probs[:, 2]
                semantic_ambiguity = torch.max(stem_leaf, stem_branch)
        
        # 参数预测（如果启用）
        predicted_params = None
        if self.use_param_predictor:
            predicted_params = self.param_predictor(output_feats, semantic_scores.softmax(dim=-1))

        # point wise losses
        if self.use_boundary_attention and boundary_scores is not None:
            # 使用边界感知的offset loss
            valid_mask = (semantic_labels != self.ignore_label).float()
            offset_loss = self.boundary_loss_fn(
                pt_offsets, pt_offset_labels, boundary_scores, valid_mask, coords=coords_float)
            # 替换原有的offset_loss
            point_wise_loss = self.point_wise_loss(semantic_scores, pt_offsets, semantic_labels,
                                                   instance_labels, pt_offset_labels,
                                                   coords_float=coords_float,
                                                   instance_pointnum=instance_pointnum,
                                                   instance_cls=instance_cls,
                                                   **kwargs)
            point_wise_loss['offset_loss'] = offset_loss  # 覆盖
            losses.update(point_wise_loss)
        else:
            point_wise_loss = self.point_wise_loss(semantic_scores, pt_offsets, semantic_labels,
                                                   instance_labels, pt_offset_labels,
                                                   coords_float=coords_float,
                                                   instance_pointnum=instance_pointnum,
                                                   instance_cls=instance_cls,
                                                   **kwargs)
            losses.update(point_wise_loss)

        # instance losses
        if not self.semantic_only and not self.use_query_mask_head:
            # 使用预测的参数或配置文件的参数
            grouping_cfg_to_use = self.grouping_cfg
            if predicted_params is not None:
                # 临时覆盖grouping_cfg中的参数
                grouping_cfg_to_use = type('obj', (object,), {
                    **{k: v for k, v in vars(self.grouping_cfg).items()},
                    'class_specific_radius': [p.item() for p in predicted_params['radius']],
                    'class_npoint_thr': [p.item() for p in predicted_params['npoint_thr']]
                })()
            
            proposals_idx, proposals_offset = self.forward_grouping(semantic_scores, pt_offsets,
                                                                    batch_idxs, coords_float,
                                                                    grouping_cfg_to_use)
            
            # ============ 训练阶段：禁用按类别保底的proposal保留 ============
            # 原因：proposal保留需要额外的forward_instance（翻倍计算量）+ CPU操作（.cpu(), .numpy()等）
            # 这些操作会严重拖慢训练，只在验证/推理阶段使用
            # 训练时只使用简单的截断策略
            
            # 如果proposal数量超过限制，进行截断
            if proposals_offset.shape[0] > self.train_cfg.max_proposal_num + 1:
                proposals_offset = proposals_offset[:self.train_cfg.max_proposal_num + 1]
                proposals_idx = proposals_idx[:proposals_offset[-1]]
                assert proposals_idx.shape[0] == proposals_offset[-1]
            
            inst_feats, inst_map = self.clusters_voxelization(
                proposals_idx,
                proposals_offset,
                output_feats,
                coords_float,
                rand_quantize=True,
                **self.instance_voxel_cfg)
            (instance_batch_idxs,
             cls_scores,
             iou_scores,
             mask_scores,
             proposal_feats,
             t_mask,
             mask_quality) = self.forward_instance(inst_feats, inst_map)
            
            reliability = None
            if self.use_instance_adaptive and proposals_offset.numel() > 1:
                cls_scores, reliability = self.instance_adaptive(
                    proposal_feats,
                    proposals_idx,
                    proposals_offset,
                    coords_float,
                    cls_scores,
                    output_feats)
            instance_loss = self.instance_loss(cls_scores, mask_scores, iou_scores, proposals_idx,
                                               proposals_offset, instance_labels, instance_pointnum,
                                               instance_cls, instance_batch_idxs, t_mask=t_mask, mask_quality=mask_quality)
            losses.update(instance_loss)
            
            # Reliability监督loss（如果启用instance_adaptive）
            if self.use_instance_adaptive and reliability is not None and reliability.numel() > 0:
                # 计算每个proposal的真实IoU作为监督目标
                proposals_idx_cuda = proposals_idx[:, 1].int().cuda()
                proposals_offset_cuda = proposals_offset.cuda()
                ious_on_cluster = get_mask_iou_on_cluster(
                    proposals_idx_cuda, proposals_offset_cuda, instance_labels, instance_pointnum)
                # 取每个proposal与最匹配GT的IoU
                if ious_on_cluster.numel() > 0:
                    gt_ious, _ = ious_on_cluster.max(dim=1)  # (K,)
                    # 对leaf类别的reliability loss给予更高权重
                    # 通过cls_scores预测类别，或从instance_cls获取GT类别
                    num_proposals = reliability.size(0)
                    reliability_loss_weight = reliability.new_ones(num_proposals)
                    # 尝试从cls_scores获取预测类别（用于加权）
                    if cls_scores.size(0) == num_proposals:
                        pred_classes = cls_scores.argmax(dim=1)  # (K,)
                        leaf_inds = (pred_classes == 1)  # leaf类别索引为1
                        leaf_reliability_weight = getattr(self.train_cfg, 'leaf_reliability_weight', 1.4)  # 默认1.4倍权重
                        if isinstance(leaf_reliability_weight, torch.Tensor):
                            leaf_reliability_weight = leaf_reliability_weight.item()
                        elif not isinstance(leaf_reliability_weight, (int, float)):
                            leaf_reliability_weight = float(leaf_reliability_weight)
                        reliability_loss_weight[leaf_inds] = reliability_loss_weight[leaf_inds] * leaf_reliability_weight
                    
                    # 计算加权reliability loss
                    # ReliabilityLoss内部使用smooth_l1_loss，我们手动计算逐样本loss然后加权
                    reliability_loss_raw = F.smooth_l1_loss(reliability, gt_ious.detach(), reduction='none')  # (K,)
                    reliability_loss = (reliability_loss_raw * reliability_loss_weight).mean()
                    reliability_weight = getattr(self.train_cfg, 'reliability_weight', 0.2)  # 降低到0.2
                    # 确保权重是Python标量
                    if isinstance(reliability_weight, torch.Tensor):
                        reliability_weight = reliability_weight.item()
                    elif not isinstance(reliability_weight, (int, float)):
                        reliability_weight = float(reliability_weight)
                    losses['reliability_loss'] = reliability_loss * reliability_weight

            # InstanceAdaptiveModule中prior_logits的L2正则，防止先验整体漂移
            if self.use_instance_adaptive:
                prior = self.instance_adaptive.prior_logits
                prior_reg = (prior ** 2).mean()
                prior_reg_weight = getattr(self.train_cfg, 'prior_reg_weight', 1e-4)
                # 确保权重是Python标量，而不是tensor或其他类型
                if isinstance(prior_reg_weight, torch.Tensor):
                    prior_reg_weight = prior_reg_weight.item()
                elif not isinstance(prior_reg_weight, (int, float)):
                    prior_reg_weight = float(prior_reg_weight)
                losses['prior_reg_loss'] = prior_reg * prior_reg_weight
            
            # 参数预测正则化loss（如果启用）
            if self.use_param_predictor and predicted_params is not None:
                param_reg_loss = self.param_predictor.get_param_regularization_loss(predicted_params)
                losses['param_reg_loss'] = param_reg_loss * 0.1  # 权重0.1
            
            # Offset方向一致性loss（如果启用）
            if self.use_offset_direction:
                direction_loss = self.offset_direction_loss_fn(pt_offsets, instance_labels, coords_float)
                # 从配置中读取权重，默认0.1（降低以提高训练稳定性）
                offset_dir_weight = getattr(self.grouping_cfg, 'offset_direction_loss_weight', 0.1)
                if isinstance(offset_dir_weight, torch.Tensor):
                    offset_dir_weight = offset_dir_weight.item()
                elif not isinstance(offset_dir_weight, (int, float)):
                    offset_dir_weight = float(offset_dir_weight)
                losses['offset_direction_loss'] = direction_loss * offset_dir_weight
            
            # 连接处感知loss（如果启用）
            if self.use_junction_attention and junction_scores is not None:
                valid_mask = (instance_labels != self.ignore_label).float()
                junction_loss, junction_loss_dict = self.junction_loss_fn(
                    pt_offsets, pt_offset_labels, junction_scores, predicted_directions,
                    valid_mask, semantic_ambiguity)
                junction_weight = getattr(self.train_cfg, 'junction_weight', 0.15)  # 降低到0.15
                # 确保权重是Python标量
                if isinstance(junction_weight, torch.Tensor):
                    junction_weight = junction_weight.item()
                elif not isinstance(junction_weight, (int, float)):
                    junction_weight = float(junction_weight)
                losses['junction_loss'] = junction_loss * junction_weight
            
            # ============ 训练阶段：低频启用按类别约束loss（GPU加速版本）============
            # 优化：使用GPU加速的低频约束loss，只对少量实例/点计算
            # - 分离性约束：只对最多32个实例计算，使用torch.cdist（GPU加速）
            # - 连续性约束：只对最多2048个点计算，使用torch_cluster.knn或torch.cdist（GPU加速）
            # - 方向一致性约束：相对简单，保留
            # 通过constraint_loss_freq控制计算频率（默认每4个batch计算一次）
            if self.use_category_aware_constraints:
                constraint_loss_freq = getattr(self.train_cfg, 'constraint_loss_freq', 4)  # 默认每4个batch计算一次
                if isinstance(constraint_loss_freq, torch.Tensor):
                    constraint_loss_freq = constraint_loss_freq.item()
                elif not isinstance(constraint_loss_freq, (int, float)):
                    constraint_loss_freq = int(constraint_loss_freq)
                
                # 获取当前iteration（从kwargs中获取，如果没有则设为0）
                current_iter = kwargs.get('iter', 0) if isinstance(kwargs, dict) else 0
                if isinstance(current_iter, torch.Tensor):
                    current_iter = current_iter.item()
                elif not isinstance(current_iter, (int, float)):
                    current_iter = int(current_iter) if hasattr(current_iter, '__int__') else 0
                
                # 低频触发：每constraint_loss_freq个iteration计算一次
                if constraint_loss_freq > 0 and current_iter % constraint_loss_freq == 0:
                    # 计算endpoints（coords + offset）
                    endpoints = coords_float + pt_offsets
                    
                    # 分离性约束loss（只对leaf/branch）
                    if hasattr(self, 'separation_loss_fn'):
                        separation_loss = self.separation_loss_fn(
                            endpoints, instance_labels, instance_cls,
                            target_class_ids=[1, 2]  # leaf=1, branch=2
                        )
                        separation_loss_weight = getattr(self.train_cfg, 'separation_loss_weight', 0.1)
                        if isinstance(separation_loss_weight, torch.Tensor):
                            separation_loss_weight = separation_loss_weight.item()
                        elif not isinstance(separation_loss_weight, (int, float)):
                            separation_loss_weight = float(separation_loss_weight)
                        losses['separation_loss'] = separation_loss * separation_loss_weight
                    else:
                        losses['separation_loss'] = coords_float.new_tensor(0.0)
                    
                    # 连续性约束loss（只对stem）
                    if hasattr(self, 'continuity_loss_fn'):
                        continuity_loss = self.continuity_loss_fn(
                            coords_float, pt_offsets, instance_labels, instance_cls,
                            stem_class_id=0
                        )
                        continuity_loss_weight = getattr(self.train_cfg, 'continuity_loss_weight', 0.1)
                        if isinstance(continuity_loss_weight, torch.Tensor):
                            continuity_loss_weight = continuity_loss_weight.item()
                        elif not isinstance(continuity_loss_weight, (int, float)):
                            continuity_loss_weight = float(continuity_loss_weight)
                        losses['continuity_loss'] = continuity_loss * continuity_loss_weight
                    else:
                        losses['continuity_loss'] = coords_float.new_tensor(0.0)
                    
                    # 方向一致性约束loss（只对leaf）
                    if hasattr(self, 'direction_consistency_loss_fn'):
                        direction_consistency_loss = self.direction_consistency_loss_fn(
                            pt_offsets, instance_labels, instance_cls,
                            leaf_class_id=1
                        )
                        direction_consistency_loss_weight = getattr(self.train_cfg, 'direction_consistency_loss_weight', 0.05)
                        if isinstance(direction_consistency_loss_weight, torch.Tensor):
                            direction_consistency_loss_weight = direction_consistency_loss_weight.item()
                        elif not isinstance(direction_consistency_loss_weight, (int, float)):
                            direction_consistency_loss_weight = float(direction_consistency_loss_weight)
                        losses['direction_consistency_loss'] = direction_consistency_loss * direction_consistency_loss_weight
                    else:
                        losses['direction_consistency_loss'] = coords_float.new_tensor(0.0)
                else:
                    # 不计算约束loss的iteration，设为0
                    losses['separation_loss'] = coords_float.new_tensor(0.0)
                    losses['continuity_loss'] = coords_float.new_tensor(0.0)
                    losses['direction_consistency_loss'] = coords_float.new_tensor(0.0)
            else:
                losses['separation_loss'] = coords_float.new_tensor(0.0)
                losses['continuity_loss'] = coords_float.new_tensor(0.0)
                losses['direction_consistency_loss'] = coords_float.new_tensor(0.0)
        
        # Query-based mask head 分支（替代聚类和proposal实例头）
        if not self.semantic_only and self.use_query_mask_head:
            qm_outputs = self.query_mask_head(output_feats, coords_float)
            qm_losses = self.query_mask_head.compute_losses(qm_outputs, instance_labels, self.ignore_label)
            # 大幅增加loss权重，确保query head能学习
            qm_losses['qm_cls_loss'] = qm_losses['qm_cls_loss'] * 10.0   # 分类权重x10
            qm_losses['qm_mask_bce'] = qm_losses['qm_mask_bce'] * 0.1    # mask bce权重降低
            qm_losses['qm_mask_dice'] = qm_losses['qm_mask_dice'] * 0.5  # mask dice权重降低
            losses.update(qm_losses)

        return self.parse_losses(losses)

    def point_wise_loss(self, semantic_scores, pt_offsets, semantic_labels, instance_labels,
                        pt_offset_labels, coords_float=None, instance_pointnum=None, instance_cls=None, **kwargs):
        losses = {}
        if self.semantic_weight:
            weight = torch.tensor(self.semantic_weight, dtype=torch.float, device='cuda')
        else:
            weight = None
        semantic_loss = F.cross_entropy(
            semantic_scores, semantic_labels, weight=weight, ignore_index=self.ignore_label)
        losses['semantic_loss'] = semantic_loss

        pos_inds = instance_labels != self.ignore_label
        if pos_inds.sum() == 0:
            offset_loss = 0 * pt_offsets.sum()
        else:
            # ============ 训练阶段：低频启用medoid中心计算（仅对小实例）============
            # 优化：在训练后期（epoch 80+）对小实例（<1000点）启用medoid中心计算
            # 大实例仍使用质心，避免O(N²)的cdist计算
            # 通过use_category_aware_centers和enable_medoid_in_training控制
            use_category_aware_centers = getattr(self.train_cfg, 'use_category_aware_centers', False)
            enable_medoid_in_training = getattr(self.train_cfg, 'enable_medoid_in_training', False)
            current_epoch = kwargs.get('epoch', 0) if isinstance(kwargs, dict) else 0
            if isinstance(current_epoch, torch.Tensor):
                current_epoch = current_epoch.item()
            elif not isinstance(current_epoch, (int, float)):
                current_epoch = int(current_epoch) if hasattr(current_epoch, '__int__') else 0
            
            # 训练后期（epoch 80+）且启用medoid时，对小实例使用medoid中心
            if (use_category_aware_centers and enable_medoid_in_training and 
                current_epoch >= 80 and hasattr(self, 'robust_center_method')):
                from softgroup.model.category_aware_centers import compute_category_aware_offset_targets
                
                # 只对小实例使用medoid，大实例仍使用质心
                # 这里简化处理：对所有实例都尝试使用medoid，但medoid计算内部会采样大实例
                try:
                    offset_targets, _ = compute_category_aware_offset_targets(
                        coords_float, instance_labels, instance_pointnum, instance_cls,
                        ignore_label=self.ignore_label,
                        leaf_class_id=1, branch_class_id=2, stem_class_id=0,
                        robust_center_method=self.robust_center_method,
                        stem_num_anchors=None
                    )
                    # 只对positive samples计算loss
                    offset_loss = F.l1_loss(
                        pt_offsets[pos_inds], offset_targets[pos_inds], reduction='sum') / pos_inds.sum()
                except Exception:
                    # 如果medoid计算失败，回退到质心
                    offset_loss = F.l1_loss(
                        pt_offsets[pos_inds], pt_offset_labels[pos_inds], reduction='sum') / pos_inds.sum()
            else:
                # 训练阶段：使用原始offset监督目标（质心），这是纯GPU操作，速度快
             offset_loss = F.l1_loss(
                pt_offsets[pos_inds], pt_offset_labels[pos_inds], reduction='sum') / pos_inds.sum()
        losses['offset_loss'] = offset_loss
        return losses

    @force_fp32(apply_to=('cls_scores', 'mask_scores', 'iou_scores'))
    def instance_loss(self, cls_scores, mask_scores, iou_scores, proposals_idx, proposals_offset,
                      instance_labels, instance_pointnum, instance_cls, instance_batch_idxs,
                      t_mask=None, mask_quality=None):
        if proposals_idx.size(0) == 0 or (instance_cls != self.ignore_label).sum() == 0:
            cls_loss = cls_scores.sum() * 0
            mask_loss = mask_scores.sum() * 0
            iou_score_loss = iou_scores.sum() * 0
            return dict(
                cls_loss=cls_loss,
                mask_loss=mask_loss,
                iou_score_loss=iou_score_loss,
                num_pos=mask_loss,
                num_neg=mask_loss)

        losses = {}

        # 保留每一行mask对应的proposal id，用于自适应mask阈值和软Dice
        proposal_ids = proposals_idx[:, 0].long().cuda()
        proposals_idx = proposals_idx[:, 1].int().cuda()
        proposals_offset = proposals_offset.cuda()

        # cal iou of clustered instance
        ious_on_cluster = get_mask_iou_on_cluster(proposals_idx, proposals_offset, instance_labels,
                                                  instance_pointnum)

        # filter out background instances
        fg_inds = (instance_cls != self.ignore_label)
        fg_instance_cls = instance_cls[fg_inds]
        fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

        # assign proposal to gt idx. -1: negative, 0 -> num_gts - 1: positive
        num_proposals = fg_ious_on_cluster.size(0)
        num_gts = fg_ious_on_cluster.size(1)
        assigned_gt_inds = fg_ious_on_cluster.new_full((num_proposals, ), -1, dtype=torch.long)

        # overlap > thr on fg instances are positive samples
        # 支持类别特定的pos_iou_thr（对leaf类别使用更低的阈值）
        base_pos_iou_thr = self.train_cfg.pos_iou_thr
        leaf_pos_iou_thr = getattr(self.train_cfg, 'leaf_pos_iou_thr', base_pos_iou_thr)
        if isinstance(leaf_pos_iou_thr, torch.Tensor):
            leaf_pos_iou_thr = leaf_pos_iou_thr.item()
        elif not isinstance(leaf_pos_iou_thr, (int, float)):
            leaf_pos_iou_thr = float(leaf_pos_iou_thr)
        
        max_iou, argmax_iou = fg_ious_on_cluster.max(1)
        
        # 为每个proposal确定匹配阈值：如果是leaf类别GT，使用leaf_pos_iou_thr
        leaf_class_id = 1  # leaf类别索引
        pos_thr_per_proposal = torch.full((num_proposals,), base_pos_iou_thr, 
                                          device=max_iou.device, dtype=max_iou.dtype)
        # 找到每个proposal对应的GT类别
        for p_idx in range(num_proposals):
            best_gt_idx = argmax_iou[p_idx]
            if best_gt_idx < num_gts:
                gt_cls = fg_instance_cls[best_gt_idx]
                if gt_cls == leaf_class_id:
                    pos_thr_per_proposal[p_idx] = leaf_pos_iou_thr
        
        pos_inds = max_iou >= pos_thr_per_proposal
        assigned_gt_inds[pos_inds] = argmax_iou[pos_inds]

        # allow low-quality proposals with best iou to be as positive sample
        # in case pos_iou_thr is too high to achieve
        match_low_quality = getattr(self.train_cfg, 'match_low_quality', False)
        min_pos_thr = getattr(self.train_cfg, 'min_pos_thr', 0)
        if isinstance(min_pos_thr, torch.Tensor):
            min_pos_thr = min_pos_thr.item()
        elif not isinstance(min_pos_thr, (int, float)):
            min_pos_thr = float(min_pos_thr)
        if match_low_quality:
            gt_max_iou, gt_argmax_iou = fg_ious_on_cluster.max(0)
            for i in range(num_gts):
                if gt_max_iou[i] >= min_pos_thr:
                    assigned_gt_inds[gt_argmax_iou[i]] = i

        # compute cls loss. follow detection convention: 0 -> K - 1 are fg, K is bg
        labels = fg_instance_cls.new_full((num_proposals, ), self.instance_classes)
        pos_inds = assigned_gt_inds >= 0
        labels[pos_inds] = fg_instance_cls[assigned_gt_inds[pos_inds]]
        
        # 类别特定的loss权重：对leaf类别（class_id=1）给予更高权重
        leaf_class_id = 1  # leaf类别索引
        cls_weight = labels.new_ones(num_proposals)
        leaf_inds = (labels == leaf_class_id)
        leaf_cls_weight = getattr(self.train_cfg, 'leaf_cls_weight', 1.5)  # 默认1.5倍权重
        if isinstance(leaf_cls_weight, torch.Tensor):
            leaf_cls_weight = leaf_cls_weight.item()
        elif not isinstance(leaf_cls_weight, (int, float)):
            leaf_cls_weight = float(leaf_cls_weight)
        cls_weight[leaf_inds] = leaf_cls_weight
        
        cls_loss = F.cross_entropy(cls_scores, labels, weight=None, reduction='none')
        cls_loss = (cls_loss * cls_weight).mean()
        losses['cls_loss'] = cls_loss

        # compute mask loss
        mask_cls_label = labels[instance_batch_idxs.long()]
        slice_inds = torch.arange(
            0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)
        mask_scores_sigmoid_slice = mask_scores.sigmoid()[slice_inds, mask_cls_label]
        mask_label = get_mask_label(proposals_idx, proposals_offset, instance_labels, instance_cls,
                                    instance_pointnum, ious_on_cluster, self.train_cfg.pos_iou_thr)
        mask_label_weight = (mask_label != -1).float()

        # ==== 基于软阈值的可导mask监督（Soft Dice）====
        if t_mask is not None:
            # 取出当前类别对应的原始mask logits（未sigmoid）
            raw_mask_logits = mask_scores[slice_inds, mask_cls_label]  # (R,)
            # 将proposal级阈值t_mask映射到每一行mask_scores
            row_t = t_mask.squeeze(-1)[proposal_ids].to(raw_mask_logits.device)  # (R,)
            # 温度系数tau，可从配置中读取，否则使用默认0.05（更sharp的mask）
            tau = getattr(self.train_cfg, 'mask_soft_tau', 0.05)
            # 确保tau是Python标量
            if isinstance(tau, torch.Tensor):
                tau = tau.item()
            elif not isinstance(tau, (int, float)):
                tau = float(tau)
            soft_mask = torch.sigmoid((raw_mask_logits - row_t) / tau)  # (R,)

            # 使用0/1 GT（忽略-1标记的无效位置）
            gt_mask = (mask_label > 0).float()
            valid = mask_label_weight  # (R,)

            inter = (soft_mask * gt_mask * valid).sum()
            union = (soft_mask * valid).sum() + (gt_mask * valid).sum() + 1e-6
            soft_dice = 2 * inter / union
            soft_dice_loss = 1.0 - soft_dice

            soft_dice_weight = getattr(self.train_cfg, 'mask_soft_dice_weight', 0.2)  # 降低到0.2
            # 确保权重是Python标量
            if isinstance(soft_dice_weight, torch.Tensor):
                soft_dice_weight = soft_dice_weight.item()
            elif not isinstance(soft_dice_weight, (int, float)):
                soft_dice_weight = float(soft_dice_weight)
            losses['mask_soft_dice_loss'] = soft_dice_loss * soft_dice_weight

        # BCE保持原逻辑，但对leaf类别给予更高权重
        mask_label[mask_label == -1.] = 0.5  # any value is ok
        # 为leaf类别的mask loss添加额外权重
        mask_loss_weight = mask_label_weight.clone()
        leaf_mask_inds = (mask_cls_label == 1)  # leaf类别索引为1
        leaf_mask_weight = getattr(self.train_cfg, 'leaf_mask_weight', 1.3)  # 默认1.3倍权重
        if isinstance(leaf_mask_weight, torch.Tensor):
            leaf_mask_weight = leaf_mask_weight.item()
        elif not isinstance(leaf_mask_weight, (int, float)):
            leaf_mask_weight = float(leaf_mask_weight)
        mask_loss_weight[leaf_mask_inds] = mask_loss_weight[leaf_mask_inds] * leaf_mask_weight
        
        # ============ 基于IoU的mask loss权重（提升高IoU样本的边界精度）============
        # 对高IoU的proposal给予更高的mask loss权重，让模型更关注高质量样本的边界精度
        use_iou_weighted_mask_loss = getattr(self.train_cfg, 'use_iou_weighted_mask_loss', True)
        iou_weight_power = getattr(self.train_cfg, 'iou_weight_power', 1.5)  # IoU权重幂次，默认1.5
        if isinstance(iou_weight_power, torch.Tensor):
            iou_weight_power = iou_weight_power.item()
        elif not isinstance(iou_weight_power, (int, float)):
            iou_weight_power = float(iou_weight_power)
        
        if use_iou_weighted_mask_loss and pos_inds.sum() > 0:
            # 向量化实现：为每个proposal计算IoU权重，然后映射到mask points
            # 1. 为所有proposal创建IoU权重表（negative samples权重为1.0）
            proposal_iou_weights = max_iou.new_ones(num_proposals)  # (num_proposals,)
            pos_proposal_ious = max_iou[pos_inds]  # (num_pos,)
            # IoU权重：IoU越高，权重越大（使用幂次函数，范围：[1.0, 2.0]）
            pos_iou_weights = (pos_proposal_ious ** iou_weight_power) * (2.0 - 1.0) + 1.0  # (num_pos,)
            proposal_iou_weights[pos_inds] = pos_iou_weights
            
            # 2. 将proposal权重映射到mask points（向量化操作，避免Python循环）
            proposal_ids_for_mask = proposal_ids  # (R,) proposal id for each mask point
            # 使用索引操作，将proposal权重映射到mask points
            iou_weights = proposal_iou_weights[proposal_ids_for_mask]  # (R,)
            
            mask_loss_weight = mask_loss_weight * iou_weights
        
        # ============ 边界感知Mask Loss（仅对leaf类别应用）============
        # 对leaf类别的边界点给予更高权重，提升高IoU下的边界精度
        use_boundary_aware_mask_loss = getattr(self.train_cfg, 'use_boundary_aware_mask_loss', True)
        if use_boundary_aware_mask_loss and leaf_mask_inds.sum() > 0:
            # 只对leaf类别的mask points计算边界权重
            boundary_mask_weight = getattr(self.train_cfg, 'boundary_mask_weight', 2.5)  # 边界点权重，默认2.5
            if isinstance(boundary_mask_weight, torch.Tensor):
                boundary_mask_weight = boundary_mask_weight.item()
            elif not isinstance(boundary_mask_weight, (int, float)):
                boundary_mask_weight = float(boundary_mask_weight)
            
            high_iou_boundary_weight = getattr(self.train_cfg, 'high_iou_boundary_weight', 1.5)  # 高IoU样本的边界点额外权重，默认1.5
            if isinstance(high_iou_boundary_weight, torch.Tensor):
                high_iou_boundary_weight = high_iou_boundary_weight.item()
            elif not isinstance(high_iou_boundary_weight, (int, float)):
                high_iou_boundary_weight = float(high_iou_boundary_weight)
            
            # 检测GT mask的边界点（使用简单的梯度方法，GPU友好）
            # GT mask: mask_label > 0 表示正样本点
            gt_mask_binary = (mask_label > 0).float()  # (R,)
            
            # 边界检测：对于每个proposal，计算其mask的边界
            # 方法：使用proposal内的点，如果点的邻居中有正样本和负样本，则该点是边界点
            # 为了高效，我们使用简化的方法：对于每个proposal，计算mask的"边缘"
            # 边缘 = mask内部点且其预测值接近0.5的点（边界模糊区域）
            # 或者更简单：mask预测值与GT差异较大的点（边界误差大的点）
            
            # 改进的边界检测方法：结合预测误差和GT mask的梯度信息
            # 方法1：预测与GT差异较大的点（边界误差大的点）
            pred_gt_diff = torch.abs(mask_scores_sigmoid_slice - gt_mask_binary)  # (R,)
            boundary_threshold = getattr(self.train_cfg, 'boundary_detection_threshold', 0.25)  # 从配置读取，默认0.25（从0.3降低，更准确）
            if isinstance(boundary_threshold, torch.Tensor):
                boundary_threshold = boundary_threshold.item()
            elif not isinstance(boundary_threshold, (int, float)):
                boundary_threshold = float(boundary_threshold)
            
            # 方法2：GT mask的边界点（使用GT mask的局部变化来识别边界点，向量化实现，高效）
            # 方法：对于每个proposal，计算每个点的GT mask与其proposal内平均GT mask的差异
            # 边界点 = GT mask与proposal平均差异大的点（位于mask边缘）
            
            # 向量化计算每个proposal的平均GT mask（使用scatter_add，避免Python循环）
            proposal_gt_sum = torch.zeros(num_proposals, device=gt_mask_binary.device, dtype=gt_mask_binary.dtype)  # (num_proposals,)
            proposal_point_count = torch.zeros(num_proposals, device=proposal_ids.device, dtype=torch.long)  # (num_proposals,)
            
            # 使用scatter_add计算每个proposal的GT mask总和和点数
            proposal_gt_sum.scatter_add_(0, proposal_ids, gt_mask_binary)
            proposal_point_count.scatter_add_(0, proposal_ids, torch.ones_like(proposal_ids, dtype=torch.long))
            
            # 计算平均GT mask（避免除零）
            proposal_gt_mean = proposal_gt_sum / (proposal_point_count.float() + 1e-6)  # (num_proposals,)
            
            # 将proposal平均GT mask映射到mask points
            proposal_gt_mean_for_mask = proposal_gt_mean[proposal_ids]  # (R,)
            
            # 计算每个点与proposal平均GT mask的差异（边界点差异大）
            gt_variation = torch.abs(gt_mask_binary - proposal_gt_mean_for_mask)  # (R,)
            
            # 结合两种方法：预测误差大 AND GT mask变化大（更保守，减少误判）
            # 使用AND而不是OR，确保只对真正的边界点加权
            is_boundary_by_pred = (pred_gt_diff > boundary_threshold).float()  # (R,)
            is_boundary_by_gt = (gt_variation > 0.25).float()  # GT mask变化超过0.25的点（边界区域，阈值从0.3降低到0.25）
            is_boundary_point = (is_boundary_by_pred * is_boundary_by_gt).float()  # (R,) 两者都满足
            
            # 只对leaf类别应用边界权重
            boundary_weights = mask_loss_weight.new_ones(mask_loss_weight.size(0))  # (R,)
            leaf_boundary_mask = leaf_mask_inds.float() * is_boundary_point  # (R,) 只对leaf类别的边界点
            
            # 基础边界权重
            boundary_weights = boundary_weights + (boundary_mask_weight - 1.0) * leaf_boundary_mask
            
            # 高IoU样本的边界点权重更高
            high_iou_mask = (iou_weights > 1.5).float() if use_iou_weighted_mask_loss and pos_inds.sum() > 0 else mask_loss_weight.new_zeros(mask_loss_weight.size(0))
            high_iou_boundary_mask = high_iou_mask * leaf_boundary_mask
            boundary_weights = boundary_weights + (high_iou_boundary_weight - 1.0) * high_iou_boundary_mask
            
            # 应用边界权重
            mask_loss_weight = mask_loss_weight * boundary_weights
        
        mask_loss = F.binary_cross_entropy(
            mask_scores_sigmoid_slice, mask_label, weight=mask_loss_weight, reduction='sum')
        mask_loss /= (mask_loss_weight.sum() + 1)
        losses['mask_loss'] = mask_loss

        # compute iou score loss，对leaf类别给予更高权重
        ious = get_mask_iou_on_pred(proposals_idx, proposals_offset, instance_labels,
                                    instance_pointnum, mask_scores_sigmoid_slice.detach())
        fg_ious = ious[:, fg_inds]
        gt_ious, _ = fg_ious.max(1)
        slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
        iou_score_weight = (labels < self.instance_classes).float()
        # 对leaf类别给予更高权重
        leaf_inds = (labels == 1)
        leaf_iou_weight = getattr(self.train_cfg, 'leaf_iou_weight', 1.2)  # 默认1.2倍权重
        if isinstance(leaf_iou_weight, torch.Tensor):
            leaf_iou_weight = leaf_iou_weight.item()
        elif not isinstance(leaf_iou_weight, (int, float)):
            leaf_iou_weight = float(leaf_iou_weight)
        iou_score_weight[leaf_inds] = iou_score_weight[leaf_inds] * leaf_iou_weight
        
        iou_score_slice = iou_scores[slice_inds, labels]
        iou_score_loss = F.mse_loss(iou_score_slice, gt_ious, reduction='none')
        iou_score_loss = (iou_score_loss * iou_score_weight).sum() / (iou_score_weight.sum() + 1)
        losses['iou_score_loss'] = iou_score_loss

        # 额外：让每个proposal的mask阈值与其IoU相关（简单监督），使阈值真正可学习
        if t_mask is not None:
            # t_mask: (num_proposals, 1) 与 cls_scores 对齐
            t_mask_flat = t_mask.squeeze(-1)
            # 将gt_ious映射到[t_min, t_max]作为理想阈值：IoU高→阈值略高，IoU低→阈值略低
            t_min, t_max = 0.2, 0.8
            t_target = t_min + (t_max - t_min) * gt_ious.clamp(0, 1)
            mask_thr_reg_loss = F.mse_loss(t_mask_flat, t_target)
            mask_thr_weight = getattr(self.train_cfg, 'mask_thr_weight', 0.05)  # 降低到0.05
            # 确保权重是Python标量
            if isinstance(mask_thr_weight, torch.Tensor):
                mask_thr_weight = mask_thr_weight.item()
            elif not isinstance(mask_thr_weight, (int, float)):
                mask_thr_weight = float(mask_thr_weight)
            losses['mask_thr_reg_loss'] = mask_thr_reg_loss * mask_thr_weight
        
        # ============ Mask Quality Loss（质量分支监督）============
        # 预测proposal与匹配GT的mask IoU，用于提升高IoU下的AP
        if mask_quality is not None and pos_inds.sum() > 0:
            # 使用GT IoU作为监督目标
            quality_pred = mask_quality[pos_inds]  # (num_pos,)
            quality_target = gt_ious[pos_inds]  # (num_pos,)
            
            # 对leaf类别给予更高权重
            quality_loss_weight = quality_pred.new_ones(quality_pred.size(0))
            leaf_quality_weight = getattr(self.train_cfg, 'leaf_quality_weight', 1.5)  # 默认1.5倍权重
            if isinstance(leaf_quality_weight, torch.Tensor):
                leaf_quality_weight = leaf_quality_weight.item()
            elif not isinstance(leaf_quality_weight, (int, float)):
                leaf_quality_weight = float(leaf_quality_weight)
            
            # 找到leaf类别的proposal
            pos_labels = labels[pos_inds]
            leaf_quality_inds = (pos_labels == 1)  # leaf类别索引为1
            quality_loss_weight[leaf_quality_inds] = quality_loss_weight[leaf_quality_inds] * leaf_quality_weight
            
            # 使用smooth_l1_loss（更稳健）或mse_loss
            quality_loss = F.smooth_l1_loss(quality_pred, quality_target, reduction='none')
            quality_loss = (quality_loss * quality_loss_weight).sum() / (quality_loss_weight.sum() + 1)
            
            quality_loss_weight_config = getattr(self.train_cfg, 'quality_loss_weight', 0.1)  # 默认0.1
            if isinstance(quality_loss_weight_config, torch.Tensor):
                quality_loss_weight_config = quality_loss_weight_config.item()
            elif not isinstance(quality_loss_weight_config, (int, float)):
                quality_loss_weight_config = float(quality_loss_weight_config)
            losses['quality_loss'] = quality_loss * quality_loss_weight_config
        else:
            losses['quality_loss'] = cls_scores.new_tensor(0.0)

        # add logging variables
        losses['num_pos'] = (labels < self.instance_classes).sum().float()
        losses['num_neg'] = (labels >= self.instance_classes).sum().float()
        return losses

    def parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' + f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    @cuda_cast
    def forward_test(self, batch_idxs, voxel_coords, p2v_map, v2p_map, coords_float, feats,
                     semantic_labels, instance_labels, pt_offset_labels, spatial_shape, batch_size,
                     scan_ids, **kwargs):
        color_feats = feats
        boundary_scores = None  # for optional boundary attention export
        if self.with_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        # lvl_fusion directly use output point as level 1 for pyramid map for fast inference
        lvl_fusion = getattr(self.test_cfg, 'lvl_fusion', False)
        semantic_scores, pt_offsets, output_feats = self.forward_backbone(
            input, v2p_map, x4_split=self.test_cfg.x4_split, lvl_fusion=lvl_fusion)
        
        # 边界注意力增强（如果启用）
        if self.use_boundary_attention:
            semantic_probs = semantic_scores.softmax(dim=-1)
            output_feats, boundary_scores = self.boundary_attention(
                output_feats, coords_float, semantic_probs=semantic_probs)
        
        # 参数预测（如果启用）
        predicted_params = None
        if self.use_param_predictor:
            predicted_params = self.param_predictor(output_feats, semantic_scores.softmax(dim=-1))
        if self.test_cfg.x4_split:
            coords_float = self.merge_4_parts(coords_float)
            semantic_labels = self.merge_4_parts(semantic_labels)
            instance_labels = self.merge_4_parts(instance_labels)
            pt_offset_labels = self.merge_4_parts(pt_offset_labels)
        semantic_preds = semantic_scores.max(1)[1]
        ret = dict(scan_id=scan_ids[0])
        if 'semantic' in self.test_cfg.eval_tasks or 'panoptic' in self.test_cfg.eval_tasks:
            ret.update(
                dict(
                    semantic_labels=semantic_labels.cpu().numpy(),
                    instance_labels=instance_labels.cpu().numpy()))
        if 'semantic' in self.test_cfg.eval_tasks:
            point_wise_results = self.get_point_wise_results(coords_float, color_feats,
                                                             semantic_preds, pt_offsets,
                                                             pt_offset_labels, v2p_map, lvl_fusion)
            ret.update(point_wise_results)
            # 在语义评估任务下导出 boundary_scores，供可视化使用
            if self.use_boundary_attention and boundary_scores is not None:
                # 与语义/offset保持同一顺序（当前实现中 boundary_scores 与 coords_float 对齐）
                ret['boundary_scores'] = boundary_scores.detach().cpu().numpy()
        if not self.semantic_only and not self.use_query_mask_head:
            if 'instance' in self.test_cfg.eval_tasks or 'panoptic' in self.test_cfg.eval_tasks:
                if lvl_fusion:
                    batch_idxs = input.indices[:, 0].int()
                    coords_float = voxelization(coords_float, p2v_map)
                # 使用预测的参数或配置参数
                grouping_cfg_to_use = self.grouping_cfg
                if predicted_params is not None:
                    grouping_cfg_to_use = type('obj', (object,), {
                        **{k: v for k, v in vars(self.grouping_cfg).items()},
                        'class_specific_radius': [p.item() for p in predicted_params['radius']],
                        'class_npoint_thr': [p.item() for p in predicted_params['npoint_thr']]
                    })()
                
                proposals_idx, proposals_offset = self.forward_grouping(
                    semantic_scores,
                    pt_offsets,
                    batch_idxs,
                    coords_float,
                    grouping_cfg_to_use,
                    lvl_fusion=lvl_fusion)
                
                # ============ 推理阶段：按类别保底的proposal保留（可选） ============
                # 注意：这是CPU-heavy操作，只在推理时使用，训练时已禁用
                use_category_quota = getattr(self.test_cfg, 'use_category_quota', False)
                if use_category_quota and proposals_offset.shape[0] > 1:
                    # 先进行forward_instance以获取cls_scores（用于按类别保留）
                    inst_feats_temp, inst_map_temp = self.clusters_voxelization(
                        proposals_idx, proposals_offset, output_feats, coords_float,
                        **self.instance_voxel_cfg)
                    (_, cls_scores_temp, iou_scores_temp, mask_scores_temp,
                     proposal_feats_temp, t_mask_temp, mask_quality_temp) = self.forward_instance(inst_feats_temp, inst_map_temp)
                    
                    # 按类别保留proposal
                    category_quota = getattr(self.test_cfg, 'category_quota', None)
                    max_proposal_num = getattr(self.test_cfg, 'max_proposal_num', None)
                    if max_proposal_num is None:
                        max_proposal_num = getattr(self.train_cfg, 'max_proposal_num', 1500)
                    proposals_idx, proposals_offset, _ = retain_proposals_by_category(
                        proposals_idx, proposals_offset, cls_scores_temp,
                        max_proposal_num, category_quota=category_quota)
                
                inst_feats, inst_map = self.clusters_voxelization(proposals_idx, proposals_offset,
                                                                  output_feats, coords_float,
                                                                  **self.instance_voxel_cfg)
                (_, cls_scores, iou_scores, mask_scores,
                 proposal_feats,
                 t_mask,
                 mask_quality) = self.forward_instance(inst_feats, inst_map)
                
                # 初始化 reliability（如果未使用 instance_adaptive，则为 None）
                reliability = None
                if self.use_instance_adaptive and proposals_offset.numel() > 1:
                    cls_scores, reliability = self.instance_adaptive(
                        proposal_feats,
                        proposals_idx,
                        proposals_offset,
                        coords_float,
                        cls_scores,
                        output_feats)
                
                # Proposal置信度细化（如果启用）
                if self.use_proposal_refiner:
                    refined_cls_scores, petiole_probs = self.proposal_refiner(
                        proposals_idx, proposals_offset, coords_float, 
                        semantic_scores.softmax(dim=-1), cls_scores
                    )
                    cls_scores = refined_cls_scores  # 使用细化后的分数
                
                pred_instances = self.get_instances(
                    scan_ids[0],
                    proposals_idx,
                    semantic_scores,
                    cls_scores,
                    iou_scores,
                    mask_scores,
                    coords_float=coords_float,
                    v2p_map=v2p_map,
                    lvl_fusion=lvl_fusion,
                    reliability=reliability,
                    t_mask=t_mask,
                    mask_quality=mask_quality)
            if 'instance' in self.test_cfg.eval_tasks:
                gt_instances = self.get_gt_instances(semantic_labels, instance_labels)
                ret.update(dict(pred_instances=pred_instances, gt_instances=gt_instances))
            if 'panoptic' in self.test_cfg.eval_tasks:
                panoptic_preds = self.panoptic_fusion(semantic_preds.cpu().numpy(), pred_instances)
                ret.update(panoptic_preds=panoptic_preds)
        # Query-based mask head 分支（测试）
        if not self.semantic_only and self.use_query_mask_head:
            qm_outputs = self.query_mask_head(output_feats, coords_float)
            class_logits = qm_outputs['class_logits']  # [Q, C+1]
            mask_logits = qm_outputs['mask_logits']    # [Q, N]
            C = self.semantic_classes
            Q, N = class_logits.size(0), mask_logits.size(1)
            class_probs = class_logits.softmax(dim=-1)
            cls_scores, cls_ids = class_probs[:, :C].max(dim=-1)  # [Q]
            no_obj_prob = class_probs[:, C]

            cls_thr = getattr(self.test_cfg, 'query_cls_thr', 0.3)
            mask_thr = getattr(self.test_cfg, 'query_mask_thr', 0.5)
            min_npoint = getattr(self.test_cfg, 'min_npoint', 2)

            pred_instances = []
            probs = mask_logits.sigmoid()
            for q in range(Q):
                # 过滤no-object与低置信度
                if no_obj_prob[q] >= (1.0 - cls_thr):
                    continue
                if cls_scores[q] < cls_thr:
                    continue
                mask_bin = (probs[q] > mask_thr)
                npt = int(mask_bin.long().sum().item())
                if npt < min_npoint:
                    continue
                pred = {}
                pred['scan_id'] = scan_ids[0]
                pred['label_id'] = int(cls_ids[q].item()) + 1
                pred['conf'] = float(cls_scores[q].item())
                pred['pred_mask'] = rle_encode(mask_bin.detach().cpu().numpy().astype(np.uint8))
                pred_instances.append(pred)

            if 'instance' in self.test_cfg.eval_tasks:
                gt_instances = self.get_gt_instances(semantic_labels, instance_labels)
                ret.update(dict(pred_instances=pred_instances, gt_instances=gt_instances))
        return ret

    def forward_backbone(self, input, input_map, x4_split=False, lvl_fusion=False):
        if x4_split:
            assert not lvl_fusion, 'x4_split not support lvl_fusion'
            output_feats = self.forward_4_parts(input, input_map)
            output_feats = self.merge_4_parts(output_feats)
        else:
            output = self.input_conv(input)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features
            if not lvl_fusion:
                output_feats = output_feats[input_map.long()]

        semantic_scores = self.semantic_linear(output_feats)
        pt_offsets = self.offset_linear(output_feats)
        return semantic_scores, pt_offsets, output_feats

    def forward_4_parts(self, x, input_map):
        """Helper function for s3dis: devide and forward 4 parts of a scene."""
        outs = []
        for i in range(4):    # S3DIS:4
            inds = x.indices[:, 0] == i
            feats = x.features[inds]
            coords = x.indices[inds]
            coords[:, 0] = 0
            x_new = spconv.SparseConvTensor(
                indices=coords, features=feats, spatial_shape=x.spatial_shape, batch_size=1)
            out = self.input_conv(x_new)
            out = self.unet(out)
            out = self.output_layer(out)
            outs.append(out.features)
        outs = torch.cat(outs, dim=0)
        return outs[input_map.long()]

    def merge_4_parts(self, x):
        
        
        """Helper function for s3dis: take output of 4 parts and merge them."""
        inds = torch.arange(x.size(0), device=x.device)
        p1 = inds[::4]
        p2 = inds[1::4]
        p3 = inds[2::4]
        p4 = inds[3::4]
        ps = [p1, p2, p3, p4]
        x_split = torch.split(x, [p.size(0) for p in ps])
        x_new = torch.zeros_like(x)
        for i, p in enumerate(ps):
            x_new[p] = x_split[i]
        return x_new

    @force_fp32(apply_to=('semantic_scores, pt_offsets'))
    def forward_grouping(self,
                         semantic_scores,
                         pt_offsets,
                         batch_idxs,
                         coords_float,
                         grouping_cfg=None,
                         lvl_fusion=False):
        proposals_idx_list = []
        proposals_offset_list = []
        batch_size = batch_idxs.max() + 1
        semantic_scores = semantic_scores.softmax(dim=-1)

        # 获取基础参数
        base_radius = self.grouping_cfg.radius
        mean_active = self.grouping_cfg.mean_active
        base_npoint_thr = self.grouping_cfg.npoint_thr
        with_pyramid = getattr(self.grouping_cfg, 'with_pyramid', False)
        with_octree = getattr(self.grouping_cfg, 'with_octree', False)
        base_size = getattr(self.grouping_cfg, 'pyramid_base_size', 0.02)
        
        # 类别自适应参数
        class_specific_radius = getattr(self.grouping_cfg, 'class_specific_radius', None)
        class_specific_npoint = getattr(self.grouping_cfg, 'class_npoint_thr', None)
        
        class_numpoint_mean = torch.tensor(
            self.grouping_cfg.class_numpoint_mean, dtype=torch.float32)
        assert class_numpoint_mean.size(0) == self.semantic_classes
        
        for class_id in range(self.semantic_classes):
            if class_id in self.grouping_cfg.ignore_classes:
                continue
            
            # 为每个类别选择合适的半径
            if class_specific_radius and len(class_specific_radius) > class_id:
                radius = class_specific_radius[class_id]
            else:
                radius = base_radius
            
            # 为每个类别选择合适的点数阈值
            if class_specific_npoint and len(class_specific_npoint) > class_id:
                npoint_thr = class_specific_npoint[class_id]
            else:
                npoint_thr = base_npoint_thr
            
            scores = semantic_scores[:, class_id].contiguous()
            object_idxs = (scores > self.grouping_cfg.score_thr).nonzero().view(-1)
            if object_idxs.size(0) < self.test_cfg.min_npoint:
                continue
            batch_idxs_ = batch_idxs[object_idxs]
            coords_ = coords_float[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            if with_pyramid:
                num_points = coords_.size(0)
                level = self.get_level(num_points)
                radius = radius * level  # 使用类别特定的radius
                if level > 1 or not lvl_fusion:
                    coords_, pt_offsets_, batch_idxs_, l2p_map = self.pyramid_map(
                        coords_, pt_offsets_, batch_idxs_, level, base_size)
            batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
            neighbor_inds, start_len = ball_query(
                coords_ + pt_offsets_,
                batch_idxs_,
                batch_offsets_,
                radius,  # 使用类别特定的radius
                mean_active,
                with_octree=with_octree)
            proposals_idx, proposals_offset = bfs_cluster(class_numpoint_mean, neighbor_inds.cpu(),
                                                          start_len.cpu(), npoint_thr, class_id)  # 使用类别特定的npoint_thr
            if with_pyramid:
                if level > 1 or not lvl_fusion:
                    proposals_idx, proposals_offset = self.pyramid_inverse_map(
                        proposals_idx, proposals_offset, coords_.size(0), l2p_map)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

            # merge proposals
            if len(proposals_offset_list) > 0:
                proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                proposals_offset += proposals_offset_list[-1][-1]
                proposals_offset = proposals_offset[1:]
            if proposals_idx.size(0) > 0:
                proposals_idx_list.append(proposals_idx)
                proposals_offset_list.append(proposals_offset)
        if len(proposals_idx_list) > 0:
            proposals_idx = torch.cat(proposals_idx_list, dim=0)
            proposals_offset = torch.cat(proposals_offset_list)
        else:
            proposals_idx = torch.zeros((0, 2), dtype=torch.int32)
            proposals_offset = torch.zeros((0, ), dtype=torch.int32)
        return proposals_idx, proposals_offset

    def get_level(self, num_points):
        if num_points > 1000000:
            level = 3
        elif num_points > 100000:
            level = 2
        else:
            level = 1
        return level

    def pyramid_map(self, coords_float, pt_offsets, batch_idxs, level=1, base_size=0.02):
        coords = (coords_float / (base_size * level)).long()
        coords = torch.cat([batch_idxs[:, None], coords], dim=1)
        coords, l2p_map, p2l_map = voxelization_idx(coords.cpu(), batch_idxs[-1].item() + 1)
        coords_float = voxelization(coords_float, p2l_map.cuda())
        pt_offsets = voxelization(pt_offsets, p2l_map.cuda())
        batch_idxs = coords[:, 0].cuda().int()
        return coords_float, pt_offsets, batch_idxs, l2p_map

    def pyramid_inverse_map(self, proposals_idx, proposals_offset, num_points, l2p_map):
        proposals = torch.zeros((proposals_offset.size(0) - 1, num_points), dtype=torch.int)
        proposals[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
        proposals = proposals[:, l2p_map.cpu().long()]
        proposals_idx = proposals.nonzero()
        proposals_offset = torch.cumsum(proposals.sum(1), dim=0).int()
        proposals_offset = torch.cat([proposals_offset.new_zeros(1), proposals_offset])
        return proposals_idx, proposals_offset

    def forward_instance(self, inst_feats, inst_map):
        feats = self.tiny_unet(inst_feats)
        feats = self.tiny_unet_outputlayer(feats)

        # predict mask scores (point-level, for each proposal)
        mask_scores = self.mask_linear(feats.features)
        mask_scores = mask_scores[inst_map.long()]
        instance_batch_idxs = feats.indices[:, 0][inst_map.long()]

        # predict instance cls and iou scores（proposal级别）
        pooled_feats = self.global_pool(feats)
        cls_scores = self.cls_linear(pooled_feats)
        iou_scores = self.iou_score_linear(pooled_feats)

        # 预测每个proposal的mask阈值 t_mask \in [mask_t_min, mask_t_max]
        t_mask_raw = self.mask_thr_head(pooled_feats)  # (num_proposals, 1), in (0, 1)
        t_mask = self.mask_t_min + (self.mask_t_max - self.mask_t_min) * t_mask_raw
        
        # 预测每个proposal的mask质量（IoU质量）
        mask_quality = self.mask_quality_head(pooled_feats).squeeze(-1)  # (num_proposals,)

        return instance_batch_idxs, cls_scores, iou_scores, mask_scores, pooled_feats, t_mask, mask_quality

    @force_fp32(apply_to=('semantic_preds', 'offset_preds'))
    def get_point_wise_results(self, coords_float, color_feats, semantic_preds, offset_preds,
                               offset_labels, v2p_map, lvl_fusion):
        if lvl_fusion:
            semantic_preds = semantic_preds[v2p_map.long()]
            offset_preds = offset_preds[v2p_map.long()]
        return dict(
            coords_float=coords_float.cpu().numpy(),
            color_feats=color_feats.cpu().numpy(),
            semantic_preds=semantic_preds.cpu().numpy(),
            offset_preds=offset_preds.cpu().numpy(),
            offset_labels=offset_labels.cpu().numpy())

    @force_fp32(apply_to=('semantic_scores', 'cls_scores', 'iou_scores', 'mask_scores'))
    def get_instances(self,
                      scan_id,
                      proposals_idx,
                      semantic_scores,
                      cls_scores,
                      iou_scores,
                      mask_scores,
                      coords_float=None,
                      v2p_map=None,
                      lvl_fusion=False,
                      reliability=None,
                      t_mask=None,
                      mask_quality=None):
        if proposals_idx.size(0) == 0:
            return []

        num_instances = cls_scores.size(0)
        num_points = semantic_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        semantic_pred = semantic_scores.max(1)[1]
        semantic_pred_np = semantic_pred.cpu().numpy()
        cls_pred_list, score_pred_list, mask_pred_list = [], [], []
        for i in range(self.instance_classes):
            if i in self.sem2ins_classes:
                cls_pred = cls_scores.new_tensor([i + 1], dtype=torch.long)
                score_pred = cls_scores.new_tensor([1.], dtype=torch.float32)
                mask_pred = (semantic_pred == i)[None, :].int()
                if lvl_fusion:
                    mask_pred = mask_pred[:, v2p_map.long()]
            else:
                cls_pred = cls_scores.new_full((num_instances, ), i + 1, dtype=torch.long)
                cur_cls_scores = cls_scores[:, i]
                cur_iou_scores = iou_scores[:, i]
                cur_mask_scores = mask_scores[:, i]  # (num_mask_rows,)
                
                leaf_class_id = 1  # leaf类别索引
                
                # 类别特定的score重标定：只对leaf类别使用新公式
                if i == leaf_class_id:
                    # Leaf类别：使用 iou_score + reliability 融合公式
                    # score_leaf = score_leaf * sigmoid(iou_pred)^α * reliability^β
                    # 这样可以更好地排序高质量候选，提高高IoU下的AP
                    
                    # 基础score：cls_prob
                    score_pred = cur_cls_scores.clone()
                    
                    # iou_pred: 使用sigmoid将iou logits转换为概率，然后取α次方
                    iou_pred_sigmoid = torch.sigmoid(cur_iou_scores)  # (num_instances,)
                    leaf_iou_alpha = getattr(self.test_cfg, 'leaf_iou_alpha', 0.7)  # 默认0.7
                    if isinstance(leaf_iou_alpha, torch.Tensor):
                        leaf_iou_alpha = leaf_iou_alpha.item()
                    elif not isinstance(leaf_iou_alpha, (int, float)):
                        leaf_iou_alpha = float(leaf_iou_alpha)
                    iou_factor = iou_pred_sigmoid ** leaf_iou_alpha
                    score_pred = score_pred * iou_factor
                    
                    # reliability: 取β次方（如果可用），并加入"保底"偏置，避免过度压制
                    if reliability is not None and reliability.numel() > 0:
                        if reliability.size(0) == num_instances:
                            # 读取 leaf 专用的 reliability floor 和幂次
                            leaf_reliability_beta = getattr(self.test_cfg, 'leaf_reliability_beta', 1.3)  # 默认1.3
                            if isinstance(leaf_reliability_beta, torch.Tensor):
                                leaf_reliability_beta = leaf_reliability_beta.item()
                            elif not isinstance(leaf_reliability_beta, (int, float)):
                                leaf_reliability_beta = float(leaf_reliability_beta)

                            leaf_reliability_floor = getattr(self.test_cfg, 'leaf_reliability_floor', 0.3)  # 默认0.3保底
                            try:
                                leaf_reliability_floor = float(leaf_reliability_floor)
                            except Exception:
                                leaf_reliability_floor = 0.3
                            leaf_reliability_floor = max(0.0, min(leaf_reliability_floor, 1.0))

                            # 先做线性插值获得带保底的 reliability，再取幂次：
                            # r_eff = floor + (1-floor)*r ∈ [floor, 1]
                            r = reliability.clamp(0.0, 1.0)
                            r_eff = leaf_reliability_floor + (1.0 - leaf_reliability_floor) * r
                            reliability_factor = r_eff ** leaf_reliability_beta
                            score_pred = score_pred * reliability_factor
                    
                    # ============ Mask Quality分支：使用质量预测重排序（只对leaf类别）============
                    # score_leaf = cls_score × q^α，其中q是质量分支预测的mask IoU
                    if mask_quality is not None and mask_quality.numel() > 0:
                        if mask_quality.size(0) == num_instances:
                            # 读取质量分支的幂次参数
                            leaf_quality_alpha = getattr(self.test_cfg, 'leaf_quality_alpha', 2.0)  # 默认2.0
                            if isinstance(leaf_quality_alpha, torch.Tensor):
                                leaf_quality_alpha = leaf_quality_alpha.item()
                            elif not isinstance(leaf_quality_alpha, (int, float)):
                                leaf_quality_alpha = float(leaf_quality_alpha)
                            
                            # 应用质量分支：score_leaf = cls_score × q^α
                            quality_scores = mask_quality.clamp(0.0, 1.0)  # (num_instances,)
                            quality_factor = quality_scores ** leaf_quality_alpha
                            score_pred = score_pred * quality_factor
                else:
                    # Stem/Branch类别：使用质量分支重排序（新增，提升高IoU下的AP）
                    # 基础score：cls_prob * iou_pred
                    iou_pred = cur_iou_scores.clamp(0, 1)
                    score_pred = cur_cls_scores * iou_pred
                    
                    # 应用reliability（如果可用）
                    if reliability is not None and reliability.numel() > 0:
                        if reliability.size(0) == num_instances:
                            # reliability 范围 [0, 1]，映射到 [0.5, 1.0] 的权重（带0.5偏置）
                            # 低可靠性降至50%，高可靠性保持100%
                            reliability_weight = 0.5 + 0.5 * reliability
                        score_pred = score_pred * reliability_weight
                    
                    # ============ Mask Quality分支：对stem/branch也使用质量预测重排序 ===========
                    # score = cls_score × iou_pred × reliability × q^α
                    if mask_quality is not None and mask_quality.numel() > 0:
                        if mask_quality.size(0) == num_instances:
                            # 根据类别读取对应的质量分支幂次参数
                            if i == 0:  # stem类别
                                quality_alpha = getattr(self.test_cfg, 'stem_quality_alpha', 2.0)  # 默认2.0
                            elif i == 2:  # branch类别
                                quality_alpha = getattr(self.test_cfg, 'branch_quality_alpha', 2.0)  # 默认2.0
                            else:
                                quality_alpha = 2.0  # 其他类别默认2.0
                            
                            if isinstance(quality_alpha, torch.Tensor):
                                quality_alpha = quality_alpha.item()
                            elif not isinstance(quality_alpha, (int, float)):
                                quality_alpha = float(quality_alpha)
                            
                            # 应用质量分支：score = score × q^α
                            quality_scores = mask_quality.clamp(0.0, 1.0)  # (num_instances,)
                            quality_factor = quality_scores ** quality_alpha
                            score_pred = score_pred * quality_factor

                mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device='cuda')

                # 使用自适应mask阈值：为mask_scores的每一行分配对应proposal的t_mask
                if t_mask is not None:
                    # proposals_idx[:, 0] 是proposal索引（0..num_instances-1）
                    # 将其映射到对应的t_mask，放到与cur_mask_scores同设备
                    row_t = t_mask.squeeze(-1)[proposals_idx[:, 0].long().to(mask_scores.device)]
                    mask_inds = cur_mask_scores > row_t
                else:
                 mask_inds = cur_mask_scores > self.test_cfg.mask_score_thr

                cur_proposals_idx = proposals_idx[mask_inds.cpu()].long()  # 在CPU上索引
                mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1

                # filter low score instance
                # 对 leaf 使用更宽松且与重标定一致的过滤策略，避免在 cls_prob 阶段过早被裁掉
                # 其它类别仍使用 cls_prob 作为过滤依据
                base_cls_thr = float(self.test_cfg.cls_score_thr)
                leaf_class_id = 1
                if i == leaf_class_id and hasattr(self.test_cfg, 'leaf_cls_score_thr'):
                    # leaf 的专用阈值（通常更低），若未配置则回退到全局阈值
                    leaf_cls_thr = getattr(self.test_cfg, 'leaf_cls_score_thr', base_cls_thr)
                    try:
                        leaf_cls_thr = float(leaf_cls_thr)
                    except Exception:
                        leaf_cls_thr = base_cls_thr
                    # leaf：用最终 score_pred 做过滤，更符合 AP 排序逻辑
                    inds = score_pred > leaf_cls_thr
                else:
                    # stem / branch：保持原来的 cls_prob 过滤逻辑
                    inds = cur_cls_scores > base_cls_thr
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]

                if lvl_fusion:
                    mask_pred = mask_pred[:, v2p_map.long()]

                # filter too small instances
                npoint = mask_pred.sum(1)
                inds = npoint >= self.test_cfg.min_npoint
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]
                
                # Top-K 过滤：确保每个类别保留足够的实例数量（特别是leaf）
                # 先按分数降序排序
                if len(score_pred) > 0:
                    sort_inds = torch.argsort(score_pred, descending=True)
                    cls_pred = cls_pred[sort_inds]
                    score_pred = score_pred[sort_inds]
                    mask_pred = mask_pred[sort_inds]
                    
                    # 获取类别特定的top-K配置
                    top_k_per_class = getattr(self.test_cfg, 'top_k_per_class', None)
                    if top_k_per_class is not None:
                        # top_k_per_class 可以是整数（所有类别相同）或列表 [stem_k, leaf_k, branch_k]
                        if isinstance(top_k_per_class, (list, tuple)) and len(top_k_per_class) > i:
                            k = int(top_k_per_class[i])
                        elif isinstance(top_k_per_class, (int, float)):
                            k = int(top_k_per_class)
                        else:
                            k = None
                        
                        if k is not None and k > 0 and len(cls_pred) > k:
                            # 只保留top-K个实例
                            cls_pred = cls_pred[:k]
                            score_pred = score_pred[:k]
                            mask_pred = mask_pred[:k]
            
            cls_pred_list.append(cls_pred.cpu())
            score_pred_list.append(score_pred.cpu())
            mask_pred_list.append(mask_pred.cpu())
        cls_pred = torch.cat(cls_pred_list).numpy()
        score_pred = torch.cat(score_pred_list).numpy()
        mask_pred = torch.cat(mask_pred_list).numpy()

        # ================= Leaf 兜底实例：在 semantic_pred==leaf 且未被任何实例覆盖的区域，再做一次小半径聚类 =================
        use_leaf_fallback = getattr(self.test_cfg, 'use_leaf_fallback', False)
        if use_leaf_fallback and coords_float is not None and cls_pred.size > 0:
            leaf_class_id = 1
            leaf_label_id = leaf_class_id + 1  # 输出里是 1-based

            num_points_np = semantic_pred_np.shape[0]
            # 已有 leaf 实例覆盖情况
            leaf_inst_mask = (cls_pred == leaf_label_id)
            if leaf_inst_mask.any():
                leaf_masks_existing = mask_pred[leaf_inst_mask].astype(bool)  # (P_leaf, N)
                covered_leaf = leaf_masks_existing.any(axis=0)
            else:
                covered_leaf = np.zeros(num_points_np, dtype=bool)

            # 语义为 leaf 但未被任何实例覆盖的点
            uncovered_leaf = (semantic_pred_np == leaf_class_id) & (~covered_leaf)

            if uncovered_leaf.any():
                coords_np = coords_float.cpu().numpy()
                leaf_coords = coords_np[uncovered_leaf]
                leaf_indices = np.nonzero(uncovered_leaf)[0]

                radius = float(getattr(self.test_cfg, 'leaf_fallback_radius', 0.01))
                min_pts = int(getattr(self.test_cfg, 'leaf_fallback_min_points', 300))
                base_score = float(getattr(self.test_cfg, 'leaf_fallback_score', 0.05))

                # 基于 voxel 的粗聚类：先对坐标做体素化，再在 voxel 网格里做连通域搜索
                if radius > 0 and leaf_coords.shape[0] >= min_pts:
                    voxel_size = radius
                    voxel_idx = np.floor(leaf_coords / voxel_size).astype(np.int32)  # (M,3)

                    cell_to_points = {}
                    for lp, gidx in zip(voxel_idx, leaf_indices):
                        key = (int(lp[0]), int(lp[1]), int(lp[2]))
                        cell_to_points.setdefault(key, []).append(int(gidx))

                    visited_cells = set()
                    clusters = []
                    neighbor_shifts = [
                        (dx, dy, dz)
                        for dx in (-1, 0, 1)
                        for dy in (-1, 0, 1)
                        for dz in (-1, 0, 1)
                    ]

                    for cell in cell_to_points.keys():
                        if cell in visited_cells:
                            continue
                        # BFS / flood fill in voxel grid
                        queue = [cell]
                        visited_cells.add(cell)
                        cluster_points = []
                        while queue:
                            c = queue.pop()
                            cluster_points.extend(cell_to_points.get(c, []))
                            cx, cy, cz = c
                            for dx, dy, dz in neighbor_shifts:
                                nc = (cx + dx, cy + dy, cz + dz)
                                if nc in cell_to_points and nc not in visited_cells:
                                    visited_cells.add(nc)
                                    queue.append(nc)
                        if cluster_points:
                            clusters.append(np.array(cluster_points, dtype=np.int64))

                    new_masks = []
                    new_scores = []
                    new_labels = []
                    for pts_idx in clusters:
                        if pts_idx.size < min_pts:
                            continue
                        m = np.zeros(num_points_np, dtype=np.uint8)
                        m[pts_idx] = 1
                        new_masks.append(m)
                        new_scores.append(base_score)
                        new_labels.append(leaf_label_id)

                    if new_masks:
                        new_masks = np.stack(new_masks, axis=0)
                        new_scores = np.asarray(new_scores, dtype=score_pred.dtype)
                        new_labels = np.asarray(new_labels, dtype=cls_pred.dtype)

                        cls_pred = np.concatenate([cls_pred, new_labels], axis=0)
                        score_pred = np.concatenate([score_pred, new_scores], axis=0)
                        mask_pred = np.concatenate([mask_pred, new_masks], axis=0)

        # ================= 连通性约束的实例后处理 =================
        # 对 leaf/branch：Keep-largest 策略（消除多岛并集）
        # 对 stem：合并/保留策略（保持完整结构）
        use_connectivity_postprocess = getattr(self.test_cfg, 'use_connectivity_postprocess', True)
        keep_largest_classes = getattr(self.test_cfg, 'keep_largest_classes', [1, 2])  # leaf=1, branch=2
        stem_class_id = 0  # stem类别ID（输出中是1-based，所以是1，但这里用0表示stem）
        if use_connectivity_postprocess and coords_float is not None and mask_pred.shape[0] > 0:
            # 获取连通半径：leaf grouping 半径的 0.5~0.7 倍
            # 默认取 0.6 倍，可通过配置调整
            leaf_grouping_radius = self.grouping_cfg.class_specific_radius[1] if hasattr(self.grouping_cfg, 'class_specific_radius') and len(self.grouping_cfg.class_specific_radius) > 1 else 0.018
            connectivity_radius_ratio = getattr(self.test_cfg, 'connectivity_radius_ratio', 0.6)  # 默认 0.6
            connectivity_radius = leaf_grouping_radius * connectivity_radius_ratio
            
            # 也可以直接指定连通半径（优先级更高）
            if hasattr(self.test_cfg, 'connectivity_radius'):
                connectivity_radius = float(self.test_cfg.connectivity_radius)
            
            # stem 使用更大的连通半径进行补连/合并
            stem_connectivity_radius = getattr(self.test_cfg, 'stem_connectivity_radius', None)
            if stem_connectivity_radius is None:
                # 默认使用 stem grouping 半径的较大倍数
                stem_grouping_radius = self.grouping_cfg.class_specific_radius[0] if hasattr(self.grouping_cfg, 'class_specific_radius') and len(self.grouping_cfg.class_specific_radius) > 0 else 0.02
                stem_connectivity_radius = stem_grouping_radius * 1.5  # 更大的半径用于补连
            
            coords_np = coords_float.cpu().numpy()
            num_instances = mask_pred.shape[0]
            num_points = mask_pred.shape[1]
            
            # 对每个实例进行连通性分析
            refined_masks = []
            refined_cls = []
            refined_scores = []
            
            for inst_idx in range(num_instances):
                inst_mask = mask_pred[inst_idx].astype(bool)
                inst_cls = cls_pred[inst_idx]
                
                if inst_mask.sum() == 0:
                    continue
                
                # 判断类别，stem 跳过所有后处理，直接保留
                # 注意：cls_pred 是 1-based (1=stem, 2=leaf, 3=branch)
                is_stem = (inst_cls == 1)
                
                if is_stem:
                    # Stem：跳过所有后处理，直接保留原始结果
                    refined_masks.append(inst_mask)
                    refined_cls.append(inst_cls)
                    refined_scores.append(score_pred[inst_idx])
                    continue
                
                # 提取该实例的所有点坐标
                inst_point_indices = np.nonzero(inst_mask)[0]
                inst_coords = coords_np[inst_point_indices]
                
                if len(inst_point_indices) == 0:
                    continue
                if len(inst_point_indices) == 1:
                    # 单点实例，直接保留
                    refined_masks.append(inst_mask)
                    refined_cls.append(inst_cls)
                    refined_scores.append(score_pred[inst_idx])
                    continue
                
                # Leaf/Branch 处理：Keep-largest 策略
                if inst_cls in keep_largest_classes:
                        # 在原始坐标空间进行连通性分析
                        num_components, component_labels = self._analyze_connectivity(
                            inst_coords, connectivity_radius
                        )
                        
                        if num_components == 1:
                            # 已经是连通的，直接保留
                            refined_masks.append(inst_mask)
                            refined_cls.append(inst_cls)
                            refined_scores.append(score_pred[inst_idx])
                        else:
                            # 多个连通块，只保留最大的
                            component_sizes = []
                            for comp_id in range(num_components):
                                comp_mask = (component_labels == comp_id)
                                component_sizes.append(comp_mask.sum())
                            
                            largest_comp_id = np.argmax(component_sizes)
                            largest_comp_mask = (component_labels == largest_comp_id)
                            largest_point_indices = inst_point_indices[largest_comp_mask]
                            
                            # 构建新的 mask（只包含最大连通块的点）
                            new_mask = np.zeros(num_points, dtype=np.uint8)
                            new_mask[largest_point_indices] = 1
                            
                            # 只保留点数足够的实例（避免过度拆分导致的小碎片）
                            min_points_after_split = getattr(self.test_cfg, 'min_points_after_split', self.test_cfg.min_npoint)
                            if new_mask.sum() >= min_points_after_split:
                                refined_masks.append(new_mask)
                                refined_cls.append(inst_cls)
                                refined_scores.append(score_pred[inst_idx])
                            # 如果最大连通块太小，丢弃该实例
                else:
                    # 其他类别，直接保留
                    refined_masks.append(inst_mask)
                    refined_cls.append(inst_cls)
                    refined_scores.append(score_pred[inst_idx])
            
            if len(refined_masks) > 0:
                mask_pred = np.stack(refined_masks, axis=0)
                cls_pred = np.array(refined_cls, dtype=cls_pred.dtype)
                score_pred = np.array(refined_scores, dtype=score_pred.dtype)
        
        # ================= 收缩式修剪后处理：去除远端拖尾点 =================
        # 在 Keep-largest 之后，对 leaf 和 branch 实例进行收缩式修剪
        # 去除明显不属于该实例的远端拖尾点，使实例更紧致、更贴合单片叶子
        use_shrink_trim = getattr(self.test_cfg, 'use_shrink_trim', True)
        shrink_trim_classes = getattr(self.test_cfg, 'shrink_trim_classes', [1, 2])  # leaf=1, branch=2
        if use_shrink_trim and coords_float is not None and mask_pred.shape[0] > 0:
            coords_np = coords_float.cpu().numpy()
            num_instances = mask_pred.shape[0]
            num_points = mask_pred.shape[1]
            
            trimmed_masks = []
            trimmed_cls = []
            trimmed_scores = []
            
            # 获取连通半径（用于修剪后的 Keep-largest）
            leaf_grouping_radius = self.grouping_cfg.class_specific_radius[1] if hasattr(self.grouping_cfg, 'class_specific_radius') and len(self.grouping_cfg.class_specific_radius) > 1 else 0.018
            connectivity_radius_ratio = getattr(self.test_cfg, 'connectivity_radius_ratio', 0.6)
            connectivity_radius = leaf_grouping_radius * connectivity_radius_ratio
            if hasattr(self.test_cfg, 'connectivity_radius'):
                connectivity_radius = float(self.test_cfg.connectivity_radius)
            
            for inst_idx in range(num_instances):
                inst_mask = mask_pred[inst_idx].astype(bool)
                inst_cls = cls_pred[inst_idx]
                
                # 只对指定的类别进行修剪（leaf 和 branch）
                if inst_cls not in shrink_trim_classes:
                    trimmed_masks.append(inst_mask)
                    trimmed_cls.append(inst_cls)
                    trimmed_scores.append(score_pred[inst_idx])
                    continue
                
                if inst_mask.sum() == 0:
                    continue
                
                # 提取该实例的所有点坐标
                inst_point_indices = np.nonzero(inst_mask)[0]
                inst_coords = coords_np[inst_point_indices]
                
                if len(inst_point_indices) < 10:  # 点数太少，不修剪
                    trimmed_masks.append(inst_mask)
                    trimmed_cls.append(inst_cls)
                    trimmed_scores.append(score_pred[inst_idx])
                    continue
                
                # 计算实例中心：使用 medoid（更稳健）或质心
                use_medoid = getattr(self.test_cfg, 'shrink_trim_use_medoid', True)
                if use_medoid:
                    # Medoid: 距离所有其他点距离之和最小的点
                    dist_matrix = cdist(inst_coords, inst_coords)
                    total_dists = dist_matrix.sum(axis=1)
                    center_idx = np.argmin(total_dists)
                    center = inst_coords[center_idx]
                else:
                    # 质心
                    center = inst_coords.mean(axis=0)
                
                # 计算每个点到中心的距离
                dists = np.linalg.norm(inst_coords - center, axis=1)
                
                # 截断方法：95%分位 或 中位数+3倍MAD
                trim_method = getattr(self.test_cfg, 'shrink_trim_method', 'percentile')  # 'percentile' 或 'mad'
                
                if trim_method == 'percentile':
                    trim_percentile = getattr(self.test_cfg, 'shrink_trim_percentile', 95.0)
                    dist_threshold = np.percentile(dists, trim_percentile)
                else:  # 'mad'
                    median_dist = np.median(dists)
                    mad = np.median(np.abs(dists - median_dist))  # Median Absolute Deviation
                    mad_multiplier = getattr(self.test_cfg, 'shrink_trim_mad_multiplier', 3.0)
                    dist_threshold = median_dist + mad_multiplier * mad
                
                # 保留距离小于阈值的点
                keep_mask = dists <= dist_threshold
                trimmed_point_indices = inst_point_indices[keep_mask]
                
                if len(trimmed_point_indices) < 3:  # 修剪后点数太少，保留原mask
                    trimmed_masks.append(inst_mask)
                    trimmed_cls.append(inst_cls)
                    trimmed_scores.append(score_pred[inst_idx])
                    continue
                
                # 构建修剪后的 mask
                trimmed_mask = np.zeros(num_points, dtype=np.uint8)
                trimmed_mask[trimmed_point_indices] = 1
                
                # 再次执行 Keep-largest（防止修剪产生小碎块）
                trimmed_coords = coords_np[trimmed_point_indices]
                num_components, component_labels = self._analyze_connectivity(
                    trimmed_coords, connectivity_radius
                )
                
                if num_components == 1:
                    # 连通，直接使用
                    final_mask = trimmed_mask
                else:
                    # 多个连通块，只保留最大的
                    component_sizes = []
                    for comp_id in range(num_components):
                        comp_mask = (component_labels == comp_id)
                        component_sizes.append(comp_mask.sum())
                    
                    largest_comp_id = np.argmax(component_sizes)
                    largest_comp_mask = (component_labels == largest_comp_id)
                    largest_point_indices = trimmed_point_indices[largest_comp_mask]
                    
                    final_mask = np.zeros(num_points, dtype=np.uint8)
                    final_mask[largest_point_indices] = 1
                
                # 只保留点数足够的实例
                min_points_after_trim = getattr(self.test_cfg, 'min_points_after_trim', self.test_cfg.min_npoint)
                if final_mask.sum() >= min_points_after_trim:
                    trimmed_masks.append(final_mask)
                    trimmed_cls.append(inst_cls)
                    trimmed_scores.append(score_pred[inst_idx])
                else:
                    # 修剪后点数太少，丢弃该实例
                    pass
            
            if len(trimmed_masks) > 0:
                mask_pred = np.stack(trimmed_masks, axis=0)
                cls_pred = np.array(trimmed_cls, dtype=cls_pred.dtype)
                score_pred = np.array(trimmed_scores, dtype=score_pred.dtype)

        instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            instances.append(pred)
        return instances

    def panoptic_fusion(self, semantic_preds, instance_preds):
        cls_offset = self.semantic_classes - self.instance_classes - 1
        panoptic_cls = semantic_preds.copy().astype(np.uint32)
        panoptic_ids = np.zeros_like(semantic_preds).astype(np.uint32)

        # higher score has higher fusion priority
        scores = [x['conf'] for x in instance_preds]
        score_inds = np.argsort(scores)[::-1]
        prev_paste = np.zeros_like(semantic_preds, dtype=bool)
        panoptic_id = 1
        for i in score_inds:
            instance = instance_preds[i]
            cls = instance['label_id']
            mask = rle_decode(instance['pred_mask']).astype(bool)

            # check overlap with pasted instances
            intersect = (mask * prev_paste).sum()
            if intersect / (mask.sum() + 1e-5) > self.test_cfg.panoptic_skip_iou:
                continue

            paste = mask * (~prev_paste)
            panoptic_cls[paste] = cls + cls_offset
            panoptic_ids[paste] = panoptic_id
            prev_paste[paste] = 1
            panoptic_id += 1

        # if thing classes have panoptic id == 0, ignore it
        ignore_inds = (panoptic_cls >= 11) & (panoptic_ids == 0)

        # encode panoptic results
        panoptic_preds = (panoptic_cls & 0xFFFF) | (panoptic_ids << 16)
        panoptic_preds[ignore_inds] = self.semantic_classes
        panoptic_preds = panoptic_preds.astype(np.uint32)
        return panoptic_preds

    def get_gt_instances(self, semantic_labels, instance_labels):
        """Get gt instances for evaluation."""
        # convert to evaluation format 0: ignore, 1->N: valid
        label_shift = self.semantic_classes - self.instance_classes
        semantic_labels = semantic_labels - label_shift + 1
        semantic_labels[semantic_labels < 0] = 0
        instance_labels += 1
        ignore_inds = instance_labels < 0
        # scannet encoding rule
        gt_ins = semantic_labels * 1000 + instance_labels
        gt_ins[ignore_inds] = 0
        gt_ins = gt_ins.cpu().numpy()
        return gt_ins
    
    def _analyze_connectivity(self, coords, radius):
        """
        分析点云的连通分量
        
        Args:
            coords: (N, 3) 点坐标数组
            radius: 连通性判断的半径阈值
        
        Returns:
            num_components: 连通分量数
            component_labels: (N,) 每个点所属的连通分量标签
        """
        if len(coords) == 0:
            return 0, np.array([], dtype=np.int32)
        if len(coords) == 1:
            return 1, np.array([0], dtype=np.int32)
        
        try:
            # 使用 KDTree 找邻居
            tree = cKDTree(coords)
            pairs = tree.query_pairs(radius)
            pairs_list = list(pairs)
            
            if len(pairs_list) == 0:
                # 没有连接，每个点都是独立的连通分量
                return len(coords), np.arange(len(coords), dtype=np.int32)
            
            # 构建邻接矩阵
            N = len(coords)
            row = np.array([p[0] for p in pairs_list], dtype=np.int32)
            col = np.array([p[1] for p in pairs_list], dtype=np.int32)
            data = np.ones(len(pairs_list), dtype=np.float32)
            
            # 构建对称矩阵（无向图）
            row_sym = np.concatenate([row, col])
            col_sym = np.concatenate([col, row])
            data_sym = np.concatenate([data, data])
            
            adj_matrix = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
            
            # 计算连通分量
            num_components, component_labels = connected_components(
                adj_matrix, 
                directed=False, 
                return_labels=True
            )
            
            return num_components, component_labels
        except Exception as e:
            # 如果出错，返回每个点都是独立的连通分量（最坏情况）
            return len(coords), np.arange(len(coords), dtype=np.int32)

    @force_fp32(apply_to='feats')
    def clusters_voxelization(self,
                              clusters_idx,
                              clusters_offset,
                              feats,
                              coords,
                              scale,
                              spatial_shape,
                              rand_quantize=False):
        if clusters_idx.size(0) == 0:
            # create dummpy tensors
            coords = torch.tensor(
                [[0, 0, 0, 0], [0, spatial_shape - 1, spatial_shape - 1, spatial_shape - 1]],
                dtype=torch.int,
                device='cuda')
            feats = feats[0:2]
            voxelization_feats = spconv.SparseConvTensor(feats, coords, [spatial_shape] * 3, 1)
            inp_map = feats.new_zeros((1, ), dtype=torch.long)
            return voxelization_feats, inp_map

        batch_idx = clusters_idx[:, 0].cuda().long()
        c_idxs = clusters_idx[:, 1].cuda()
        feats = feats[c_idxs.long()]
        coords = coords[c_idxs.long()]

        coords_min = sec_min(coords, clusters_offset.cuda())
        coords_max = sec_max(coords, clusters_offset.cuda())

        # 0.01 to ensure voxel_coords < spatial_shape
        clusters_scale = 1 / ((coords_max - coords_min) / spatial_shape).max(1)[0] - 0.01
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        coords_min = coords_min * clusters_scale[:, None]
        coords_max = coords_max * clusters_scale[:, None]
        clusters_scale = clusters_scale[batch_idx]
        coords = coords * clusters_scale[:, None]

        if rand_quantize:
            # after this, coords.long() will have some randomness
            range = coords_max - coords_min
            coords_min -= torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3).cuda()
            coords_min -= torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3).cuda()
        coords_min = coords_min[batch_idx]
        coords -= coords_min
        assert coords.shape.numel() == ((coords >= 0) * (coords < spatial_shape)).sum()
        coords = coords.long()
        coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), coords.cpu()], 1)

        out_coords, inp_map, out_map = voxelization_idx(coords, int(clusters_idx[-1, 0]) + 1)
        out_feats = voxelization(feats, out_map.cuda())
        spatial_shape = [spatial_shape] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats,
                                                     out_coords.int().cuda(), spatial_shape,
                                                     int(clusters_idx[-1, 0]) + 1)
        return voxelization_feats, inp_map

    def get_batch_offsets(self, batch_idxs, bs):
        batch_offsets = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    @force_fp32(apply_to=('x'))
    def global_pool(self, x, expand=False):
        indices = x.indices[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1, ), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool

        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x
