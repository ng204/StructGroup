import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        dim_in = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(queries, keys, keys)
        return self.norm(queries + attn_out)


class QueryMaskHead(nn.Module):
    """
    Query-based mask head inspired by MaskFormer/OneFormer3D.

    - Projects per-point features into mask embedding space
    - Learns a set of queries that attend to point features
    - Produces per-query class logits and mask weights
    - Mask logits are computed as sigmoid(query_mask @ point_mask_emb^T)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_queries: int = 100,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_attn_layers: int = 2,
        no_object_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.no_object_weight = no_object_weight

        self.point_proj = MLP(in_channels, embed_dim, hidden_dim=max(128, embed_dim), num_layers=3)
        self.query_embed = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)

        attn_layers: List[nn.Module] = []
        for _ in range(num_attn_layers):
            attn_layers.append(SimpleCrossAttention(embed_dim, num_heads=num_heads))
        self.cross_attn = nn.ModuleList(attn_layers)

        self.cls_head = nn.Linear(embed_dim, num_classes + 1)
        self.mask_head = nn.Linear(embed_dim, embed_dim)

    @staticmethod
    def sigmoid_dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Dice loss for binary masks
        inputs: [1, N] logits
        targets: [1, N] binary targets
        """
        inputs = inputs.sigmoid()
        numerator = 2 * (inputs * targets).sum(dim=1)
        denominator = inputs.sum(dim=1) + targets.sum(dim=1) + 1e-6
        dice_coef = (numerator + 1e-6) / (denominator + 1e-6)
        # Dice loss = 1 - dice coefficient
        loss = 1 - dice_coef
        # Clamp to avoid numerical issues
        loss = torch.clamp(loss, min=0.0, max=1.0)
        return loss

    @staticmethod
    def sigmoid_ce(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(inputs, targets, reduction="none").mean(dim=1)

    def forward(
        self,
        point_features: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # point_features: [N, C]
        N, _ = point_features.shape
        mask_embeddings = self.point_proj(point_features)  # [N, E]

        queries = self.query_embed.unsqueeze(0).expand(1, self.num_queries, self.embed_dim)  # [1, Q, E]
        keys = mask_embeddings.unsqueeze(0)  # [1, N, E]
        for layer in self.cross_attn:
            queries = layer(queries, keys)

        query_feats = queries.squeeze(0)  # [Q, E]
        class_logits = self.cls_head(query_feats)  # [Q, C+1]
        mask_weights = self.mask_head(query_feats)  # [Q, E]
        mask_logits = torch.matmul(mask_weights, mask_embeddings.t())  # [Q, N]

        return {
            "class_logits": class_logits,
            "mask_logits": mask_logits,
        }

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        instance_labels: torch.Tensor,
        ignore_label: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute set prediction loss with a simple greedy matching.

        instance_labels: [N] with encoding class*1000 + k, or <=0 for background/ignore
        """
        class_logits: torch.Tensor = outputs["class_logits"]  # [Q, C+1]
        mask_logits: torch.Tensor = outputs["mask_logits"]  # [Q, N]
        device = class_logits.device

        valid_mask = instance_labels > 0
        if valid_mask.sum() == 0:
            # No instances in this sample
            tgt_labels = torch.full((self.num_queries,), self.num_classes, dtype=torch.long, device=device)
            cls_loss = F.cross_entropy(
                class_logits, tgt_labels, weight=self._get_cls_weights(device)
            )
            return {"qm_cls_loss": cls_loss, "qm_mask_bce": class_logits.new_tensor(0.0), "qm_mask_dice": class_logits.new_tensor(0.0)}

        unique_ids = torch.unique(instance_labels[valid_mask])
        gt_masks: List[torch.Tensor] = []
        gt_classes: List[int] = []
        for inst_id in unique_ids.tolist():
            inst_mask = (instance_labels == inst_id)
            gt_masks.append(inst_mask)
            class_id = int(inst_id // 1000)
            # 检查class_id是否在有效范围
            if class_id >= self.num_classes:
                print(f"警告：实例{inst_id}的类别{class_id}超出范围[0, {self.num_classes-1}]！跳过此实例")
                continue
            gt_classes.append(class_id)

        Q = class_logits.size(0)
        N = mask_logits.size(1)
        num_gt = len(gt_masks)
        
        # 调试：打印GT数量和类别
        if num_gt != len(gt_classes):
            print(f"警告：GT masks数量({num_gt})与classes数量({len(gt_classes)})不匹配！")
            # 修正：只使用有效的
            gt_masks = gt_masks[:len(gt_classes)]
            num_gt = len(gt_classes)
        
        if num_gt == 0:
            print(f"错误：没有有效的GT实例！unique_ids有{len(unique_ids)}个，但全部被过滤")
            tgt_labels = torch.full((Q,), self.num_classes, dtype=torch.long, device=device)
            cls_loss = F.cross_entropy(class_logits, tgt_labels, weight=self._get_cls_weights(device))
            return {"qm_cls_loss": cls_loss, "qm_mask_bce": class_logits.new_tensor(0.0), "qm_mask_dice": class_logits.new_tensor(0.0)}

        with torch.no_grad():
            gt_stack = torch.stack([m.to(device) for m in gt_masks], dim=0).float()  # [G, N]
            pred_probs = mask_logits.sigmoid()  # [Q, N]
            # Dice cost
            inter = torch.einsum("qn,gn->qg", pred_probs, gt_stack)
            sums = pred_probs.sum(dim=1, keepdim=True) + gt_stack.sum(dim=1).unsqueeze(0)
            dice = (2 * inter + 1e-6) / (sums + 1e-6)
            dice_cost = 1 - dice  # [Q, G]
            # Class cost: pick predicted class prob at gt class
            cls_logit = class_logits[:, : self.num_classes]
            cls_prob = cls_logit.softmax(dim=-1)  # [Q, C]
            gt_cls_idx = torch.tensor(gt_classes, device=device, dtype=torch.long)
            cls_cost = 1.0 - cls_prob[:, gt_cls_idx]  # [Q, G]
            total_cost = 2.0 * dice_cost + 1.0 * cls_cost
            # Greedy matching: for each gt, pick best query not used
            assigned_q = [-1] * num_gt
            used = set()
            for g in range(num_gt):
                costs = total_cost[:, g]
                # find argmin among unused
                sorted_idx = torch.argsort(costs)
                for q in sorted_idx.tolist():
                    if q not in used:
                        assigned_q[g] = q
                        used.add(q)
                        break

        # Build targets
        tgt_labels = torch.full((Q,), self.num_classes, dtype=torch.long, device=device)
        mask_bce_list: List[torch.Tensor] = []
        mask_dice_list: List[torch.Tensor] = []
        
        # 调试：打印匹配信息
        matched_count = sum(1 for q in assigned_q if q >= 0)
        if matched_count != num_gt:
            print(f"警告：只匹配了{matched_count}/{num_gt}个GT实例")
        
        for g, q in enumerate(assigned_q):
            if q < 0:
                continue
            tgt_labels[q] = gt_classes[g]
            tgt_mask = gt_masks[g].to(device).float().unsqueeze(0)  # [1, N]
            pred_mask = mask_logits[q : q + 1, :]  # [1, N]
            mask_bce = self.sigmoid_ce(pred_mask, tgt_mask)
            mask_dice = self.sigmoid_dice_loss(pred_mask, tgt_mask)
            mask_bce_list.append(mask_bce)
            mask_dice_list.append(mask_dice)

        cls_loss = F.cross_entropy(
            class_logits, tgt_labels, weight=self._get_cls_weights(device)
        )
        if len(mask_bce_list) == 0:
            # 没有匹配到GT，所有query都预测no-object
            mask_bce_loss = class_logits.new_tensor(0.0)
            mask_dice_loss = class_logits.new_tensor(0.0)
        else:
            mask_bce_loss = torch.mean(torch.cat(mask_bce_list, dim=0))
            mask_dice_loss = torch.mean(torch.cat(mask_dice_list, dim=0))
        
        # 添加负样本的mask loss（unmatched queries应该预测empty mask）
        unmatched_queries = [q for q in range(Q) if q not in used]
        if len(unmatched_queries) > 0:
            # 负样本：预测的mask应该全为0
            unmatched_masks = mask_logits[unmatched_queries, :]  # [U, N]
            # 惩罚预测为1的区域
            negative_bce = F.binary_cross_entropy_with_logits(
                unmatched_masks, 
                torch.zeros_like(unmatched_masks),
                reduction='mean'
            )
            mask_bce_loss = mask_bce_loss + 0.5 * negative_bce

        return {
            "qm_cls_loss": cls_loss,
            "qm_mask_bce": mask_bce_loss,
            "qm_mask_dice": mask_dice_loss,
        }

    def _get_cls_weights(self, device: torch.device) -> torch.Tensor:
        weights = torch.ones(self.num_classes + 1, device=device)
        weights[-1] = self.no_object_weight
        return weights



