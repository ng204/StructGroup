"""
按类别保底的proposal保留机制
训练和推理时按类别分配topK名额，确保每个类别都有足够的proposal
"""
import torch
import torch.nn as nn


def retain_proposals_by_category(proposals_idx, proposals_offset, cls_scores, 
                                 max_proposal_num, category_quota=None):
    """
    按类别保底保留proposal
    
    Args:
        proposals_idx: (sumNPoint, 2) proposal点索引
        proposals_offset: (num_proposals + 1,) proposal偏移
        cls_scores: (num_proposals, num_classes + 1) 类别分数
        max_proposal_num: 最大proposal数量
        category_quota: dict {class_id: quota} 每个类别的配额，如果None则均匀分配
    
    Returns:
        retained_proposals_idx: 保留的proposal索引
        retained_proposals_offset: 保留的proposal偏移
        retained_indices: 保留的proposal原始索引
    """
    num_proposals = cls_scores.size(0)
    num_classes = cls_scores.size(1) - 1  # 排除背景类
    
    if num_proposals == 0:
        return proposals_idx, proposals_offset, torch.zeros(0, dtype=torch.long, device=proposals_idx.device)
    
    # 预测每个proposal的类别
    pred_classes = cls_scores[:, :num_classes].argmax(dim=1)  # (num_proposals,)
    
    # 如果没有指定配额，则均匀分配
    if category_quota is None:
        quota_per_class = max_proposal_num // num_classes
        category_quota = {i: quota_per_class for i in range(num_classes)}
        # 剩余的数量分配给前几个类别
        remainder = max_proposal_num % num_classes
        for i in range(remainder):
            category_quota[i] = category_quota.get(i, 0) + 1
    else:
        # 如果category_quota是字典，但键是字符串（如'stem', 'leaf', 'branch'），需要转换
        if isinstance(category_quota, dict):
            # 检查是否有字符串键
            if any(isinstance(k, str) for k in category_quota.keys()):
                # 映射字符串到类别ID：stem=0, leaf=1, branch=2
                str_to_id = {'stem': 0, 'leaf': 1, 'branch': 2}
                category_quota = {str_to_id.get(k, k): v for k, v in category_quota.items()}
    
    # 按类别收集proposal
    retained_indices_list = []
    
    for class_id in range(num_classes):
        class_mask = (pred_classes == class_id)
        class_indices = torch.nonzero(class_mask).squeeze(1)
        
        if class_indices.numel() == 0:
            continue
        
        # 获取该类别的分数（使用该类别的logit）
        class_scores = cls_scores[class_indices, class_id]
        
        # 按分数排序
        sorted_indices = torch.argsort(class_scores, descending=True)
        class_indices_sorted = class_indices[sorted_indices]
        
        # 保留配额内的proposal
        quota = category_quota.get(class_id, 0)
        if quota > 0:
            retained_class_indices = class_indices_sorted[:min(quota, len(class_indices_sorted))]
            retained_indices_list.append(retained_class_indices)
    
    if len(retained_indices_list) == 0:
        # 如果没有保留任何proposal，返回空
        empty_idx = torch.zeros((0, 2), dtype=proposals_idx.dtype, device=proposals_idx.device)
        empty_offset = torch.zeros((1,), dtype=proposals_offset.dtype, device=proposals_offset.device)
        return empty_idx, empty_offset, torch.zeros(0, dtype=torch.long, device=proposals_idx.device)
    
    # 合并所有保留的proposal索引
    retained_indices = torch.cat(retained_indices_list)
    retained_indices = torch.sort(retained_indices)[0]  # 排序
    
    # 根据保留的索引重新构建proposals_idx和proposals_offset
    num_retained = len(retained_indices)
    
    # 构建新的proposals_offset
    retained_proposals_offset = torch.zeros((num_retained + 1,), dtype=proposals_offset.dtype, device=proposals_offset.device)
    
    # 计算每个保留proposal的点数
    for i, prop_idx in enumerate(retained_indices):
        start = proposals_offset[prop_idx].item()
        end = proposals_offset[prop_idx + 1].item()
        num_points = end - start
        retained_proposals_offset[i + 1] = retained_proposals_offset[i] + num_points
    
    # 构建新的proposals_idx
    total_points = retained_proposals_offset[-1].item()
    retained_proposals_idx = torch.zeros((total_points, 2), dtype=proposals_idx.dtype, device=proposals_idx.device)
    
    current_offset = 0
    for i, prop_idx in enumerate(retained_indices):
        start = proposals_offset[prop_idx].item()
        end = proposals_offset[prop_idx + 1].item()
        num_points = end - start
        
        # 复制原始proposal的点
        original_points = proposals_idx[start:end]
        retained_proposals_idx[current_offset:current_offset + num_points, 1] = original_points[:, 1]
        retained_proposals_idx[current_offset:current_offset + num_points, 0] = i
        
        current_offset += num_points
    
    return retained_proposals_idx, retained_proposals_offset, retained_indices

