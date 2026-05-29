import argparse
import os
import os.path as osp
from operator import itemgetter

import numpy as np
import colorsys

# yapf:disable
def generate_hsv_colors(num_colors=256, s=0.9, v=0.95):
    """均匀 HSV 采样生成颜色，返回 shape=(num_colors,3) 的 0~255 RGB"""
    colors = []
    for i in range(num_colors):
        h = float(i) / num_colors  # [0,1)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(colors, dtype=np.float32)

# 生成 256 个均匀分布的高饱和度颜色
COLOR_DETECTRON2 = generate_hsv_colors(256)

def select_contrasting_colors(base_color_idx, num_colors, total_colors):
    """
    为同一类别内的多个实例选择差异最大的颜色

    Args:
        base_color_idx: 起始颜色索引
        num_colors: 需要的颜色数量
        total_colors: 总颜色数量

    Returns:
        list: 选择的颜色索引列表
    """
    if num_colors <= 1:
        return [base_color_idx]

    # 计算颜色之间的HSV差异
    selected_indices = [base_color_idx]

    for i in range(1, num_colors):
        max_min_diff = -1
        best_color_idx = None

        for color_idx in range(total_colors):
            if color_idx in selected_indices:
                continue

            # 计算与已选颜色的最小HSV差异
            min_diff = float('inf')
            for selected_idx in selected_indices:
                # 简化的颜色差异计算（RGB欧几里得距离）
                color1 = COLOR_DETECTRON2[selected_idx]
                color2 = COLOR_DETECTRON2[color_idx]
                diff = np.linalg.norm(color1 - color2)
                min_diff = min(min_diff, diff)

            if min_diff > max_min_diff:
                max_min_diff = min_diff
                best_color_idx = color_idx

        if best_color_idx is not None:
            selected_indices.append(best_color_idx)

    return selected_indices

# yapf:enable

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_NAMES = np.array([
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink',
    'bathtub', 'otherfurniture'
])
CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160],
    # Panax数据集颜色映射
    'stem': [139, 69, 19],      # 棕色
    'leaf': [34, 139, 34],      # 森林绿
    'branch': [30, 144, 255]     # 赭石色
}
SEMANTIC_IDX2NAME = {
    1: 'wall',
    2: 'floor',
    3: 'cabinet',
    4: 'bed',
    5: 'chair',
    6: 'sofa',
    7: 'table',
    8: 'door',
    9: 'window',
    10: 'bookshelf',
    11: 'picture',
    12: 'counter',
    14: 'desk',
    16: 'curtain',
    24: 'refridgerator',
    28: 'shower curtain',
    33: 'toilet',
    34: 'sink',
    36: 'bathtub',
    39: 'otherfurniture'
}


def get_coords_color(opt):

    coord_file = osp.join(opt.prediction_path, 'coords', opt.room_name + '.npy')
    color_file = osp.join(opt.prediction_path, 'colors', opt.room_name + '.npy')
    label_file = osp.join(opt.prediction_path, 'semantic_label', opt.room_name + '.npy')
    inst_label_file = osp.join(opt.prediction_path, 'gt_instance', opt.room_name + '.txt')
    xyz = np.load(coord_file)
    rgb = np.load(color_file)
    label = np.load(label_file)
    # 对于Panax/S3DIS数据集，rgb已经是原始颜色特征，无需归一化处理

    if (opt.task == 'semantic_gt'):
        label = label.astype(int)
        label_rgb = np.zeros(rgb.shape)
        # 为Panax数据集创建类别名称映射
        panax_names = ['stem', 'leaf', 'branch']
        for i, class_name in enumerate(panax_names):
            mask = label == i
            if mask.any():
                label_rgb[mask] = CLASS_COLOR[class_name]
        rgb = label_rgb

    elif (opt.task == 'semantic_pred'):
        semantic_file = os.path.join(opt.prediction_path, 'semantic_pred', opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(int)  # 0~2 for Panax dataset
        # 为Panax数据集创建类别名称映射
        panax_names = ['stem', 'leaf', 'branch']
        label_pred_rgb = np.zeros(rgb.shape)
        for i, class_name in enumerate(panax_names):
            mask = label_pred == i
            if mask.any():
                label_pred_rgb[mask] = CLASS_COLOR[class_name]
        rgb = label_pred_rgb

    elif (opt.task == 'offset_semantic_pred'):
        semantic_file = os.path.join(opt.prediction_path, 'semantic_pred', opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

        offset_file = os.path.join(opt.prediction_path, 'offset_pred', opt.room_name + '.npy')
        assert os.path.isfile(offset_file), 'No offset result - {}.'.format(offset_file)
        offset_coords = np.load(offset_file)
        xyz += offset_coords

    # same color order according to instance pointnum
    elif (opt.task == 'instance_gt'):
        # 仅在需要 GT 实例时读取，避免其他任务缺文件报错
        inst_label_file = osp.join(opt.prediction_path, 'gt_instance', opt.room_name + '.txt')
        assert os.path.isfile(inst_label_file), f'No gt_instance file: {inst_label_file}'
        raw_inst = np.array(open(inst_label_file).read().splitlines(), dtype=int)
        # S3DIS 编码：semantic_id * 1000 + inst_id（inst_id 已经是 0,1,2,...）
        semantic_from_inst = raw_inst // 1000
        inst_id_raw = raw_inst % 1000
        
        # 修复：为每个语义类别分别处理实例ID，避免不同类别的相同inst_id被映射到相同颜色
        # 使用完整的 raw_inst 值作为唯一标识，而不是只使用 inst_id
        # 这样不同类别的实例（即使inst_id相同）也会有不同的颜色
        unique_inst_values, remapped = np.unique(raw_inst, return_inverse=True)
        inst_label = remapped  # 连续编号，每个唯一的 raw_inst 值对应一个编号

        print('Instance number: {}'.format(inst_label.max() + 1))
        print('Semantic distribution: stem={}, leaf={}, branch={}'.format(
            (semantic_from_inst == 0).sum(), 
            (semantic_from_inst == 1).sum(), 
            (semantic_from_inst == 2).sum()))
        
        # 为不同语义类别分配不同的调色盘，确保 stem/leaf/branch 颜色明显区分
        # 打散基础调色盘，避免相邻颜色过近；step 与 256 互质更好
        def spread_palette(palette, step=97):
            n = len(palette)
            order = [(i * step) % n for i in range(n)]
            return palette[order]

        base_spread = spread_palette(COLOR_DETECTRON2, step=97)
        class_palettes = {
            0: base_spread,                    # stem
            1: np.roll(base_spread, 85, axis=0),   # leaf
            2: np.roll(base_spread, 170, axis=0),  # branch
        }

        inst_label_rgb = np.zeros(rgb.shape)
        
        # 改进：为每个类别内的实例选择差异最大的颜色，避免色系相近
        # 1. 统计每个类别内的实例及其点数（动态处理所有semantic_id）
        class_instances = {}  # {sem_id: [(inst_val, point_count), ...]}
        for inst_val in unique_inst_values:
            sem_id = inst_val // 1000
            if sem_id not in class_instances:
                class_instances[sem_id] = []
            point_count = (raw_inst == inst_val).sum()
            class_instances[sem_id].append((inst_val, point_count))
        
        # 2. 为每个类别按点数排序，并选择差异最大的颜色
        for sem_id in sorted(class_instances.keys()):
            if len(class_instances[sem_id]) == 0:
                continue
            
            # 按点数降序排序（大实例优先）
            instances_sorted = sorted(class_instances[sem_id], key=lambda x: x[1], reverse=True)
            num_instances = len(instances_sorted)
            
            # 为每个类别选择差异最大的颜色索引
            # 如果sem_id不在预定义的调色盘中，使用默认调色盘
            if sem_id in class_palettes:
                palette = class_palettes[sem_id]
            else:
                # 对于未知类别，使用打散的调色盘并滚动
                palette = np.roll(base_spread, (sem_id * 85) % len(base_spread), axis=0)
            base_color_idx = (sem_id * 85) % len(COLOR_DETECTRON2)  # 不同类别起始位置不同
            selected_color_indices = select_contrasting_colors(base_color_idx, num_instances, len(COLOR_DETECTRON2))
            
            # 分配颜色
            for idx, (inst_val, _) in enumerate(instances_sorted):
                color_idx = selected_color_indices[idx]
                color = COLOR_DETECTRON2[color_idx]
                inst_label_rgb[raw_inst == inst_val] = color

        rgb = inst_label_rgb

    elif (opt.task == 'instance_pred'):
        # 优先使用模型导出的实例结果（pred_instance/predicted_masks）
        instance_file = os.path.join(opt.prediction_path, 'pred_instance', opt.room_name + '.txt')
        if os.path.isfile(instance_file):
            with open(instance_file, 'r') as f:
                # 格式: relative_mask_path class_id score
                masks = [line.rstrip().split() for line in f.readlines()]

            # 参数：分数阈值与最小点数过滤
            score_thr = float(getattr(opt, 'score_thr', 0.0))
            min_points = int(getattr(opt, 'min_points', 1))
            # AP_50过滤：只显示IoU>=0.5的预测实例
            use_ap50_filter = int(getattr(opt, 'use_ap50_filter', 0))
            iou_threshold = float(getattr(opt, 'iou_threshold', 0.5))

            num_points = rgb.shape[0]
            inst_label_rgb = np.zeros(rgb.shape, dtype=float)  # 背景黑色，0~255
            best_priority = np.full(num_points, -np.inf, dtype=float)  # 竞争指标：score/sqrt(area)

            # 分数降序，以高分实例优先
            scores = np.array([float(x[-1]) for x in masks])
            sort_inds = np.argsort(scores)[::-1]

            # 语义参考：优先使用 GT semantic_label；若无则使用 semantic_pred（避免跨类误覆盖）
            # 诊断模式：可通过 --use_semantic_ref 0 禁用语义一致性约束
            semantic_ref = None
            use_semantic_ref = int(getattr(opt, 'use_semantic_ref', 1))
            if use_semantic_ref:
                semantic_pred_file = os.path.join(opt.prediction_path, 'semantic_pred', opt.room_name + '.npy')
                semantic_label_file = os.path.join(opt.prediction_path, 'semantic_label', opt.room_name + '.npy')
                if os.path.isfile(semantic_label_file):
                    semantic_ref = np.load(semantic_label_file).astype(int)
                elif os.path.isfile(semantic_pred_file):
                    semantic_ref = np.load(semantic_pred_file).astype(int)
            else:
                print('[diagnosis] 语义一致性约束已禁用 (--use_semantic_ref 0)，将直接使用实例mask，不进行语义裁剪')

            # AP_50过滤：加载GT实例并计算IoU
            gt_instances_dict = None  # {sem_id: {inst_id: mask_bool}}
            if use_ap50_filter:
                gt_instance_file = os.path.join(opt.prediction_path, 'gt_instance', opt.room_name + '.txt')
                if os.path.isfile(gt_instance_file):
                    print(f'[AP_50] 加载GT实例文件: {gt_instance_file}')
                    gt_inst_raw = np.array(open(gt_instance_file).read().splitlines(), dtype=int)
                    if gt_inst_raw.shape[0] == num_points:
                        # 解析GT实例：格式为 semantic_id * 1000 + inst_id
                        unique_gt_inst = np.unique(gt_inst_raw)
                        gt_instances_dict = {}  # {sem_id: {inst_id: mask_bool}}
                        for inst_val in unique_gt_inst:
                            if inst_val < 1000:  # 跳过背景
                                continue
                            sem_id = inst_val // 1000
                            inst_id = inst_val % 1000
                            if sem_id not in gt_instances_dict:
                                gt_instances_dict[sem_id] = {}
                            gt_instances_dict[sem_id][inst_id] = (gt_inst_raw == inst_val)
                        print(f'[AP_50] GT实例统计: {sum(len(v) for v in gt_instances_dict.values())} 个实例')
                    else:
                        print(f'[AP_50] GT实例点数不匹配: {gt_inst_raw.shape[0]} vs {num_points}, 禁用AP_50过滤')
                        use_ap50_filter = 0
                else:
                    print(f'[AP_50] GT实例文件不存在: {gt_instance_file}, 禁用AP_50过滤')
                    use_ap50_filter = 0

            # 类别调色盘：先做一次"间隔取色"打散相邻色，避免连续实例颜色过于相近
            def spread_palette(palette, step):
                """间隔取色打散相邻色，step 与 256 互质更好"""
                n = len(palette)
                order = [(i * step) % n for i in range(n)]
                return palette[order]

            # 采用步长 101（与 256 互质）打散，并为不同类别做大幅滚动偏移
            base_spread = spread_palette(COLOR_DETECTRON2, step=101)  # (n,3) 0~255
            class_palettes = {
                0: base_spread,                               # stem
                1: np.roll(base_spread, 85, axis=0),          # leaf
                2: np.roll(base_spread, 170, axis=0),         # branch
            }
            
            # 改进：预先统计每个类别内的实例，使用select_contrasting_colors选择差异最大的颜色
            # 先收集所有符合条件的实例信息
            valid_instances_by_class = {0: [], 1: [], 2: []}  # {cls_id: [(score, pts, mask_bool, priority), ...]}

            # 调试统计
            debug_stats = {
                'total': 0,
                'score_filtered': 0,
                'cls_filtered': 0,
                'mask_not_found': 0,
                'mask_mismatch': 0,
                'semantic_filtered': 0,
                'min_points_filtered': 0,
                'ap50_filtered': 0,
                'kept': 0
            }

            for idx in sort_inds:
                debug_stats['total'] += 1
                rel_path, cls_raw, score_str = masks[idx][0], masks[idx][1], masks[idx][2]
                try:
                    score = float(score_str)
                except Exception:
                    continue
                if score < score_thr:
                    debug_stats['score_filtered'] += 1
                    continue

                # class_id 映射：原文件 1/2/3 -> 0/1/2
                try:
                    cls_id = int(cls_raw) - 1
                except Exception:
                    debug_stats['cls_filtered'] += 1
                    continue
                if cls_id not in [0, 1, 2]:
                    debug_stats['cls_filtered'] += 1
                    continue

                # mask_path 以 prediction_path 为根，兼容含 "/" 或 "\" 的相对路径
                mask_rel = rel_path.lstrip('/').replace('\\', '/')
                # 优先在 prediction_path 下找；若不存在，回退到 prediction_path/pred_instance 下找
                mask_path = os.path.join(opt.prediction_path, mask_rel)
                if not os.path.isfile(mask_path):
                    alt_mask_path = os.path.join(opt.prediction_path, 'pred_instance', mask_rel)
                    if os.path.isfile(alt_mask_path):
                        mask_path = alt_mask_path
                    else:
                        debug_stats['mask_not_found'] += 1
                        if debug_stats['mask_not_found'] <= 3:  # 只打印前3个警告
                            print(f"[warn] mask file not found: {mask_path} (also tried: {alt_mask_path})")
                        continue
                mask = np.array(open(mask_path).read().splitlines(), dtype=int)
                if mask.shape[0] != num_points:
                    debug_stats['mask_mismatch'] += 1
                    if debug_stats['mask_mismatch'] <= 3:  # 只打印前3个警告
                        print(f"[warn] mask length mismatch: {mask.shape[0]} vs {num_points}, skip {mask_path}")
                    continue

                mask_bool_original = mask.astype(bool)  # 保存原始mask用于IoU计算
                mask_bool = mask_bool_original.copy()  # 用于语义过滤

                # 语义一致性约束：智能版本，平衡覆盖率和质量
                # 策略：允许一定程度的跨类容忍，但过滤明显错误的预测
                if semantic_ref is not None:
                    # 计算mask中各类别的点数
                    covered_sem = semantic_ref[mask_bool]
                    if len(covered_sem) == 0:
                        debug_stats['semantic_filtered'] += 1
                        continue
                    
                    unique, counts = np.unique(covered_sem, return_counts=True)
                    sem_dict = dict(zip(unique, counts))
                    correct_pts = sem_dict.get(cls_id, 0)
                    total_pts = len(covered_sem)
                    correct_ratio = correct_pts / total_pts
                    
                    # 找到主导类别
                    dominant_cls = unique[np.argmax(counts)]
                    dominant_ratio = counts.max() / total_pts
                    
                    # 智能过滤策略（更宽松，平衡覆盖率和质量）：
                    # 1. 如果对应语义类占比>=50%，直接保留（高质量预测）
                    # 2. 如果对应语义类占比>=30%，保留（中等质量）
                    # 3. 如果对应语义类是主导类（>=50%），保留（即使占比<30%，但主导类正确）
                    # 4. 如果对应语义类占比>=15%且总点数较大(>=50)，保留（边界模糊，但有一定正确性）
                    # 5. 否则过滤（明显错误的预测：<15%且不是主导类）
                    if correct_ratio >= 0.5:
                        # 高质量：对应语义类占主导，直接保留
                        pass  # mask_bool保持不变
                    elif correct_ratio >= 0.3:
                        # 中等质量：对应语义类占比>=30%，保留
                        pass  # mask_bool保持不变
                    elif dominant_cls == cls_id and dominant_ratio >= 0.5:
                        # 主导类正确：即使占比<30%，但主导类正确，保留
                        pass  # mask_bool保持不变
                    elif correct_ratio >= 0.15 and total_pts >= 50:
                        # 边界情况：点数较多，可能是边界模糊，保留但裁剪到对应语义类
                        strict_mask = mask_bool & (semantic_ref == cls_id)
                        strict_pts = int(strict_mask.sum())
                        if strict_pts >= min_points:
                            mask_bool = strict_mask
                        else:
                            # 裁剪后点数不足，跳过
                            debug_stats['semantic_filtered'] += 1
                            continue
                    else:
                        # 明显错误：对应语义类占比太低且不是主导类，跳过
                        debug_stats['semantic_filtered'] += 1
                        continue

                pts = int(mask_bool.sum())
                if pts < min_points:
                    debug_stats['min_points_filtered'] += 1
                    continue

                # AP_50过滤：计算与GT的IoU，只保留IoU>=0.5的预测实例
                # 注意：使用原始mask_bool_original计算IoU，因为评估时不会先做语义过滤
                if use_ap50_filter and gt_instances_dict is not None:
                    max_iou = 0.0
                    # 只与对应语义类的GT实例计算IoU
                    if cls_id in gt_instances_dict:
                        for gt_inst_id, gt_mask_bool in gt_instances_dict[cls_id].items():
                            # 使用原始mask计算IoU，与评估逻辑一致
                            intersection = np.sum(mask_bool_original & gt_mask_bool)
                            union = np.sum(mask_bool_original | gt_mask_bool)
                            if union > 0:
                                iou = intersection / union
                                max_iou = max(max_iou, iou)
                    # 如果最大IoU < 阈值，跳过该预测实例
                    if max_iou < iou_threshold:
                        debug_stats['ap50_filtered'] += 1
                        continue

                debug_stats['kept'] += 1

                # 计算优先级（面积惩罚，偏好小实例）
                priority = score / (np.sqrt(pts) + 1e-6)
                # 先收集实例信息，稍后统一分配颜色
                valid_instances_by_class[cls_id].append((score, pts, mask_bool, priority))
            
            # 第二遍：为每个类别选择差异最大的颜色，然后分配
            # Top-K 过滤：与评估时的逻辑一致，确保每个类别保留足够的实例数量
            top_k_per_class = getattr(opt, 'top_k_per_class', None)
            if top_k_per_class is not None:
                # top_k_per_class 可以是整数（所有类别相同）或列表 [stem_k, leaf_k, branch_k]
                if isinstance(top_k_per_class, (list, tuple)) and len(top_k_per_class) >= 3:
                    top_k_dict = {0: int(top_k_per_class[0]), 1: int(top_k_per_class[1]), 2: int(top_k_per_class[2])}
                elif isinstance(top_k_per_class, (int, float)):
                    k = int(top_k_per_class)
                    top_k_dict = {0: k, 1: k, 2: k}
                else:
                    top_k_dict = None
            else:
                top_k_dict = None
            
            kept_count = {0: 0, 1: 0, 2: 0}
            for cls_id in [0, 1, 2]:
                if len(valid_instances_by_class[cls_id]) == 0:
                    continue
                
                # 按score降序排序（与评估时一致，评估时按score_pred排序）
                instances_sorted = sorted(valid_instances_by_class[cls_id], key=lambda x: x[0], reverse=True)
                
                # 应用Top-K过滤（与评估时一致）
                if top_k_dict is not None and cls_id in top_k_dict:
                    k = top_k_dict[cls_id]
                    instances_sorted = instances_sorted[:k]
                
                num_instances = len(instances_sorted)
                
                # 为每个类别选择差异最大的颜色索引
                base_color_idx = (cls_id * 85) % len(COLOR_DETECTRON2)  # 不同类别起始位置不同
                selected_color_indices = select_contrasting_colors(base_color_idx, num_instances, len(COLOR_DETECTRON2))
                
                # 分配颜色（按优先级顺序）
                for idx, (score, pts, mask_bool, priority) in enumerate(instances_sorted):
                    # 逐点竞争：仅在 mask 内且分数更高时覆盖颜色
                    overwrite = mask_bool & (priority > best_priority)
                    if overwrite.any():
                        color_idx = selected_color_indices[idx]
                        color = COLOR_DETECTRON2[color_idx]
                        inst_label_rgb[overwrite] = color
                        best_priority[overwrite] = priority
                        kept_count[cls_id] += 1

            rgb = inst_label_rgb  # 背景保持黑色
            semantic_status = "enabled" if semantic_ref is not None else "disabled"
            ap50_status = f"enabled(IoU>={iou_threshold})" if use_ap50_filter else "disabled"
            print(f"[instance_pred] done. stem={kept_count[0]}, leaf={kept_count[1]}, branch={kept_count[2]}, "
                  f"score_thr={score_thr}, min_points={min_points}, semantic_ref={semantic_status}, AP_50_filter={ap50_status}")
            print(f"[debug] 过滤统计: total={debug_stats['total']}, score_filtered={debug_stats['score_filtered']}, "
                  f"cls_filtered={debug_stats['cls_filtered']}, mask_not_found={debug_stats['mask_not_found']}, "
                  f"mask_mismatch={debug_stats['mask_mismatch']}, semantic_filtered={debug_stats['semantic_filtered']}, "
                  f"min_points_filtered={debug_stats['min_points_filtered']}, ap50_filtered={debug_stats['ap50_filtered']}, "
                  f"kept={debug_stats['kept']}")
            if inst_label_rgb.max() == 0:
                print("[warn] 所有实例颜色仍为0，可能没有有效mask或阈值过高/掩码为空")

            covered_ratio = (best_priority > -np.inf).mean()
            print(f"[viz] covered_ratio={covered_ratio:.3f}, black_ratio={1.0 - covered_ratio:.3f}")

            # 若已成功基于 pred_instance 着色，直接返回，避免落入备选的 DBSCAN 流程覆盖颜色
            return xyz, rgb
        else:
            # 若没有 pred_instance，退回到语义+DBSCAN 的备选方案（每实例独色情色）
            print('未找到 pred_instance 结果，使用语义 + DBSCAN 进行实例着色（备选方案）')
            semantic_file = os.path.join(opt.prediction_path, 'semantic_pred', opt.room_name + '.npy')
            if os.path.isfile(semantic_file):
                from sklearn.cluster import DBSCAN

                semantic_pred = np.load(semantic_file).astype(int)
                global_instance_id = 0
                for target_class in [0, 1, 2]:  # stem, leaf, branch
                    class_mask = semantic_pred == target_class
                    if class_mask.sum() == 0:
                        continue
                    class_points = xyz[class_mask]
                    if len(class_points) <= 30:
                        continue

                    clustering = DBSCAN(eps=0.015, min_samples=5).fit(class_points)
                    class_instance_labels = clustering.labels_
                    valid_mask = class_instance_labels != -1
                    if valid_mask.sum() == 0:
                        continue

                    class_instance_labels = class_instance_labels[valid_mask]
                    valid_original_indices = np.where(class_mask)[0][valid_mask]
                    inst_label = np.zeros(len(rgb), dtype=int)
                    inst_label[valid_original_indices] = class_instance_labels + 1
                    unique_instances = np.unique(inst_label)
                    for inst_id in unique_instances:
                        if inst_id > 0:
                            color_idx = global_instance_id % len(COLOR_DETECTRON2)
                            rgb[inst_label == inst_id] = COLOR_DETECTRON2[color_idx]
                            global_instance_id += 1

    sem_valid = (label != -100)
    xyz = xyz[sem_valid]
    rgb = rgb[sem_valid]

    return xyz, rgb


def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    # ---- 防御性清洗，避免 NaN/Inf 或颜色超界导致读取失败 ----
    verts = np.asarray(verts)
    colors = np.asarray(colors, dtype=np.float32)
    # 若颜色长度与点不一致，截断到一致长度
    n = min(len(verts), len(colors))
    verts = verts[:n]
    colors = colors[:n]
    # 只保留有限值的点
    finite_mask = np.isfinite(verts).all(axis=1) & np.isfinite(colors).all(axis=1)
    verts = verts[finite_mask]
    colors = colors[finite_mask]
    # 若颜色是 0~255，先归一化到 0~1，再裁剪
    if colors.max() > 1.5:
        colors = colors / 255.0
    colors = np.clip(colors, 0.0, 1.0)

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    # 仅在存在面时写 face 元素，避免某些查看器对 0 面的警告
    if len(indices) > 0:
     file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        r = int(color[0] * 255)
        g = int(color[1] * 255)
        b = int(color[2] * 255)
        file.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(vert[0], vert[1], vert[2], r, g, b))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prediction_path', help='path to the prediction results', default='./results')
    parser.add_argument('--room_name', help='room_name', default='scene0011_00')
    parser.add_argument(
        '--task',
        help='input/semantic_gt/semantic_pred/offset_semantic_pred/instance_gt/instance_pred',
        default='instance_pred')
    parser.add_argument('--out', help='output point cloud file in FILE.ply format', default='')
    # 默认阈值收紧，避免噪声实例把全云涂满：score_thr=0.1, min_points=50
    parser.add_argument('--score_thr', type=float, default=0.1, help='score threshold for instance_pred')
    parser.add_argument('--min_points', type=int, default=50, help='minimum points for an instance to be kept')
    parser.add_argument('--use_semantic_ref', type=int, default=1, 
                       help='whether to use semantic consistency constraint (1=enable, 0=disable for diagnosis)')
    parser.add_argument('--use_ap50_filter', type=int, default=0,
                       help='whether to filter predictions by IoU>=0.5 with GT (1=enable AP_50 filter, 0=disable)')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for AP_50 filter (default: 0.5)')
    parser.add_argument('--top_k_per_class', type=str, default=None,
                       help='Top-K per class filter: integer or list like "[200,500,200]" (stem,leaf,branch). Default: None (no top-k filter)')
    opt = parser.parse_args()
    
    # 解析 top_k_per_class 参数
    if opt.top_k_per_class is not None:
        try:
            # 尝试解析为列表格式 "[200,500,200]"
            if opt.top_k_per_class.startswith('[') and opt.top_k_per_class.endswith(']'):
                opt.top_k_per_class = [int(x.strip()) for x in opt.top_k_per_class[1:-1].split(',')]
            else:
                # 尝试解析为单个整数
                opt.top_k_per_class = int(opt.top_k_per_class)
        except:
            print(f"[warn] 无法解析 top_k_per_class={opt.top_k_per_class}，将禁用top-k过滤")
            opt.top_k_per_class = None

    xyz, rgb = get_coords_color(opt)
    points = xyz[:, :3]
    # 自适应归一化：若原本是 0~255 则除以 255，否则保持 0~1
    rgb = rgb.astype(np.float32)
    colors = rgb / 255.0 if rgb.max() > 1.5 else rgb

    if opt.out:
        assert '.ply' in opt.out, 'output cloud file should be in FILE.ply format'
        write_ply(points, colors, None, opt.out)
    else:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pc)
        vis.get_render_option().point_size = 1.5
        vis.run()
        vis.destroy_window()
