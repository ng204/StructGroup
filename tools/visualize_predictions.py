"""
直接可视化模型预测的实例，不使用DBSCAN重新聚类
"""
import argparse
import os
import os.path as osp
import numpy as np

# 扩展颜色调色板
COLORS = np.array([
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
    [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
    [128, 0, 128], [0, 128, 128], [255, 128, 0], [255, 0, 128], [128, 255, 0],
    [0, 255, 128], [128, 0, 255], [255, 128, 128], [128, 255, 128], [128, 128, 255],
    [255, 215, 0], [255, 20, 147], [0, 191, 255], [255, 105, 180], [50, 205, 50],
    [255, 140, 0], [75, 0, 130], [255, 69, 0], [173, 216, 230], [144, 238, 144],
    [255, 182, 193], [221, 160, 221], [176, 196, 222], [255, 160, 122], [135, 206, 250],
    [240, 128, 128], [152, 251, 152], [230, 230, 250], [255, 228, 181], [176, 224, 230],
    [250, 128, 114], [147, 112, 219], [100, 149, 237], [72, 209, 204], [199, 21, 133],
    [255, 99, 71], [138, 43, 226], [205, 133, 63], [210, 105, 30], [218, 112, 214],
], dtype=np.float64)

def write_ply(points, colors, labels, output_file):
    """写入PLY文件"""
    output_dir = osp.dirname(output_file)
    if output_dir:  # 避免当前目录时dirname返回空字符串
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        
        for i in range(len(points)):
            f.write(f'{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} ')
            f.write(f'{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n')

def visualize_predictions(pred_path, room_name, conf_threshold=0.0):
    """直接使用模型预测的实例mask"""
    
    # 加载坐标
    coord_file = osp.join(pred_path, 'coords', f'{room_name}.npy')
    coords = np.load(coord_file)
    
    # 加载原始颜色
    color_file = osp.join(pred_path, 'colors', f'{room_name}.npy')
    colors = np.load(color_file)
    
    # 读取预测实例列表
    pred_file = osp.join(pred_path, 'pred_instance', f'{room_name}.txt')
    
    print(f"加载场景: {room_name}")
    print(f"总点数: {len(coords)}")
    print()
    
    # 初始化实例标签
    instance_labels = np.zeros(len(coords), dtype=int)
    instance_colors = np.zeros((len(coords), 3), dtype=np.float64)
    
    # 读取每个预测的实例
    instances_info = []
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            mask_file = parts[0]
            class_id = int(parts[1])
            conf = float(parts[2])
            instances_info.append((mask_file, class_id, conf))
    
    print(f"预测实例总数: {len(instances_info)}")
    
    # 按类别统计
    class_counts = {}
    for _, class_id, conf in instances_info:
        if conf >= conf_threshold:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    print(f"应用置信度阈值 {conf_threshold} 后:")
    for cls_id in sorted(class_counts.keys()):
        cls_name = ['stem', 'leaf', 'branch'][cls_id - 1]
        print(f"  类别{cls_id}({cls_name}): {class_counts[cls_id]}个实例")
    print()
    
    # 加载并合并所有实例mask
    color_idx = 0
    filtered_count = 0
    
    for mask_file, class_id, conf in instances_info:
        if conf < conf_threshold:
            filtered_count += 1
            continue
        
        # 读取mask
        full_mask_path = osp.join(pred_path, 'pred_instance', mask_file)
        mask = np.loadtxt(full_mask_path, dtype=int)
        
        # 检查mask长度是否与coords匹配
        if len(mask) != len(coords):
            print(f"  警告: {mask_file} mask长度({len(mask)})与coords长度({len(coords)})不匹配，跳过")
            filtered_count += 1
            continue
        
        # 为该实例分配颜色
        color = COLORS[color_idx % len(COLORS)]
        
        # 应用mask
        mask_bool = mask.astype(bool)
        instance_labels[mask_bool] = color_idx + 1
        instance_colors[mask_bool] = color
        
        color_idx += 1
    
    if filtered_count > 0:
        print(f"过滤掉{filtered_count}个低置信度实例")
    print(f"最终可视化{color_idx}个实例")
    print()
    
    # 统计每个类别的点数和覆盖率
    # 加载语义标签以计算覆盖率
    semantic_file = osp.join(pred_path, 'semantic_pred', f'{room_name}.npy')
    if osp.exists(semantic_file):
        semantic_pred = np.load(semantic_file).astype(int)
        
        for cls_id in [1, 2, 3]:
            cls_name = ['stem', 'leaf', 'branch'][cls_id - 1]
            # 统计该类别所有实例的点数
            cls_point_count = sum([
                (np.loadtxt(osp.join(pred_path, 'pred_instance', mask_file), dtype=int).sum())
                for mask_file, cid, conf in instances_info 
                if cid == cls_id and conf >= conf_threshold
            ])
            
            # 计算覆盖率
            total_cls_points = (semantic_pred == (cls_id - 1)).sum()
            coverage = (cls_point_count / total_cls_points * 100) if total_cls_points > 0 else 0
            
            print(f"{cls_name}类别: 预测点数={cls_point_count}, 总点数={total_cls_points}, 覆盖率={coverage:.1f}%")
    else:
        for cls_id in [1, 2, 3]:
            cls_name = ['stem', 'leaf', 'branch'][cls_id - 1]
            cls_point_count = sum([
                (np.loadtxt(osp.join(pred_path, 'pred_instance', mask_file), dtype=int).sum())
                for mask_file, cid, conf in instances_info 
                if cid == cls_id and conf >= conf_threshold
            ])
            print(f"{cls_name}类别总点数: {cls_point_count}")
    
    # 未被预测的点（黑色点）用灰色显示
    unpredicted_mask = instance_colors.sum(axis=1) == 0
    if unpredicted_mask.sum() > 0:
        instance_colors[unpredicted_mask] = [50, 50, 50]  # 深灰色
        print(f"\n未被预测的点: {unpredicted_mask.sum()}个 (显示为灰色)")
    
    return coords, instance_colors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化实例分割预测结果')
    parser.add_argument('--prediction_path', required=True, help='测试结果路径')
    parser.add_argument('--room_name', required=True, help='场景名称（例如: Area_18_Panax_17）')
    parser.add_argument('--conf_threshold', type=float, default=0.1, 
                       help='置信度阈值 (0.01-0.2)，越低检测越多实例但可能有噪声')
    parser.add_argument('--out', required=True, help='输出.ply文件路径')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Panax实例分割可视化")
    print("=" * 80)
    print()
    
    coords, colors = visualize_predictions(args.prediction_path, args.room_name, args.conf_threshold)
    
    print()
    print("=" * 80)
    print(f"保存PLY文件到: {args.out}")
    write_ply(coords, colors, None, args.out)
    
    print()
    print("✅ 完成！")
    print()
    print("下一步:")
    print("  1. 使用CloudCompare打开PLY文件查看")
    print(f"     cloudcompare.CloudCompare {args.out}")
    print()
    print("  2. 或进行表型分析:")
    print(f"     python3 tools/phenotype_analysis.py --prediction_path {args.prediction_path} --room_name {args.room_name} --conf_threshold {args.conf_threshold} --out_csv phenotype_results/{args.room_name}.csv")
    print()
    print("=" * 80)

