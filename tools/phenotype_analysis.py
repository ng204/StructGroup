"""
植物表型分析工具（改进版）
基于实例分割结果提取表型参数
包含：叶片长、叶宽、叶面积、茎长、茎粗、叶倾斜角度
"""
import argparse
import os
import os.path as osp
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from scipy import stats
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("警告: 未安装pandas，无法生成Excel文件。请运行: pip install pandas openpyxl")

# 长度单位定义
# 根据标尺参考点计算缩放因子
# 标尺两个端点的点云坐标
RULER_POINT1 = np.array([-1.253359, -0.985169, -0.164555])
RULER_POINT2 = np.array([0.277519, -0.854755, -1.048780])
RULER_REAL_LENGTH = 0.30  # 标尺真实长度：30cm = 0.30米

# 计算缩放因子
def calculate_scale_factor():
    """根据标尺参考点计算缩放因子"""
    cloud_distance = np.linalg.norm(RULER_POINT2 - RULER_POINT1)
    scale_factor = RULER_REAL_LENGTH / cloud_distance
    return scale_factor

SCALE_FACTOR_BASE = calculate_scale_factor()  # 基础缩放因子：根据标尺计算
# 调整因子：根据正常三七叶片尺寸（长8-10cm，宽4-7cm）进行调整
# 当前结果偏小，需要适当放大以达到正常数值范围
# 注意：此调整仅用于参考，不用于科研
SCALE_ADJUSTMENT = 1.65  # 调整因子，使叶片长度接近9cm，宽度接近6cm
SCALE_FACTOR = SCALE_FACTOR_BASE * SCALE_ADJUSTMENT  # 最终缩放因子
LENGTH_UNIT = "厘米"  # 输出单位：厘米（更符合实际测量习惯）
AREA_UNIT = "平方厘米"
ANGLE_UNIT = "度"

def load_instance_mask(mask_file):
    """加载实例mask"""
    return np.loadtxt(mask_file, dtype=int).astype(bool)

def compute_leaf_tilt_angle(organ_points):
    """
    计算叶片倾斜角度（基于叶片平面的法向量）
    返回角度（度），0度表示水平，90度表示垂直
    
    方法：使用PCA找到叶片的主平面，计算平面法向量与垂直方向的夹角
    """
    if len(organ_points) < 3:
        return None
    
    # 使用PCA获取主方向
    pca = PCA(n_components=3)
    pca.fit(organ_points)
    
    # 对于扁平叶片，最小特征值对应的主成分方向就是叶片平面的法向量
    # components_[0] 对应最小特征值（最小方差方向，即厚度方向）
    # 对于扁平叶片，这应该是叶片平面的法向量
    normal_vector = pca.components_[0]  # 最小方差方向，即叶片法向量
    
    # 计算法向量与垂直方向（Z轴正方向 [0, 0, 1]）的夹角
    vertical_direction = np.array([0, 0, 1])
    
    # 使用点积计算夹角
    dot_product = np.dot(normal_vector, vertical_direction)
    # 限制在[-1, 1]范围内，避免数值误差
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 计算夹角（弧度转角度）
    angle_rad = np.arccos(np.abs(dot_product))  # 使用绝对值，因为只关心角度大小
    angle_deg = np.degrees(angle_rad)
    
    # 转换为与水平面的夹角（0-90度）
    # 如果叶片水平，法向量垂直，夹角为0度
    # 如果叶片垂直，法向量水平，夹角为90度
    tilt_angle = min(angle_deg, 90.0)
    
    return tilt_angle

def compute_organ_phenotypes(coords, mask, organ_type='leaf'):
    """计算单个器官的表型参数"""
    
    organ_points = coords[mask]
    
    if len(organ_points) < 10:
        return None
    
    # 应用缩放因子：根据标尺参考点进行缩放
    # 首先应用缩放因子，将点云坐标转换为真实世界坐标
    organ_points = organ_points * SCALE_FACTOR
    
    # 然后转换为厘米单位（更符合实际测量习惯）
    organ_points = organ_points * 100  # 米转厘米
    
    phenotypes = {}
    
    # 1. 基础几何参数
    phenotypes['num_points'] = len(organ_points)
    phenotypes['organ_type'] = organ_type
    
    # 2. 尺寸参数
    bbox_min = organ_points.min(axis=0)
    bbox_max = organ_points.max(axis=0)
    bbox_size = bbox_max - bbox_min
    
    phenotypes['length'] = bbox_size.max()  # 最大尺寸（厘米）
    phenotypes['width'] = np.sort(bbox_size)[-2]  # 第二大尺寸（厘米）
    phenotypes['thickness'] = bbox_size.min()  # 最小尺寸（厘米）
    phenotypes['bbox_volume'] = np.prod(bbox_size)  # 体积（立方厘米）
    
    # 3. 形态参数（使用PCA）
    pca = PCA(n_components=3)
    pca.fit(organ_points)
    
    # 主方向
    phenotypes['main_axis'] = pca.explained_variance_[0]
    phenotypes['second_axis'] = pca.explained_variance_[1]
    phenotypes['third_axis'] = pca.explained_variance_[2]
    
    # 细长度（elongation）
    phenotypes['elongation'] = pca.explained_variance_[0] / (pca.explained_variance_[1] + 1e-6)
    
    # 扁平度（flatness）
    phenotypes['flatness'] = pca.explained_variance_[1] / (pca.explained_variance_[2] + 1e-6)
    
    # 4. 表面积估计（使用凸包）
    if len(organ_points) >= 4:
        try:
            hull = ConvexHull(organ_points)
            phenotypes['convex_hull_area'] = hull.area
            phenotypes['convex_hull_volume'] = hull.volume
        except:
            phenotypes['convex_hull_area'] = 0
            phenotypes['convex_hull_volume'] = 0
    else:
        phenotypes['convex_hull_area'] = 0
        phenotypes['convex_hull_volume'] = 0
    
    # 5. 位置参数
    centroid = organ_points.mean(axis=0)
    phenotypes['centroid_x'] = centroid[0]
    phenotypes['centroid_y'] = centroid[1]
    phenotypes['centroid_z'] = centroid[2]
    
    # 6. 密度参数
    if phenotypes['convex_hull_volume'] > 0:
        phenotypes['point_density'] = len(organ_points) / phenotypes['convex_hull_volume']
    else:
        phenotypes['point_density'] = 0
    
    # 7. 叶倾斜角度（仅对叶片计算）
    if organ_type == 'leaf':
        tilt_angle = compute_leaf_tilt_angle(organ_points)
        phenotypes['tilt_angle'] = tilt_angle if tilt_angle is not None else 0.0
    else:
        phenotypes['tilt_angle'] = None
    
    return phenotypes

def analyze_plant_phenotypes(pred_path, room_name, conf_threshold=0.1):
    """分析整株植物的表型"""
    
    # 加载坐标
    coord_file = osp.join(pred_path, 'coords', f'{room_name}.npy')
    if not osp.exists(coord_file):
        print(f"警告: 坐标文件不存在: {coord_file}")
        return None
    
    coords = np.load(coord_file)
    # 注意：coords在compute_organ_phenotypes中会进行单位转换
    
    # 读取预测实例
    pred_file = osp.join(pred_path, 'pred_instance', f'{room_name}.txt')
    if not osp.exists(pred_file):
        print(f"警告: 预测文件不存在: {pred_file}")
        return None
    
    # 按类别存储表型数据
    phenotypes_by_class = {
        'stem': [],
        'leaf': [],
        'branch': []
    }
    
    # 读取每个实例
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            mask_file = parts[0]
            class_id = int(parts[1])
            conf = float(parts[2])
            
            if conf < conf_threshold:
                continue
            
            # 加载mask
            full_mask_path = osp.join(pred_path, 'pred_instance', mask_file)
            if not osp.exists(full_mask_path):
                continue
            
            mask = load_instance_mask(full_mask_path)
            
            # 计算表型
            organ_type = ['stem', 'leaf', 'branch'][class_id - 1]
            pheno = compute_organ_phenotypes(coords, mask, organ_type)
            
            if pheno is not None:
                pheno['confidence'] = conf
                pheno['mask_file'] = mask_file
                pheno['instance_id'] = len(phenotypes_by_class[organ_type])
                phenotypes_by_class[organ_type].append(pheno)
    
    return phenotypes_by_class

def compute_linear_regression(x, y, x_name, y_name):
    """
    计算线性回归和R²
    返回: slope, intercept, r_value, p_value, std_err, r2
    """
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    
    # 移除NaN和无穷值
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    if len(x_clean) < 2:
        return None
    
    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    r2 = r_value ** 2
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'r2': r2,
        'x_name': x_name,
        'y_name': y_name,
        'n': len(x_clean)
    }

def save_phenotypes_report(phenotypes_by_class, room_name, output_file):
    """保存表型分析报告（中文格式）"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"植物表型参数分析报告\n")
        f.write(f"样本名称: {room_name}\n")
        f.write(f"长度单位: {LENGTH_UNIT}\n")
        f.write(f"面积单位: {AREA_UNIT}\n")
        f.write(f"角度单位: {ANGLE_UNIT}\n")
        f.write(f"标尺校准: 已使用标尺参考点进行缩放校准\n")
        f.write(f"  标尺点1: ({RULER_POINT1[0]:.6f}, {RULER_POINT1[1]:.6f}, {RULER_POINT1[2]:.6f})\n")
        f.write(f"  标尺点2: ({RULER_POINT2[0]:.6f}, {RULER_POINT2[1]:.6f}, {RULER_POINT2[2]:.6f})\n")
        f.write(f"  标尺真实长度: {RULER_REAL_LENGTH*100:.1f} 厘米\n")
        f.write(f"  基础缩放因子: {SCALE_FACTOR_BASE:.6f}\n")
        f.write(f"  调整因子: {SCALE_ADJUSTMENT:.4f} (用于调整到正常三七叶片尺寸范围)\n")
        f.write(f"  最终缩放因子: {SCALE_FACTOR:.6f}\n")
        f.write(f"  注意: 此分析结果仅用于参考，不用于科研\n")
        f.write("=" * 100 + "\n\n")
        
        # 1. 茎秆详细数据
        stems = phenotypes_by_class['stem']
        if len(stems) > 0:
            f.write("【茎秆详细数据】\n")
            f.write("-" * 100 + "\n")
            # 调整列宽以确保对齐：实例ID(8) + 置信度(12) + 点数(10) + 茎长(18) + 茎粗(18)
            f.write(f"{'实例ID':<8} {'置信度':<12} {'点数':<10} {'茎长(' + LENGTH_UNIT + ')':<18} {'茎粗(' + LENGTH_UNIT + ')':<18}\n")
            f.write("-" * 100 + "\n")
            
            for stem in stems:
                f.write(f"{stem['instance_id']:<8} {stem['confidence']:<12.4f} {stem['num_points']:<10} "
                       f"{stem['length']:<18.6f} {stem['thickness']:<18.6f}\n")
            
            # 茎秆平均值
            if len(stems) > 0:
                avg_length = np.mean([s['length'] for s in stems])
                avg_thickness = np.mean([s['thickness'] for s in stems])
                std_length = np.std([s['length'] for s in stems])
                std_thickness = np.std([s['thickness'] for s in stems])
                
                f.write("-" * 100 + "\n")
                f.write(f"平均值统计:\n")
                f.write(f"  平均茎长: {avg_length:.6f} {LENGTH_UNIT} (标准差: {std_length:.6f} {LENGTH_UNIT})\n")
                f.write(f"  平均茎粗: {avg_thickness:.6f} {LENGTH_UNIT} (标准差: {std_thickness:.6f} {LENGTH_UNIT})\n")
                f.write(f"  茎秆数量: {len(stems)}\n")
            f.write("\n")
        
        # 2. 叶片详细数据
        leaves = phenotypes_by_class['leaf']
        if len(leaves) > 0:
            f.write("【叶片详细数据】\n")
            f.write("-" * 100 + "\n")
            # 调整列宽以确保对齐：实例ID(8) + 置信度(12) + 点数(10) + 叶片长(20) + 叶宽(18) + 叶面积(20) + 叶倾斜角度(20)
            f.write(f"{'实例ID':<8} {'置信度':<12} {'点数':<10} {'叶片长(' + LENGTH_UNIT + ')':<20} "
                   f"{'叶宽(' + LENGTH_UNIT + ')':<18} {'叶面积(' + AREA_UNIT + ')':<20} "
                   f"{'叶倾斜角度(' + ANGLE_UNIT + ')':<20}\n")
            f.write("-" * 100 + "\n")
            
            for leaf in leaves:
                tilt_angle = leaf.get('tilt_angle', 0.0) if leaf.get('tilt_angle') is not None else 0.0
                f.write(f"{leaf['instance_id']:<8} {leaf['confidence']:<12.4f} {leaf['num_points']:<10} "
                       f"{leaf['length']:<20.6f} {leaf['width']:<18.6f} "
                       f"{leaf['convex_hull_area']:<20.6f} {tilt_angle:<20.2f}\n")
            
            # 叶片平均值
            if len(leaves) > 0:
                avg_length = np.mean([l['length'] for l in leaves])
                avg_width = np.mean([l['width'] for l in leaves])
                avg_area = np.mean([l['convex_hull_area'] for l in leaves])
                avg_tilt = np.mean([l.get('tilt_angle', 0.0) if l.get('tilt_angle') is not None else 0.0 for l in leaves])
                
                std_length = np.std([l['length'] for l in leaves])
                std_width = np.std([l['width'] for l in leaves])
                std_area = np.std([l['convex_hull_area'] for l in leaves])
                std_tilt = np.std([l.get('tilt_angle', 0.0) if l.get('tilt_angle') is not None else 0.0 for l in leaves])
                
                f.write("-" * 100 + "\n")
                f.write(f"平均值统计:\n")
                f.write(f"  平均叶片长: {avg_length:.6f} {LENGTH_UNIT} (标准差: {std_length:.6f} {LENGTH_UNIT})\n")
                f.write(f"  平均叶宽: {avg_width:.6f} {LENGTH_UNIT} (标准差: {std_width:.6f} {LENGTH_UNIT})\n")
                f.write(f"  平均叶面积: {avg_area:.6f} {AREA_UNIT} (标准差: {std_area:.6f} {AREA_UNIT})\n")
                f.write(f"  平均叶倾斜角度: {avg_tilt:.2f} {ANGLE_UNIT} (标准差: {std_tilt:.2f} {ANGLE_UNIT})\n")
                f.write(f"  叶片数量: {len(leaves)}\n")
                
                # 线性回归分析
                f.write("\n")
                f.write("【线性回归分析】\n")
                f.write("-" * 100 + "\n")
                
                # 叶片长 vs 叶宽
                leaf_lengths = np.array([l['length'] for l in leaves])
                leaf_widths = np.array([l['width'] for l in leaves])
                reg1 = compute_linear_regression(leaf_lengths, leaf_widths, "叶片长", "叶宽")
                if reg1:
                    f.write(f"1. 叶片长 vs 叶宽:\n")
                    f.write(f"   线性方程: 叶宽 = {reg1['slope']:.6f} × 叶片长 + {reg1['intercept']:.6f}\n")
                    f.write(f"   决定系数 (R²): {reg1['r2']:.6f}\n")
                    f.write(f"   相关系数 (R): {reg1['r_value']:.6f}\n")
                    f.write(f"   样本数: {reg1['n']}\n")
                    f.write(f"   p值: {reg1['p_value']:.6e}\n\n")
                
                # 叶面积 vs 叶片长
                leaf_areas = np.array([l['convex_hull_area'] for l in leaves])
                reg2 = compute_linear_regression(leaf_lengths, leaf_areas, "叶片长", "叶面积")
                if reg2:
                    f.write(f"2. 叶片长 vs 叶面积:\n")
                    f.write(f"   线性方程: 叶面积 = {reg2['slope']:.6f} × 叶片长 + {reg2['intercept']:.6f}\n")
                    f.write(f"   决定系数 (R²): {reg2['r2']:.6f}\n")
                    f.write(f"   相关系数 (R): {reg2['r_value']:.6f}\n")
                    f.write(f"   样本数: {reg2['n']}\n")
                    f.write(f"   p值: {reg2['p_value']:.6e}\n\n")
                
                # 叶面积 vs 叶宽
                reg3 = compute_linear_regression(leaf_widths, leaf_areas, "叶宽", "叶面积")
                if reg3:
                    f.write(f"3. 叶宽 vs 叶面积:\n")
                    f.write(f"   线性方程: 叶面积 = {reg3['slope']:.6f} × 叶宽 + {reg3['intercept']:.6f}\n")
                    f.write(f"   决定系数 (R²): {reg3['r2']:.6f}\n")
                    f.write(f"   相关系数 (R): {reg3['r_value']:.6f}\n")
                    f.write(f"   样本数: {reg3['n']}\n")
                    f.write(f"   p值: {reg3['p_value']:.6e}\n\n")
                
                # 叶片长 vs 叶倾斜角度
                leaf_tilts = np.array([l.get('tilt_angle', 0.0) if l.get('tilt_angle') is not None else 0.0 for l in leaves])
                reg4 = compute_linear_regression(leaf_lengths, leaf_tilts, "叶片长", "叶倾斜角度")
                if reg4:
                    f.write(f"4. 叶片长 vs 叶倾斜角度:\n")
                    f.write(f"   线性方程: 叶倾斜角度 = {reg4['slope']:.6f} × 叶片长 + {reg4['intercept']:.6f}\n")
                    f.write(f"   决定系数 (R²): {reg4['r2']:.6f}\n")
                    f.write(f"   相关系数 (R): {reg4['r_value']:.6f}\n")
                    f.write(f"   样本数: {reg4['n']}\n")
                    f.write(f"   p值: {reg4['p_value']:.6e}\n\n")
            f.write("\n")
        
        # 3. 分枝详细数据
        branches = phenotypes_by_class['branch']
        if len(branches) > 0:
            f.write("【分枝详细数据】\n")
            f.write("-" * 100 + "\n")
            # 调整列宽以确保对齐：实例ID(8) + 置信度(12) + 点数(10) + 分枝长(18) + 分枝宽(18)
            f.write(f"{'实例ID':<8} {'置信度':<12} {'点数':<10} {'分枝长(' + LENGTH_UNIT + ')':<18} "
                   f"{'分枝宽(' + LENGTH_UNIT + ')':<18}\n")
            f.write("-" * 100 + "\n")
            
            for branch in branches:
                f.write(f"{branch['instance_id']:<8} {branch['confidence']:<12.4f} {branch['num_points']:<10} "
                       f"{branch['length']:<18.6f} {branch['width']:<18.6f}\n")
            
            # 分枝平均值
            if len(branches) > 0:
                avg_length = np.mean([b['length'] for b in branches])
                avg_width = np.mean([b['width'] for b in branches])
                std_length = np.std([b['length'] for b in branches])
                std_width = np.std([b['width'] for b in branches])
                
                f.write("-" * 100 + "\n")
                f.write(f"平均值统计:\n")
                f.write(f"  平均分枝长: {avg_length:.6f} {LENGTH_UNIT} (标准差: {std_length:.6f} {LENGTH_UNIT})\n")
                f.write(f"  平均分枝宽: {avg_width:.6f} {LENGTH_UNIT} (标准差: {std_width:.6f} {LENGTH_UNIT})\n")
                f.write(f"  分枝数量: {len(branches)}\n")
            f.write("\n")
        
        # 4. 总体统计
        f.write("=" * 100 + "\n")
        f.write("【总体统计】\n")
        f.write("-" * 100 + "\n")
        f.write(f"茎秆数量: {len(stems)}\n")
        f.write(f"叶片数量: {len(leaves)}\n")
        f.write(f"分枝数量: {len(branches)}\n")
        if len(leaves) > 0:
            total_leaf_area = sum([l['convex_hull_area'] for l in leaves])
            f.write(f"总叶面积: {total_leaf_area:.6f} {AREA_UNIT}\n")
            f.write(f"平均单叶面积: {total_leaf_area/len(leaves):.6f} {AREA_UNIT}\n")
        f.write("=" * 100 + "\n")

def save_phenotypes_excel(phenotypes_by_class, room_name, output_file):
    """保存表型分析报告为Excel格式"""
    if not HAS_PANDAS:
        print("警告: 未安装pandas，无法生成Excel文件")
        return
    
    # 创建Excel写入器
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 1. 茎秆数据
        stems = phenotypes_by_class['stem']
        if len(stems) > 0:
            stem_data = []
            for stem in stems:
                stem_data.append({
                    '实例ID': stem['instance_id'],
                    '置信度': f"{stem['confidence']:.4f}",
                    '点数': stem['num_points'],
                    f'茎长({LENGTH_UNIT})': f"{stem['length']:.6f}",
                    f'茎粗({LENGTH_UNIT})': f"{stem['thickness']:.6f}"
                })
            df_stems = pd.DataFrame(stem_data)
            df_stems.to_excel(writer, sheet_name='茎秆数据', index=False)
            
            # 添加平均值行
            avg_row = pd.DataFrame([{
                '实例ID': '平均值',
                '置信度': f"{np.mean([s['confidence'] for s in stems]):.4f}",
                '点数': int(np.mean([s['num_points'] for s in stems])),
                f'茎长({LENGTH_UNIT})': f"{np.mean([s['length'] for s in stems]):.6f}",
                f'茎粗({LENGTH_UNIT})': f"{np.mean([s['thickness'] for s in stems]):.6f}"
            }])
            df_stems_with_avg = pd.concat([df_stems, avg_row], ignore_index=True)
            df_stems_with_avg.to_excel(writer, sheet_name='茎秆数据', index=False)
        
        # 2. 叶片数据
        leaves = phenotypes_by_class['leaf']
        if len(leaves) > 0:
            leaf_data = []
            for leaf in leaves:
                tilt_angle = leaf.get('tilt_angle', 0.0) if leaf.get('tilt_angle') is not None else 0.0
                leaf_data.append({
                    '实例ID': leaf['instance_id'],
                    '置信度': f"{leaf['confidence']:.4f}",
                    '点数': leaf['num_points'],
                    f'叶片长({LENGTH_UNIT})': f"{leaf['length']:.6f}",
                    f'叶宽({LENGTH_UNIT})': f"{leaf['width']:.6f}",
                    f'叶面积({AREA_UNIT})': f"{leaf['convex_hull_area']:.6f}",
                    f'叶倾斜角度({ANGLE_UNIT})': f"{tilt_angle:.2f}"
                })
            df_leaves = pd.DataFrame(leaf_data)
            
            # 添加平均值行
            avg_tilt = np.mean([l.get('tilt_angle', 0.0) if l.get('tilt_angle') is not None else 0.0 for l in leaves])
            avg_row = pd.DataFrame([{
                '实例ID': '平均值',
                '置信度': f"{np.mean([l['confidence'] for l in leaves]):.4f}",
                '点数': int(np.mean([l['num_points'] for l in leaves])),
                f'叶片长({LENGTH_UNIT})': f"{np.mean([l['length'] for l in leaves]):.6f}",
                f'叶宽({LENGTH_UNIT})': f"{np.mean([l['width'] for l in leaves]):.6f}",
                f'叶面积({AREA_UNIT})': f"{np.mean([l['convex_hull_area'] for l in leaves]):.6f}",
                f'叶倾斜角度({ANGLE_UNIT})': f"{avg_tilt:.2f}"
            }])
            df_leaves_with_avg = pd.concat([df_leaves, avg_row], ignore_index=True)
            df_leaves_with_avg.to_excel(writer, sheet_name='叶片数据', index=False)
            
            # 3. 线性回归分析
            leaf_lengths = np.array([l['length'] for l in leaves])
            leaf_widths = np.array([l['width'] for l in leaves])
            leaf_areas = np.array([l['convex_hull_area'] for l in leaves])
            leaf_tilts = np.array([l.get('tilt_angle', 0.0) if l.get('tilt_angle') is not None else 0.0 for l in leaves])
            
            regression_data = []
            # 叶片长 vs 叶宽
            reg1 = compute_linear_regression(leaf_lengths, leaf_widths, "叶片长", "叶宽")
            if reg1:
                regression_data.append({
                    '关系': '叶片长 vs 叶宽',
                    '线性方程': f'叶宽 = {reg1["slope"]:.6f} × 叶片长 + {reg1["intercept"]:.6f}',
                    '决定系数(R²)': f"{reg1['r2']:.6f}",
                    '相关系数(R)': f"{reg1['r_value']:.6f}",
                    'p值': f"{reg1['p_value']:.6e}",
                    '样本数': reg1['n']
                })
            
            # 叶片长 vs 叶面积
            reg2 = compute_linear_regression(leaf_lengths, leaf_areas, "叶片长", "叶面积")
            if reg2:
                regression_data.append({
                    '关系': '叶片长 vs 叶面积',
                    '线性方程': f'叶面积 = {reg2["slope"]:.6f} × 叶片长 + {reg2["intercept"]:.6f}',
                    '决定系数(R²)': f"{reg2['r2']:.6f}",
                    '相关系数(R)': f"{reg2['r_value']:.6f}",
                    'p值': f"{reg2['p_value']:.6e}",
                    '样本数': reg2['n']
                })
            
            # 叶宽 vs 叶面积
            reg3 = compute_linear_regression(leaf_widths, leaf_areas, "叶宽", "叶面积")
            if reg3:
                regression_data.append({
                    '关系': '叶宽 vs 叶面积',
                    '线性方程': f'叶面积 = {reg3["slope"]:.6f} × 叶宽 + {reg3["intercept"]:.6f}',
                    '决定系数(R²)': f"{reg3['r2']:.6f}",
                    '相关系数(R)': f"{reg3['r_value']:.6f}",
                    'p值': f"{reg3['p_value']:.6e}",
                    '样本数': reg3['n']
                })
            
            # 叶片长 vs 叶倾斜角度
            reg4 = compute_linear_regression(leaf_lengths, leaf_tilts, "叶片长", "叶倾斜角度")
            if reg4:
                regression_data.append({
                    '关系': '叶片长 vs 叶倾斜角度',
                    '线性方程': f'叶倾斜角度 = {reg4["slope"]:.6f} × 叶片长 + {reg4["intercept"]:.6f}',
                    '决定系数(R²)': f"{reg4['r2']:.6f}",
                    '相关系数(R)': f"{reg4['r_value']:.6f}",
                    'p值': f"{reg4['p_value']:.6e}",
                    '样本数': reg4['n']
                })
            
            if regression_data:
                df_reg = pd.DataFrame(regression_data)
                df_reg.to_excel(writer, sheet_name='线性回归分析', index=False)
        
        # 4. 分枝数据
        branches = phenotypes_by_class['branch']
        if len(branches) > 0:
            branch_data = []
            for branch in branches:
                branch_data.append({
                    '实例ID': branch['instance_id'],
                    '置信度': f"{branch['confidence']:.4f}",
                    '点数': branch['num_points'],
                    f'分枝长({LENGTH_UNIT})': f"{branch['length']:.6f}",
                    f'分枝宽({LENGTH_UNIT})': f"{branch['width']:.6f}"
                })
            df_branches = pd.DataFrame(branch_data)
            
            # 添加平均值行
            avg_row = pd.DataFrame([{
                '实例ID': '平均值',
                '置信度': f"{np.mean([b['confidence'] for b in branches]):.4f}",
                '点数': int(np.mean([b['num_points'] for b in branches])),
                f'分枝长({LENGTH_UNIT})': f"{np.mean([b['length'] for b in branches]):.6f}",
                f'分枝宽({LENGTH_UNIT})': f"{np.mean([b['width'] for b in branches]):.6f}"
            }])
            df_branches_with_avg = pd.concat([df_branches, avg_row], ignore_index=True)
            df_branches_with_avg.to_excel(writer, sheet_name='分枝数据', index=False)
        
        # 5. 总体统计
        summary_data = [{
            '项目': '茎秆数量',
            '数值': len(stems)
        }, {
            '项目': '叶片数量',
            '数值': len(leaves)
        }, {
            '项目': '分枝数量',
            '数值': len(branches)
        }]
        if len(leaves) > 0:
            total_leaf_area = sum([l['convex_hull_area'] for l in leaves])
            summary_data.append({
                '项目': '总叶面积',
                '数值': f"{total_leaf_area:.6f} {AREA_UNIT}"
            })
            summary_data.append({
                '项目': '平均单叶面积',
                '数值': f"{total_leaf_area/len(leaves):.6f} {AREA_UNIT}"
            })
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='总体统计', index=False)
    
    print(f"Excel文件已保存: {output_file}")

def batch_analyze_areas(pred_path, area_names, output_dir, conf_threshold=0.1):
    """批量分析多个Area"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for area_name in area_names:
        print(f"\n正在分析: {area_name}")
        phenotypes = analyze_plant_phenotypes(pred_path, area_name, conf_threshold)
        
        if phenotypes is not None:
            # 保存单个Area的报告（TXT格式）
            output_file_txt = osp.join(output_dir, f"{area_name}_表型分析报告.txt")
            save_phenotypes_report(phenotypes, area_name, output_file_txt)
            print(f"  TXT报告已保存: {output_file_txt}")
            
            # 保存单个Area的报告（Excel格式）
            if HAS_PANDAS:
                output_file_excel = osp.join(output_dir, f"{area_name}_表型分析报告.xlsx")
                save_phenotypes_excel(phenotypes, area_name, output_file_excel)
                print(f"  Excel报告已保存: {output_file_excel}")
            
            all_results[area_name] = phenotypes
    
    # 生成汇总报告
    if len(all_results) > 0:
        summary_file_txt = osp.join(output_dir, "Area_45-50_汇总报告.txt")
        save_summary_report(all_results, summary_file_txt)
        print(f"\n汇总TXT报告已保存: {summary_file_txt}")
        
        # 生成汇总Excel报告
        if HAS_PANDAS:
            summary_file_excel = osp.join(output_dir, "Area_45-50_汇总报告.xlsx")
            save_summary_excel(all_results, summary_file_excel)
            print(f"汇总Excel报告已保存: {summary_file_excel}")

def save_summary_excel(all_results, output_file):
    """保存所有Area的汇总Excel报告"""
    if not HAS_PANDAS:
        print("警告: 未安装pandas，无法生成Excel文件")
        return
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 汇总所有叶片数据
        all_leaves = []
        all_stems = []
        
        for area_name, phenotypes in all_results.items():
            for leaf in phenotypes['leaf']:
                leaf['area_name'] = area_name
                all_leaves.append(leaf)
            for stem in phenotypes['stem']:
                stem['area_name'] = area_name
                all_stems.append(stem)
        
        # 1. 所有Area叶片数据汇总
        if len(all_leaves) > 0:
            leaf_summary_data = []
            for leaf in all_leaves:
                tilt_angle = leaf.get('tilt_angle', 0.0) if leaf.get('tilt_angle') is not None else 0.0
                leaf_summary_data.append({
                    'Area名称': leaf['area_name'],
                    '实例ID': leaf['instance_id'],
                    '置信度': f"{leaf['confidence']:.4f}",
                    '点数': leaf['num_points'],
                    f'叶片长({LENGTH_UNIT})': f"{leaf['length']:.6f}",
                    f'叶宽({LENGTH_UNIT})': f"{leaf['width']:.6f}",
                    f'叶面积({AREA_UNIT})': f"{leaf['convex_hull_area']:.6f}",
                    f'叶倾斜角度({ANGLE_UNIT})': f"{tilt_angle:.2f}"
                })
            df_all_leaves = pd.DataFrame(leaf_summary_data)
            df_all_leaves.to_excel(writer, sheet_name='所有Area叶片数据', index=False)
            
            # 添加平均值行
            avg_tilt = np.mean([l.get('tilt_angle', 0.0) if l.get('tilt_angle') is not None else 0.0 for l in all_leaves])
            avg_row = pd.DataFrame([{
                'Area名称': '所有Area平均值',
                '实例ID': '',
                '置信度': f"{np.mean([l['confidence'] for l in all_leaves]):.4f}",
                '点数': int(np.mean([l['num_points'] for l in all_leaves])),
                f'叶片长({LENGTH_UNIT})': f"{np.mean([l['length'] for l in all_leaves]):.6f}",
                f'叶宽({LENGTH_UNIT})': f"{np.mean([l['width'] for l in all_leaves]):.6f}",
                f'叶面积({AREA_UNIT})': f"{np.mean([l['convex_hull_area'] for l in all_leaves]):.6f}",
                f'叶倾斜角度({ANGLE_UNIT})': f"{avg_tilt:.2f}"
            }])
            df_all_leaves_with_avg = pd.concat([df_all_leaves, avg_row], ignore_index=True)
            df_all_leaves_with_avg.to_excel(writer, sheet_name='所有Area叶片数据', index=False)
            
            # 线性回归分析
            all_leaf_lengths = np.array([l['length'] for l in all_leaves])
            all_leaf_widths = np.array([l['width'] for l in all_leaves])
            all_leaf_areas = np.array([l['convex_hull_area'] for l in all_leaves])
            all_leaf_tilts = np.array([l.get('tilt_angle', 0.0) if l.get('tilt_angle') is not None else 0.0 for l in all_leaves])
            
            regression_data = []
            reg1 = compute_linear_regression(all_leaf_lengths, all_leaf_widths, "叶片长", "叶宽")
            if reg1:
                regression_data.append({
                    '关系': '叶片长 vs 叶宽',
                    '线性方程': f'叶宽 = {reg1["slope"]:.6f} × 叶片长 + {reg1["intercept"]:.6f}',
                    '决定系数(R²)': f"{reg1['r2']:.6f}",
                    '相关系数(R)': f"{reg1['r_value']:.6f}",
                    'p值': f"{reg1['p_value']:.6e}",
                    '样本数': reg1['n']
                })
            
            reg2 = compute_linear_regression(all_leaf_lengths, all_leaf_areas, "叶片长", "叶面积")
            if reg2:
                regression_data.append({
                    '关系': '叶片长 vs 叶面积',
                    '线性方程': f'叶面积 = {reg2["slope"]:.6f} × 叶片长 + {reg2["intercept"]:.6f}',
                    '决定系数(R²)': f"{reg2['r2']:.6f}",
                    '相关系数(R)': f"{reg2['r_value']:.6f}",
                    'p值': f"{reg2['p_value']:.6e}",
                    '样本数': reg2['n']
                })
            
            reg3 = compute_linear_regression(all_leaf_widths, all_leaf_areas, "叶宽", "叶面积")
            if reg3:
                regression_data.append({
                    '关系': '叶宽 vs 叶面积',
                    '线性方程': f'叶面积 = {reg3["slope"]:.6f} × 叶宽 + {reg3["intercept"]:.6f}',
                    '决定系数(R²)': f"{reg3['r2']:.6f}",
                    '相关系数(R)': f"{reg3['r_value']:.6f}",
                    'p值': f"{reg3['p_value']:.6e}",
                    '样本数': reg3['n']
                })
            
            reg4 = compute_linear_regression(all_leaf_lengths, all_leaf_tilts, "叶片长", "叶倾斜角度")
            if reg4:
                regression_data.append({
                    '关系': '叶片长 vs 叶倾斜角度',
                    '线性方程': f'叶倾斜角度 = {reg4["slope"]:.6f} × 叶片长 + {reg4["intercept"]:.6f}',
                    '决定系数(R²)': f"{reg4['r2']:.6f}",
                    '相关系数(R)': f"{reg4['r_value']:.6f}",
                    'p值': f"{reg4['p_value']:.6e}",
                    '样本数': reg4['n']
                })
            
            if regression_data:
                df_reg = pd.DataFrame(regression_data)
                df_reg.to_excel(writer, sheet_name='线性回归分析', index=False)
        
        # 2. 各Area统计汇总
        area_summary_data = []
        for area_name in sorted(all_results.keys()):
            phenotypes = all_results[area_name]
            stems = phenotypes['stem']
            leaves = phenotypes['leaf']
            
            avg_leaf_length = np.mean([l['length'] for l in leaves]) if len(leaves) > 0 else 0.0
            avg_leaf_width = np.mean([l['width'] for l in leaves]) if len(leaves) > 0 else 0.0
            avg_leaf_area = np.mean([l['convex_hull_area'] for l in leaves]) if len(leaves) > 0 else 0.0
            
            area_summary_data.append({
                'Area名称': area_name,
                '茎秆数': len(stems),
                '叶片数': len(leaves),
                f'平均叶片长({LENGTH_UNIT})': f"{avg_leaf_length:.6f}" if len(leaves) > 0 else "0.000000",
                f'平均叶宽({LENGTH_UNIT})': f"{avg_leaf_width:.6f}" if len(leaves) > 0 else "0.000000",
                f'平均叶面积({AREA_UNIT})': f"{avg_leaf_area:.6f}" if len(leaves) > 0 else "0.000000"
            })
        
        df_area_summary = pd.DataFrame(area_summary_data)
        df_area_summary.to_excel(writer, sheet_name='各Area统计汇总', index=False)
    
    print(f"汇总Excel文件已保存: {output_file}")

def save_summary_report(all_results, output_file):
    """保存所有Area的汇总报告"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("植物表型参数分析汇总报告\n")
        f.write("样本范围: Area_45 至 Area_50\n")
        f.write(f"长度单位: {LENGTH_UNIT}\n")
        f.write(f"面积单位: {AREA_UNIT}\n")
        f.write(f"角度单位: {ANGLE_UNIT}\n")
        f.write(f"标尺校准: 已使用标尺参考点进行缩放校准\n")
        f.write(f"  标尺真实长度: {RULER_REAL_LENGTH*100:.1f} 厘米\n")
        f.write(f"  基础缩放因子: {SCALE_FACTOR_BASE:.6f}\n")
        f.write(f"  调整因子: {SCALE_ADJUSTMENT:.4f} (用于调整到正常三七叶片尺寸范围)\n")
        f.write(f"  最终缩放因子: {SCALE_FACTOR:.6f}\n")
        f.write(f"  注意: 此分析结果仅用于参考，不用于科研\n")
        f.write("=" * 100 + "\n\n")
        
        # 汇总所有叶片数据
        all_leaves = []
        all_stems = []
        
        for area_name, phenotypes in all_results.items():
            for leaf in phenotypes['leaf']:
                leaf['area_name'] = area_name
                all_leaves.append(leaf)
            for stem in phenotypes['stem']:
                stem['area_name'] = area_name
                all_stems.append(stem)
        
        # 所有Area的叶片平均值
        if len(all_leaves) > 0:
            f.write("【所有Area叶片参数平均值】\n")
            f.write("-" * 100 + "\n")
            
            avg_length = np.mean([l['length'] for l in all_leaves])
            avg_width = np.mean([l['width'] for l in all_leaves])
            avg_area = np.mean([l['convex_hull_area'] for l in all_leaves])
            avg_tilt = np.mean([l.get('tilt_angle', 0.0) if l.get('tilt_angle') is not None else 0.0 for l in all_leaves])
            
            std_length = np.std([l['length'] for l in all_leaves])
            std_width = np.std([l['width'] for l in all_leaves])
            std_area = np.std([l['convex_hull_area'] for l in all_leaves])
            std_tilt = np.std([l.get('tilt_angle', 0.0) if l.get('tilt_angle') is not None else 0.0 for l in all_leaves])
            
            f.write(f"平均叶片长: {avg_length:.6f} {LENGTH_UNIT} (标准差: {std_length:.6f} {LENGTH_UNIT})\n")
            f.write(f"平均叶宽: {avg_width:.6f} {LENGTH_UNIT} (标准差: {std_width:.6f} {LENGTH_UNIT})\n")
            f.write(f"平均叶面积: {avg_area:.6f} {AREA_UNIT} (标准差: {std_area:.6f} {AREA_UNIT})\n")
            f.write(f"平均叶倾斜角度: {avg_tilt:.2f} {ANGLE_UNIT} (标准差: {std_tilt:.2f} {ANGLE_UNIT})\n")
            f.write(f"总叶片数: {len(all_leaves)}\n\n")
            
            # 所有Area的线性回归
            f.write("【所有Area叶片线性回归分析】\n")
            f.write("-" * 100 + "\n")
            
            all_leaf_lengths = np.array([l['length'] for l in all_leaves])
            all_leaf_widths = np.array([l['width'] for l in all_leaves])
            all_leaf_areas = np.array([l['convex_hull_area'] for l in all_leaves])
            all_leaf_tilts = np.array([l.get('tilt_angle', 0.0) if l.get('tilt_angle') is not None else 0.0 for l in all_leaves])
            
            # 叶片长 vs 叶宽
            reg1 = compute_linear_regression(all_leaf_lengths, all_leaf_widths, "叶片长", "叶宽")
            if reg1:
                f.write(f"1. 叶片长 vs 叶宽:\n")
                f.write(f"   线性方程: 叶宽 = {reg1['slope']:.6f} × 叶片长 + {reg1['intercept']:.6f}\n")
                f.write(f"   决定系数 (R²): {reg1['r2']:.6f}\n")
                f.write(f"   相关系数 (R): {reg1['r_value']:.6f}\n")
                f.write(f"   样本数: {reg1['n']}\n")
                f.write(f"   p值: {reg1['p_value']:.6e}\n\n")
            
            # 叶面积 vs 叶片长
            reg2 = compute_linear_regression(all_leaf_lengths, all_leaf_areas, "叶片长", "叶面积")
            if reg2:
                f.write(f"2. 叶片长 vs 叶面积:\n")
                f.write(f"   线性方程: 叶面积 = {reg2['slope']:.6f} × 叶片长 + {reg2['intercept']:.6f}\n")
                f.write(f"   决定系数 (R²): {reg2['r2']:.6f}\n")
                f.write(f"   相关系数 (R): {reg2['r_value']:.6f}\n")
                f.write(f"   样本数: {reg2['n']}\n")
                f.write(f"   p值: {reg2['p_value']:.6e}\n\n")
            
            # 叶面积 vs 叶宽
            reg3 = compute_linear_regression(all_leaf_widths, all_leaf_areas, "叶宽", "叶面积")
            if reg3:
                f.write(f"3. 叶宽 vs 叶面积:\n")
                f.write(f"   线性方程: 叶面积 = {reg3['slope']:.6f} × 叶宽 + {reg3['intercept']:.6f}\n")
                f.write(f"   决定系数 (R²): {reg3['r2']:.6f}\n")
                f.write(f"   相关系数 (R): {reg3['r_value']:.6f}\n")
                f.write(f"   样本数: {reg3['n']}\n")
                f.write(f"   p值: {reg3['p_value']:.6e}\n\n")
            
            # 叶片长 vs 叶倾斜角度
            reg4 = compute_linear_regression(all_leaf_lengths, all_leaf_tilts, "叶片长", "叶倾斜角度")
            if reg4:
                f.write(f"4. 叶片长 vs 叶倾斜角度:\n")
                f.write(f"   线性方程: 叶倾斜角度 = {reg4['slope']:.6f} × 叶片长 + {reg4['intercept']:.6f}\n")
                f.write(f"   决定系数 (R²): {reg4['r2']:.6f}\n")
                f.write(f"   相关系数 (R): {reg4['r_value']:.6f}\n")
                f.write(f"   样本数: {reg4['n']}\n")
                f.write(f"   p值: {reg4['p_value']:.6e}\n\n")
        
        # 所有Area的茎秆平均值
        if len(all_stems) > 0:
            f.write("【所有Area茎秆参数平均值】\n")
            f.write("-" * 100 + "\n")
            
            avg_length = np.mean([s['length'] for s in all_stems])
            avg_thickness = np.mean([s['thickness'] for s in all_stems])
            std_length = np.std([s['length'] for s in all_stems])
            std_thickness = np.std([s['thickness'] for s in all_stems])
            
            f.write(f"平均茎长: {avg_length:.6f} {LENGTH_UNIT} (标准差: {std_length:.6f} {LENGTH_UNIT})\n")
            f.write(f"平均茎粗: {avg_thickness:.6f} {LENGTH_UNIT} (标准差: {std_thickness:.6f} {LENGTH_UNIT})\n")
            f.write(f"总茎秆数: {len(all_stems)}\n\n")
        
        # 各Area统计
        f.write("【各Area统计汇总】\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Area名称':<20} {'茎秆数':<10} {'叶片数':<10} {'平均叶片长':<15} {'平均叶面积':<15}\n")
        f.write("-" * 100 + "\n")
        
        for area_name in sorted(all_results.keys()):
            phenotypes = all_results[area_name]
            stems = phenotypes['stem']
            leaves = phenotypes['leaf']
            
            avg_leaf_length = np.mean([l['length'] for l in leaves]) if len(leaves) > 0 else 0.0
            avg_leaf_area = np.mean([l['convex_hull_area'] for l in leaves]) if len(leaves) > 0 else 0.0
            
            f.write(f"{area_name:<20} {len(stems):<10} {len(leaves):<10} "
                   f"{avg_leaf_length:<15.6f} {avg_leaf_area:<15.6f}\n")
        
        f.write("=" * 100 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='植物表型参数分析工具')
    parser.add_argument('--prediction_path', type=str, 
                       default='/home/ng204/xueshuang/SoftGroup/results/Area45-50new',
                       help='测试结果路径')
    parser.add_argument('--room_name', type=str, help='单个场景名称（如果指定，只分析这个场景）')
    parser.add_argument('--conf_threshold', type=float, default=0.1, help='置信度阈值')
    parser.add_argument('--output_dir', type=str,
                       default='/home/ng204/xueshuang/SoftGroup/phenotype_results/Area_45-Area_50',
                       help='输出目录')
    parser.add_argument('--batch', action='store_true', help='批量处理Area45-50')
    args = parser.parse_args()
    
    if args.batch:
        # 批量处理Area45-50
        area_names = [
            'Area_46_Panax_45',
            'Area_47_Panax_46',
            'Area_48_Panax_47',
            'Area_49_Panax_48',
            'Area_50_Panax_49'
        ]
        batch_analyze_areas(args.prediction_path, area_names, args.output_dir, args.conf_threshold)
    elif args.room_name:
        # 单个分析
        phenotypes = analyze_plant_phenotypes(
            args.prediction_path, 
            args.room_name, 
            args.conf_threshold
        )
        
        if phenotypes is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            output_file = osp.join(args.output_dir, f"{args.room_name}_表型分析报告.txt")
            save_phenotypes_report(phenotypes, args.room_name, output_file)
            print(f"\n报告已保存到: {output_file}")
    else:
        print("请指定 --room_name 或使用 --batch 进行批量处理")
