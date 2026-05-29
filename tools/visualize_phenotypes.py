"""
表型数据可视化工具
读取CSV文件并生成统计图表
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_phenotype_data(csv_file):
    """加载表型数据"""
    df = pd.read_csv(csv_file)
    print("=" * 80)
    print(f"表型数据概览: {csv_file}")
    print("=" * 80)
    print(f"\n总器官数: {len(df)}")
    print(f"\n各类器官数量:")
    print(df['organ_type'].value_counts())
    print("\n" + "=" * 80)
    return df

def print_statistics(df):
    """打印详细统计信息"""
    print("\n详细统计:")
    print("-" * 80)
    
    for organ_type in ['stem', 'leaf', 'branch']:
        organ_df = df[df['organ_type'] == organ_type]
        if len(organ_df) == 0:
            continue
        
        print(f"\n【{organ_type.upper()}】 ({len(organ_df)}个实例)")
        print("-" * 80)
        
        # 主要参数统计
        params = {
            '长度 (length)': 'length',
            '宽度 (width)': 'width',
            '厚度 (thickness)': 'thickness',
            '表面积 (area)': 'convex_hull_area',
            '体积 (volume)': 'convex_hull_volume',
            '细长度 (elongation)': 'elongation',
            '扁平度 (flatness)': 'flatness',
            '点数 (num_points)': 'num_points',
            '置信度 (confidence)': 'confidence'
        }
        
        for name, col in params.items():
            if col in organ_df.columns:
                values = organ_df[col].dropna()
                if len(values) > 0:
                    print(f"  {name:20s}: 均值={values.mean():8.3f}, "
                          f"标准差={values.std():8.3f}, "
                          f"范围=[{values.min():7.3f}, {values.max():7.3f}]")

def create_summary_table(df, output_file=None):
    """创建汇总表格"""
    summary_data = []
    
    for organ_type in ['stem', 'leaf', 'branch']:
        organ_df = df[df['organ_type'] == organ_type]
        if len(organ_df) == 0:
            continue
        
        row = {
            '器官类型': organ_type,
            '实例数': len(organ_df),
            '平均长度': organ_df['length'].mean(),
            '平均宽度': organ_df['width'].mean(),
            '平均表面积': organ_df['convex_hull_area'].mean(),
            '平均体积': organ_df['convex_hull_volume'].mean(),
            '平均细长度': organ_df['elongation'].mean(),
            '平均置信度': organ_df['confidence'].mean()
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "=" * 80)
    print("汇总表格:")
    print("=" * 80)
    print(summary_df.to_string(index=False, float_format='%.3f'))
    
    if output_file:
        summary_df.to_csv(output_file, index=False, float_format='%.3f')
        print(f"\n汇总表已保存到: {output_file}")
    
    return summary_df

def plot_distributions(df, output_dir):
    """绘制分布图"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("生成可视化图表...")
    print("=" * 80)
    
    # 设置绘图风格
    sns.set_style("whitegrid")
    colors = {'stem': '#FF6B6B', 'leaf': '#4ECDC4', 'branch': '#95E1D3'}
    
    # 1. 各器官数量柱状图
    fig, ax = plt.subplots(figsize=(8, 6))
    organ_counts = df['organ_type'].value_counts()
    bars = ax.bar(organ_counts.index, organ_counts.values, 
                   color=[colors[o] for o in organ_counts.index])
    ax.set_xlabel('Organ Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Number of Detected Organs', fontsize=14, fontweight='bold')
    
    # 在柱子上显示数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_organ_counts.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 已生成: {output_dir}/1_organ_counts.png")
    plt.close()
    
    # 2. 叶片长度分布直方图
    leaf_df = df[df['organ_type'] == 'leaf']
    if len(leaf_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(leaf_df['length'], bins=15, color=colors['leaf'], 
                edgecolor='black', alpha=0.7)
        ax.axvline(leaf_df['length'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {leaf_df["length"].mean():.3f}')
        ax.set_xlabel('Leaf Length', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Leaf Length Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/2_leaf_length_dist.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ 已生成: {output_dir}/2_leaf_length_dist.png")
        plt.close()
    
    # 3. 叶片面积分布
    if len(leaf_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(leaf_df['convex_hull_area'], bins=15, color=colors['leaf'],
                edgecolor='black', alpha=0.7)
        ax.axvline(leaf_df['convex_hull_area'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {leaf_df["convex_hull_area"].mean():.3f}')
        ax.set_xlabel('Leaf Area', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Leaf Area Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/3_leaf_area_dist.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ 已生成: {output_dir}/3_leaf_area_dist.png")
        plt.close()
    
    # 4. 长度 vs 宽度散点图（各器官对比）
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for organ_type in ['stem', 'leaf', 'branch']:
        organ_df = df[df['organ_type'] == organ_type]
        if len(organ_df) > 0:
            ax.scatter(organ_df['length'], organ_df['width'], 
                      s=100, alpha=0.6, c=colors[organ_type], 
                      label=f'{organ_type} (n={len(organ_df)})',
                      edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Length', fontsize=12)
    ax.set_ylabel('Width', fontsize=12)
    ax.set_title('Organ Size Comparison (Length vs Width)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_length_width_scatter.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 已生成: {output_dir}/4_length_width_scatter.png")
    plt.close()
    
    # 5. 细长度对比箱线图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    organ_types = []
    elongations = []
    for organ_type in ['stem', 'leaf', 'branch']:
        organ_df = df[df['organ_type'] == organ_type]
        if len(organ_df) > 0:
            organ_types.extend([organ_type] * len(organ_df))
            elongations.extend(organ_df['elongation'].tolist())
    
    if len(organ_types) > 0:
        plot_df = pd.DataFrame({'Organ': organ_types, 'Elongation': elongations})
        
        # 使用对数刻度（因为stem的细长度可能很大）
        bp = ax.boxplot([plot_df[plot_df['Organ'] == o]['Elongation'].values 
                         for o in ['stem', 'leaf', 'branch'] if o in plot_df['Organ'].unique()],
                        labels=[o for o in ['stem', 'leaf', 'branch'] 
                               if o in plot_df['Organ'].unique()],
                        patch_artist=True)
        
        # 设置颜色
        for patch, organ in zip(bp['boxes'], [o for o in ['stem', 'leaf', 'branch'] 
                                               if o in plot_df['Organ'].unique()]):
            patch.set_facecolor(colors[organ])
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Organ Type', fontsize=12)
        ax.set_ylabel('Elongation', fontsize=12)
        ax.set_title('Elongation Comparison (Higher = More Elongated)', 
                    fontsize=14, fontweight='bold')
        ax.set_yscale('log')  # 对数刻度
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/5_elongation_boxplot.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ 已生成: {output_dir}/5_elongation_boxplot.png")
        plt.close()
    
    # 6. 叶片大小排序（前20片）
    if len(leaf_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 按面积排序
        leaf_sorted = leaf_df.sort_values('convex_hull_area', ascending=False).head(20)
        x_pos = np.arange(len(leaf_sorted))
        
        bars = ax.bar(x_pos, leaf_sorted['convex_hull_area'].values, 
                     color=colors['leaf'], edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Leaf Index (sorted by area)', fontsize=12)
        ax.set_ylabel('Leaf Area', fontsize=12)
        ax.set_title('Top 20 Largest Leaves', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'#{i+1}' for i in range(len(leaf_sorted))], rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/6_top_leaves_area.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ 已生成: {output_dir}/6_top_leaves_area.png")
        plt.close()
    
    # 7. 置信度分布
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, organ_type in enumerate(['stem', 'leaf', 'branch']):
        organ_df = df[df['organ_type'] == organ_type]
        if len(organ_df) > 0:
            axes[idx].hist(organ_df['confidence'], bins=20, 
                          color=colors[organ_type], edgecolor='black', alpha=0.7)
            axes[idx].axvline(organ_df['confidence'].mean(), color='red', 
                            linestyle='--', linewidth=2)
            axes[idx].set_xlabel('Confidence', fontsize=11)
            axes[idx].set_ylabel('Frequency', fontsize=11)
            axes[idx].set_title(f'{organ_type.capitalize()} Confidence', 
                              fontsize=12, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/7_confidence_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 已生成: {output_dir}/7_confidence_distribution.png")
    plt.close()
    
    print()

def compare_organs_radar(df, output_file):
    """创建器官特征雷达图"""
    from math import pi
    
    # 准备数据（归一化）
    features = ['length', 'width', 'convex_hull_area', 'elongation', 'flatness']
    feature_labels = ['Length', 'Width', 'Area', 'Elongation', 'Flatness']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = [n / float(len(features)) * 2 * pi for n in range(len(features))]
    angles += angles[:1]
    
    colors_list = {'stem': '#FF6B6B', 'leaf': '#4ECDC4', 'branch': '#95E1D3'}
    
    for organ_type in ['stem', 'leaf', 'branch']:
        organ_df = df[df['organ_type'] == organ_type]
        if len(organ_df) == 0:
            continue
        
        values = []
        for feat in features:
            # 归一化到0-1
            max_val = df[feat].max()
            min_val = df[feat].min()
            if max_val > min_val:
                norm_val = (organ_df[feat].mean() - min_val) / (max_val - min_val)
            else:
                norm_val = 0.5
            values.append(norm_val)
        
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=organ_type.capitalize(), color=colors_list[organ_type])
        ax.fill(angles, values, alpha=0.15, color=colors_list[organ_type])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Organ Feature Comparison (Normalized)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ 已生成: {output_file}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='表型数据可视化工具')
    parser.add_argument('--csv', required=True, help='表型CSV文件路径')
    parser.add_argument('--output_dir', default='phenotype_visualizations', 
                       help='图表输出目录')
    args = parser.parse_args()
    
    # 加载数据
    df = load_phenotype_data(args.csv)
    
    # 打印统计
    print_statistics(df)
    
    # 创建汇总表
    summary_file = args.csv.replace('.csv', '_summary.csv')
    create_summary_table(df, summary_file)
    
    # 生成图表
    plot_distributions(df, args.output_dir)
    
    # 雷达图
    compare_organs_radar(df, f'{args.output_dir}/8_organ_comparison_radar.png')
    
    print("\n" + "=" * 80)
    print("✅ 全部完成！")
    print("=" * 80)
    print("\n生成的文件:")
    print(f"  - 汇总表: {summary_file}")
    print(f"  - 图表目录: {args.output_dir}/")
    print(f"    包含8张图表:")
    print(f"    1. 器官数量柱状图")
    print(f"    2. 叶片长度分布")
    print(f"    3. 叶片面积分布")
    print(f"    4. 长度vs宽度散点图")
    print(f"    5. 细长度箱线图")
    print(f"    6. 最大20片叶子")
    print(f"    7. 置信度分布")
    print(f"    8. 器官特征雷达图")
    print("\n" + "=" * 80)


