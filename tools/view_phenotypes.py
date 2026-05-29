"""
表型数据查看工具（无需matplotlib）
"""
import argparse
import pandas as pd
import numpy as np

def view_phenotypes(csv_file):
    """查看表型数据"""
    
    # 读取CSV
    df = pd.read_csv(csv_file)
    
    print("=" * 100)
    print(f"表型数据文件: {csv_file}")
    print("=" * 100)
    print(f"\n总器官数: {len(df)}")
    print("\n器官类型分布:")
    print(df['organ_type'].value_counts())
    
    # 显示前10行
    print("\n" + "=" * 100)
    print("前10个器官的数据:")
    print("=" * 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    print(df.head(10).to_string(index=False))
    
    # 各器官统计
    print("\n" + "=" * 100)
    print("各器官统计汇总:")
    print("=" * 100)
    
    for organ_type in ['stem', 'leaf', 'branch']:
        organ_df = df[df['organ_type'] == organ_type]
        if len(organ_df) == 0:
            continue
        
        print(f"\n【{organ_type.upper()}】 共{len(organ_df)}个实例")
        print("-" * 100)
        
        # 主要参数
        params = [
            ('长度', 'length'),
            ('宽度', 'width'),
            ('厚度', 'thickness'),
            ('表面积', 'convex_hull_area'),
            ('体积', 'convex_hull_volume'),
            ('细长度', 'elongation'),
            ('扁平度', 'flatness'),
            ('点数', 'num_points'),
            ('置信度', 'confidence')
        ]
        
        print(f"{'参数':<12} {'平均值':>12} {'标准差':>12} {'最小值':>12} {'最大值':>12}")
        print("-" * 100)
        
        for name, col in params:
            if col in organ_df.columns:
                values = organ_df[col].dropna()
                if len(values) > 0:
                    print(f"{name:<12} {values.mean():>12.4f} {values.std():>12.4f} "
                          f"{values.min():>12.4f} {values.max():>12.4f}")
    
    # 植株级别统计
    print("\n" + "=" * 100)
    print("植株级别表型:")
    print("=" * 100)
    
    leaf_df = df[df['organ_type'] == 'leaf']
    branch_df = df[df['organ_type'] == 'branch']
    stem_df = df[df['organ_type'] == 'stem']
    
    if len(leaf_df) > 0:
        total_leaf_area = leaf_df['convex_hull_area'].sum()
        print(f"\n叶片统计:")
        print(f"  总叶片数: {len(leaf_df)}")
        print(f"  总叶面积: {total_leaf_area:.4f}")
        print(f"  平均叶面积: {total_leaf_area / len(leaf_df):.4f}")
        print(f"  最大叶片长度: {leaf_df['length'].max():.4f}")
        print(f"  最小叶片长度: {leaf_df['length'].min():.4f}")
        print(f"  叶片大小变异系数: {leaf_df['convex_hull_area'].std() / leaf_df['convex_hull_area'].mean():.4f}")
    
    if len(branch_df) > 0:
        print(f"\n分枝统计:")
        print(f"  总分枝数: {len(branch_df)}")
        print(f"  平均分枝长度: {branch_df['length'].mean():.4f}")
    
    if len(stem_df) > 0:
        print(f"\n茎秆统计:")
        print(f"  检测到的茎秆数: {len(stem_df)}")
        print(f"  总茎秆长度: {stem_df['length'].sum():.4f}")
    
    # 创建汇总表
    print("\n" + "=" * 100)
    print("汇总对比表:")
    print("=" * 100)
    
    summary_data = []
    for organ_type in ['stem', 'leaf', 'branch']:
        organ_df = df[df['organ_type'] == organ_type]
        if len(organ_df) > 0:
            summary_data.append({
                '器官': organ_type,
                '数量': len(organ_df),
                '平均长度': f"{organ_df['length'].mean():.3f}",
                '平均宽度': f"{organ_df['width'].mean():.3f}",
                '平均面积': f"{organ_df['convex_hull_area'].mean():.3f}",
                '平均细长度': f"{organ_df['elongation'].mean():.1f}",
                '平均置信度': f"{organ_df['confidence'].mean():.3f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # 保存汇总表
    summary_file = csv_file.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n汇总表已保存到: {summary_file}")
    
    # 保存详细统计报告
    report_file = csv_file.replace('.csv', '_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"表型分析报告\n")
        f.write(f"数据文件: {csv_file}\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        
        for organ_type in ['stem', 'leaf', 'branch']:
            organ_df = df[df['organ_type'] == organ_type]
            if len(organ_df) > 0:
                f.write(f"\n{organ_type.upper()} 详细数据:\n")
                f.write("-" * 100 + "\n")
                f.write(organ_df[['instance_id', 'confidence', 'num_points', 'length', 
                                 'width', 'convex_hull_area']].to_string(index=False))
                f.write("\n\n")
    
    print(f"详细报告已保存到: {report_file}")
    
    print("\n" + "=" * 100)
    print("✅ 完成！")
    print("=" * 100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='查看表型数据（无需matplotlib）')
    parser.add_argument('--csv', required=True, help='表型CSV文件路径')
    args = parser.parse_args()
    
    view_phenotypes(args.csv)


