"""
Graph Cut后处理 - 分离叶柄-分支连接处
这是一个无需重新训练的后处理方法，基于图论和几何约束
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
import argparse
import os


class GraphCutRefiner:
    """
    基于Graph Cut的实例细化
    专门用于分离叶柄-分支连接处
    """
    def __init__(self, radius=0.02, min_instance_points=100):
        self.radius = radius  # 构图半径
        self.min_instance_points = min_instance_points  # 最小实例点数
    
    def compute_local_features(self, coords, k=10):
        """
        计算每个点的局部几何特征
        """
        tree = cKDTree(coords)
        features = []
        
        for i in range(len(coords)):
            # 找k个最近邻
            dists, indices = tree.query(coords[i], k=min(k, len(coords)))
            neighbors = coords[indices[1:]]  # 排除自己
            
            if len(neighbors) < 3:
                # 邻域太小
                features.append([1.0, 1.0, 0.01])
                continue
            
            # PCA计算形状
            pca = PCA(n_components=3)
            try:
                pca.fit(neighbors)
                eigenvalues = pca.explained_variance_
                
                # 细长度和扁平度
                elongation = eigenvalues[0] / (eigenvalues[1] + 1e-6)
                flatness = eigenvalues[1] / (eigenvalues[2] + 1e-6)
                thickness = np.sqrt(eigenvalues[2])
                
                features.append([elongation, flatness, thickness])
            except:
                features.append([1.0, 1.0, 0.01])
        
        return np.array(features)
    
    def build_graph(self, coords, features, class_id):
        """
        构建点云图
        边权重基于：
        1. 距离（近的权重高）
        2. 几何相似性（形状相似的权重高）
        3. 叶柄检测（叶柄-非叶柄连接权重低，容易被切断）
        """
        N = len(coords)
        tree = cKDTree(coords)
        
        # 找邻域
        pairs = tree.query_pairs(self.radius, output_type='ndarray')
        
        if len(pairs) == 0:
            # 没有连接
            return None, None
        
        # 计算边权重
        weights = []
        
        for i, j in pairs:
            # 1. 距离权重（近的连接强）
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_weight = np.exp(-dist / self.radius)  # 0-1
            
            # 2. 几何相似性
            elong_i, flat_i, thick_i = features[i]
            elong_j, flat_j, thick_j = features[j]
            
            # 几何差异
            elong_diff = abs(elong_i - elong_j) / (max(elong_i, elong_j) + 1e-6)
            flat_diff = abs(flat_i - flat_j) / (max(flat_i, flat_j) + 1e-6)
            
            # 差异小 = 相似 = 权重高
            geom_similarity = np.exp(-(elong_diff + flat_diff))
            
            # 3. 叶柄检测
            # 对于leaf类别：
            if class_id == 1:  # leaf
                # 如果一个点是叶片主体（扁平），另一个是叶柄（细长）
                # 判断标准：flatness相差大 且 一个很扁平一个不扁平
                is_petiole_i = (elong_i > 3) and (flat_i < 5)  # 细长但不太扁
                is_petiole_j = (elong_j > 3) and (flat_j < 5)
                
                is_leaf_body_i = (flat_i > 10)  # 很扁平 = 叶片主体
                is_leaf_body_j = (flat_j > 10)
                
                # 如果一个是叶片主体，一个是叶柄 → 可能是内部连接，保持
                # 如果一个是叶柄，另一个既不是叶片也不是叶柄 → 可能是叶柄-分支连接，降低权重
                if (is_petiole_i or is_petiole_j) and not (is_leaf_body_i or is_leaf_body_j):
                    # 叶柄到非叶片区域 - 降低权重（容易被切）
                    petiole_penalty = 0.3
                else:
                    petiole_penalty = 1.0
            else:
                petiole_penalty = 1.0
            
            # 综合权重
            weight = dist_weight * geom_similarity * petiole_penalty
            weights.append(weight)
        
        weights = np.array(weights)
        
        # 构建稀疏邻接矩阵
        if len(pairs) > 0:
            row_indices = pairs[:, 0]
            col_indices = pairs[:, 1]
            
            # 对称矩阵
            row = np.concatenate([row_indices, col_indices])
            col = np.concatenate([col_indices, row_indices])
            weights_sym = np.concatenate([weights, weights])
            
            adj_matrix = csr_matrix((weights_sym, (row, col)), shape=(N, N))
        else:
            adj_matrix = csr_matrix((N, N))
        
        if len(pairs) > 0:
            return adj_matrix, pairs, weights
        else:
            return adj_matrix, pairs, np.array([])
    
    def cut_weak_connections(self, coords, adj_matrix, threshold_percentile=10):
        """
        切断弱连接（权重低的边）
        
        Args:
            threshold_percentile: 切断权重最低的X%的边
        """
        # 转换为COO格式以访问边
        coo = adj_matrix.tocoo()
        
        # 只看上三角（避免重复）
        upper_triangle_mask = coo.row < coo.col
        edges = np.stack([coo.row[upper_triangle_mask], coo.col[upper_triangle_mask]], axis=1)
        edge_weights = coo.data[upper_triangle_mask]
        
        if len(edge_weights) == 0:
            return adj_matrix
        
        # 找到阈值
        threshold = np.percentile(edge_weights, threshold_percentile)
        
        # 切断弱边
        strong_edges_mask = edge_weights > threshold
        strong_edges = edges[strong_edges_mask]
        strong_weights = edge_weights[strong_edges_mask]
        
        # 重建图
        N = adj_matrix.shape[0]
        row = np.concatenate([strong_edges[:, 0], strong_edges[:, 1]])
        col = np.concatenate([strong_edges[:, 1], strong_edges[:, 0]])
        weights = np.concatenate([strong_weights, strong_weights])
        
        refined_adj = csr_matrix((weights, (row, col)), shape=(N, N))
        
        return refined_adj
    
    def extract_components(self, adj_matrix):
        """
        提取连通分量（每个分量是一个实例）
        """
        n_components, labels = connected_components(
            adj_matrix, 
            directed=False, 
            return_labels=True
        )
        
        return n_components, labels
    
    def refine_instance(self, instance_mask, coords, class_id):
        """
        对单个实例进行细化
        
        Returns:
            refined_instances: list of masks，可能从1个变成多个
        """
        instance_coords = coords[instance_mask]
        instance_indices = np.where(instance_mask)[0]
        
        if len(instance_coords) < 50:
            # 太小，不处理
            return [instance_mask]
        
        # 1. 计算局部特征
        features = self.compute_local_features(instance_coords)
        
        # 2. 构建图
        adj_matrix, pairs, edge_weights = self.build_graph(instance_coords, features, class_id)
        
        if adj_matrix is None:
            return [instance_mask]
        
        # 3. 切断弱连接（叶柄-分支连接处）
        refined_adj = self.cut_weak_connections(instance_coords, adj_matrix, threshold_percentile=5)
        
        # 4. 提取连通分量
        n_components, component_labels = self.extract_components(refined_adj)
        
        if n_components <= 1:
            # 没有分离出新实例
            return [instance_mask]
        
        # 5. 创建新的实例masks
        refined_instances = []
        for comp_id in range(n_components):
            comp_mask_local = (component_labels == comp_id)
            comp_indices = instance_indices[comp_mask_local]
            
            # 过滤太小的分量
            if len(comp_indices) >= self.min_instance_points:
                new_mask = np.zeros_like(instance_mask, dtype=bool)
                new_mask[comp_indices] = True
                refined_instances.append(new_mask)
        
        if len(refined_instances) == 0:
            return [instance_mask]
        
        print(f"  实例分离: 1 → {len(refined_instances)}个sub-instances")
        
        return refined_instances


def refine_predictions(pred_path, room_name, output_path, min_instance_points=100):
    """
    对预测结果进行Graph Cut细化
    """
    print("=" * 80)
    print(f"Graph Cut细化: {room_name}")
    print("=" * 80)
    
    # 加载数据
    coords = np.load(f'{pred_path}/coords/{room_name}.npy')
    sem_pred = np.load(f'{pred_path}/semantic_pred/{room_name}.npy')
    
    # 读取预测实例
    pred_file = f'{pred_path}/pred_instance/{room_name}.txt'
    
    refiner = GraphCutRefiner(radius=0.02, min_instance_points=min_instance_points)
    
    refined_instances = []
    
    with open(pred_file, 'r') as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()
            mask_file = parts[0]
            class_id = int(parts[1])
            conf = float(parts[2])
            
            # 加载mask
            mask = np.loadtxt(f'{pred_path}/pred_instance/{mask_file}', dtype=bool)
            
            if len(mask) != len(coords):
                print(f"  跳过{mask_file}: 维度不匹配")
                continue
            
            # 只对leaf类别进行细化（分离叶柄）
            if class_id == 2 and conf > 0.05:  # leaf类别
                sub_instances = refiner.refine_instance(mask, coords, class_id)
                
                for sub_mask in sub_instances:
                    refined_instances.append({
                        'mask': sub_mask,
                        'class_id': class_id,
                        'conf': conf
                    })
            else:
                refined_instances.append({
                    'mask': mask,
                    'class_id': class_id,
                    'conf': conf
                })
    
    print(f"\n原始实例数: {len(open(pred_file).readlines())}")
    print(f"细化后实例数: {len(refined_instances)}")
    
    # 保存细化结果（简化：只打印统计）
    leaf_count = sum(1 for inst in refined_instances if inst['class_id'] == 2 and inst['conf'] > 0.05)
    print(f"Leaf实例数（conf>0.05）: {leaf_count}")
    
    return refined_instances


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Cut后处理细化')
    parser.add_argument('--prediction_path', required=True, help='预测结果路径')
    parser.add_argument('--room_name', required=True, help='场景名称')
    parser.add_argument('--output_path', default='results/graph_cut_refined', help='输出路径')
    parser.add_argument('--min_points', type=int, default=100, help='最小实例点数')
    args = parser.parse_args()
    
    refined = refine_predictions(
        args.prediction_path, 
        args.room_name, 
        args.output_path,
        args.min_points
    )
    
    print("\n" + "=" * 80)
    print("细化完成！")
    print("=" * 80)

