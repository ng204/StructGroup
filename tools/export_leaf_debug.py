#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
导出单片 GT 叶子及其对应的预测 leaf 实例，用于在 CloudCompare 中可视化：

- 从 Panax 预处理数据 (.pth) 中读取坐标和颜色
- 从 val_gt/*.txt 中解析语义 / 实例 GT（label = sem*1000 + inst）
- 从 results/.../pred_instance/*.txt 中读取 leaf 预测 mask
- 选择指定场景的一片 GT 叶子，找出所有与其有交集的预测 leaf 实例
- 导出一个 PLY，包含：
    x, y, z, r, g, b, gt_leaf(0/1), pred_leaf_id(0..K)
  方便在 CloudCompare 里按标量着色，看是被“切碎”还是整体偏移
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm


def load_preprocess_scene(pre_root, scene_name):
    """加载 Panax 预处理 .pth，返回 coords(N,3), colors(N,3)"""
    path = os.path.join(pre_root, f"{scene_name}_inst_nostuff.pth")
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, (list, tuple)) or len(obj) < 2:
        raise RuntimeError(f"预处理文件格式异常: {path}")
    coords = np.asarray(obj[0])  # (N,3), float32
    colors = np.asarray(obj[1])  # (N,3)
    return coords, colors


def load_gt_from_val_txt(gt_root, scene_name):
    """从 val_gt/*.txt 读取 GT 语义和实例标签

    当前 Panax 的 val_gt 采用类似 S3DIS 的编码方式：
      label = semantic_id * 1000 + instance_id
    其中：
      - semantic_id: 0=stem, 1=leaf, 2=branch
      - instance_id: 从 1 开始的实例编号（0 表示无实例）
    我们解码为：
      semantic = label // 1000
      instance = label % 1000 - 1  （使实例 id 从 0 开始，-1 表示无实例）
    """
    txt_path = os.path.join(gt_root, f"{scene_name}.txt")
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"GT 文件不存在: {txt_path}")
    data = np.loadtxt(txt_path).astype(int)
    if data.ndim == 1:
        labels = data
    else:
        labels = data[:, -1]
    semantic = labels // 1000
    instance = labels % 1000 - 1
    return semantic, instance


def load_leaf_pred_masks(pred_root, scene_name, leaf_cls_id, num_points):
    """读取某个场景的 leaf 预测实例，返回一个 (P_leaf, N) 的 bool 矩阵"""
    inst_txt = os.path.join(pred_root, "pred_instance", f"{scene_name}.txt")
    if not os.path.isfile(inst_txt):
        print(f"[warn] 预测实例文件不存在: {inst_txt}")
        return np.zeros((0, num_points), dtype=bool)

    masks = []
    with open(inst_txt, "r") as f:
        for line in f:
            items = line.strip().split()
            if len(items) < 3:
                continue
            rel_path, cls_id_str, score_str = items[0], items[1], items[2]
            try:
                cls_id = int(cls_id_str)
                _ = float(score_str)
            except Exception:
                continue
            if cls_id != leaf_cls_id:
                continue

            mask_path = os.path.join(pred_root, rel_path)
            if not os.path.isfile(mask_path):
                alt = os.path.join(pred_root, "pred_instance", rel_path)
                if os.path.isfile(alt):
                    mask_path = alt
                else:
                    print(f"[warn] 找不到 mask 文件: {rel_path}")
                    continue

            # mask 保存的是点索引列表
            idx = np.loadtxt(mask_path, dtype=np.int64)
            idx = np.atleast_1d(idx)
            valid = (idx >= 0) & (idx < num_points)
            idx = idx[valid]
            m = np.zeros(num_points, dtype=bool)
            m[idx] = True
            masks.append(m)

    if not masks:
        return np.zeros((0, num_points), dtype=bool)
    return np.stack(masks, axis=0)


def write_ply_with_scalars(path, coords, colors, gt_leaf, pred_leaf_id):
    """写出 PLY，包含 x y z r g b gt_leaf pred_leaf_id"""
    assert coords.shape[0] == colors.shape[0] == gt_leaf.shape[0] == pred_leaf_id.shape[0]
    N = coords.shape[0]

    # 颜色自适应到 0-255 uint8
    rgb = colors.astype(np.float32)
    if rgb.max() <= 1.5:
        rgb = (rgb * 255.0).clip(0, 255)
    else:
        rgb = rgb.clip(0, 255)
    rgb = rgb.astype(np.uint8)

    gt_leaf = gt_leaf.astype(np.int32)
    pred_leaf_id = pred_leaf_id.astype(np.int32)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int gt_leaf\n")
        f.write("property int pred_leaf_id\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = coords[i]
            r, g, b = rgb[i]
            gl = gt_leaf[i]
            pid = pred_leaf_id[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {int(gl)} {int(pid)}\n")


def main():
    parser = argparse.ArgumentParser(description="导出单片 GT 叶子及对应预测实例用于可视化调试")
    parser.add_argument(
        "--scene",
        default="Area_46_Panax_45",
        help="场景名，例如 Area_46_Panax_45",
    )
    parser.add_argument(
        "--pre_root",
        default="dataset/Panax/preprocess",
        help="预处理数据根目录，包含 *_inst_nostuff.pth",
    )
    parser.add_argument(
        "--gt_root",
        default="dataset/Panax/val_gt",
        help="GT 根目录，包含 Area_xxx_yyy.txt",
    )
    parser.add_argument(
        "--prediction_path",
        required=True,
        help="预测结果根目录（包含 pred_instance 子目录）",
    )
    parser.add_argument(
        "--leaf_cls_id",
        type=int,
        default=1,
        help="leaf 语义类别 id（默认 1）",
    )
    parser.add_argument(
        "--out_dir",
        default="vis_output",
        help="输出 PLY 存放目录",
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scene = args.scene
    print(f"[info] scene = {scene}")

    coords, colors = load_preprocess_scene(args.pre_root, scene)
    N = coords.shape[0]
    print(f"[info] loaded preprocess: coords shape = {coords.shape}, colors shape = {colors.shape}")

    gt_sem, gt_ins = load_gt_from_val_txt(args.gt_root, scene)
    if gt_sem.shape[0] != N:
        print(f"[warn] GT 点数 ({gt_sem.shape[0]}) 与 preprocess 点数 ({N}) 不一致，可能顺序不同")
    else:
        print(f"[info] loaded GT: semantic/instance shape = {gt_sem.shape}")

    # 找出 leaf GT 实例 id（当前每个场景 1 片目标叶子）
    mask_leaf = (gt_sem == args.leaf_cls_id) & (gt_ins >= 0)
    leaf_gt_ids = np.unique(gt_ins[mask_leaf])
    leaf_gt_ids = leaf_gt_ids[leaf_gt_ids >= 0]
    if leaf_gt_ids.size == 0:
        print("[warn] 该场景没有 leaf GT 实例")
        return
    if leaf_gt_ids.size > 1:
        print(f"[info] 该场景有 {leaf_gt_ids.size} 个 leaf GT 实例，默认选择第一个 id={leaf_gt_ids[0]}")
    leaf_gt_id = int(leaf_gt_ids[0])

    gt_leaf_mask = (gt_ins == leaf_gt_id)
    print(f"[info] 选中 GT leaf id={leaf_gt_id}, 点数 = {int(gt_leaf_mask.sum())}")

    # 读取预测 leaf 实例，并找出与该 GT leaf 有交集的实例
    pred_leaf_masks = load_leaf_pred_masks(args.prediction_path, scene, args.leaf_cls_id, N)
    print(f"[info] 预测 leaf 实例数 = {pred_leaf_masks.shape[0]}")

    pred_leaf_id_per_point = np.zeros(N, dtype=np.int32)  # 0 表示未被选中 leaf 实例覆盖
    if pred_leaf_masks.shape[0] > 0:
        # 与该 GT leaf 有交集的实例
        inter = (pred_leaf_masks & gt_leaf_mask).sum(axis=1)
        overlapping = np.nonzero(inter > 0)[0]
        print(f"[info] 与 GT leaf 有交集的预测 leaf 实例数 = {overlapping.size}")
        for local_id, inst_idx in enumerate(overlapping, start=1):
            m = pred_leaf_masks[inst_idx]
            pred_leaf_id_per_point[m] = local_id

    # 构建标量：gt_leaf (0/1)
    gt_leaf_scalar = gt_leaf_mask.astype(np.int32)

    # 仅保留“与 GT leaf 相关”的点：既包括 GT leaf 自身，也包括所有与之相交的预测 leaf 实例点
    related_mask = gt_leaf_mask | (pred_leaf_id_per_point > 0)
    coords_out = coords[related_mask]
    colors_out = colors[related_mask]
    gt_leaf_out = gt_leaf_scalar[related_mask]
    pred_leaf_id_out = pred_leaf_id_per_point[related_mask]

    out_path = os.path.join(args.out_dir, f"{scene}_leaf_debug.ply")
    write_ply_with_scalars(out_path, coords_out, colors_out, gt_leaf_out, pred_leaf_id_out)
    print(f"[done] 导出调试 PLY: {out_path}")
    print("CloudCompare 中建议：")
    print("  - 用原始 RGB 看整体形状")
    print("  - 用标量 gt_leaf（0/1）着色，看 GT 叶片轮廓")
    print("  - 用标量 pred_leaf_id 着色，看同一片 GT 叶子被几个预测实例覆盖 / 是否整体偏移")


if __name__ == "__main__":
    main()


