#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断验证集上 leaf GT 的召回情况：
- 对每个 leaf GT，计算与所有 leaf 预测实例的最大 IoU
- 统计 IoU 分布：<small, small~mid, >=mid
- 统计 leaf GT 点数直方图，判断是否主要是“小叶片”丢失

注意：
- 不依赖现有训练 / 推理代码，作为独立脚本使用
- 仅使用 numpy / tqdm 等基础库
"""

import os
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def load_pred_instances(prediction_path, room_name, leaf_cls_id):
    """读取某个 room 的预测实例，只保留 leaf 类别

    预测目录结构假定为：
      prediction_path/
        pred_instance/
          {room_name}.txt        # 每行: mask_rel_path cls_id score
          predicted_masks/xxx.txt 或 .npy
    """
    inst_txt = os.path.join(prediction_path, "pred_instance", f"{room_name}.txt")
    if not os.path.isfile(inst_txt):
        return []

    preds = []
    with open(inst_txt, "r") as f:
        for line in f:
            items = line.strip().split()
            if len(items) < 3:
                continue
            rel_path, cls_id_str, score_str = items[0], items[1], items[2]
            try:
                cls_id = int(cls_id_str)
                score = float(score_str)
            except Exception:
                continue

            if cls_id != leaf_cls_id:
                continue

            mask_path = os.path.join(prediction_path, rel_path)
            if not os.path.isfile(mask_path):
                alt = os.path.join(prediction_path, "pred_instance", rel_path)
                if os.path.isfile(alt):
                    mask_path = alt
                else:
                    # 找不到 mask，跳过
                    continue

            # mask 文件可能是 txt(索引) 或 npy(bool/int)
            if mask_path.endswith(".npy"):
                mask = np.load(mask_path)
            else:
             mask = np.loadtxt(mask_path, dtype=np.int64)

            # 统一成 bool 向量（1 表示属于该实例）
            mask = mask.astype(bool)
            preds.append(mask)

    if len(preds) == 0:
        return []

    # (num_pred, N)
    return np.stack(preds, axis=0)


def load_gt_from_val_txt(gt_root, room_name):
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
    txt_path = os.path.join(gt_root, f"{room_name}.txt")
    if not os.path.isfile(txt_path):
        return None, None

    data = np.loadtxt(txt_path).astype(int)
    # 兼容单列或多列格式：我们只关心最后一列 label
    if data.ndim == 1:
        labels = data
    else:
        labels = data[:, -1]

    semantic = labels // 1000
    instance = labels % 1000 - 1

    return semantic, instance


def main():
    parser = argparse.ArgumentParser(description="诊断 leaf GT 召回情况（直接读取 val_gt/*.txt）")
    parser.add_argument(
        "--gt_root",
        required=True,
        help="GT 根目录，例如 /home/.../dataset/Panax/val_gt，内部是 Area_xxx_yyy.txt",
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
        help="leaf 的类别 id（默认 1）",
    )
    parser.add_argument(
        "--iou_thr_small",
        type=float,
        default=0.25,
        help="IoU < iou_thr_small 记为严重遗漏",
    )
    parser.add_argument(
        "--iou_thr_mid",
        type=float,
        default=0.5,
        help="iou_thr_small <= IoU < iou_thr_mid 记为中等覆盖",
    )

    args = parser.parse_args()

    # 从 gt_root 自动推断 room 列表
    rooms = []
    for fn in os.listdir(args.gt_root):
        if not fn.endswith(".txt"):
            continue
        room = os.path.splitext(fn)[0]
        rooms.append(room)
    rooms.sort()

    if len(rooms) == 0:
        print("[ERROR] 未在 gt_root 中找到任何 txt GT 文件，请检查路径")
        return

    print(f"[info] GT 根目录: {args.gt_root}")
    print(f"[info] 预测根目录: {args.prediction_path}")
    print(f"[info] 将在 {len(rooms)} 个 room 上进行诊断")

    # 全局统计量
    bins_small = 0  # IoU < small
    bins_mid = 0    # small <= IoU < mid
    bins_good = 0   # IoU >= mid
    pts_hist = defaultdict(int)  # 以点数做粗直方图
    per_room_leaf_gt = []

    total_leaf_gt = 0

    for room in tqdm(rooms, desc="rooms"):
        gt_sem, gt_ins = load_gt_from_val_txt(args.gt_root, room)
        if gt_sem is None:
            continue

        # 找出 leaf GT 实例 id
        mask_leaf = (gt_sem == args.leaf_cls_id) & (gt_ins >= 0)
        leaf_gt_ids = np.unique(gt_ins[mask_leaf])
        leaf_gt_ids = leaf_gt_ids[leaf_gt_ids >= 0]
        if leaf_gt_ids.size == 0:
            continue

        # 读取预测的 leaf 实例 mask
        pred_leaf_masks = load_pred_instances(args.prediction_path, room, args.leaf_cls_id)

        per_room_leaf_gt.append((room, int(leaf_gt_ids.size)))
        total_leaf_gt += int(leaf_gt_ids.size)

        if pred_leaf_masks == []:
            # 没有任何 leaf 预测，所有 leaf GT 都算严重遗漏
            for gid in leaf_gt_ids:
                gt_mask = (gt_ins == gid)
                pts = int(gt_mask.sum())
                bucket = min(pts // 50 * 50, 1000)
                pts_hist[bucket] += 1
                bins_small += 1
            continue

        pred_leaf_masks = pred_leaf_masks.astype(bool)  # (P, N)

        # 对每个 leaf GT 计算最大 IoU
        for gid in leaf_gt_ids:
            gt_mask = (gt_ins == gid)
            pts = int(gt_mask.sum())
            bucket = min(pts // 50 * 50, 1000)
            pts_hist[bucket] += 1

            inter = (pred_leaf_masks & gt_mask).sum(axis=1)
            union = pred_leaf_masks.sum(axis=1) + gt_mask.sum() - inter
            union = np.clip(union, 1, None)
            ious = inter / union
            best_iou = float(ious.max()) if ious.size > 0 else 0.0

            if best_iou < args.iou_thr_small:
                bins_small += 1
            elif best_iou < args.iou_thr_mid:
                bins_mid += 1
            else:
                bins_good += 1

    print("\n================ Leaf GT 覆盖统计 ================")
    print(f"总 leaf GT 数: {total_leaf_gt}")
    print(f"IoU < {args.iou_thr_small:.2f}: {bins_small}")
    print(f"{args.iou_thr_small:.2f} <= IoU < {args.iou_thr_mid:.2f}: {bins_mid}")
    print(f"IoU >= {args.iou_thr_mid:.2f}: {bins_good}")

    print("\n================ Leaf GT 点数直方图（桶宽约 50 点，>1000 合并） ================")
    for k in sorted(pts_hist.keys()):
        print(f"{k:4d} ~ {k+49:4d}: {pts_hist[k]}")

    print("\n================ 每个 room 的 leaf GT 数 ================")
    for room, cnt in per_room_leaf_gt:
        print(f"{room}: {cnt}")


if __name__ == "__main__":
    main()


