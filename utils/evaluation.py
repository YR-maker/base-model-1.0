from pathlib import Path

import numpy as np
import torch
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.measure import euler_number, label
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import SimpleITK as sitk

# [新增] 导入 MONAI 指标库
from monai.metrics import compute_average_surface_distance, compute_surface_dice


class Evaluator:
    """
    血管分割评估器类

    实现评估指标，包括：
    - Dice系数：分割重叠度评估
    - clDice：拓扑结构保持度评估
    - Betti数误差：拓扑特征评估
    - NSD: 归一化表面Dice (Surface Overlap)
    - ASD: 平均表面距离 (Boundary Accuracy)
    - 传统分类指标：精确度、召回率等
    """

    def extract_labels(self, gt_array, pred_array):
        """提取真实值和预测值中的标签"""
        labels_gt = np.unique(gt_array)
        labels_pred = np.unique(pred_array)
        labels = list(set().union(labels_gt, labels_pred))
        labels = [int(x) for x in labels]
        return labels

    def betti_number_error(self, gt, pred):
        """计算Betti数误差 - 评估拓扑结构差异"""
        labels = self.extract_labels(gt_array=gt, pred_array=pred)
        if 0 in labels:
            labels.remove(0)

        if len(labels) == 0:
            return 0, 0

        # 处理可能的非二值情况，这里简化处理只看前景
        # assert len(labels) == 1 and 1 in labels, "Invalid binary segmentation."

        gt_betti_numbers = self.betti_number(gt)
        pred_betti_numbers = self.betti_number(pred)
        betti_0_error = abs(pred_betti_numbers[0] - gt_betti_numbers[0])
        betti_1_error = abs(pred_betti_numbers[1] - gt_betti_numbers[1])
        return betti_0_error, betti_1_error

    def betti_number(self, img):
        """计算3D二值图像的Betti数"""
        assert img.ndim == 3
        N6 = 1
        N26 = 3

        padded = np.pad(img, pad_width=1)
        # assert set(np.unique(padded)).issubset({0, 1})

        _, b0 = label(padded, return_num=True, connectivity=N26)
        euler_char_num = euler_number(padded, connectivity=N26)
        _, b2 = label(1 - padded, return_num=True, connectivity=N6)

        b2 -= 1
        b1 = b0 + b2 - euler_char_num
        return [b0, b1, b2]

    def cl_dice(self, v_p, v_l):
        """计算拓扑感知的clDice系数"""

        def cl_score(v, s):
            return np.sum(v * s) / np.sum(s)

        if len(v_p.shape) == 2:
            tprec = cl_score(v_p, skeletonize(v_l))
            tsens = cl_score(v_l, skeletonize(v_p))
        elif len(v_p.shape) == 3:
            tprec = cl_score(v_p, skeletonize_3d(v_l))
            tsens = cl_score(v_l, skeletonize_3d(v_p))
        else:
            raise ValueError(f"Invalid shape for cl_dice: {v_p.shape}")

        return 2 * tprec * tsens / (tprec + tsens + np.finfo(float).eps)

    def estimate_metrics(self, pred_seg, gt_seg, threshold=0.5, spacing=(1, 1, 1), nsd_tolerance=1.0, fast=False):
        """
        计算全面的分割评估指标 (新增 ASD 和 NSD)

        Args:
            pred_seg: 模型预测的分割概率图 (Tensor)
            gt_seg: 真实分割标签 (Tensor)
            threshold: 二值化阈值
            spacing: 图像的物理间距 (z, y, x)，默认 (1,1,1) 计算 voxel 距离
            nsd_tolerance: NSD 的容差阈值，默认 1.0
            fast: 快速模式标志

        Returns:
            metrics: 包含所有评估指标的字典
        """
        metrics = {}

        # 1. 基础数据准备
        # 确保数据在 CPU 上用于 sklearn/skimage 计算
        pred_seg_thresh = (pred_seg >= threshold).float().cpu()
        gt_seg_cpu = gt_seg.cpu()

        # 2. 计算混淆矩阵 (基于 flatten 数组)
        tn, fp, fn, tp = confusion_matrix(
            gt_seg_cpu.flatten().clone().numpy(),
            pred_seg_thresh.flatten().clone().numpy(),
            labels=[0, 1],
        ).ravel()

        # --- 快速模式返回 ---
        if fast:
            metrics["dice"] = (2 * tp) / (2 * tp + fp + fn + 1e-6)
            return metrics

        # 3. 计算 ASD 和 NSD (使用 MONAI)
        # MONAI 需要输入格式为 (Batch, Channel, Spatial...)
        # 假设输入 pred_seg 是 (D, H, W) 或 (B, D, H, W)，我们需要标准化为 (1, 1, D, H, W)

        # 转换并增加维度
        pred_monai = pred_seg_thresh.clone()
        gt_monai = gt_seg_cpu.clone()

        # 递归增加维度直到变成 5维 (B, C, D, H, W)
        while pred_monai.ndim < 5:
            pred_monai = pred_monai.unsqueeze(0)
        while gt_monai.ndim < 5:
            gt_monai = gt_monai.unsqueeze(0)

        # [新增] 计算 Average Surface Distance (ASD)
        # symmetric=True 表示计算双向平均距离
        try:
            asd = compute_average_surface_distance(
                y_pred=pred_monai,
                y=gt_monai,
                symmetric=True,
                spacing=spacing
            )
            # 处理 MONAI 返回 tensor 的情况
            metrics["asd"] = asd.item() if isinstance(asd, torch.Tensor) else asd
            if np.isinf(metrics["asd"]) or np.isnan(metrics["asd"]):
                metrics["asd"] = 0.0
        except Exception as e:
            # 如果预测为空或全满，可能导致计算失败
            metrics["asd"] = 0.0

        # [新增] 计算 Normalized Surface Dice (NSD)
        # class_thresholds 控制容忍距离
        try:
            nsd = compute_surface_dice(
                y_pred=pred_monai,
                y=gt_monai,
                class_thresholds=[nsd_tolerance],
                spacing=spacing
            )
            metrics["nsd"] = nsd.item() if isinstance(nsd, torch.Tensor) else nsd
            # NSD 默认为 0-1，如果需要百分比请在外部 * 100，这里保持小数
        except Exception as e:
            metrics["nsd"] = 0.0

        # 4. 计算其他复杂指标 (基于 numpy)
        gt_np = gt_seg_cpu.flatten().clone().detach().numpy()
        pred_np = pred_seg.flatten().cpu().clone().detach().numpy()
        pred_thresh_np = pred_seg_thresh.flatten().clone().detach().numpy()

        # ROC / PR AUC
        try:
            roc_auc = roc_auc_score(gt_np, pred_np)
        except ValueError:
            roc_auc = 0.0

        try:
            pr_auc = average_precision_score(gt_np, pred_np)
        except ValueError:
            pr_auc = 0.0

        # clDice 和 拓扑指标 (需要 3D numpy 数组)
        pred_3d = pred_seg_thresh.squeeze().clone().detach().byte().numpy()
        gt_3d = gt_seg_cpu.squeeze().clone().detach().byte().numpy()

        cldice = self.cl_dice(pred_3d, gt_3d)

        # Betti Errors
        # 注意：转为 int 避免类型问题
        betti_0_error, betti_1_error = self.betti_number_error(
            gt_3d.astype(int),
            pred_3d.astype(int)
        )

        # 预测的 Betti Numbers
        betti_0, betti_1, betti_2 = self.betti_number(pred_3d.astype(int))

        # 5. 汇总所有指标
        epsilon = 1e-6
        metrics["recall_tpr_sensitivity"] = tp / (tp + fn + epsilon)
        metrics["fpr"] = fp / (fp + tn + epsilon)
        metrics["precision"] = tp / (tp + fp + epsilon)
        metrics["specificity"] = tn / (tn + fp + epsilon)
        metrics["jaccard_iou"] = tp / (tp + fp + fn + epsilon)
        metrics["dice"] = (2 * tp) / (2 * tp + fp + fn + epsilon)
        metrics["cldice"] = cldice
        metrics["accuracy"] = (tp + tn) / (tn + fp + tp + fn + epsilon)
        metrics["roc_auc"] = roc_auc
        metrics["pr_auc_ap"] = pr_auc
        metrics["betti_0_error"] = betti_0_error
        metrics["betti_1_error"] = betti_1_error
        metrics["betti_0"] = betti_0
        metrics["betti_1"] = betti_1
        metrics["betti_2"] = betti_2

        return metrics


def read_nifti(path: str):
    """读取NIfTI格式的医学图像文件"""
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def calculate_mean_metrics(results, round_to=2):
    """
    计算多个结果的均值指标

    Args:
        results: 多个评估结果列表
        round_to: 结果舍入的小数位数

    Returns:
        mean: 平均指标字典
    """
    if not results:
        return {}

    mean = {}
    for k in results[0].keys():  # 遍历所有指标键
        numbers = [r[k] for r in results]  # 收集所有结果
        # 过滤 NaN 和 Inf
        numbers = [n for n in numbers if not np.isnan(n) and not np.isinf(n)]

        if not numbers:
            mean[k] = 0.0
            continue

        mean[k] = np.mean(numbers)

        # 某些指标习惯用百分数表示
        if k in ["dice", "cldice", "nsd", "jaccard_iou", "recall_tpr_sensitivity", "precision", "accuracy"]:
            mean[k] = mean[k] * 100

        mean[k] = np.round(mean[k], round_to)
    return mean