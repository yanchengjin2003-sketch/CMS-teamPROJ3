"""Batch voxel-wise multiclass accuracy evaluation over predicted crops.

Mirrors the "计算accuracy" cell in notebooks/test_sa2.ipynb
(argmax(pred) == argmax(groundtruth), averaged over voxels), with two fixes
over that cell:

1. Groundtruth channels are loaded in the exact order of `class_names`
   instead of via the positional slice `class_names[:-1]`. That slice only
   isolates 'bg' when 'bg' happens to be the last entry (true in the old
   4-class setup); with 6 classes it drops 'pm' instead and leaves a
   stale/duplicate 'bg' channel in the mix, silently misaligning
   groundtruth labels against prediction labels.

2. Predictions and groundtruth are not always stored at corresponding scale
   levels (some crops are resampled to isotropic voxels during prediction,
   which changes the z voxel count relative to the anisotropic groundtruth
   pyramid). Rather than assume a fixed `pred_scale`/`gt_scale` pair, we
   search the groundtruth scale pyramid for the level whose shape matches
   the prediction exactly, and skip the crop if none matches.
"""

from pathlib import Path

import numpy as np
import torch
import zarr

from post_process import util


class ShapeMismatchError(Exception):
    pass


def _label_group_path(domain: str, dataset: str, crop: str, class_name: str) -> Path:
    return Path(domain) / dataset / f'{dataset}.zarr' / 'recon-1' / 'labels' / 'groundtruth' / crop / class_name


def find_matching_gt_scale(
    pred_shape: tuple,
    origin_domain: str,
    bg_data_domain: str,
    dataset: str,
    crop: str,
    class_names: list[str],
    bg_label: str = 'bg',
    preferred_scale: str = 's1',
) -> str | None:
    """Find the groundtruth scale level whose array shape equals `pred_shape`.

    Checks the first class in `class_names` only (all classes in a crop
    share the same scale grid, per util.load_em_crop's assumption).
    """
    ref_name = class_names[0]
    domain = bg_data_domain if ref_name == bg_label else origin_domain
    group = zarr.open(str(_label_group_path(domain, dataset, crop, ref_name)), mode='r')

    scale_paths = [k for k in group.keys() if k.startswith('s')]
    scale_paths.sort(key=lambda s: (s != preferred_scale, s))

    for scale_path in scale_paths:
        if group[scale_path].shape == pred_shape:
            return scale_path
    return None


def load_groundtruth_multiclass_aligned(
    class_names: list[str],
    origin_domain: str,
    bg_data_domain: str,
    dataset: str,
    crop: str,
    scale: str,
    bg_label: str = 'bg',
) -> np.ndarray:
    """Load groundtruth channels in the exact order of `class_names`.

    The `bg_label` channel is read from `bg_data_domain` (the corrected
    bg-only package); every other channel is read from `origin_domain`.

    :return: (W, H, D, len(class_names))
    """
    channels = [
        util.load_groundtruth(
            bg_data_domain if name == bg_label else origin_domain,
            dataset, crop, name, scale,
        )
        for name in class_names
    ]
    return np.stack(channels, axis=-1)


def compute_iou(
    pred_labels: torch.Tensor | np.ndarray,
    gt_labels: torch.Tensor | np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """Per-class IoU (intersection over union) between two voxel label maps.

    :param pred_labels: (W, H, D) integer tensor/array of predicted class ids.
    :param gt_labels: (W, H, D) integer tensor/array of groundtruth class ids, same shape.
    :param class_names: class names indexed by label id, i.e. `class_names[i]` is the
        name of the class with id `i` (as produced by `argmax` over a `(..., len(class_names))`
        multiclass tensor).
    :return: {class_name: IoU} for every class present in `pred_labels` or `gt_labels`,
        plus 'mean_iou' (macro average over those classes). Classes absent from both
        pred and gt are skipped so they don't inflate the mean.
    :raises ShapeMismatchError: `pred_labels.shape != gt_labels.shape`.
    """
    pred_labels = torch.as_tensor(pred_labels)
    gt_labels = torch.as_tensor(gt_labels)

    if pred_labels.shape != gt_labels.shape:
        raise ShapeMismatchError(
            f"prediction shape {tuple(pred_labels.shape)} != groundtruth shape {tuple(gt_labels.shape)}"
        )

    iou_per_class = {}
    for label_id, name in enumerate(class_names):
        pred_mask = pred_labels == label_id
        gt_mask = gt_labels == label_id
        union = torch.count_nonzero(pred_mask | gt_mask)
        if union == 0:
            continue
        intersection = torch.count_nonzero(pred_mask & gt_mask)
        iou_per_class[name] = (intersection / union).item()

    iou_per_class['mean_iou'] = float(np.mean(list(iou_per_class.values()))) if iou_per_class else 0.0
    return iou_per_class


def compute_crop_accuracy(
    class_names: list[str],
    predictions_domain: str,
    origin_domain: str,
    bg_data_domain: str,
    dataset: str,
    crop: str,
    pred_scale: str = 's0',
    gt_scale: str = 's1',
) -> float:
    """Voxel-wise multiclass accuracy for a single crop: mean(argmax(pred) == argmax(gt)).

    :raises ShapeMismatchError: no groundtruth scale level matches the prediction shape.
    """
    pred_logits = util.load_multiclass_result(class_names, predictions_domain, crop, scale=pred_scale)
    pred_labels = pred_logits.argmax(axis=-1)

    matched_scale = find_matching_gt_scale(
        pred_labels.shape, origin_domain, bg_data_domain, dataset, crop, class_names,
        preferred_scale=gt_scale,
    )
    if matched_scale is None:
        raise ShapeMismatchError(
            f"no groundtruth scale matches prediction shape {pred_labels.shape}"
        )

    gt = load_groundtruth_multiclass_aligned(
        class_names, origin_domain, bg_data_domain, dataset, crop, scale=matched_scale,
    )
    gt_labels = gt.argmax(axis=-1)

    return float((pred_labels == gt_labels).mean())


def evaluate_predictions(
    predictions_domain: str,
    origin_domain: str,
    bg_data_domain: str,
    class_names: list[str],
    threshold: float = 0.7,
    pred_scale: str = 's0',
    gt_scale: str = 's1',
    verbose: bool = True,
) -> dict[str, dict]:
    """Compute per-crop accuracy for every dataset under `predictions_domain`.

    Crops with no matching groundtruth (missing labels, or no groundtruth
    scale level whose shape matches the prediction) are skipped.

    :return: {
        dataset_name: {
            'crop_accuracy': {crop_name: accuracy},
            'num_crops': int,               # crops actually scored
            'mean_accuracy': float,
            'above_threshold_ratio': float, # fraction of scored crops with accuracy > threshold
        },
        ...
    }
    """
    results = {}

    for dataset_dir in sorted(Path(predictions_domain).glob('*.zarr')):
        dataset = dataset_dir.name.removesuffix('.zarr')
        crop_names = sorted(p.name for p in dataset_dir.glob('crop*') if p.is_dir())

        crop_accuracy_map = {}
        for crop in crop_names:
            try:
                crop_accuracy_map[crop] = compute_crop_accuracy(
                    class_names, str(dataset_dir), origin_domain, bg_data_domain,
                    dataset, crop, pred_scale, gt_scale,
                )
            except (zarr.errors.PathNotFoundError, ShapeMismatchError) as e:
                if verbose:
                    print(f"  [skip] {dataset}/{crop}: {e}")

        if not crop_accuracy_map:
            continue

        accs = np.array(list(crop_accuracy_map.values()))
        results[dataset] = {
            'crop_accuracy': crop_accuracy_map,
            'num_crops': len(accs),
            'mean_accuracy': float(accs.mean()),
            'above_threshold_ratio': float((accs > threshold).mean()),
        }

        if verbose:
            r = results[dataset]
            print(
                f"{dataset}: {r['num_crops']} crops, "
                f"mean_accuracy={r['mean_accuracy']:.4f}, "
                f">{threshold:.0%} ratio={r['above_threshold_ratio']:.2%}"
            )

    return results