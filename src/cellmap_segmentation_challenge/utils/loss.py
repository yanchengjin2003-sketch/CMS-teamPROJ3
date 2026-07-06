import torch
import torch.nn.functional as F


class CellMapLossWrapper(torch.nn.modules.loss._Loss):
    """
    Wrapper for any PyTorch loss function that is applied to the output of a model and the target.

    Because the target can contain NaN values, the loss function is applied only to the non-NaN values.
    This is done by multiplying the loss by a mask that is 1 where the target is not NaN and 0 where the target is NaN.
    The loss is then averaged across the non-NaN values.

    Parameters
    ----------
    loss_fn : torch.nn.modules.loss._Loss or torch.nn.modules.loss._WeightedLoss
        The loss function to apply to the output and target.
    **kwargs
        Keyword arguments to pass to the loss function.
    """

    def __init__(
        self,
        loss_fn: torch.nn.modules.loss._Loss | torch.nn.modules.loss._WeightedLoss,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        self.kwargs["reduction"] = "none"
        self.loss_fn = loss_fn(**self.kwargs)

    def calc_loss(self, outputs: torch.Tensor, target: torch.Tensor):
        loss = self.loss_fn(outputs, target.nan_to_num(0))
        loss = (loss * target.isnan().logical_not()).nanmean()
        return loss

    def forward(
        self,
        outputs: dict | torch.Tensor,
        targets: dict | torch.Tensor,
    ):
        if isinstance(targets, dict):
            loss = 0
            if isinstance(outputs, dict):
                for key, target in targets.items():
                    loss += self.calc_loss(outputs[key], target)
            else:
                # Assumes outputs is a list or tuple of tensors aligned with targets
                for i, target in enumerate(targets.values()):
                    loss += self.calc_loss(outputs[i], target)
            loss /= len(targets)
        else:
            loss = self.calc_loss(outputs, targets)  # type: ignore
        return loss


class CellMapCrossEntropyLoss(torch.nn.Module):
    """
    Multi-class cross entropy for CellMap-style one-channel-per-class targets.

    Converts a target tensor of shape (B, C, ...) into class indices of shape
    (B, ...) using argmax over the class dimension, then applies CE to raw
    logits of shape (B, C, ...).
    """

    def __init__(self, ignore_index: int = -100, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.kwargs = kwargs

    def _target_to_indices(self, targets: torch.Tensor) -> torch.Tensor:
        valid = targets.isnan().logical_not().any(dim=1)
        target_indices = targets.nan_to_num(0).argmax(dim=1).long()
        return target_indices.masked_fill(valid.logical_not(), self.ignore_index)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        target_indices = self._target_to_indices(targets)
        return F.cross_entropy(
            outputs,
            target_indices,
            ignore_index=self.ignore_index,
            **self.kwargs,
        )


class CellMapDynamicWeightedCrossEntropyLoss(CellMapCrossEntropyLoss):
    """Patch-wise inverse-frequency weighted cross-entropy.

    For each patch independently, an active class ``c`` receives weight
    ``N / (C_active * n_c)``, where ``N`` is the number of valid voxels,
    ``C_active`` is the number of classes present in the patch, and ``n_c`` is
    the class voxel count. Missing classes receive weight zero. Optional lower
    and upper bounds control majority-class downweighting and rare-class
    amplification.
    """

    def __init__(
        self,
        min_class_weight: float | None = None,
        max_class_weight: float | None = 25.0,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(ignore_index=ignore_index, **kwargs)
        if min_class_weight is not None and min_class_weight <= 0:
            raise ValueError("min_class_weight must be positive or None")
        if max_class_weight is not None and max_class_weight <= 0:
            raise ValueError("max_class_weight must be positive or None")
        if (
            min_class_weight is not None
            and max_class_weight is not None
            and min_class_weight > max_class_weight
        ):
            raise ValueError(
                "min_class_weight cannot be greater than max_class_weight"
            )
        self.min_class_weight = min_class_weight
        self.max_class_weight = max_class_weight

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        target_indices = self._target_to_indices(targets)
        voxel_losses = F.cross_entropy(
            outputs,
            target_indices,
            ignore_index=self.ignore_index,
            reduction="none",
            **self.kwargs,
        )

        patch_losses = []
        num_classes = outputs.shape[1]
        for patch_index in range(outputs.shape[0]):
            patch_targets = target_indices[patch_index]
            valid = patch_targets != self.ignore_index
            valid_count = valid.sum()
            if valid_count == 0:
                patch_losses.append(outputs[patch_index].sum() * 0.0)
                continue

            class_counts = torch.bincount(
                patch_targets[valid],
                minlength=num_classes,
            ).to(device=outputs.device, dtype=outputs.dtype)
            active = class_counts > 0
            active_count = active.sum().to(dtype=outputs.dtype)

            class_weights = torch.zeros_like(class_counts)
            class_weights[active] = valid_count.to(outputs.dtype) / (
                active_count * class_counts[active]
            )
            if self.min_class_weight is not None:
                class_weights[active] = class_weights[active].clamp(
                    min=self.min_class_weight
                )
            if self.max_class_weight is not None:
                class_weights[active] = class_weights[active].clamp(
                    max=self.max_class_weight
                )

            voxel_weights = class_weights[patch_targets[valid]]
            weighted_loss = voxel_losses[patch_index][valid] * voxel_weights
            patch_losses.append(weighted_loss.sum() / voxel_weights.sum())

        return torch.stack(patch_losses).mean()


class CellMapForegroundCEBackgroundRejectionLoss(torch.nn.Module):
    """Foreground CE with confidence rejection on a separate background mask.

    The model predicts ``C`` foreground classes while the target contains
    ``C + 1`` channels. The final target channel is a background mask:

    - foreground voxels use ordinary multi-class cross-entropy;
    - background voxels do not enter CE;
    - background voxels are penalized when the largest foreground softmax
      probability exceeds ``confidence_threshold``.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        background_penalty_weight: float = 1.0,
        penalty_power: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        if not 0.0 < confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if background_penalty_weight < 0.0:
            raise ValueError("background_penalty_weight must be non-negative")
        if penalty_power <= 0.0:
            raise ValueError("penalty_power must be positive")

        self.confidence_threshold = confidence_threshold
        self.background_penalty_weight = background_penalty_weight
        self.penalty_power = penalty_power
        self.ce_kwargs = kwargs

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        num_foreground_classes = outputs.shape[1]
        expected_target_channels = num_foreground_classes + 1
        if targets.shape[1] != expected_target_channels:
            raise ValueError(
                "Background-rejection loss expects one target channel per "
                f"foreground class plus one bg channel: expected "
                f"{expected_target_channels}, got {targets.shape[1]}."
            )

        foreground_targets = targets[:, :num_foreground_classes].nan_to_num(0)
        background_mask = targets[:, -1].nan_to_num(0) > 0.5
        foreground_mask = foreground_targets.sum(dim=1) > 0.5

        target_indices = foreground_targets.argmax(dim=1)
        if foreground_mask.any():
            voxel_ce = F.cross_entropy(
                outputs,
                target_indices,
                reduction="none",
                **self.ce_kwargs,
            )
            foreground_ce = voxel_ce[foreground_mask].mean()
        else:
            foreground_ce = outputs.sum() * 0.0

        if background_mask.any() and self.background_penalty_weight > 0.0:
            max_foreground_probability = F.softmax(outputs, dim=1).amax(dim=1)
            excess_confidence = F.relu(
                max_foreground_probability - self.confidence_threshold
            )
            background_penalty = (
                excess_confidence[background_mask].pow(self.penalty_power).mean()
            )
        else:
            background_penalty = outputs.sum() * 0.0

        return (
            foreground_ce
            + self.background_penalty_weight * background_penalty
        )


class CellMapDiceCELoss(CellMapCrossEntropyLoss):
    """Combined Dice + CE loss for mutually exclusive CellMap labels."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
        include_background: bool = True,
        class_weights: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(ignore_index=ignore_index, **kwargs)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        self.include_background = include_background
        self.class_weights = (
            None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        )

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        target_indices = self._target_to_indices(targets)
        weight = (
            None
            if self.class_weights is None
            else self.class_weights.to(device=outputs.device, dtype=outputs.dtype)
        )
        ce_loss = F.cross_entropy(
            outputs,
            target_indices,
            weight=weight,
            ignore_index=self.ignore_index,
            **self.kwargs,
        )

        valid = target_indices != self.ignore_index
        safe_target = target_indices.masked_fill(valid.logical_not(), 0)
        target_one_hot = F.one_hot(
            safe_target, num_classes=outputs.shape[1]
        ).movedim(-1, 1)
        target_one_hot = target_one_hot.to(dtype=outputs.dtype)
        probs = F.softmax(outputs, dim=1)

        valid = valid.unsqueeze(1)
        probs = probs * valid
        target_one_hot = target_one_hot * valid

        if not self.include_background and outputs.shape[1] > 1:
            probs = probs[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

        reduce_dims = tuple(range(2, outputs.ndim))
        intersection = (probs * target_one_hot).sum(dim=reduce_dims)
        denominator = probs.sum(dim=reduce_dims) + target_one_hot.sum(dim=reduce_dims)
        dice_score = (2 * intersection + self.dice_smooth) / (
            denominator + self.dice_smooth
        )
        dice_loss = 1 - dice_score.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class CellMapAdjacencyPriorDiceCELoss(CellMapDiceCELoss):
    """Dice + CE with soft biologically-informed prior losses.

    ``adjacency_pairs`` is a list of ``(lumen_class, membrane_class)`` pairs.
    For each pair, the prior penalizes lumen probability next to
    classes that are neither that lumen nor its paired membrane. The prior uses
    all 26 neighbours in 3D and is normalized by the number of checked neighbour
    slots, so the raw adjacency prior is clamped to ``[0, 1]``.

    The optional size prior penalizes predictions where selected classes, such
    as ``ecs`` and ``bg``, occupy more than ``size_ratio_threshold`` of the
    patch. The size prior is also normalized to ``[0, 1]`` and uses a squared
    excess ratio so that small threshold violations are penalized gently.
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        dice_smooth: float = 0.4,
        include_background: bool = True,
        adjacency_pairs: list[tuple[str, str]]
        | tuple[tuple[str, str], ...]
        | None = None,
        prior_weight: float = 0.1,
        adjacency_weight: float = 0.8,
        size_weight: float = 0.2,
        size_class_names: list[str] | tuple[str, ...] | None = None,
        size_ratio_threshold: float = 0.65,
        class_names: list[str] | tuple[str, ...] | None = None,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            dice_smooth=dice_smooth,
            include_background=include_background,
            ignore_index=ignore_index,
            **kwargs,
        )
        self.class_names = None if class_names is None else tuple(class_names)
        self.adjacency_pairs = self._parse_adjacency_pairs(adjacency_pairs)
        if prior_weight < 0:
            raise ValueError("prior_weight must be non-negative")
        if adjacency_weight < 0 or size_weight < 0:
            raise ValueError("adjacency_weight and size_weight must be non-negative")
        if adjacency_weight + size_weight > 1.0:
            raise ValueError(
                "adjacency_weight + size_weight must be <= 1 to keep the "
                "combined prior term in the 0-1 range."
            )
        if not 0.0 <= size_ratio_threshold < 1.0:
            raise ValueError("size_ratio_threshold must be in [0, 1)")

        self.prior_weight = float(prior_weight)
        self.adjacency_weight = float(adjacency_weight)
        self.size_weight = float(size_weight)
        self.size_ratio_threshold = float(size_ratio_threshold)
        self.size_class_indices = self._parse_size_class_names(size_class_names)
        self._offsets = tuple(
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dz, dy, dx) != (0, 0, 0)
        )

    def _parse_adjacency_pairs(
        self,
        adjacency_pairs: list[tuple[str, str]]
        | tuple[tuple[str, str], ...]
        | None,
    ) -> tuple[tuple[int, int], ...]:
        if not adjacency_pairs:
            return ()
        if self.class_names is None:
            raise ValueError("class_names is required when adjacency_pairs use names")

        name_to_index = {name: index for index, name in enumerate(self.class_names)}
        parsed_pairs = []
        for lumen_name, membrane_name in adjacency_pairs:
            if lumen_name not in name_to_index:
                raise ValueError(f"Unknown adjacency lumen class: {lumen_name!r}")
            if membrane_name not in name_to_index:
                raise ValueError(f"Unknown adjacency membrane class: {membrane_name!r}")
            lumen_index = name_to_index[lumen_name]
            membrane_index = name_to_index[membrane_name]
            if lumen_index == membrane_index:
                raise ValueError("Adjacency lumen and membrane classes must differ")
            parsed_pairs.append((lumen_index, membrane_index))
        return tuple(parsed_pairs)

    def _parse_size_class_names(
        self,
        size_class_names: list[str] | tuple[str, ...] | None,
    ) -> tuple[int, ...]:
        if not size_class_names:
            return ()
        if self.class_names is None:
            raise ValueError("class_names is required when size_class_names use names")

        name_to_index = {name: index for index, name in enumerate(self.class_names)}
        parsed_indices = []
        for class_name in size_class_names:
            if class_name not in name_to_index:
                raise ValueError(f"Unknown size-prior class: {class_name!r}")
            parsed_indices.append(name_to_index[class_name])
        return tuple(parsed_indices)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        base_loss = super().forward(outputs, targets)
        if self.prior_weight == 0:
            return base_loss

        prior_loss = outputs.new_zeros(())
        if self.adjacency_pairs and self.adjacency_weight > 0:
            prior_loss = prior_loss + (
                self.adjacency_weight * self._adjacency_prior_loss(outputs)
            )
        if self.size_class_indices and self.size_weight > 0:
            prior_loss = prior_loss + self.size_weight * self._size_prior_loss(outputs)

        return base_loss + self.prior_weight * prior_loss.clamp(min=0.0, max=1.0)

    def _adjacency_prior_loss(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.ndim != 5:
            raise ValueError(
                f"Adjacency prior expects outputs [B, C, D, H, W], got {outputs.shape}"
            )

        probs = F.softmax(outputs, dim=1)
        topology_loss = outputs.new_zeros(())
        for lumen_index, membrane_index in self.adjacency_pairs:
            pair_loss = self._pair_adjacency_violation(
                probs[:, lumen_index],
                probs[:, membrane_index],
            )
            topology_loss = topology_loss + pair_loss
        topology_loss = topology_loss / len(self.adjacency_pairs)
        return topology_loss.clamp(min=0.0, max=1.0)

    def _size_prior_loss(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.ndim != 5:
            raise ValueError(
                f"Size prior expects outputs [B, C, D, H, W], got {outputs.shape}"
            )

        probs = F.softmax(outputs, dim=1)
        selected_prob = probs[:, self.size_class_indices].sum(dim=1)
        selected_ratio = selected_prob.mean(dim=tuple(range(1, selected_prob.ndim)))
        excess_ratio = F.relu(selected_ratio - self.size_ratio_threshold)
        normalized_excess = excess_ratio / (1.0 - self.size_ratio_threshold)
        return normalized_excess.pow(2).mean().clamp(min=0.0, max=1.0)

    def _pair_adjacency_violation(
        self,
        lumen_prob: torch.Tensor,
        membrane_prob: torch.Tensor,
    ) -> torch.Tensor:
        forbidden_prob = (1.0 - lumen_prob - membrane_prob).clamp(min=0.0)
        violation = lumen_prob.new_zeros(())
        slot_count = 0

        for dz, dy, dx in self._offsets:
            source, neighbour = self._shift_pair(
                lumen_prob,
                forbidden_prob,
                dz,
                dy,
                dx,
            )
            violation = violation + (source * neighbour).sum()
            slot_count += source.numel()

        if slot_count == 0:
            return lumen_prob.new_zeros(())
        return violation / slot_count

    @staticmethod
    def _shift_pair(
        source_volume: torch.Tensor,
        neighbour_volume: torch.Tensor,
        dz: int,
        dy: int,
        dx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, depth, height, width = source_volume.shape

        source_z0, source_z1 = max(0, -dz), depth - max(0, dz)
        source_y0, source_y1 = max(0, -dy), height - max(0, dy)
        source_x0, source_x1 = max(0, -dx), width - max(0, dx)

        neighbour_z0, neighbour_z1 = max(0, dz), depth - max(0, -dz)
        neighbour_y0, neighbour_y1 = max(0, dy), height - max(0, -dy)
        neighbour_x0, neighbour_x1 = max(0, dx), width - max(0, -dx)

        source = source_volume[
            :,
            source_z0:source_z1,
            source_y0:source_y1,
            source_x0:source_x1,
        ]
        neighbour = neighbour_volume[
            :,
            neighbour_z0:neighbour_z1,
            neighbour_y0:neighbour_y1,
            neighbour_x0:neighbour_x1,
        ]
        return source, neighbour


class CellMapFilteredDynamicWeightedDiceCELoss(CellMapCrossEntropyLoss):
    """Patch-filtered dynamic inverse-frequency CE + Dice loss.

    This loss is for mutually exclusive CellMap labels stored as one binary
    channel per class. It can ignore whole patches when selected classes occupy
    too much of the patch, then trains on the remaining patches with:

    ``ce_weight * dynamic_inverse_frequency_CE + dice_weight * Dice``.
    """

    def __init__(
        self,
        ce_weight: float = 0.4,
        dice_weight: float = 0.6,
        dice_smooth: float = 1.0,
        min_class_weight: float | None = None,
        max_class_weight: float | None = 100.0,
        filter_class_indices: list[int] | tuple[int, ...] | None = None,
        filter_ratio_threshold: float | None = None,
        include_background: bool = True,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(ignore_index=ignore_index, **kwargs)
        if ce_weight < 0 or dice_weight < 0:
            raise ValueError("ce_weight and dice_weight must be non-negative")
        if min_class_weight is not None and min_class_weight <= 0:
            raise ValueError("min_class_weight must be positive or None")
        if max_class_weight is not None and max_class_weight <= 0:
            raise ValueError("max_class_weight must be positive or None")
        if (
            min_class_weight is not None
            and max_class_weight is not None
            and min_class_weight > max_class_weight
        ):
            raise ValueError(
                "min_class_weight cannot be greater than max_class_weight"
            )
        if filter_ratio_threshold is not None and not 0 <= filter_ratio_threshold <= 1:
            raise ValueError("filter_ratio_threshold must be between 0 and 1")

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        self.min_class_weight = min_class_weight
        self.max_class_weight = max_class_weight
        self.filter_class_indices = (
            None if filter_class_indices is None else tuple(filter_class_indices)
        )
        self.filter_ratio_threshold = filter_ratio_threshold
        self.include_background = include_background

    def _patch_is_kept(self, patch_targets: torch.Tensor, valid: torch.Tensor) -> bool:
        if self.filter_class_indices is None or self.filter_ratio_threshold is None:
            return True

        valid_count = valid.sum()
        if valid_count == 0:
            return False

        filter_mask = torch.zeros_like(valid)
        for class_index in self.filter_class_indices:
            filter_mask |= patch_targets == class_index
        filter_ratio = (filter_mask & valid).sum().to(torch.float32) / valid_count
        return bool(filter_ratio <= self.filter_ratio_threshold)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        target_indices = self._target_to_indices(targets)
        voxel_losses = F.cross_entropy(
            outputs,
            target_indices,
            ignore_index=self.ignore_index,
            reduction="none",
            **self.kwargs,
        )
        probabilities = F.softmax(outputs, dim=1)
        num_classes = outputs.shape[1]

        patch_losses = []
        for patch_index in range(outputs.shape[0]):
            patch_targets = target_indices[patch_index]
            valid = patch_targets != self.ignore_index
            if not self._patch_is_kept(patch_targets, valid):
                continue

            valid_count = valid.sum()
            if valid_count == 0:
                continue

            class_counts = torch.bincount(
                patch_targets[valid],
                minlength=num_classes,
            ).to(device=outputs.device, dtype=outputs.dtype)
            active = class_counts > 0
            active_count = active.sum().to(dtype=outputs.dtype)

            class_weights = torch.zeros_like(class_counts)
            class_weights[active] = valid_count.to(outputs.dtype) / (
                active_count * class_counts[active]
            )
            if self.min_class_weight is not None:
                class_weights[active] = class_weights[active].clamp(
                    min=self.min_class_weight
                )
            if self.max_class_weight is not None:
                class_weights[active] = class_weights[active].clamp(
                    max=self.max_class_weight
                )
            active_weight_mean = class_weights[active].mean()
            if active_weight_mean > 0:
                class_weights[active] = class_weights[active] / active_weight_mean

            voxel_weights = class_weights[patch_targets[valid]]
            ce_loss = (
                voxel_losses[patch_index][valid] * voxel_weights
            ).sum() / voxel_weights.sum()

            safe_target = patch_targets.masked_fill(valid.logical_not(), 0)
            target_one_hot = F.one_hot(
                safe_target,
                num_classes=num_classes,
            ).movedim(-1, 0)
            target_one_hot = target_one_hot.to(dtype=outputs.dtype)
            patch_probs = probabilities[patch_index]
            valid_for_dice = valid.unsqueeze(0)
            patch_probs = patch_probs * valid_for_dice
            target_one_hot = target_one_hot * valid_for_dice

            if not self.include_background and num_classes > 1:
                patch_probs = patch_probs[:-1]
                target_one_hot = target_one_hot[:-1]

            reduce_dims = tuple(range(1, patch_probs.ndim))
            intersection = (patch_probs * target_one_hot).sum(dim=reduce_dims)
            denominator = patch_probs.sum(dim=reduce_dims) + target_one_hot.sum(
                dim=reduce_dims
            )
            dice_score = (2 * intersection + self.dice_smooth) / (
                denominator + self.dice_smooth
            )
            dice_loss = 1 - dice_score.mean()

            patch_losses.append(self.ce_weight * ce_loss + self.dice_weight * dice_loss)

        if not patch_losses:
            return outputs.sum() * 0.0
        return torch.stack(patch_losses).mean()


class CellMapWeightedDiceIoUCELoss(CellMapCrossEntropyLoss):
    """Weighted CE + per-class Dice + per-class IoU for CellMap labels.

    Targets are expected as one channel per class, while outputs are raw
    multi-class logits. CE uses either explicit ``class_weights`` or dynamic
    inverse-frequency weights computed independently for each patch. Dice and
    IoU are computed per class and then averaged, so each included class gets
    one vote regardless of its voxel volume.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 0.0,
        iou_weight: float = 0.0,
        dice_smooth: float = 1.0,
        iou_smooth: float = 1.0,
        min_class_weight: float | None = None,
        max_class_weight: float | None = 10.0,
        normalize_class_weights: bool = True,
        include_background: bool = True,
        class_weights: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(ignore_index=ignore_index, **kwargs)
        if ce_weight < 0 or dice_weight < 0 or iou_weight < 0:
            raise ValueError("ce_weight, dice_weight, and iou_weight must be non-negative")
        if min_class_weight is not None and min_class_weight <= 0:
            raise ValueError("min_class_weight must be positive or None")
        if max_class_weight is not None and max_class_weight <= 0:
            raise ValueError("max_class_weight must be positive or None")
        if (
            min_class_weight is not None
            and max_class_weight is not None
            and min_class_weight > max_class_weight
        ):
            raise ValueError(
                "min_class_weight cannot be greater than max_class_weight"
            )

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.dice_smooth = dice_smooth
        self.iou_smooth = iou_smooth
        self.min_class_weight = min_class_weight
        self.max_class_weight = max_class_weight
        self.normalize_class_weights = normalize_class_weights
        self.include_background = include_background
        self.class_weights = (
            None
            if class_weights is None
            else torch.as_tensor(class_weights, dtype=torch.float32)
        )

    def _dynamic_class_weights(
        self,
        patch_targets: torch.Tensor,
        valid: torch.Tensor,
        num_classes: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        valid_count = valid.sum()
        class_counts = torch.bincount(
            patch_targets[valid],
            minlength=num_classes,
        ).to(device=patch_targets.device, dtype=dtype)
        active = class_counts > 0
        active_count = active.sum().to(dtype=dtype)

        class_weights = torch.zeros_like(class_counts)
        class_weights[active] = valid_count.to(dtype) / (
            active_count * class_counts[active]
        )
        if self.min_class_weight is not None:
            class_weights[active] = class_weights[active].clamp(
                min=self.min_class_weight
            )
        if self.max_class_weight is not None:
            class_weights[active] = class_weights[active].clamp(
                max=self.max_class_weight
            )
        if self.normalize_class_weights and active.any():
            active_weight_mean = class_weights[active].mean()
            if active_weight_mean > 0:
                class_weights[active] = class_weights[active] / active_weight_mean
        return class_weights

    def _configured_class_weights(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_classes: int,
    ) -> torch.Tensor:
        if self.class_weights is None:
            raise ValueError("class_weights are not configured")
        class_weights = self.class_weights.to(device=device, dtype=dtype)
        if class_weights.numel() != num_classes:
            raise ValueError(
                f"class_weights must contain {num_classes} values, "
                f"but got {class_weights.numel()}."
            )
        if self.min_class_weight is not None:
            class_weights = class_weights.clamp(min=self.min_class_weight)
        if self.max_class_weight is not None:
            class_weights = class_weights.clamp(max=self.max_class_weight)
        if self.normalize_class_weights:
            class_weights = class_weights / class_weights.mean()
        return class_weights

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        target_indices = self._target_to_indices(targets)
        voxel_losses = F.cross_entropy(
            outputs,
            target_indices,
            ignore_index=self.ignore_index,
            reduction="none",
            **self.kwargs,
        )
        probabilities = F.softmax(outputs, dim=1)
        num_classes = outputs.shape[1]

        configured_weights = None
        if self.class_weights is not None:
            configured_weights = self._configured_class_weights(
                outputs.device,
                outputs.dtype,
                num_classes,
            )

        patch_losses = []
        for patch_index in range(outputs.shape[0]):
            patch_targets = target_indices[patch_index]
            valid = patch_targets != self.ignore_index
            valid_count = valid.sum()
            if valid_count == 0:
                patch_losses.append(outputs[patch_index].sum() * 0.0)
                continue

            if configured_weights is None:
                class_weights = self._dynamic_class_weights(
                    patch_targets,
                    valid,
                    num_classes,
                    outputs.dtype,
                )
            else:
                class_weights = configured_weights

            voxel_weights = class_weights[patch_targets[valid]]
            ce_loss = (
                voxel_losses[patch_index][valid] * voxel_weights
            ).sum() / voxel_weights.sum().clamp_min(torch.finfo(outputs.dtype).eps)

            safe_target = patch_targets.masked_fill(valid.logical_not(), 0)
            target_one_hot = F.one_hot(
                safe_target,
                num_classes=num_classes,
            ).movedim(-1, 0)
            target_one_hot = target_one_hot.to(dtype=outputs.dtype)

            patch_probs = probabilities[patch_index]
            valid_for_region_losses = valid.unsqueeze(0)
            patch_probs = patch_probs * valid_for_region_losses
            target_one_hot = target_one_hot * valid_for_region_losses

            if not self.include_background and num_classes > 1:
                patch_probs = patch_probs[:-1]
                target_one_hot = target_one_hot[:-1]

            reduce_dims = tuple(range(1, patch_probs.ndim))
            intersection = (patch_probs * target_one_hot).sum(dim=reduce_dims)
            pred_volume = patch_probs.sum(dim=reduce_dims)
            target_volume = target_one_hot.sum(dim=reduce_dims)

            dice_denominator = pred_volume + target_volume
            dice_score = (2 * intersection + self.dice_smooth) / (
                dice_denominator + self.dice_smooth
            )
            dice_loss = 1 - dice_score.mean()

            union = pred_volume + target_volume - intersection
            iou_score = (intersection + self.iou_smooth) / (
                union + self.iou_smooth
            )
            iou_loss = 1 - iou_score.mean()

            patch_losses.append(
                self.ce_weight * ce_loss
                + self.dice_weight * dice_loss
                + self.iou_weight * iou_loss
            )

        return torch.stack(patch_losses).mean()


class CellMapFocalDiceLoss(CellMapCrossEntropyLoss):
    """Combined focal + Dice loss for imbalanced mutually exclusive labels."""

    def __init__(
        self,
        alpha: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        gamma: float = 1.5,
        focal_weight: float = 0.75,
        dice_weight: float = 0.25,
        dice_smooth: float = 1.0,
        include_background: bool = True,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(ignore_index=ignore_index, **kwargs)
        self.alpha = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        self.include_background = include_background

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        target_indices = self._target_to_indices(targets)
        valid = target_indices != self.ignore_index
        safe_target = target_indices.masked_fill(valid.logical_not(), 0)

        ce_loss = F.cross_entropy(
            outputs,
            target_indices,
            ignore_index=self.ignore_index,
            reduction="none",
            **self.kwargs,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt).pow(self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(device=outputs.device, dtype=outputs.dtype)
            alpha_t = alpha[safe_target]
            focal_loss = alpha_t * focal_loss

        focal_loss = focal_loss[valid].mean()

        target_one_hot = F.one_hot(
            safe_target, num_classes=outputs.shape[1]
        ).movedim(-1, 1)
        target_one_hot = target_one_hot.to(dtype=outputs.dtype)
        probs = F.softmax(outputs, dim=1)

        valid = valid.unsqueeze(1)
        probs = probs * valid
        target_one_hot = target_one_hot * valid

        if not self.include_background and outputs.shape[1] > 1:
            probs = probs[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

        reduce_dims = tuple(range(2, outputs.ndim))
        intersection = (probs * target_one_hot).sum(dim=reduce_dims)
        denominator = probs.sum(dim=reduce_dims) + target_one_hot.sum(dim=reduce_dims)
        dice_score = (2 * intersection + self.dice_smooth) / (
            denominator + self.dice_smooth
        )
        dice_loss = 1 - dice_score.mean()

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
