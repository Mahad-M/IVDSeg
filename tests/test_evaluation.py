from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from ivdseg.evaluation import (
    InstanceAPAccumulator,
    compute_subject_slice_dice,
    compute_subject_metrics,
    reconstruct_canonical_prediction,
    select_fixed_test_records,
    write_native_prediction,
)
from ivdseg.spatial import canonicalize


def _result(scores: list[float], labels: list[int], masks: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "scores": torch.tensor(scores),
        "labels": torch.tensor(labels),
        "masks": torch.stack(masks).unsqueeze(1).bool(),
    }


def test_reconstruction_uses_the_locked_strict_score_threshold_and_unions_masks() -> None:
    prediction = reconstruct_canonical_prediction(
        slice_results=(
            _result(
                [0.35, 0.36, 0.99],
                [0, 0, 1],
                [
                    torch.tensor([[1, 0], [0, 0]]),
                    torch.tensor([[0, 1], [0, 0]]),
                    torch.tensor([[0, 0], [1, 1]]),
                ],
            ),
            _result([0.8], [0], [torch.tensor([[0, 0], [1, 0]])]),
        ),
        volume_shape=(2, 2, 2),
        score_threshold=0.35,
    )

    expected = np.array(
        [
            [[False, True], [False, False]],
            [[False, False], [True, False]],
        ]
    )
    np.testing.assert_array_equal(prediction, expected)


def test_subject_metrics_use_world_space_surface_and_centroid_distances() -> None:
    target = np.zeros((3, 3, 3), dtype=bool)
    prediction = np.zeros_like(target)
    target[0, 1, 1] = True
    prediction[1, 1, 1] = True
    affine = np.diag([2.0, 1.0, 1.0, 1.0])

    metrics = compute_subject_metrics("03", prediction, target, affine)

    assert metrics.dice == pytest.approx(0.0)
    assert metrics.asd_mm == pytest.approx(2.0)
    assert metrics.hd95_mm == pytest.approx(2.0)
    assert metrics.localization_distance_mm == pytest.approx(2.0)
    assert metrics.component_false_positives == 1
    assert metrics.component_false_negatives == 1


def test_slice_dice_separates_foreground_quality_from_empty_slice_inflation() -> None:
    target = np.zeros((3, 2, 2), dtype=bool)
    prediction = np.zeros_like(target)
    target[0, 0, 0] = True
    prediction[0, 0, 0] = True
    target[1, 0, 0] = True

    metrics = compute_subject_slice_dice("03", prediction, target)

    assert metrics.foreground_only_mean_dice == pytest.approx(0.5)
    assert metrics.all_slice_mean_dice == pytest.approx(2.0 / 3.0)
    assert metrics.foreground_slice_count == 2
    assert metrics.empty_target_slice_count == 1
    assert metrics.empty_target_true_negative_slice_count == 1


def test_native_prediction_restores_the_source_label_grid_and_affine(tmp_path: Path) -> None:
    raw_data = np.zeros((2, 3, 4), dtype=np.uint8)
    raw_data[0, 1, 2] = 1
    raw_affine = np.diag([-2.0, 1.5, 1.0, 1.0])
    raw_affine[0, 3] = 2.0
    raw_path = tmp_path / "label.nii.gz"
    output_path = tmp_path / "prediction.nii.gz"
    nib.save(nib.Nifti1Image(raw_data, raw_affine), raw_path)
    canonical_prediction = np.asarray(canonicalize(nib.load(raw_path)).dataobj, dtype=np.uint8).astype(bool)

    write_native_prediction(canonical_prediction, raw_label_path=raw_path, output_path=output_path)

    restored = nib.load(output_path)
    assert restored.shape == raw_data.shape
    np.testing.assert_allclose(restored.affine, raw_affine)
    np.testing.assert_array_equal(np.asarray(restored.dataobj), raw_data)
    assert set(np.unique(np.asarray(restored.dataobj))) <= {0, 1}


def test_fixed_test_record_selection_rejects_a_manifest_with_an_unexpected_test_split() -> None:
    manifest = {
        "subjects": [
            {"subject_id": "03", "partition": "test"},
            {"subject_id": "07", "partition": "test"},
            {"subject_id": "10", "partition": "test"},
            {"subject_id": "14", "partition": "train"},
        ]
    }

    with pytest.raises(ValueError, match="fixed test subjects"):
        select_fixed_test_records(manifest)


def test_instance_ap_accumulator_uses_global_score_order_and_greedy_mask_matching() -> None:
    accumulator = InstanceAPAccumulator(iou_thresholds=(0.5,))
    target_masks = torch.tensor(
        [
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
        ],
        dtype=torch.bool,
    )
    prediction_masks = torch.tensor(
        [
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],  # true positive, score 0.9
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],  # false positive, score 0.8
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],  # true positive, score 0.7
        ],
        dtype=torch.bool,
    )

    accumulator.add_image(
        scores=torch.tensor([0.9, 0.8, 0.7]),
        labels=torch.tensor([0, 0, 0]),
        masks=prediction_masks.unsqueeze(1),
        target_masks=target_masks,
    )
    summary = accumulator.finalize()

    assert summary.target_instance_count == 2
    assert summary.detection_count == 3
    assert summary.ap50 == pytest.approx(5.0 / 6.0)
    assert summary.map_50_95 == pytest.approx(5.0 / 6.0)
