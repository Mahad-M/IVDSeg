import torch

from ivdseg.augmentations import (
    AugmentationConfig,
    Synchronized2p5DAugmentation,
    boxes_from_masks,
    normalize_boxes_xyxy,
    resize_tensor_and_target,
    target_from_masks,
)


def _target_for_mask(mask: torch.Tensor) -> dict:
    return target_from_masks(
        mask.unsqueeze(0),
        labels=torch.tensor([0]),
        template={"image_id": torch.tensor([7])},
        original_size=tuple(mask.shape),
    )


def test_geometric_transform_is_shared_by_all_channels_and_instance_masks() -> None:
    torch.manual_seed(17)
    image = torch.zeros((12, 16, 16), dtype=torch.float32)
    image[:, 4:12, 5:11] = 1.0
    mask = torch.zeros((16, 16), dtype=torch.bool)
    mask[4:12, 5:11] = True
    augment = Synchronized2p5DAugmentation(
        AugmentationConfig(
            geometric_probability=1.0,
            max_rotation_degrees=7.0,
            max_translate_fraction=0.1,
            minimum_scale=0.95,
            maximum_scale=1.05,
            intensity_probability=0.0,
        )
    )

    augmented_image, augmented_target = augment(image, _target_for_mask(mask))
    boxes_xyxy, keep = boxes_from_masks(augmented_target["masks"])

    for channel_index in range(1, 12):
        torch.testing.assert_close(augmented_image[0], augmented_image[channel_index])
    assert keep.tolist() == [True]
    torch.testing.assert_close(
        augmented_target["boxes"],
        normalize_boxes_xyxy(boxes_xyxy, height=16, width=16),
    )
    assert augmented_target["masks"].dtype == torch.bool


def test_modality_intensity_transform_is_shared_within_each_three_slice_block() -> None:
    image = torch.arange(12, dtype=torch.float32).reshape(12, 1, 1)
    image[:, 0, 0] += 1.0
    image[0, 0, 0] = 0.0  # Background stays background even with an intensity bias.
    mask = torch.ones((1, 1), dtype=torch.bool)
    augment = Synchronized2p5DAugmentation(
        AugmentationConfig(
            geometric_probability=0.0,
            intensity_probability=1.0,
            minimum_intensity_gain=2.0,
            maximum_intensity_gain=2.0,
            maximum_intensity_bias=0.0,
        )
    )

    augmented_image, _ = augment(image, _target_for_mask(mask))

    torch.testing.assert_close(augmented_image[1:3], image[1:3] * 2.0)
    torch.testing.assert_close(augmented_image[3:6], image[3:6] * 2.0)
    torch.testing.assert_close(augmented_image[6:9], image[6:9] * 2.0)
    torch.testing.assert_close(augmented_image[9:12], image[9:12] * 2.0)
    assert augmented_image[0, 0, 0].item() == 0.0
    assert not augment.config.uses_anatomical_axis_flips


def test_resize_rebuilds_normalized_boxes_from_nearest_instance_masks() -> None:
    image = torch.zeros((12, 4, 8), dtype=torch.float32)
    mask = torch.zeros((4, 8), dtype=torch.bool)
    mask[1:3, 2:6] = True

    resized_image, resized_target = resize_tensor_and_target(image, _target_for_mask(mask), resolution=24)
    boxes_xyxy, keep = boxes_from_masks(resized_target["masks"])

    assert resized_image.shape == (12, 24, 24)
    assert keep.tolist() == [True]
    torch.testing.assert_close(
        resized_target["boxes"],
        normalize_boxes_xyxy(boxes_xyxy, height=24, width=24),
    )
    assert resized_target["orig_size"].tolist() == [4, 8]
    assert resized_target["size"].tolist() == [24, 24]
