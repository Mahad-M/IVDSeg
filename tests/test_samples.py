from pathlib import Path

import numpy as np

from ivdseg.manifest import build_manifest
from ivdseg.normalization import load_profile
from ivdseg.samples import (
    MINIMUM_COMPONENT_SIZE_VOXELS,
    channel_names,
    connected_components_3d,
    instance_targets_for_slice,
    make_2_5d_tensor,
    neighbour_indices,
    prepare_subject,
)


def test_tensor_channel_order_and_edge_replication_are_fixed() -> None:
    modalities = {
        "fat": np.arange(3, dtype=np.float32).reshape(3, 1, 1),
        "inn": np.arange(10, 13, dtype=np.float32).reshape(3, 1, 1),
        "opp": np.arange(20, 23, dtype=np.float32).reshape(3, 1, 1),
        "water": np.arange(30, 33, dtype=np.float32).reshape(3, 1, 1),
    }

    tensor = make_2_5d_tensor(modalities, slice_index=0)

    assert channel_names() == (
        "fat[-1]", "fat[+0]", "fat[+1]",
        "inn[-1]", "inn[+0]", "inn[+1]",
        "opp[-1]", "opp[+0]", "opp[+1]",
        "water[-1]", "water[+0]", "water[+1]",
    )
    np.testing.assert_array_equal(tensor[:, 0, 0], [0, 0, 1, 10, 10, 11, 20, 20, 21, 30, 30, 31])
    assert neighbour_indices(2, 3) == (1, 2, 2)
    assert tensor.dtype == np.float32


def test_26_connected_components_remove_small_specks_and_emit_2d_targets() -> None:
    label = np.zeros((4, 4, 4), dtype=bool)
    label[0, 0, 0] = True
    label[1, 1, 1] = True
    label[2, 2, 2] = True  # Diagonal adjacency is connected under 26-connectivity.
    label[0, 3, 3] = True  # Separate speck.

    components, sizes = connected_components_3d(label, minimum_size_voxels=3)
    targets = instance_targets_for_slice(components, 1)

    assert sizes == {1: 3}
    assert set(np.unique(components)) == {0, 1}
    assert len(targets) == 1
    assert targets[0].component_id == 1
    assert targets[0].bbox_xyxy == (1, 1, 2, 2)
    assert targets[0].class_id == 0


def test_real_subject_preparation_obeys_tensor_and_component_contracts() -> None:
    manifest = build_manifest(Path("IVDM3Seg"))
    profile = load_profile(Path("artifacts/normalization/ivdm3seg-training-pool-v1.json"))
    record = next(record for record in manifest["subjects"] if record["subject_id"] == "01")

    prepared = prepare_subject(record, Path("IVDM3Seg"), profile)
    tensor = prepared.tensor_for_slice(0)

    assert tensor.shape == (12, 256, 256)
    assert tensor.dtype == np.float32
    assert all(size >= MINIMUM_COMPONENT_SIZE_VOXELS for size in prepared.component_sizes.values())
    assert np.array_equal(tensor[0], tensor[1])
