import torch
from PIL import Image

from flowerclf.config import IMAGE_SIZE
from flowerclf.data import capped_indices, eval_transform


def test_capped_indices_limits_each_class():
    targets = [0] * 10 + [1] * 10 + [2] * 5
    generator = torch.Generator().manual_seed(0)
    selected = capped_indices(targets, max_per_class=4, generator=generator)
    counts = {label: 0 for label in (0, 1, 2)}
    for i in selected:
        counts[targets[i]] += 1
    assert counts == {0: 4, 1: 4, 2: 4}


def test_capped_indices_none_returns_all():
    generator = torch.Generator().manual_seed(0)
    assert sorted(capped_indices([0, 1, 2], None, generator)) == [0, 1, 2]


def test_eval_transform_returns_normalized_tensor():
    tensor = eval_transform()(Image.new("RGB", (200, 150), color="white"))
    assert tuple(tensor.shape) == (3, IMAGE_SIZE, IMAGE_SIZE)
