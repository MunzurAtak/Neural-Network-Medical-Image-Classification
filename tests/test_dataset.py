import os
import sys
import pytest
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset import HAM10000Dataset, get_transforms, get_class_weights


CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


@pytest.fixture
def dummy_data(tmp_path):
    image_dir = tmp_path / 'images'
    image_dir.mkdir()

    rows = []
    for i, cls in enumerate(CLASSES):
        for j in range(2):
            image_id = f'IMG_{i}_{j}'
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img.save(image_dir / f'{image_id}.jpg')
            rows.append({'image_id': image_id, 'dx': cls})

    df = pd.DataFrame(rows)
    return df, str(image_dir)


def test_len(dummy_data):
    df, image_dir = dummy_data
    dataset = HAM10000Dataset(df, [image_dir], get_transforms('val'))
    assert len(dataset) == len(df)


def test_getitem_returns_tensor_and_int(dummy_data):
    import torch
    df, image_dir = dummy_data
    dataset = HAM10000Dataset(df, [image_dir], get_transforms('val'))
    image, label = dataset[0]
    assert hasattr(image, 'shape'), 'image should be a tensor'
    assert isinstance(label, int)


def test_tensor_shape(dummy_data):
    df, image_dir = dummy_data
    dataset = HAM10000Dataset(df, [image_dir], get_transforms('val'))
    image, _ = dataset[0]
    assert image.shape == (3, 224, 224)


def test_class_weights_length(dummy_data):
    df, image_dir = dummy_data
    dataset = HAM10000Dataset(df, [image_dir], get_transforms('val'))
    weights = get_class_weights(dataset)
    assert len(weights) == len(CLASSES)


def test_train_val_transforms_differ(dummy_data):
    df, image_dir = dummy_data
    train_ds = HAM10000Dataset(df, [image_dir], get_transforms('train'))
    val_ds = HAM10000Dataset(df, [image_dir], get_transforms('val'))
    assert str(train_ds.transform) != str(val_ds.transform)
