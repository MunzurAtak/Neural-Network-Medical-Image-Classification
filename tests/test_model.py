import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import build_model, get_device


def test_output_shape():
    model = build_model(num_classes=7)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 7)


def test_frozen_has_fewer_trainable_params():
    frozen_model = build_model(num_classes=7, freeze_backbone=True)
    full_model = build_model(num_classes=7, freeze_backbone=False)

    frozen_trainable = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
    full_trainable = sum(p.numel() for p in full_model.parameters() if p.requires_grad)

    assert frozen_trainable < full_trainable


def test_get_device_returns_valid():
    device = get_device()
    assert device in ('cuda', 'cpu')
