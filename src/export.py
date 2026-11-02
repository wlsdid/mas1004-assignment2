"""Write a trained model as the ONNX file the grader runs, and run it back.

This file is complete. Both tracks use it: src/official.py calls it at the end
of a Track A run, and in Track B you call `export_onnx` yourself on the model
you want to hand in.

The file holds the whole network plus the normalisation in front of it, so it
takes the photo exactly as rules.grader_input gives it (values 0..1) and needs
nothing else to run.
"""

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rules import CITIES, MEAN, STD


class WithNormalisation(nn.Module):
    """The network with ImageNet normalisation in front of it."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, image):
        return self.net((image - self.mean) / self.std)


def export_onnx(net, path, size):
    """Write `net`, which takes normalised (N, 3, size, size) images, to `path`.

    Checks afterwards that the ONNX file gives the answers PyTorch gives.
    Returns the largest difference between the two.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = WithNormalisation(copy.deepcopy(net).float().cpu().eval()).eval()
    example = torch.rand(2, 3, size, size)
    with torch.no_grad():
        expected = model(example).numpy()
    if expected.shape != (2, len(CITIES)):
        raise ValueError(f"The model gives shape {expected.shape[1:]} for one photo, not ({len(CITIES)},).")
    torch.onnx.export(
        model, (example,), str(path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}},
        opset_version=17, dynamo=False,
    )
    gap = float(np.abs(run_onnx(path, example.numpy()) - expected).max())
    if gap > 1e-3:
        raise RuntimeError(f"{path.name} answers differently from PyTorch (largest gap {gap:.4f}).")
    return gap


def run_onnx(path, images, session=None):
    """Logits for a float32 batch (N, 3, S, S) from an ONNX file."""
    import onnxruntime
    if session is None:
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(["logits"], {"image": np.asarray(images, dtype=np.float32)})[0]
