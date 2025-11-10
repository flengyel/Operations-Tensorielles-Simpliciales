"""Smoke tests for the SimplicialTensors package."""

import importlib


def test_package_importable() -> None:
    pkg = importlib.import_module("simplicial_tensors")
    assert set(pkg.__all__) == {"tensor_ops", "symbolic_tensor_ops"}
