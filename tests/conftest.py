import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture()
def create_safetensors(tmp_path):
    tensors = {
        "weight0": torch.zeros(1, 1),
        "weight1": torch.ones(2, 2),
        "weight2": torch.rand(3, 3),
    }
    save_file(tensors, tmp_path / "mock.safetensors")
    return tmp_path
