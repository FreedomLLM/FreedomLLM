import pytest
from model_loader import GPUMaker, ModelLoader


@pytest.mark.model_loader
def test_load_model(create_safetensors):
    file_path = str(create_safetensors)
    model_loader = ModelLoader(GPUMaker.NVIDIA)
    tensors = model_loader.load_model(file_path)
    assert len(tensors) == 3, f"Number of weights (expected: 3): {len(tensors)}"
    assert tensors["weight0"].shape == (
        1,
        1,
    ), f"Shape of weight0 (expected: (1, 1)): {tensors['weight0'].shape}"
    assert tensors["weight1"].shape == (
        2,
        2,
    ), f"Shape of weight1 (expected: (2, 2)): {tensors['weight1'].shape}"
    assert tensors["weight2"].shape == (
        3,
        3,
    ), f"Shape of weight2 (expected: (3, 3)): {tensors['weight2'].shape}"
