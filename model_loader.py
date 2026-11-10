import glob
import os
from enum import Enum, auto
import torch
from safetensors import safe_open


class GPUMaker(Enum):
    NVIDIA = auto()
    AMD = auto()
    INTEL = auto()

# TODO: Current load_model() method load every model weight into one GPU.
# Change algorithm so that it can efficiently distributed model weights to multiple GPUs
class ModelLoader:
    def __init__(self, gpu_maker: GPUMaker):
        self.gpu_maker = gpu_maker
        self.device = None

    def load_model(self, model_path: str):
        tensors: dict[str, torch.Tensor] = {}
        if self.gpu_maker == GPUMaker.NVIDIA or self.gpu_maker == GPUMaker.AMD:
            self.device = "cuda"
        elif self.gpu_maker == GPUMaker.INTEL:
            self.device = "xpu"

        search_pattern = os.path.join(model_path, "**", "*.safetensors")
        for safetensors_file in glob.glob(search_pattern, recursive=True):
            with safe_open(safetensors_file, framework="pt", device=self.device) as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
        return tensors
