import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class Explainability:
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def __init__(self, name):
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_name(self):
        return self.name

    def filename2tensor(self, mosaic_path: str):
        img = Image.open(mosaic_path)
        img = img.convert('RGB')
        img_tensor = self.transformation(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        return img_tensor

    def explain(self, mosaic_filepath: str, target_class: int) -> np.ndarray:
        mosaic_tensor = self.filename2tensor(mosaic_filepath)
        mosaic_explanation = self.eval_image(mosaic_tensor, target_class)
        return mosaic_explanation

    def eval_image(self, img: torch.Tensor, target_class: int) -> np.ndarray:
        raise NotImplementedError

    def heatmap_visualization(self, heatmap: np.ndarray, img: torch.Tensor) -> np.ndarray:
        raise NotImplementedError
    