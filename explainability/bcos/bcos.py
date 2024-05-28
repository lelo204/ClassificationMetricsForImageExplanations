import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from explainability.explainability_class import Explainability


# AddInverse is copied from https://github.com/B-cos/B-cos-v2/blob/main/bcos/data/transforms.py
class AddInverse(torch.nn.Module):
    """To a [B, C, H, W] input add the inverse channels of the given one to it.
    Results in a [B, 2C, H, W] output. Single image [C, H, W] is also accepted.

    Args:
        dim (int): where to add channels to. Default: -3
    """

    def __init__(self, dim: int = -3):
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([in_tensor, 1 - in_tensor], dim=self.dim)

class Bcos(Explainability):    
    def __init__(self, model):
        super().__init__('Bcos')
        self.model = model.to(self.device)

        self.bcos_transformation = transforms.Compose([
                                transforms.Resize(size=224),
                                transforms.ToTensor(),
                                transforms.ConvertImageDtype(torch.float),
                                AddInverse(),
                            ])

    def filename2tensor(self, mosaic_path: str):
        img = Image.open(mosaic_path)
        #img_tensor = self.model.transform(img)             # model.transform uses CenterCrop
        img_tensor = self.bcos_transformation(img)
        img_tensor = img_tensor[None]
        return img_tensor

    def explain(self, mosaic_path: str, target_class: int) -> np.ndarray:
        mosaic_tensor = self.filename2tensor(mosaic_path)
        mosaic_explanation = self.eval_image(mosaic_tensor, target_class)
        return mosaic_explanation

    def eval_image(self, img: torch.Tensor, target_class: int) -> np.ndarray:
        img = img.to(self.device)
        self.bcos_explanation = self.model.explain(img, target_class)
        bcos_feature_attrib = self.bcos_explanation['contribution_map'].squeeze().detach().cpu().numpy()
        return bcos_feature_attrib

    def heatmap_visualization(self, heatmap: np.ndarray = None, img: torch.Tensor = None):
        fig = plt.figure(figsize=(15, 15))
        fig.add_subplot(self.bcos_explanation['explanation'])
        plt.axis('off')

        return fig
