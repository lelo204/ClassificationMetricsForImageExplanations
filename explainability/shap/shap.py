import cv2
import torch
import numpy as np
from captum.attr import KernelShap

from explainability.explainability_class import Explainability


class SHAP(Explainability):
    def __init__(self, model):
        super().__init__('SHAP')
        self.model = model.to(self.device)
        self.shap = KernelShap(self.model)
        
    def _get_feature_mask(self, num_superpixs: int, img_size) -> torch.Tensor:
        """This function generates the feature mask needed to apply KernelShap from captum library.
        The input image is divided into num_superpixs superpixels, their size is determined automatically. 
        The superpixels will be blurred / masked for computing SHAP values (i.e. varying subsets of superpixels will be masked).

        Args:
            num_superpixs (int):            Total number of superpixels the image will be divided into.
            img_size (Tuple):               width and height of image represented as a tuple             

        Returns:
            feature_mask (torch.Tensor):    Feature mask that contains superpixel info and can be used to compute SHAP values using KernelShap.
        """
        width = img_size[0]
        height = img_size[1]
        superpixel_size = int(np.ceil(np.sqrt(width * height / num_superpixs)))

        # calculate number of superpixels in each row / column
        num_superpixels_x = int(np.ceil(width / superpixel_size))
        num_superpixels_y = int(np.ceil(height / superpixel_size))

        # create list that stores superpixel blocks used in the feature mask
        blocks = []

        # divide image into superpixels
        for i in range(num_superpixels_y):
            for j in range(num_superpixels_x):
                left = j * superpixel_size
                upper = i * superpixel_size
                right = min(left + superpixel_size, width)
                lower = min(upper + superpixel_size, height)

                # create array for superpixel blocks
                blocks.append((i*num_superpixels_x + j) * np.ones((right - left, lower - upper)))

        # arrange numpy array with superpixel blocks
        vertical_con = []
        for m in range(num_superpixels_y):
            horizontal_con = [blocks[m*num_superpixels_x + n] for n in range(num_superpixels_x)]
            horizontal_con = np.concatenate(horizontal_con, axis=0)
            vertical_con.append(horizontal_con)
        feature_mask = np.concatenate(vertical_con, axis=1)

        return torch.from_numpy(feature_mask)

    def eval_image(self, img: torch.Tensor, target_class: int) -> np.ndarray:
        img = img.to(self.device)
        feature_mask = self._get_feature_mask(16, img.shape[-2:]).to(self.device)
        attr = self.shap.attribute(img, target=target_class, feature_mask=feature_mask.long(), show_progress=True, n_samples=100)
        attr = np.array(attr.cpu())
        attr = attr[0].transpose(1,2,0)
        return attr[:,:,0]
        
    def heatmap_visualization(self, heatmap: np.ndarray, img: torch.Tensor) -> np.ndarray:
        img = np.array(img.cpu())[0, :, :, :]
        img = img.transpose(1,2,0)

        heatmap = ((heatmap - heatmap.min()) *(1 / (heatmap.max() - heatmap.min()) * 255)).astype('uint8') 
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255.0
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)
    