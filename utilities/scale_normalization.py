"""
Adapted from NiftyNet
"""

from typing import Union, Optional, Callable
from torchio import DATA, TypeCallable
from torchio.transforms.preprocessing.intensity.normalization_transform import NormalizationTransform
import torch
import numpy as np

class ScaleNormalization(NormalizationTransform):
    def __init__(
            self,
            masking_method: Optional[Union[str, TypeCallable]] = None,
            verbose: bool = False,
            ):
        super().__init__(masking_method=masking_method, verbose=verbose)

    def apply_normalization(
            self,
            sample: dict,
            image_name: str,
            mask: torch.Tensor,
            ) -> None:

        image_dict = sample[image_name]
        image_dict[DATA] = self.scalenorm(
            image_dict[DATA],
            mask,
        )


    @staticmethod
    def scalenorm(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        data = tensor.numpy()
        thre = np.percentile(data,99.5)
        data[data > thre] = thre
        thre_min = data[data > 0].min()
        data[data==0] = thre_min
        data -= thre_min
        data /= (thre-thre_min)/2
        data -= 1
        return  torch.from_numpy(data)



