import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder_vit import ImageEncoderViT
from .mask_decoder import MaskDecoder

class VITU(nn.Module):
    def __init__(self, image_encoder, mask_decoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
    
    def forward(self, x) -> List[Dict[str, torch.Tensor]]:
    
        image_embeddings = self.image_encoder(x)
        
        det_pred, mask = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.image_encoder.get_dense_pe(),
            # prompt_embeddings=prompt_embeddings
        )

        return det_pred, mask