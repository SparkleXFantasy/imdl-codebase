import torch
from functools import partial

from .image_encoder_vit import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .transformer import TwoWayTransformer
from .vit_u import VITU

__all__ = [
    'build_model'
]
def build_model(
    encoder_embed_dim=768,
    encoder_depth=12,
    encoder_num_heads=12,
    encoder_global_attn_indexes=[2, 5, 8, 11],
    checkpoint=None,
):
    image_size = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    prompt_embed_dim = 256
    model = VITU(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        mask_decoder=MaskDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
        ),
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    return model