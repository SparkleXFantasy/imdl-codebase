
from functools import partial
import torch
from dataset import IMLDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.LGMNet import LGMNet
from models.swin_transformer import SwinTransformer
model = SwinTransformer(
            pretrain_img_size=256,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1)
a = torch.zeros((2, 3, 256, 256))
o = model(a)
print(o[1].shape, o[2].shape, o[3].shape, o[0].shape)
# casia2 = IMLDataset('/home/aya/workspace/data/imdl/test/loc/manip_trad/CASIA2.0_revised')
# l = DataLoader(casia2, batch_size=1, shuffle=False)
# real_count = 0

# fake_count = 0

# for id, x in tqdm(enumerate(l)):
#     cls, img, mask = x
#     if cls.item() == 0:
#         real_count += 1
#     else:
#         fake_count += 1
# print(real_count, fake_count)


# print(len(casia2))
