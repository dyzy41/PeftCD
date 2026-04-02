import torch
from change_detection.SAM2CD.sam2.build_sam import build_sam2
import os
from change_detection.utils.decode_block import *


model_cfg = 'sam2_hiera_l'
checkpoint_path = os.path.join(os.getenv('PRETRAIN'), f'sam2/sam2_hiera_large.pt')
if checkpoint_path:
    model = build_sam2(model_cfg, checkpoint_path)

del model.sam_mask_decoder
del model.sam_prompt_encoder
del model.memory_encoder
del model.memory_attention
del model.mask_downsample
del model.obj_ptr_tpos_proj
del model.obj_ptr_proj
del model.image_encoder.neck
encoder = model.image_encoder.trunk


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
encoder.eval()

print(model)


x = torch.randn(2, 3, 256, 256).to(device)

out = encoder(x)

for v in out:
    print(v.shape)