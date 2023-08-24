from ViT.vits import create_vit
import torch

model, _ = create_vit('base', 224, False, 0, 0)
checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True)  # 从url中加载vit参数
state_dict = checkpoint["model"]
msg = model.load_state_dict(state_dict, strict=False)