import torch
from sat.model.official import ChatGLMModel
from sat.model.base_model import BaseMixin
from copy import deepcopy
import json
from .blip2 import BLIP2
import torch.nn as nn
from ViT.vits import create_vit

from sat.resources.urls import MODEL_URLS
MODEL_URLS['visualglm-6b'] = 'r2://visualglm-6b.zip'

class ViT_12(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, _ = create_vit('base', 224, False, 0, 0)
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True)  # 从url中加载vit参数
        state_dict = checkpoint["model"]
        msg = self.model.load_state_dict(state_dict, strict=False)
        self.proj = nn.Linear(768, 1408)

    def forward(self, img):
        outputs = self.proj(self.model(img))
        return outputs
class ImageMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        self.args = deepcopy(args)
        if hasattr(args, 'model_parallel_size'):
            args.eva_args['model_parallel_size'] = args.model_parallel_size
            args.qformer_args['model_parallel_size'] = args.model_parallel_size
        self.model = BLIP2(args.eva_args, args.qformer_args)
        # self.model = BLIP2(eva_args=args.eva_args, qformer_args=args.qformer_args, vit=ViT_12())

    def word_embedding_forward(self, input_ids, output_cross_layer, **kw_args):
        # if kw_args["pre_image"] > input_ids.shape[1] or kw_args.get("image", None) is None:
        #     return self.transformer.word_embeddings(input_ids)
        if kw_args["pre_image"] > input_ids.shape[1]:
            return self.transformer.word_embeddings(input_ids)
        # image_emb = self.model(**kw_args)
        img0_emb = self.model(image=kw_args['img0'], **kw_args)
        img1_emb = self.model(image=kw_args['img1'], **kw_args)
        # the image is inserted after 问：<img>, override 32 pads
        pre_id, pads, post_id = torch.tensor_split(input_ids, [kw_args["pre_image"], kw_args["pre_image"]+self.args.image_length], dim=1)
        pre_txt_emb = self.transformer.word_embeddings(pre_id)
        post_txt_emb = self.transformer.word_embeddings(post_id)
        # return torch.cat([pre_txt_emb, image_emb, post_txt_emb], dim=1)
        return torch.cat([pre_txt_emb, img0_emb, img1_emb, post_txt_emb], dim=1)

class VisualGLMModel(ChatGLMModel):
    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer=transformer, **kwargs)
        self.image_length = args.image_length
        self.add_mixin("eva", ImageMixin(args))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('VisualGLM', 'VisualGLM Configurations')
        group.add_argument('--image_length', type=int, default=32)
        group.add_argument('--eva_args', type=json.loads, default={})
        group.add_argument('--qformer_args', type=json.loads, default={})
        return super().add_model_specific_args(parser)
    
