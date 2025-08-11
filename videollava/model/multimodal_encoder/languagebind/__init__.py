import torch
from torch import nn
from transformers import AutoConfig

from .image.configuration_image import LanguageBindImageConfig
from .image.modeling_image import LanguageBindImage
from .image.tokenization_image import LanguageBindImageTokenizer
from .image.processing_image import LanguageBindImageProcessor

from .video.configuration_video import LanguageBindVideoConfig
from .video.modeling_video import LanguageBindVideo
from .video.tokenization_video import LanguageBindVideoTokenizer
from .video.processing_video import LanguageBindVideoProcessor

from .depth.configuration_depth import LanguageBindDepthConfig
from .depth.modeling_depth import LanguageBindDepth
from .depth.tokenization_depth import LanguageBindDepthTokenizer
from .depth.processing_depth import LanguageBindDepthProcessor

from .audio.configuration_audio import LanguageBindAudioConfig
from .audio.modeling_audio import LanguageBindAudio
from .audio.tokenization_audio import LanguageBindAudioTokenizer
from .audio.processing_audio import LanguageBindAudioProcessor

from .thermal.configuration_thermal import LanguageBindThermalConfig
from .thermal.modeling_thermal import LanguageBindThermal
from .thermal.tokenization_thermal import LanguageBindThermalTokenizer
from .thermal.processing_thermal import LanguageBindThermalProcessor

import torch.nn as nn
from typing import Optional
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
from PIL import Image
import json
import os


config_dict = {
    'thermal': LanguageBindThermalConfig,
    'image': LanguageBindImageConfig,
    'video': LanguageBindVideoConfig,
    'depth': LanguageBindDepthConfig,
    'audio': LanguageBindAudioConfig
}
model_dict = {
    'thermal': LanguageBindImage,
    'image': LanguageBindImage,
    'video': LanguageBindVideo,
    'depth': LanguageBindImage,
    'audio': LanguageBindImage
}
transform_dict = {
    'video': LanguageBindVideoProcessor,
    'audio': LanguageBindAudioProcessor,
    'depth': LanguageBindDepthProcessor,
    'thermal': LanguageBindThermalProcessor,
    'image': LanguageBindImageProcessor,
}

class TemporalPooler(nn.Module):
    """
    Per spatial site, attend across time (T) -> 1 temporally-pooled token per site.
    Inputs:
      x:    [B, T, S, C]
      tmask:[B, T]  (1=valid, 0=pad). Optional if you don't pad T.
    Output:
      y:    [B, S, C]
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, tmask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, S, C = x.shape
        x = self.ln(x)
        # Process each spatial site independently: reshape to [B*S, T, C]
        x_bs = x.permute(0, 2, 1, 3).reshape(B * S, T, C)               # [B*S, T, C]
        q = x_bs.mean(1, keepdim=True)                                   # [B*S, 1, C] simple init query
        # key_padding_mask: True = ignore (so invert tmask)
        kpm = None
        if tmask is not None:
            kpm = (tmask == 0).unsqueeze(1).expand(B, S, T).reshape(B * S, T)  # [B*S, T]
        y, _ = self.attn(q, x_bs, x_bs, key_padding_mask=kpm)            # [B*S, 1, C]
        y = y.squeeze(1).reshape(B, S, C)                                 # [B, S, C]
        return y


class SpatialPooler(nn.Module):
    """
    Global spatial pooling with K learnable queries.
    Inputs:
      x: [B, S, C]  (already temporally pooled)
    Output:
      y: [B, K, C]
    """
    def __init__(self, hidden_dim: int, num_queries: int = 32, num_heads: int = 8):
        super().__init__()
        self.K = num_queries
        self.query = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.ln = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        x = self.ln(x)                              # [B, S, C]
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # [B, K, C]
        y, _ = self.attn(q, x, x)                  # [B, K, C]
        return y                              # [B, K, C]


class TwoStagePooler(nn.Module):
    """
    Temporal (per site) -> Spatial (global) pooling.
    Inputs:
      x:     [B, T, S, C]
      tmask: [B, T] (optional)
    Output:
      y:     [B, K, C]
    """
    def __init__(self, hidden_dim: int, num_queries: int = 32, num_heads: int = 8):
        super().__init__()
        self.temporal = TemporalPooler(hidden_dim, num_heads=num_heads)
        self.spatial  = SpatialPooler(hidden_dim, num_queries=num_queries, num_heads=num_heads)

    def forward(self, x: torch.Tensor, tmask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_t = self.temporal(x, tmask=tmask)   # [B, S, C]
        y   = self.spatial(x_t)               # [B, K, C]
        return y                              # [B, K, C]


class LanguageBind(nn.Module):
    def __init__(self, clip_type=('thermal', 'image', 'video', 'depth', 'audio'), use_temp=True, cache_dir='./cache_dir'):
        super(LanguageBind, self).__init__()
        self.use_temp = use_temp
        self.modality_encoder = {}
        self.modality_proj = {}
        self.modality_scale = {}
        self.modality_config = {}
        for c in clip_type:
            pretrained_ckpt = f'LanguageBind/LanguageBind_{c.capitalize()}'
            model = model_dict[c].from_pretrained(pretrained_ckpt, cache_dir=cache_dir)
            self.modality_encoder[c] = model.vision_model
            self.modality_proj[c] = model.visual_projection
            self.modality_scale[c] = model.logit_scale
            self.modality_config[c] = model.config
        
        # Only add language model if it exists on the model
        if hasattr(model, 'text_model') and hasattr(model, 'text_projection'):
            self.modality_encoder['language'] = model.text_model
            self.modality_proj['language'] = model.text_projection

        self.modality_encoder = nn.ModuleDict(self.modality_encoder)
        self.modality_proj = nn.ModuleDict(self.modality_proj)

    def forward(self, inputs):
        outputs = {}
        for key, value in inputs.items():
            value = self.modality_encoder[key](**value)[1]
            value = self.modality_proj[key](value)
            value = value / value.norm(p=2, dim=-1, keepdim=True)
            if self.use_temp:
                if key != 'language':
                    value = value * self.modality_scale[key].exp()
            outputs[key] = value
        return outputs

def to_device(x, device):
    out_dict = {k: v.to(device) for k, v in x.items()}
    return out_dict




class LanguageBindImageTower(nn.Module):
    def __init__(self, image_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.image_tower_name = image_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = LanguageBindImageConfig.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)

    ############################################################
    def load_model(self):
        model = LanguageBindImage.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)
        self.image_tower = model.vision_model
        self.image_tower.requires_grad_(False)

        self.image_processor = LanguageBindImageProcessor(model.config)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.image_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # print('images', images.shape)
            image_forward_outs = self.image_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            # print('image_forward_outs', len(image_forward_outs), image_forward_outs[0].shape)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            # print('image_features', image_features.shape)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.image_tower.embeddings.class_embedding.dtype  #############

    @property
    def device(self):
        return self.image_tower.embeddings.class_embedding.device  ##############

    @property
    def config(self):
        if self.is_loaded:
            return self.image_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class LanguageBindVideoTower(nn.Module):
    def __init__(self, video_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.video_tower_name = video_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.cache_dir = cache_dir
        
        # Initialize two-stage pooler for video features
        self.two_stage_pooler = None  # Will be initialized after loading the model

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = LanguageBindVideoConfig.from_pretrained(self.video_tower_name, cache_dir=self.cache_dir)

    ############################################################
    def load_model(self):
        model = LanguageBindVideo.from_pretrained(self.video_tower_name, cache_dir=self.cache_dir)
        self.video_processor = LanguageBindVideoProcessor(model.config)

        # model = LanguageBindImage.from_pretrained('LanguageBind/LanguageBind_Image', cache_dir=self.cache_dir)
        self.video_tower = model.vision_model
        self.video_tower.requires_grad_(False)
        
        # Initialize two-stage pooler with the model's hidden size
        hidden_dim = self.video_tower.config.hidden_size
        self.two_stage_pooler = TwoStagePooler(
            hidden_dim=hidden_dim,
            num_queries=32,  # Configurable number of output tokens
            num_heads=8
        )

        self.is_loaded = True


    def feature_select(self, video_forward_outs):
        video_features = video_forward_outs.hidden_states[self.select_layer]  # b t n c
        return video_features  # return all
        # b, t, n, c = video_features.shape
        # if self.select_feature == 'patch':
        #     video_features = video_features[:, :, 1:]
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
        # return video_features

    @torch.no_grad()
    def forward(self, videos):
        if type(videos) is list:
            video_features = []
            for video in videos:
                video_forward_out = self.video_tower(video.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                video_feature = self.feature_select(video_forward_out).to(video.dtype)
                
                # Apply two-stage pooling: Temporal -> Spatial
                if self.two_stage_pooler is not None:
                    # video_feature shape: [1, T, N, C] where T=32, N=256, C=1024
                    pooled_feature = self.two_stage_pooler(video_feature)
                    video_features.append(pooled_feature)
                else:
                    video_features.append(video_feature)
        else:
            video_forward_outs = self.video_tower(videos.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            video_features = self.feature_select(video_forward_outs).to(videos.dtype)
            
            # Apply two-stage pooling: Temporal -> Spatial
            if self.two_stage_pooler is not None:
                # video_features shape: [B, T, N, C] where T=32, N=256, C=1024
                video_features = self.two_stage_pooler(video_features)

        return video_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.video_tower.embeddings.class_embedding.dtype  #############
        # return torch.randn(1).cuda().dtype

    @property
    def device(self):
        return self.video_tower.embeddings.class_embedding.device  ##############
        # return torch.randn(1).cuda().device

    @property
    def config(self):
        if self.is_loaded:
            return self.video_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


