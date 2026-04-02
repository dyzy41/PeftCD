import timm
import os
import torch
from ..vmamba_cd.vmamba import vanilla_vmamba_small, vmamba_tiny_s1l8



def build_backbone(model_name='resnet'):
    if model_name=='resnet50':
        try:
            model = timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'resnet50.a1_in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model(model_name, pretrained=True, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [256, 512, 1024, 2048], 'out_channels': 256, 'num_outs': 4}
        num_stages = 5

    elif model_name=='mobilenet':
        try:
            model = timm.create_model('mobilenetv4_conv_small_050', pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'mobilenetv4_conv_small_050.e3000_r224_in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model('mobilenetv4_conv_small_050', pretrained=True, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [16, 32, 48, 480], 'out_channels': 256, 'num_outs': 4}
        num_stages = 5

    elif model_name=='convnext':
        try:
            model = timm.create_model('convnext_base.clip_laion2b_augreg_ft_in12k_in1k', pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'convnext_base.clip_laion2b_augreg_ft_in12k_in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model('convnext_base.clip_laion2b_augreg_ft_in12k_in1k', pretrained=True, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [128, 256, 512, 1024], 'out_channels': 256, 'num_outs': 4}
        num_stages = 4

    elif model_name=='mamba':
        try:
            model = timm.create_model('mambaout_base', pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'mobilenetv4_conv_small_050.e3000_r224_in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model('mambaout_base', pretrained=True, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [128, 256, 512, 768], 'out_channels': 256, 'num_outs': 4}
        num_stages = 4

    elif model_name=='hrnet':
        try:
            model = timm.create_model('hrnet_w64', pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'hrnet_w64.ms_in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model('hrnet_w64', pretrained=True, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [128, 256, 512, 1024], 'out_channels': 256, 'num_outs': 4}
        num_stages = 5
    
    elif model_name=='vmamba':
        model = vmamba_tiny_s1l8(pretrained=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [192, 384, 768, 768], 'out_channels': 256, 'num_outs': 4}
        num_stages = 4
    
    elif model_name=='mambaout_tiny':
        try:
            model = timm.create_model('mambaout_tiny', pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'mambaout_tiny.in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model('mambaout_tiny.in1k', pretrained=True, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [96, 192, 384, 576], 'out_channels': 256, 'num_outs': 4}
        num_stages = 4

    elif model_name=='mambaout_small':
        try:
            model = timm.create_model('mambaout_small', pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'mambaout_small.in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model('mambaout_small.in1k', pretrained=True, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [96, 192, 384, 576], 'out_channels': 256, 'num_outs': 4}
        num_stages = 4

    return model, num_stages, FPN_DICT
    

    