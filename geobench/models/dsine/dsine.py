import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from copy import deepcopy
from easydict import EasyDict

from diffusers import AutoencoderKL, AutoencoderTiny
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.unets.unet_2d_blocks import AutoencoderTinyBlock

from geobench.utils.rotation import axis_angle_to_matrix
from geobench.models.autoencoder_wrapper import NormalAutoencoderDecoderOutput
from geobench.models.dsine.submodules import (
    Encoder, 
    ConvGRU, 
    UpSampleBN, 
    UpSampleGN, 
    RayReLU, 
    Decoder, 
    convex_upsampling, 
    get_unfold, 
    get_prediction_head, 
    INPUT_CHANNELS_DICT
    upsample_via_bilinear, 
    get_pixel_coords,
    normal_activation,
    get_intrins_from_fov,
    pad_input,
    load_checkpoint,
    get_intrins_from_txt
)
   
class DSINE(ModelMixin, ConfigMixin):
    """DSINE Model"""
    @register_to_config
    def __init__(self, B=5):
        super(DSINE, self).__init__()
        self.downsample_ratio = 8
        self.ps = 5           # patch size
        self.num_iter = 5     # num iterations

        # define encoder
        self.encoder = Encoder(B=B, pretrained=True, rm_bn2=True)

        # define decoder
        self.output_dim = output_dim = 3
        self.feature_dim = feature_dim = 64
        self.hidden_dim = hidden_dim = 64
        self.decoder = Decoder([output_dim, feature_dim, hidden_dim], B=5, NF=2048, BN=False)

        # ray direction-based ReLU
        self.ray_relu = RayReLU(eps=1e-2)

        # pixel_coords (1, 3, H, W)
        # NOTE: this is set to some arbitrarily high number, 
        # if your input is 2000+ pixels wide/tall, increase these values
        h = 2000
        w = 2000
        pixel_coords = np.ones((3, h, w)).astype(np.float32)
        x_range = np.concatenate([np.arange(w).reshape(1, w)] * h, axis=0)
        y_range = np.concatenate([np.arange(h).reshape(h, 1)] * w, axis=1)
        pixel_coords[0, :, :] = x_range + 0.5
        pixel_coords[1, :, :] = y_range + 0.5
        # self.register_buffer(pixel_coords = torch.from_numpy(pixel_coords).unsqueeze(0)
        self.register_buffer('pixel_coords', torch.from_numpy(pixel_coords).unsqueeze(0), persistent=False)

        # define ConvGRU cell
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=feature_dim+2, ks=self.ps)

        # padding used during NRN
        self.pad = (self.ps - 1) // 2

        # prediction heads
        self.prob_head = get_prediction_head(self.hidden_dim+2, 64, self.ps*self.ps)   # weights assigned for each nghbr pixel 
        self.xy_head = get_prediction_head(self.hidden_dim+2, 64, self.ps*self.ps*2)   # rotation axis for each nghbr pixel
        self.angle_head = get_prediction_head(self.hidden_dim+2, 64, self.ps*self.ps)  # rotation angle for each nghbr pixel

        # prediction heads - weights used for upsampling the coarse resolution output
        self.up_prob_head = get_prediction_head(self.hidden_dim+2, 64, 9 * self.downsample_ratio * self.downsample_ratio)


    def get_ray(self, intrins, H, W, orig_H, orig_W, return_uv=False):
        B, _, _ = intrins.shape
        fu = intrins[:, 0, 0][:,None,None] * (W / orig_W)
        cu = intrins[:, 0, 2][:,None,None] * (W / orig_W)
        fv = intrins[:, 1, 1][:,None,None] * (H / orig_H)
        cv = intrins[:, 1, 2][:,None,None] * (H / orig_H)

        # (B, 2, H, W)
        ray = self.pixel_coords[:, :, :H, :W].repeat(B, 1, 1, 1)
        ray[:, 0, :, :] = (ray[:, 0, :, :] - cu) / fu
        ray[:, 1, :, :] = (ray[:, 1, :, :] - cv) / fv

        if return_uv:
            return ray[:, :2, :, :]
        else:
            return F.normalize(ray, dim=1)

    
    def upsample(self, h, pred_norm, uv_8):
        up_mask = self.up_prob_head(torch.cat([h, uv_8], dim=1))
        up_pred_norm = convex_upsampling(pred_norm, up_mask, self.downsample_ratio)
        up_pred_norm = F.normalize(up_pred_norm, dim=1)
        return up_pred_norm

    
    def refine(self, h, feat_map, pred_norm, intrins, orig_H, orig_W, uv_8, ray_8):
        B, C, H, W = pred_norm.shape
        fu = intrins[:, 0, 0][:,None,None,None] * (W / orig_W)  # (B, 1, 1, 1)
        cu = intrins[:, 0, 2][:,None,None,None] * (W / orig_W)
        fv = intrins[:, 1, 1][:,None,None,None] * (H / orig_H)
        cv = intrins[:, 1, 2][:,None,None,None] * (H / orig_H)

        h_new = self.gru(h, feat_map)

        # get nghbr prob (B, 1, ps*ps, h, w)
        nghbr_prob = self.prob_head(torch.cat([h_new, uv_8], dim=1)).unsqueeze(1)
        nghbr_prob = torch.sigmoid(nghbr_prob)

        # get nghbr normals (B, 3, ps*ps, h, w)
        nghbr_normals = get_unfold(pred_norm, ps=self.ps, pad=self.pad)

        # get nghbr xy (B, 2, ps*ps, h, w)
        nghbr_xys = self.xy_head(torch.cat([h_new, uv_8], dim=1))
        nghbr_xs, nghbr_ys = torch.split(nghbr_xys, [self.ps*self.ps, self.ps*self.ps], dim=1)
        nghbr_xys = torch.cat([nghbr_xs.unsqueeze(1), nghbr_ys.unsqueeze(1)], dim=1)        
        nghbr_xys = F.normalize(nghbr_xys, dim=1)

        # get nghbr theta (B, 1, ps*ps, h, w)
        nghbr_angle = self.angle_head(torch.cat([h_new, uv_8], dim=1)).unsqueeze(1)
        nghbr_angle = torch.sigmoid(nghbr_angle) * np.pi

        # get nghbr pixel coord (1, 3, ps*ps, h, w)
        nghbr_pixel_coord = get_unfold(self.pixel_coords[:, :, :H, :W], ps=self.ps, pad=self.pad)

        # nghbr axes (B, 3, ps*ps, h, w)
        nghbr_axes = torch.zeros_like(nghbr_normals)

        du_over_fu = nghbr_xys[:, 0, ...] / fu                                      # (B, ps*ps, h, w)
        dv_over_fv = nghbr_xys[:, 1, ...] / fv                                      # (B, ps*ps, h, w)

        term_u = (nghbr_pixel_coord[:, 0, ...] + nghbr_xys[:, 0, ...] - cu) / fu    # (B, ps*ps, h, w)
        term_v = (nghbr_pixel_coord[:, 1, ...] + nghbr_xys[:, 1, ...] - cv) / fv    # (B, ps*ps, h, w)

        nx = nghbr_normals[:, 0, ...]                                               # (B, ps*ps, h, w)
        ny = nghbr_normals[:, 1, ...]                                               # (B, ps*ps, h, w)
        nz = nghbr_normals[:, 2, ...]                                               # (B, ps*ps, h, w)

        nghbr_delta_z_num = - (du_over_fu * nx + dv_over_fv * ny)
        nghbr_delta_z_denom = (term_u * nx + term_v * ny + nz)
        nghbr_delta_z_denom[torch.abs(nghbr_delta_z_denom) < 1e-8] = 1e-8 * torch.sign(nghbr_delta_z_denom[torch.abs(nghbr_delta_z_denom) < 1e-8])
        nghbr_delta_z = nghbr_delta_z_num / nghbr_delta_z_denom

        nghbr_axes[:, 0, ...] = du_over_fu + nghbr_delta_z * term_u
        nghbr_axes[:, 1, ...] = dv_over_fv + nghbr_delta_z * term_v
        nghbr_axes[:, 2, ...] = nghbr_delta_z
        nghbr_axes = F.normalize(nghbr_axes, dim=1)                                 # (B, 3, ps*ps, h, w)

        # make sure axes are all valid
        invalid = torch.sum(torch.logical_or(torch.isnan(nghbr_axes), torch.isinf(nghbr_axes)).float(), dim=1) > 0.5    # (B, ps*ps, h, w)
        nghbr_axes[:, 0, ...][invalid] = 0.0
        nghbr_axes[:, 1, ...][invalid] = 0.0
        nghbr_axes[:, 2, ...][invalid] = 0.0

        # nghbr_axes_angle (B, 3, ps*ps, h, w)
        nghbr_axes_angle = nghbr_axes * nghbr_angle
        nghbr_axes_angle = nghbr_axes_angle.permute(0, 2, 3, 4, 1)  # (B, ps*ps, h, w, 3)
        nghbr_R = axis_angle_to_matrix(nghbr_axes_angle)            # (B, ps*ps, h, w, 3, 3)

        # (B, 3, ps*ps, h, w)
        nghbr_normals_rot = torch.bmm(
            nghbr_R.reshape(B * self.ps * self.ps * H * W, 3, 3),
            nghbr_normals.permute(0, 2, 3, 4, 1).reshape(B * self.ps * self.ps * H * W, 3).unsqueeze(-1)
        ).reshape(B, self.ps*self.ps, H, W, 3, 1).squeeze(-1).permute(0, 4, 1, 2, 3)        # (B, 3, ps*ps, h, w)
        nghbr_normals_rot = F.normalize(nghbr_normals_rot, dim=1)

        # ray ReLU
        nghbr_normals_rot = torch.cat([
            self.ray_relu(nghbr_normals_rot[:, :, i, :, :], ray_8).unsqueeze(2) 
            for i in range(nghbr_normals_rot.size(2))
            ], dim=2)

        # (B, 1, ps*ps, h, w) * (B, 3, ps*ps, h, w)
        pred_norm = torch.sum(nghbr_prob * nghbr_normals_rot, dim=2)    # (B, C, H, W)
        pred_norm = F.normalize(pred_norm, dim=1)
    
        up_mask = self.up_prob_head(torch.cat([h_new, uv_8], dim=1))
        up_pred_norm = convex_upsampling(pred_norm, up_mask, self.downsample_ratio)
        up_pred_norm = F.normalize(up_pred_norm, dim=1)

        return h_new, pred_norm, up_pred_norm

    
    def forward(self, img, intrins=None, mode='infer'):
        # Step 1. encoder
        features = self.encoder(img)
        # Step 2. get uv encoding
        B, _, orig_H, orig_W = img.shape

        if intrins is None:
            intrins = get_intrins_from_fov(
            new_fov=60.0, H=orig_H, W=orig_W, device=img.device).unsqueeze(0).repeat(B, 1, 1)

        intrins[:, 0, 2] += 0.5
        intrins[:, 1, 2] += 0.5
        uv_32 = self.get_ray(intrins, orig_H//32, orig_W//32, orig_H, orig_W, return_uv=True)
        uv_16 = self.get_ray(intrins, orig_H//16, orig_W//16, orig_H, orig_W, return_uv=True)
        uv_8 = self.get_ray(intrins, orig_H//8, orig_W//8, orig_H, orig_W, return_uv=True)
        ray_8 = self.get_ray(intrins, orig_H//8, orig_W//8, orig_H, orig_W)

        # ray_1 = self.get_ray(intrins, orig_H, orig_W, orig_H, orig_W)
        # Step 3. decoder - initial prediction
        # pred_norm: bs, 3, 160, 120
        # feat_map: bs, 66, 160, 120
        # h: bs, 64, 160, 120
        # import pdb;pdb.set_trace()
        pred_norm, feat_map, h = self.decoder(features, uvs=(uv_32, uv_16, uv_8))
        pred_norm = self.ray_relu(pred_norm, ray_8)

        # Step 4. add ray direction encoding
        feat_map = torch.cat([feat_map, uv_8], dim=1)

        # iterative refinement
        up_pred_norm = self.upsample(h, pred_norm, uv_8)
        pred_list = [up_pred_norm]

        for i in range(self.num_iter):
            h, pred_norm, up_pred_norm = self.refine(h, feat_map, 
                                                     pred_norm.detach(), 
                                                     intrins, 
                                                     orig_H, orig_W, uv_8, ray_8)
            pred_list.append(up_pred_norm)

        return pred_list


    def get_1x_lr_params(self):
        modules = [self.encoder]
        for m in modules:
            yield from m.parameters()

    
    def get_10x_lr_params(self):
        modules = [
            self.decoder, 
            self.gru, 
            self.prob_head, 
            self.xy_head, 
            self.angle_head, 
            # self.kappa_head,
            self.up_prob_head,
        ]
        for m in modules:
            yield from m.parameters()


