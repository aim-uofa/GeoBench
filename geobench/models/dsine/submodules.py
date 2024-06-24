import torch
import torch.nn as nn
import torch.nn.functional as F
import geffnet
import numpy as np

INPUT_CHANNELS_DICT = {
    0: [1280, 112, 40, 24, 16],
    1: [1280, 112, 40, 24, 16],
    2: [1408, 120, 48, 24, 16],
    3: [1536, 136, 48, 32, 24],
    4: [1792, 160, 56, 32, 24],
    5: [2048, 176, 64, 40, 24],
    6: [2304, 200, 72, 40, 32],
    7: [2560, 224, 80, 48, 32],
    
}

class Encoder(nn.Module):
    def __init__(self, B=5, pretrained=True, rm_bn2=True):
        """ e.g. B=5 will return EfficientNet-B5
        """
        super(Encoder, self).__init__()

        basemodel = geffnet.create_model('tf_efficientnet_b%s_ap' % B, pretrained=pretrained)
        # Remove last layer
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        if rm_bn2:
            basemodel.bn2 = nn.Identity()

        self.original_model = basemodel


    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            # print(k)
            # if k=='bn2':
                # import pdb;pdb.set_trace()

            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        
        # import pdb;pdb.set_trace()
        return features


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, ks=3):
        super(ConvGRU, self).__init__()
        p = (ks - 1) // 2
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, ks, padding=p)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, ks, padding=p)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, ks, padding=p)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h
    

class RayReLU(nn.Module):
    def __init__(self, eps=1e-2):
        super(RayReLU, self).__init__()
        self.eps = eps

    def forward(self, pred_norm, ray):
        # angle between the predicted normal and ray direction
        cos = torch.cosine_similarity(pred_norm, ray, dim=1).unsqueeze(1) # (B, 1, H, W)

        # component of pred_norm along view
        norm_along_view = ray * cos

        # cos should be bigger than eps
        norm_along_view_relu = ray * (torch.relu(cos - self.eps) + self.eps)

        # difference        
        diff = norm_along_view_relu - norm_along_view

        # updated pred_norm
        new_pred_norm = pred_norm + diff
        new_pred_norm = F.normalize(new_pred_norm, dim=1)

        return new_pred_norm    


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features, align_corners=True):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())
        self.align_corners = align_corners

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=self.align_corners)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class Conv2d_WS(nn.Conv2d):
    """ weight standardization
    """ 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class UpSampleGN(nn.Module):
    """ UpSample with GroupNorm
    """
    def __init__(self, skip_input, output_features, align_corners=True):
        super(UpSampleGN, self).__init__()
        self._net = nn.Sequential(Conv2d_WS(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(8, output_features),
                                  nn.LeakyReLU(),
                                  Conv2d_WS(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.GroupNorm(8, output_features),
                                  nn.LeakyReLU())
        self.align_corners = align_corners

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=self.align_corners)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class Decoder(nn.Module):
    def __init__(self, output_dims, B=5, NF=2048, BN=False, downsample_ratio=8):
        super(Decoder, self).__init__()
        input_channels = INPUT_CHANNELS_DICT[B]
        output_dim, feature_dim, hidden_dim = output_dims
        features = bottleneck_features = NF
        self.downsample_ratio = downsample_ratio

        UpSample = UpSampleBN if BN else UpSampleGN 
        self.conv2 = nn.Conv2d(bottleneck_features + 2, features, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSample(skip_input=features // 1 + input_channels[1] + 2, output_features=features // 2, align_corners=False)
        self.up2 = UpSample(skip_input=features // 2 + input_channels[2] + 2, output_features=features // 4, align_corners=False)

        # prediction heads
        i_dim = features // 4
        h_dim = 128
        self.normal_head = get_prediction_head(i_dim+2, h_dim, output_dim)
        self.feature_head = get_prediction_head(i_dim+2, h_dim, feature_dim)
        self.hidden_head = get_prediction_head(i_dim+2, h_dim, hidden_dim)

        # if learned_upsampling:
        #     self.mask_head = get_prediction_head(i_dim+2, 128, 9 * self.downsample_ratio * self.downsample_ratio)
        #     self.upsample_fn = upsample_via_mask
        # else:
        #     self.mask_head = lambda a: None
        #     self.upsample_fn = upsample_via_bilinear


    def forward(self, features, uvs):
        _, _, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]
        uv_32, uv_16, uv_8 = uvs

        x_d0 = self.conv2(torch.cat([x_block4, uv_32], dim=1))
        x_d1 = self.up1(x_d0, torch.cat([x_block3, uv_16], dim=1))
        x_feat = self.up2(x_d1, torch.cat([x_block2, uv_8], dim=1))
        x_feat = torch.cat([x_feat, uv_8], dim=1)

        normal = self.normal_head(x_feat)
        normal = F.normalize(normal, dim=1)
        f = self.feature_head(x_feat)
        h = self.hidden_head(x_feat)
        return normal, f, h


def upsample_via_bilinear(out, up_mask, downsample_ratio):
    """ bilinear upsampling (up_mask is a dummy variable)
    """
    return F.interpolate(out, scale_factor=downsample_ratio, mode='bilinear', align_corners=True)


def upsample_via_mask(out, up_mask, downsample_ratio, padding='zero'):
    """ convex upsampling
    """
    # out: low-resolution output (B, o_dim, H, W)
    # up_mask: (B, 9*k*k, H, W)
    k = downsample_ratio

    B, C, H, W = out.shape
    up_mask = up_mask.view(B, 1, 9, k, k, H, W)
    up_mask = torch.softmax(up_mask, dim=2)             # (B, 1, 9, k, k, H, W)

    if padding == 'zero':
        # with zero padding
        up_out = F.unfold(out, [3, 3], padding=1)       # (B, 2, H, W) -> (B, 2 X 3*3, H*W)
    elif padding == 'replicate':
        # with replicate padding
        out = F.pad(out, pad=(1,1,1,1), mode='replicate')
        up_out = F.unfold(out, [3, 3], padding=0)           # (B, C, H, W) -> (B, C X 3*3, H*W)
    else:
        raise Exception('invalid padding for convex upsampling')

    up_out = up_out.view(B, C, 9, 1, 1, H, W)           # (B, C, 9, 1, 1, H, W)

    up_out = torch.sum(up_mask * up_out, dim=2)         # (B, C, k, k, H, W)
    up_out = up_out.permute(0, 1, 4, 2, 5, 3)           # (B, C, H, k, W, k)
    return up_out.reshape(B, C, k*H, k*W)               # (B, C, kH, kW)


def upsample_via_mask_1(out, up_mask, downsample_ratio):
    """ convex upsampling
    """
    # out: low-resolution output (B, o_dim, H, W)
    # up_mask: (B, 9*k*k, H, W)
    k = downsample_ratio

    N, o_dim, H, W = out.shape
    up_mask = up_mask.view(N, 1, 9, k, k, H, W)
    up_mask = torch.softmax(up_mask, dim=2)             # (B, 1, 9, k, k, H, W)

    up_out = F.unfold(out, [3, 3], padding=1)       # (B, 2, H, W) -> (B, 2 X 3*3, H*W)
    up_out = up_out.view(N, o_dim, 9, 1, 1, H, W)   # (B, 2, 3*3, 1, 1, H, W)
    up_out = torch.sum(up_mask * up_out, dim=2)     # (B, 2, k, k, H, W)

    up_out = up_out.permute(0, 1, 4, 2, 5, 3)       # (B, 2, H, k, W, k)
    return up_out.reshape(N, o_dim, k*H, k*W)   # (B, 2, kH, kW)


def normal_activation(out, elu_kappa=True):
    normal, kappa = out[:, :3, :, :], out[:, 3:, :, :]
    normal = F.normalize(normal, p=2, dim=1)
    if elu_kappa:
        kappa = F.elu(kappa) + 1.0
    return torch.cat([normal, kappa], dim=1)

def convex_upsampling(out, up_mask, k):
    # out: low-resolution output    (B, C, H, W)
    # up_mask:                      (B, 9*k*k, H, W)
    B, C, H, W = out.shape
    up_mask = up_mask.view(B, 1, 9, k, k, H, W)
    up_mask = torch.softmax(up_mask, dim=2)             # (B, 1, 9, k, k, H, W)

    out = F.pad(out, pad=(1,1,1,1), mode='replicate')
    up_out = F.unfold(out, [3, 3], padding=0)           # (B, C, H, W) -> (B, C X 3*3, H*W)
    up_out = up_out.view(B, C, 9, 1, 1, H, W)           # (B, C, 9, 1, 1, H, W)

    up_out = torch.sum(up_mask * up_out, dim=2)         # (B, C, k, k, H, W)
    up_out = up_out.permute(0, 1, 4, 2, 5, 3)           # (B, C, H, k, W, k)
    return up_out.reshape(B, C, k*H, k*W)               # (B, C, kH, kW)


def get_unfold(pred_norm, ps, pad):
    B, C, H, W = pred_norm.shape
    pred_norm = F.pad(pred_norm, pad=(pad,pad,pad,pad), mode='replicate')       # (B, C, h, w)
    pred_norm_unfold = F.unfold(pred_norm, [ps, ps], padding=0)                 # (B, C X ps*ps, h*w)
    pred_norm_unfold = pred_norm_unfold.view(B, C, ps*ps, H, W)                 # (B, C, ps*ps, h, w)
    return pred_norm_unfold


def get_prediction_head(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Conv2d(input_dim, hidden_dim, 3, padding=1), 
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, 1), 
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, output_dim, 1),
    )

def load_checkpoint(fpath, model):
    print('loading checkpoint... {}'.format(fpath))

    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if "encoder.original_model.bn2" in k:
            continue
            
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    print('loading checkpoint... / done')
    return model


def pad_input(orig_H, orig_W):
    if orig_W % 32 == 0:
        l = 0
        r = 0
    else:
        new_W = 32 * ((orig_W // 32) + 1)
        l = (new_W - orig_W) // 2
        r = (new_W - orig_W) - l

    if orig_H % 32 == 0:
        t = 0
        b = 0
    else:
        new_H = 32 * ((orig_H // 32) + 1)
        t = (new_H - orig_H) // 2
        b = (new_H - orig_H) - t

    return l, r, t, b


def pad_input_14(orig_H, orig_W):
    if orig_W % 14 == 0:
        l = 0
        r = 0
    else:
        new_W = 14 * ((orig_W // 14) + 1)
        l = (new_W - orig_W) // 2
        r = (new_W - orig_W) - l

    if orig_H % 14 == 0:
        t = 0
        b = 0
    else:
        new_H = 14 * ((orig_H // 14) + 1)
        t = (new_H - orig_H) // 2
        b = (new_H - orig_H) - t

    return l, r, t, b


def get_intrins_from_fov(new_fov, H, W, device):
    # NOTE: top-left pixel should be (0,0)
    if W >= H:
        new_fu = (W / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
        new_fv = (W / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
    else:
        new_fu = (H / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
        new_fv = (H / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))

    new_cu = (W / 2.0) - 0.5
    new_cv = (H / 2.0) - 0.5

    new_intrins = torch.tensor([
        [new_fu,    0,          new_cu  ],
        [0,         new_fv,     new_cv  ],
        [0,         0,          1       ]
    ], dtype=torch.float32, device=device)

    return new_intrins


def get_intrins_from_txt(intrins_path, device):
    # NOTE: top-left pixel should be (0,0)
    with open(intrins_path, 'r') as f:
        intrins_ = f.readlines()[0].split()[0].split(',')
        intrins_ = [float(i) for i in intrins_]
        fx, fy, cx, cy = intrins_

    intrins = torch.tensor([
        [fx, 0,cx],
        [ 0,fy,cy],
        [ 0, 0, 1]
    ], dtype=torch.float32, device=device)

    return intrins

def get_pixel_coords(h, w):
    # pixel array (1, 2, H, W)
    pixel_coords = np.ones((3, h, w)).astype(np.float32)
    x_range = np.concatenate([np.arange(w).reshape(1, w)] * h, axis=0)
    y_range = np.concatenate([np.arange(h).reshape(h, 1)] * w, axis=1)
    pixel_coords[0, :, :] = x_range + 0.5
    pixel_coords[1, :, :] = y_range + 0.5
    return torch.from_numpy(pixel_coords).unsqueeze(0)