import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import importlib

from typing import List, Dict, Union
import einops
import torch.nn.functional as F
import skimage
from functools import partial
from torchdiffeq import odeint
import cv2


from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    ControlNetModel
)

from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection, AutoImageProcessor
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

from geobench.utils.image_util import chw2hwc, colorize_depth_maps, resize_max_res, norm_to_rgb
from geobench.utils.batchsize import find_batch_size
from geobench.utils.depth_ensemble import ensemble_depths
from geobench.utils.normal_ensemble import ensemble_normals
# from geobench.models.dsine import pad_input
# from geobench.models.image_projector import ImageProjModel

from diffusers import logging
logging.set_verbosity_error()


def exists(val):
    return val is not None

def sigmoid(x):
    if isinstance(x, torch.Tensor):
        return 1 / (1 + torch.exp(-x))
    else:
        return 1 / (1 + np.exp(-x))

def cosine_log_snr(t, eps=0.00001):
    """
    Returns log Signal-to-Noise ratio for time step t and image size 64
    eps: avoid division by zero
    """
    if isinstance(t, torch.Tensor):
        return -2 * torch.log(torch.tan((torch.pi * t) / 2) + eps)
    else:
        return -2 * np.log(np.tan((np.pi * t) / 2) + eps)


def cosine_alpha_bar(t):
    return sigmoid(cosine_log_snr(t))


def q_sample(x_start: torch.Tensor, t: int, noise: torch.Tensor = None, n_diffusion_timesteps: int = 1000):
    """
    Diffuse the data for a given number of diffusion steps. In other
    words sample from q(x_t | x_0).
    """
    dev = x_start.device
    dtype = x_start.dtype

    if noise is None:
        noise = torch.randn_like(x_start)
    
    alpha_bar_t = cosine_alpha_bar(t / n_diffusion_timesteps)
    alpha_bar_t = torch.tensor(alpha_bar_t).to(dev).to(dtype)[...,None,None,None]
    # print(t, torch.sqrt(alpha_bar_t))
    

    return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise


def per_sample_min_max_normalization(x):
    """ Normalize each sample in a batch independently
    with min-max normalization to [0, 1] """
    bs, *shape = x.shape
    x_ = einops.rearrange(x, "b ... -> b (...)")
    min_val = einops.reduce(x_, "b ... -> b", "min")[..., None]
    max_val = einops.reduce(x_, "b ... -> b", "max")[..., None]
    x_ = (x_ - min_val) / (max_val - min_val)
    return x_.reshape(bs, *shape)


class FlowMatchOutput(BaseOutput):
    depth_np: np.ndarray
    depth_colored: Image.Image
    normal_np: np.ndarray
    normal_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class FlowMatchNormalOutput(BaseOutput):
    normal_np: np.ndarray
    normal_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class FlowMatchDepthOutput(BaseOutput):
    """
    Output class for FlowMatch monocular depth prediction pipeline.

    Args:
        depth_np (np.ndarray):
            Predicted depth map, with depth values in the range of [0, 1]
        depth_colored (PIL.Image.Image):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1]
        uncertainty (None` or `np.ndarray):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class FlowMatchSegOutput(BaseOutput):
    """
    Output class for FlowMatch monocular depth prediction pipeline.

    Args:
        depth_np (np.ndarray):
            Predicted depth map, with depth values in the range of [0, 1]
        depth_colored (PIL.Image.Image):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1]
        uncertainty (None` or `np.ndarray):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    # seg_np: np.ndarray
    seg_colored: Image.Image
    seg_colored_wo_remap: Image.Image
    argmax_output: Union[None, np.ndarray]
    uncertainty: Union[None, np.ndarray]


class FlowMatchPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using FlowMatch: https://arxiv.org/abs/2312.02145.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (UNet2DConditionModel):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (AutoencoderKL):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (DDIMScheduler):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (CLIPTextModel):
            Text-encoder, for empty text embedding.
        tokenizer (CLIPTokenizer):
            CLIP tokenizer.
    """

    rgb_latent_scale_factor = 0.18215
    target_latent_scale_factor = 0.18215
    seg_latent_scale_factor = 0.18215
    normal_latent_scale_factor = 0.18215

    config_name = "model_index.json"
    sd_version = 'sd21'

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        decode_type: str = 'argmax', # rgb
        normal_head = None,
        seg_head = None,
        tokenizer_2: CLIPTokenizer = None,
        text_encoder_2: CLIPTextModel = None,
        text_embeds: torch.Tensor = None,
        image_encoder: CLIPVisionModelWithProjection = None,
        image_processor: AutoImageProcessor = None,
        controlnet: ControlNetModel = None,
        image_mean_std = [0.5, 0.5],
        gt_mean_std = [0., 1.],
        noise_type = 'rgb',
        noise_factor = 0.3,
        use_cold_diffusion = False,
        denoiser_type='unet',
        use_vae_decoder_features = False,
        enable_log_normalization = True
    ):
        super().__init__()

        
        if isinstance(vae, AutoencoderKL):
            self.vae_version = 'base'
        else:
            self.vae_version = 'tiny'

        self.enable_log_normalization = enable_log_normalization
        self.use_vae_decoder_features = use_vae_decoder_features
        self.image_mean_std = image_mean_std
        self.gt_mean_std = gt_mean_std
        self.noise_type = noise_type
        self.noise_factor = noise_factor
        self.use_cold_diffusion = use_cold_diffusion
        self.denoiser_type = denoiser_type

        if self.sd_version == 'sdxl':
            text_embeds = torch.from_numpy(
                np.load('geobench/utils/text_embeds_sdxl.npy', allow_pickle=True)
            )

        elif self.sd_version == 'sd21':
            text_embeds = torch.from_numpy(
                np.load('geobench/utils/text_embeds_sd21.npy', allow_pickle=True)
            )
        self.empty_text_embed = text_embeds

        if seg_head is not None:
            self.register_modules(seg_head=seg_head)
            self.seg_head.eval()
        else:
            self.seg_head = seg_head

        if normal_head is not None:
            self.register_modules(normal_head=normal_head)
            self.normal_head.eval()
        else:
            self.normal_head = normal_head

        self.decode_type = decode_type

        if isinstance(controlnet, ControlNetModel):
            self.register_modules(
                unet=unet,
                vae=vae,
                scheduler=scheduler,
                tokenizer=tokenizer,
                controlnet=controlnet,
                
            )
        else:
            self.controlnet = None
            self.register_modules(
                unet=unet,
                vae=vae,
                scheduler=scheduler,
                tokenizer=tokenizer,
            )


        self.register_modules(text_encoder=text_encoder)
        self.register_modules(text_encoder_2=text_encoder_2)

        if tokenizer_2 is not None:
            self.register_modules(tokenizer_2=tokenizer_2)
        else:
            self.tokenizer_2 = None

        if image_encoder is not None:
            self.register_modules(
                image_encoder=image_encoder,
            )
        else:
            self.image_encoder = None
        
        self.image_processor = image_processor
        

        if image_projector is not None:
            self.register_modules(
                image_projector=image_projector,
                )
        else:
            self.image_projector = None 

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb


    @property
    def guidance_scale(self):
        return self._guidance_scale

    
    @torch.no_grad()
    def __call__(
        self,
        input_image: Image,
        input_prompt: str = "",
        denoising_steps: int = 5,
        # num_inference_steps: int = 10,
        ensemble_size: int = 1,
        processing_res: int = 768,
        match_input_res: bool = True,
        batch_size: int = 0,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        mode: str = 'depth', # depth normal joint
        q_sample_timestep: int = 400,
    ) -> FlowMatchDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (Image):
                Input RGB (or gray-scale) image.
            processing_res (int, optional):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
                Defaults to 768.
            match_input_res (bool, optional):
                Resize depth prediction to match input resolution.
                Only valid if `limit_input_res` is not None.
                Defaults to True.
            denoising_steps (int, optional):
                Number of diffusion denoising steps (DDIM) during inference.
                Defaults to 10.
            ensemble_size (int, optional):
                Number of predictions to be ensembled.
                Defaults to 10.
            batch_size (int, optional):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
                Defaults to 0.
            show_progress_bar (bool, optional):
                Display a progress bar of diffusion denoising.
                Defaults to True.
            color_map (str, optional):
                Colormap used to colorize the depth map.
                Defaults to "Spectral".
            ensemble_kwargs ()
        Returns:
            `FlowMatchDepthOutput`
        """

        device = self.device

        self.empty_text_embed = self.empty_text_embed.to(self.device)

        if isinstance(input_image, Image.Image):
            input_size = input_image.size
            if not match_input_res:
                assert (
                    processing_res is not None
                ), "Value error: `resize_output_back` is only valid with "
            assert processing_res >= 0
            assert denoising_steps >= 1
            assert ensemble_size >= 1

            # ----------------- Image Preprocess -----------------
            # Resize image
            if processing_res > 0:
                input_image = resize_max_res(
                    input_image, max_edge_resolution=processing_res
                )
            # Convert the image to RGB, to 1.remove the alpha channel 2.convert B&W to 3-channel
            input_image = input_image.convert("RGB")
            # clip_image = CLIPImageProcessor()(
            #     images=input_image, 
            #     return_tensors="pt").pixel_values[0]

            if self.image_processor is not None:
                dino_image = self.image_processor(
                    images=input_image, 
                    return_tensors="pt").pixel_values[0]

                
                dino_image = F.interpolate(
                    dino_image[None], 
                    size=(384, 384), 
                    mode='bilinear',
                    align_corners=False
                )[0]

            image = np.asarray(input_image)

            # Normalize rgb values
            rgb = np.transpose(image, (2, 0, 1))  # [H, W, rgb] -> [rgb, H, W]
            rgb_norm = rgb / 255.0  
            
            # rgb_norm = (rgb_norm - 0.5) * 2.0
            rgb_norm = (rgb_norm - self.image_mean_std[0]) / self.image_mean_std[1]

            rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
            rgb_norm = rgb_norm.to(device)

        elif isinstance(input_image, torch.Tensor):
            rgb_norm = input_image / 255.0
            rgb_norm = (rgb_norm - self.image_mean_std[0]) / self.image_mean_std[1]
            rgb_norm = rgb_norm.to(device).to(self.dtype)
            input_size = (rgb_norm.shape[2], rgb_norm.shape[1])

            if self.image_processor is not None:
                raise NotImplementedError

                dino_image = self.image_processor(
                    images=input_image, 
                    return_tensors="pt").pixel_values[0]
                

        else:
            raise NotImplementedError

        # rgb_norm = 2 * (rgb_norm - 0.5)
        if self.image_mean_std[0] == 0.5 and self.image_mean_std[1] == 0.5:
            assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        if self.image_mean_std[0] == 0.0 and self.image_mean_std[1] == 0.0:
            assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)

        if self.image_processor is None:
            single_rgb_dataset = TensorDataset(duplicated_rgb)
        else:
            duplicated_dino_rgb = torch.stack([dino_image] * ensemble_size).to(self.image_encoder.dtype)
            single_rgb_dataset = TensorDataset(duplicated_rgb, duplicated_dino_rgb)

        # single_rgb_dataset = TensorDataset(duplicated_rgb, duplicated_clip_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size, 
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        # print('batch size: {}'.format(_bs))
        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict depth maps (batched)
        depth_pred_ls = []
        normal_pred_ls = []

        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader

        for batch in iterable:
            # (batched_img, batch_clip_img) = batch
            batched_img = batch[0]
            if self.image_processor is not None:
                batched_dino_img = batch[1]
            else:
                batched_dino_img = None

            depth_pred_raw, normal_pred_raw = self.single_infer(
                prompt=[input_prompt] * batched_img.shape[0],
                rgb_in=batched_img,
                dino_rgb_in=batched_dino_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                mode=mode,
                q_sample_timestep=q_sample_timestep
            )

            if depth_pred_raw is not None:
                depth_pred_ls.append(depth_pred_raw.detach().clone())
            if normal_pred_raw is not None:
                normal_pred_ls.append(normal_pred_raw.detach().clone())

        torch.cuda.empty_cache()  # clear vram cache for ensembling

        
        # ----------------- Test-time ensembling -----------------
        
        if 'depth' in mode:
            depth_preds = torch.concat(depth_pred_ls, axis=0).squeeze(1)
            
            if ensemble_size > 1:
                # depth_pred, pred_uncert = ensemble_depths(
                #     depth_preds, **(ensemble_kwargs or {})
                # )
                depth_pred = depth_preds.mean(dim=0)
                pred_uncert = None
                
            else:
                depth_pred = depth_preds.mean(dim=0).repeat(3,1,1).permute(1,2,0).mean(dim=2)
                pred_uncert = None
        else:
            depth_pred=None
            depth_colored_img=None
            pred_uncert = None

        # ----------------- Post processing -----------------
        if 'normal' in mode:
            normal_preds = torch.concat(normal_pred_ls, axis=0).squeeze(1)
            
            # if ensemble_size > 1:
            normal_preds = ensemble_normals(normal_preds)
            # elif len(normal_preds.shape) == 4:
            #     normal_preds = normal_preds.mean(dim=0)
        
            normal = normal_preds.clip(-1, 1).cpu().numpy().astype(np.float32) # [-1, 1]
            normal_colored = norm_to_rgb(normal)
            normal_colored_hwc = chw2hwc(normal_colored)
            normal_colored_img = Image.fromarray(normal_colored_hwc)
            normal = chw2hwc(normal)

            if match_input_res:

                
                normal_colored_img = normal_colored_img.resize(input_size)
                # normal = np.asarray(Image.fromarray(normal).resize(input_size))
                normal = cv2.resize(normal, input_size, interpolation = cv2.INTER_NEAREST)
        else:
            normal = None
            normal_colored_img = None
        
        if 'depth' in mode:
            # Scale prediction to [0, 1]
            min_d = torch.min(depth_pred)
            max_d = torch.max(depth_pred)
            depth_pred = (depth_pred - min_d) / (max_d - min_d)
            # depth_pred = per_sample_min_max_normalization(depth_pred)
            # Convert to numpy
            depth_pred = depth_pred.cpu().numpy().astype(np.float32)
            
            # Resize back to original resolution
            if match_input_res:
                pred_img = Image.fromarray(depth_pred)
                pred_img = pred_img.resize(input_size)
                depth_pred = np.asarray(pred_img)

            # Clip output range
            depth_pred = depth_pred.clip(0, 1)

            if color_map is not None:
                # Colorize
                depth_colored = colorize_depth_maps(
                    depth_pred, 0, 1, cmap=color_map
                ).squeeze()  # [3, H, W], value in (0, 1)
                depth_colored = (depth_colored * 255).astype(np.uint8)
                depth_colored_hwc = chw2hwc(depth_colored)
                depth_colored_img = Image.fromarray(depth_colored_hwc)
            else:
                depth_colored_img = None

        if mode == 'seg':
            if self.decode_type == 'argmax':
                depth_preds = depth_pred_ls[0]
                if len(depth_pred_ls[0].shape) == 3:
                    depth_preds = depth_preds[None]
            else:
                depth_preds = torch.stack(depth_pred_ls, axis=0).mean(0)

            if self.seg_head is not None:
                height, width = depth_preds.shape[2:]
                pred_seg = depth_preds[0]
                pred_seg = chw2hwc(pred_seg.cpu().numpy().astype(np.uint8))[...,0]

                # palette = np.array(hypersim_palette())
                palette = np.array(seg_300_classes_palette())
                seg_colored_hwc = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
                # color_seg = np.zeros_like(np.array(input))

                for label, color in enumerate(palette):
                    seg_colored_hwc[pred_seg == label, :] = color

                seg_colored_hwc = seg_colored_hwc[..., ::-1]  # convert to BGR
                # return color_seg
                seg_colored_img = Image.fromarray(seg_colored_hwc).resize(input_size)
                seg_colored_img_wo_remap = seg_colored_img

            elif self.decode_type == 'argmax':
                height, width = batched_img.shape[2:]
                # upsampled_logits = nn.functional.interpolate(
                #         depth_preds.float(),
                #         # size=labels.shape[-2:], 
                #         size=[height, width],
                #         mode="bilinear", 
                #         align_corners=False
                # )
                pred_seg = depth_preds[0]
                pred_seg = chw2hwc(pred_seg.cpu().numpy().astype(np.uint8))[...,0]

                palette = np.array(seg_300_classes_palette())
                seg_colored_hwc = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
                # color_seg = np.zeros_like(np.array(input))

                for label, color in enumerate(palette):
                    seg_colored_hwc[pred_seg == label, :] = color

                seg_colored_hwc = seg_colored_hwc[..., ::-1]  # convert to BGR
                # return color_seg
                seg_colored_img = Image.fromarray(seg_colored_hwc).resize(input_size)
                seg_colored_img_wo_remap = seg_colored_img

            else:
                # Clip output range
                seg_colored = depth_preds.clip(0, 255).cpu().numpy().astype(np.uint8)
                seg_colored_hwc = chw2hwc(seg_colored)

                # Image.fromarray(seg_colored_hwc_wo_remap).resize(input_size)
                # *********************************** remap ****************************************
                # if False:
                if False:
                    # save non remap results
                    seg_colored_img_wo_remap = Image.fromarray(seg_colored_hwc).resize(input_size)

                    height, width = seg_colored_hwc.shape[:2]
                    # without remap color
                    # seg_colored_hwc_wo_remap = seg_colored_hwc.cpu().numpy().astype(np.uint8)
                    seg_lab_hwc = skimage.color.rgb2lab(seg_colored_hwc /255).reshape(-1, 1, 3)
                    # compute cosine similarity here
                    # seg_colored_hwc = seg_colored_hwc.reshape(-1, 3).unsqueeze(0)
                    # seg_colored_lab = rgb_to_lab(seg_colored_torch[None] / 255)[0]
                    rgb_palette = hypersim_palette()
                    lab_palette = skimage.color.rgb2lab(rgb_palette[None]/ 255).reshape(1, -1, 3)

                    # color_palette = torch.from_numpy(color_palette).to(self.device).float()
                    # color_palette = color_palette.reshape(-1, 3).unsqueeze(0)

                    # diff = seg_colored_hwc - color_palette
                    diff = seg_lab_hwc - lab_palette
                    dis = np.linalg.norm(diff, axis=2)

                    # dis = torch.norm(diff, dim=2)

                    # similarity = torch.exp(-torch.norm(diff, dim=2) * 0.5)
                    # dist = torch.cdist(seg_colored_hwc, color_palette, p=1)**0.5
                    # similarity = torch.exp(-dist[0])
                    # similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

                    # seg_colored_hwc = seg_colored_hwc.reshape(-1, 3).unsqueeze(1)
                    # color_palette = torch.from_numpy(hypersim_palette()).to(self.device).float()
                    # color_palette = color_palette.reshape(-1, 3).unsqueeze(0)
                    # similarity_matrix = F.cosine_similarity(
                    #     seg_colored_hwc, color_palette, dim=2)
                    # seg_colored_hwc = seg_colored_hwc.cpu().numpy().astype(np.uint8)
                    seg_colored_hwc = rgb_palette[dis.argmin(axis=1)].reshape(height, width, 3).astype(np.uint8)
                    seg_colored_img = Image.fromarray(seg_colored_hwc).resize(input_size)

                # ***************************************************************************
                else:
                    seg_colored_img = Image.fromarray(seg_colored_hwc).resize(input_size)
                    seg_colored_img_wo_remap = seg_colored_img
              
            return FlowMatchSegOutput(
                seg_colored=seg_colored_img,
                seg_colored_wo_remap=seg_colored_img_wo_remap,
                uncertainty=None,
                argmax_output=pred_seg if self.decode_type == 'argmax' else None
            )

        return FlowMatchOutput(
                depth_np=depth_pred,
                depth_colored=depth_colored_img,
                normal_np=normal,
                normal_colored=normal_colored_img,
                uncertainty=pred_uncert,
                )

    def encode_clip_text_feature(self, batched_prompt):
        """
        Encode text embedding for empty prompt
        """
        if self.sd_version == 'sd21':
            assert self.text_encoder is not None
            text_inputs = self.tokenizer(
                batched_prompt,
                # padding="do_not_pad",
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
            text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)
            
            return text_embed

        if self.sd_version == 'sdxl':
            assert self.text_encoder is not None and self.text_encoder_2 is not None

            text_inputs = self.tokenizer(
                batched_prompt,
                # padding="do_not_pad",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(self.text_encoder.device)

            text_inputs_2 = self.tokenizer_2(
                batched_prompt,
                # padding="do_not_pad",
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids_2 = text_inputs_2.input_ids.to(self.text_encoder.device)

            encoder_output = self.text_encoder(input_ids, output_hidden_states=True)
            text_embeds = encoder_output.hidden_states[-2]
            encoder_output_2 = self.text_encoder_2(input_ids_2, output_hidden_states=True)
            pooled_text_embeds = encoder_output_2[0]
            text_embeds_2 = encoder_output_2.hidden_states[-2]
            # bs, 77, 2048
            text_embed = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat
                
            return text_embed
    
    
    def encode_clip_image_feature(self, clip_rgb_in=None):

        if self.image_encoder is not None and clip_rgb_in is not None:
            image_embeds = self.image_encoder(clip_rgb_in.to(self.image_encoder.device)).image_embeds
            
            if self.image_projector is not None:
                image_embeds = self.image_projector(image_embeds)
                # bs, 1, 1024
                # image_embeds = image_embeds.unsqueeze(1)
                # image_embeds = torch.cat([image_embeds, ip_tokens], dim=1)
            else:
                image_embeds = image_embeds.unsqueeze(1)

            return image_embeds

    
    def encode_dino_image_feature(self, dino_rgb_in=None):

        if self.image_encoder is not None and dino_rgb_in is not None:
            image_embeds = self.image_encoder(dino_rgb_in.to(self.image_encoder.device)).pooler_output #.image_embeds
            
            if self.image_projector is not None:

                image_embeds = self.image_projector(image_embeds)
                # bs, 1, 1024
                # image_embeds = image_embeds.unsqueeze(1)
                # image_embeds = torch.cat([image_embeds, ip_tokens], dim=1)
            else:
                image_embeds = image_embeds.unsqueeze(1)

            return image_embeds

    @torch.no_grad()
    def single_infer(
        self, 
        prompt: list,
        rgb_in: torch.Tensor, 
        dino_rgb_in: torch.Tensor, 
        num_inference_steps: int, 
        show_pbar: bool,
        mode: str = 'depth',
        q_sample_timestep: int = 400,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (torch.Tensor):
                Input RGB image.
            num_inference_steps (int):
                Number of diffusion denoising steps (DDIM) during inference.
            show_pbar (bool):
                Display a progress bar of diffusion denoising.

        Returns:
            torch.Tensor: Predicted depth map.
        """
        device = rgb_in.device
        bsz = rgb_in.shape[0]

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # timestep_interval = timesteps[0] - timesteps[1]
        # if self.scheduler.config.beta_start == 1.0 and self.scheduler.config.beta_end == 1.0:
        #     timesteps = torch.ones_like(timesteps)

        # Encode image
        torch.manual_seed(0)
        rgb_latent = self.encode_rgb(rgb_in)
        torch.manual_seed(0)

        if self.empty_text_embed is not None:
            text_encoder_hidden_states = self.empty_text_embed.repeat(
                rgb_latent.shape[0], 1, 1).to(device)  # [B, 2, 1024]
        else:
            # batched_prompt=[""] * rgb_latent.shape[0]
            text_encoder_hidden_states = self.encode_clip_text_feature(prompt)

        # if self.text_encoder is not None: or self.image_encoder is None:
        #     # Batched empty text embedding
        #     # if self.empty_text_embed is None:
        #         # text_embed = self.encode_clip_feature(clip_rgb_in)
        #     encoder_hidden_states = self.empty_text_embed.repeat(
        #         rgb_latent.shape[0], 1, 1).to(device)  # [B, 2, 1024]
        if self.image_encoder is not None: # and self.image_projector is not None
            # rgb_latent.dtype
            image_encoder_hidden_states = self.encode_dino_image_feature(dino_rgb_in.to(device))
        else:
            image_encoder_hidden_states = None
            
        # encoder_hidden_states = torch.cat([image_encoder_hidden_states, text_encoder_hidden_states], axis=1)
        if self.seg_head is not None and self.seg_head.has_class_embed:
            encoder_hidden_states = self.seg_head.get_class_embed
            encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)
        elif image_encoder_hidden_states is not None:
            encoder_hidden_states = image_encoder_hidden_states
            # encoder_hidden_states = torch.cat([text_encoder_hidden_states, image_encoder_hidden_states], axis=1)
        else:
            encoder_hidden_states = text_encoder_hidden_states

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        if self.denoiser_type == 'dit':
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        else:
            added_cond_kwargs = None

        if self.seg_head is not None and self.seg_head.has_time_embed:
            seg_time_embed = self.seg_head.get_time_embed
        else:
            seg_time_embed = None

        # hack 
        # if mode == 'seg':
        #     target_latent = q_sample(rgb_latent, t=100)
        # else:
        target_latent = q_sample(rgb_latent, t=q_sample_timestep)

        if mode == 'depth_and_normal':
            target_latent = target_latent.repeat(1, 2, 1, 1)
            # z_1 = torch.cat([depth_latents, normal_latents], dim=1)

        # if self.unet.config.in_channels == 8:
        #     unet_input = torch.cat(
        #         [rgb_latent, target_latent], dim=1
        #     )  # this order is important
        # else:
        #     unet_input = target_latent

        # https://github.com/NNSSA/Rec21/blob/main/run_flowmatching.py

        unet_kwargs = dict(
            # unet_input=unet_input,
            rgb_latent=rgb_latent,
            encoder_hidden_states=encoder_hidden_states.float(),
            # class_labels=class_embedding,
            added_cond_kwargs=added_cond_kwargs,
        )

        ode_fn = partial(
            self.ode_fn,
            **unet_kwargs
        )
        # if mode == 'seg':
        #     num_steps = 5
        # else:
        num_steps = num_inference_steps
        
        n_intermediates = num_steps - 2
        timestep = torch.linspace(0, 1, n_intermediates + 2, dtype=rgb_latent.dtype).to(device)
        ode_kwargs = dict(method="euler", rtol=1e-5, atol=1e-5, options=dict(step_size=1.0 / num_steps))
        ode_results = odeint(ode_fn, target_latent, timestep, **ode_kwargs)
        # predict the noise residual

        if mode == 'depth':
            depth_target_latent = ode_results[-1]
            depth = self.decode_depth(depth_target_latent)
            
            # normalize depth maps to range [0, 1]
            if self.enable_log_normalization:
                # depth = depth.exp()
                depth = per_sample_min_max_normalization(depth.exp())
            else:
                depth = per_sample_min_max_normalization(depth)

            return depth, None

        elif mode == 'normal':
            target_latent = ode_results[-1]
            normal = self.decode_normal(target_latent)
            normal = torch.clip(normal, -1.0, 1.0)
        
            return None, normal

        elif mode == 'depth_and_normal':
            target_latent = ode_results[-1]
            depth_target_latent = target_latent[:,:4,:,:]
            normal_target_latent = target_latent[:,4:,:,:]

            depth = self.decode_depth(depth_target_latent)
            
            # normalize depth maps to range [0, 1]
            if self.enable_log_normalization:
                depth = per_sample_min_max_normalization(depth.exp())
            else:
                depth = per_sample_min_max_normalization(depth)
                
            normal = self.decode_normal(normal_target_latent)
            normal = torch.clip(normal, -1.0, 1.0)

            
            return depth, normal
            # return torch.cat([depth.repeat(1,3,1,1), normal], dim=-1)

        else:
            raise NotImplementedError

    
    def ode_fn(self, t: Tensor, x: Tensor, **kwargs):
        if t.numel() == 1:
            t = t.expand(x.size(0))

        rgb_latent = kwargs.pop('rgb_latent')
        # class_labels = kwargs.pop('class_labels')
        # this order is important
        if self.unet.config.in_channels == 8 or self.unet.config.in_channels == 12:
            unet_input = torch.cat([x, rgb_latent], dim=1)
        else:
            unet_input = x

        return self.unet(
            sample=unet_input, 
            timestep=t, 
            # class_labels=class_labels,
        **kwargs).sample


    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (torch.Tensor):
                Input RGB image to be encoded.

        Returns:
            torch.Tensor: Image latent
        """
        # encode
        # h = self.vae.encoder(rgb_in)
        # moments = self.vae.quant_conv(h)
        if self.vae_version == 'base':
            
            posterior = self.vae.encode(rgb_in).latent_dist
            rgb_latent = posterior.mean * self.rgb_latent_scale_factor
        else:
            
            rgb_latent = self.vae.encode(rgb_in).latents 

        # mean, logvar =  posterior.mean, posterior.logvar
        # mean, logvar = torch.chunk(posteriormoments, 2, dim=1)
        # scale latent
        # rgb_latent = posterior.mean * self.rgb_latent_scale_factor
        return rgb_latent
    
    
    def decode_depth(self, target_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            target_latent (torch.Tensor):
                Depth latent to be decoded.

        Returns:
            torch.Tensor: Decoded depth map.
        """
        # scale latent
        # target_latent = target_latent / self.target_latent_scale_factor
        target_latent = target_latent / self.vae.config.scaling_factor

        # decode
        if self.vae_version=='base':
            z = self.vae.post_quant_conv(target_latent)
            stacked = self.vae.decoder(z)
        else:
            stacked = self.vae.decoder(target_latent)

        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean

    
    def decode_seg(self, seg_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            target_latent (torch.Tensor):
                Depth latent to be decoded.

        Returns:
            torch.Tensor: Decoded depth map.
        """
        # scale latent
        # decode
        # if self.segvae is not None:
        if self.decode_type == 'argmax':
            seg_latent = seg_latent / self.seg_latent_scale_factor
            seg = self.vae.decode(seg_latent).sample
            # seg = nn.functional.interpolate(seg, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            seg_latent = seg_latent / self.seg_latent_scale_factor
            z = self.vae.post_quant_conv(seg_latent)
            seg = self.vae.decoder(z)
            seg = seg.clip(-1, 1)

        return seg

    
    def decode_normal(self, normal_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            target_latent (torch.Tensor):
                Depth latent to be decoded.

        Returns:
            torch.Tensor: Decoded depth map.
        """
        # scale latent
        normal_latent = normal_latent / self.vae.config.scaling_factor
        # decode
        
        if self.vae_version=='base':
            z = self.vae.post_quant_conv(normal_latent)
            normal = self.vae.decoder(z)
        else:
            normal = self.vae.decoder(normal_latent)

        # self.unet.config.in_channels != 12:
        if self.normal_head is not None:
            normal_1 = self.normal_head(normal_latent).sample_normal #[-1]
            # normal = torch.cat([normal, normal_1], dim=-1)
            return normal_1
        else:
            return normal


