# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
from typing import Dict, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
    Transformer2DModel
)
from geobench.schedulers.scheduler_perflow import PeRFlowScheduler
from diffusers.utils import BaseOutput
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import resize, pil_to_tensor
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from geobench.utils.batchsize import find_batch_size
from geobench.utils.depth_ensemble import ensemble_depths
from geobench.utils.normal_ensemble import ensemble_normals
from geobench.utils.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
    norm_to_rgb
)
from geobench.pipelines.flow_pipeline import q_sample
import cv2
from diffusers import AutoencoderTiny
import einops

def per_sample_min_max_normalization(x):
    """ Normalize each sample in a batch independently
    with min-max normalization to [0, 1] """
    bs, *shape = x.shape
    x_ = einops.rearrange(x, "b ... -> b (...)")
    min_val = einops.reduce(x_, "b ... -> b", "min")[..., None]
    max_val = einops.reduce(x_, "b ... -> b", "max")[..., None]
    x_ = (x_ - min_val) / (max_val - min_val)
    return x_.reshape(bs, *shape)


class MarigoldOutput(BaseOutput):

    normal_np: np.ndarray
    normal_colored: Image.Image
    uncertainty: Union[None, np.ndarray]
    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]


class MarigoldNormalOutput(BaseOutput):

    normal_np: np.ndarray
    normal_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]
    uncertainty: Union[None, np.ndarray]


class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
    """

    # rgb_latent_scale_factor = 0.18215
    # depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler, PeRFlowScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        enable_log_normalization = False
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.enable_log_normalization = enable_log_normalization
        self.empty_text_embed = None
        self.rgb_latent_scale_factor = vae.config.scaling_factor
        self.depth_latent_scale_factor = vae.config.scaling_factor

        if isinstance(self.unet, Transformer2DModel):
            self.sd_version = 'pixart'
        elif isinstance(self.unet, UNet2DConditionModel) and vae.config.scaling_factor == 0.13025:
            self.sd_version = 'sdxl'
        else:
            self.sd_version = 'sd21'

        

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        mode: str = 'depth',
        q_sample_timestep: int = 400,
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            processing_res (`int`, *optional*, defaults to `768`):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            denoising_steps (`int`, *optional*, defaults to `10`):
                Number of diffusion denoising steps (DDIM) during inference.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")

            # if self.sd_version == 'pixart':
            #     original_width, original_height = input_image.size
            #     new_width  = round(original_width / 8) * 8
            #     new_height = round(original_height / 8) * 8
            #     input_image = input_image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)

            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image.squeeze()
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")

        input_size = rgb.shape
        # print(input_size)
        

        assert (
            3 == rgb.dim() and 3 == input_size[0]
        ), f"Wrong input shape {input_size}, expected [rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

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
            (batched_img,) = batch
            depth_pred_raw, normal_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
                mode=mode,
                q_sample_timestep=q_sample_timestep,
            )
            if depth_pred_raw is not None:
                depth_pred_ls.append(depth_pred_raw.detach().clone())
            if normal_pred_raw is not None:
                normal_pred_ls.append(normal_pred_raw.detach().clone())

            # depth_pred_ls.append(depth_pred_raw.detach())
       
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        if 'depth' in mode:
            depth_preds = torch.concat(depth_pred_ls, dim=0).squeeze()
            # ----------------- Test-time ensembling -----------------
            if ensemble_size > 1:
                depth_pred, pred_uncert = ensemble_depths(
                    depth_preds, **(ensemble_kwargs or {})
                )
            else:
                depth_pred = depth_preds
                pred_uncert = None
            

            # ----------------- Post processing -----------------
            # Scale prediction to [0, 1]
            min_d = torch.min(depth_pred)
            max_d = torch.max(depth_pred)
            depth_pred = (depth_pred - min_d) / (max_d - min_d)

            # Resize back to original resolution
            if match_input_res:
                depth_pred = resize(
                    depth_pred.unsqueeze(0),
                    input_size[1:],
                    interpolation=resample_method,
                    antialias=True,
                ).squeeze()

            # Convert to numpy
            depth_pred = depth_pred.cpu().numpy()

            # Clip output range
            depth_pred = depth_pred.clip(0, 1)

            # Colorize
            if color_map is not None:
                depth_colored = colorize_depth_maps(
                    depth_pred, 
                    min_depth=0, 
                    max_depth=1, 
                    cmap=color_map
                ).squeeze()  # [3, H, W], value in (0, 1)
                depth_colored = (depth_colored * 255).astype(np.uint8)
                depth_colored_hwc = chw2hwc(depth_colored)
                depth_colored_img = Image.fromarray(depth_colored_hwc)
            else:
                depth_colored_img = None
        else:
            depth_pred=None
            depth_colored_img=None
            pred_uncert = None

            # return MarigoldDepthOutput(
            #     depth_np=depth_pred,
            #     depth_colored=depth_colored_img,
            #     uncertainty=pred_uncert,
            # )


        if 'normal' in mode:

            normal_preds = torch.concat(normal_pred_ls, axis=0).squeeze(1)
            normal_preds = ensemble_normals(normal_preds)
        
            normal = normal_preds.clip(-1, 1).cpu().numpy().astype(np.float32) # [-1, 1]
            normal[0,:,:] = -normal[0,:,:]
            normal_colored = norm_to_rgb(normal)
            normal_colored_hwc = chw2hwc(normal_colored)
            
            normal_colored_img = Image.fromarray(normal_colored_hwc)
            normal = chw2hwc(normal)

            if match_input_res:
                # input_size = (rgb_norm.shape[2], rgb_norm.shape[1])

                normal_colored_img = normal_colored_img.resize((input_size[2],input_size[1]))
                # normal = np.asarray(Image.fromarray(normal).resize(input_size))
                
                normal = cv2.resize(normal, (input_size[2], input_size[1]), interpolation = cv2.INTER_NEAREST)
        else:
            normal = None
            normal_colored_img = None

        return MarigoldOutput(
                depth_np=depth_pred,
                depth_colored=depth_colored_img,
                normal_np=normal,
                normal_colored=normal_colored_img,
                uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler) or isinstance(self.scheduler, DPMSolverMultistepScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler) or isinstance(self.scheduler, PeRFlowScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        

        if self.sd_version == 'sdxl':
            self.empty_text_embed = torch.from_numpy(
            np.load('geobench/utils/text_embeds_sdxl.npy', allow_pickle=True)
            ).to(self.dtype).to(self.unet.device)
        elif 'pixart' in self.sd_version:
            self.empty_text_embed = torch.from_numpy(
            np.load('geobench/utils/text_embeds_t5.npy', allow_pickle=True)
            ).to(self.dtype).to(self.unet.device)
        elif self.text_encoder is None:
            self.empty_text_embed = torch.from_numpy(
            np.load('geobench/utils/text_embeds_sd21.npy', allow_pickle=True)
            ).to(self.dtype).to(self.unet.device)
        else:
            prompt = ""
            text_inputs = self.tokenizer(
                prompt,
                padding="do_not_pad",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.unet.device)
            self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)
        

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        mode: str,
        q_sample_timestep: int = 400,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_in = rgb_in.to(device)
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        if mode == 'depth' or mode == 'normal' or mode == 'disparity':
            # Initial depth map (noise)
            target_latent = torch.randn(
                rgb_latent.shape,
                device=device,
                dtype=self.dtype,
                generator=generator,
            )  # [B, 4, h, w]
        # elif mode == 'normal':
        #     target_latent = q_sample(rgb_latent, t=q_sample_timestep)

        elif mode == 'depth_and_normal' or mode == 'disparity_and_normal':
            normal_target_latent = q_sample(rgb_latent, t=q_sample_timestep)
            # target_latent = target_latent.repeat(1,2,1,1)
            target_latent = torch.cat([target_latent, normal_target_latent], dim=1)
        else:
            raise NotImplementedError

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()

        

        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]

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

        for i, t in iterable:
            if self.unet.config.in_channels == 8:
                unet_input = torch.cat(
                    [rgb_latent, target_latent], dim=1
                )  # this order is important
            else:
                unet_input = target_latent

            if self.sd_version == 'pixart':
                added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            else:
                added_cond_kwargs = {}

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, 
                timestep=t[None], 
                encoder_hidden_states=batch_empty_text_embed,
                added_cond_kwargs=added_cond_kwargs
            ).sample  # [B, 4, h, w]

            # if isinstance(self.unet, Transformer2DModel):
            if self.sd_version=='pixart' or isinstance(self.unet, Transformer2DModel):
                 noise_pred = noise_pred.chunk(2, 1)[0]
                 t = t[None]

            
            # compute the previous noisy sample x_t -> x_t-1
            
            target_latent = self.scheduler.step(
                noise_pred, t, target_latent, generator=generator
            ).prev_sample

            # torch.Size([1, 4, 96, 72]
            # print(noise_pred.shape, target_latent.shape)
            

        if mode == 'depth' or mode == 'disparity':
            depth = self.decode_depth(target_latent)

            if self.enable_log_normalization:
                # depth = depth.exp()
                depth = per_sample_min_max_normalization(depth.exp())
            else:
                # clip prediction
                depth = torch.clip(depth, -1.0, 1.0)
                # shift to [0, 1]
                depth = (depth + 1.0) / 2.0

            return depth, None

        elif mode == 'normal':

            normal = self.decode_normal(target_latent)
            normal /= (torch.norm(normal, p=2, dim=1, keepdim=True)+1e-5)
            # normal = torch.clip(normal, -1.0, 1.0)

            return None, normal

        elif mode == 'depth_and_normal' or mode == 'disparity_and_normal':
            depth_target_latent = target_latent[:,:4,:,:]
            normal_target_latent = target_latent[:,4:,:,:]

            depth = self.decode_depth(depth_target_latent)

            # clip prediction
            depth = torch.clip(depth, -1.0, 1.0)
            # shift to [0, 1]
            depth = (depth + 1.0) / 2.0

            normal = self.decode_normal(normal_target_latent)
            normal /= (torch.norm(normal, p=2, dim=1, keepdim=True)+1e-5)

            return depth, normal


    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        if isinstance(self.vae, AutoencoderTiny):
            return self.vae.encode(rgb_in).latents
        else:
            h = self.vae.encoder(rgb_in)
            moments = self.vae.quant_conv(h)
            mean, logvar = torch.chunk(moments, 2, dim=1)
            # scale latent
            rgb_latent = mean * self.rgb_latent_scale_factor
            return rgb_latent

    
    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        if isinstance(self.vae, AutoencoderTiny):
            z = depth_latent
        else:
            z = self.vae.post_quant_conv(depth_latent)

        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean

    
    def decode_normal(self, normal_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (torch.Tensor):
                Depth latent to be decoded.

        Returns:
            torch.Tensor: Decoded depth map.
        """
        # scale latent
        normal_latent = normal_latent / self.depth_latent_scale_factor
        # decode
        if isinstance(self.vae, AutoencoderTiny):
            z = normal_latent
        else:
            z = self.vae.post_quant_conv(normal_latent)

        normal = self.vae.decoder(z)

        return normal


class MarigoldXLPipeline(MarigoldPipeline):

    rgb_latent_scale_factor = 0.13025
    depth_latent_scale_factor = 0.13025
    seg_latent_scale_factor = 0.13025
    normal_latent_scale_factor = 0.13025

    config_name = "model_index.json"
    sd_version = 'sdxl'

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler, PeRFlowScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__(
            unet,
            vae,
            scheduler,
            text_encoder,
            tokenizer,
        )