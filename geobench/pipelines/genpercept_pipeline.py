from typing import List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import importlib

from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
    ControlNetModel,
    Transformer2DModel
)
from torchvision.transforms.functional import resize, pil_to_tensor
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from geobench.utils.image_util import (
    chw2hwc, 
    colorize_depth_maps, 
    resize_max_res, 
    norm_to_rgb,
    colorize_depth_maps,
    get_tv_resample_method,
)
from geobench.utils.batchsize import find_batch_size
from geobench.utils.depth_ensemble import ensemble_depths
# from geobench.models.image_projector import ImageProjModel
# from geobench.models.mask2former_head import Mask2FormerHead
from geobench.pipelines.flow_pipeline import q_sample

from geobench.utils.seg_util import (
    ade_palette, 
    hypersim_palette,
    seg_300_classes_palette)


import torch.nn.functional as F
import skimage
import cv2

# from kornia.color.lab import rgb_to_lab, lab_to_rgb

class GenPerceptNormalOutput(BaseOutput):

    normal_np: np.ndarray
    normal_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class GenPerceptDepthOutput(BaseOutput):
    """
    Output class for GenPercept monocular depth prediction pipeline.

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


class GenPerceptSegOutput(BaseOutput):
    """
    Output class for GenPercept monocular depth prediction pipeline.

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


class GenPerceptPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using GenPercept: https://arxiv.org/abs/2312.02145.

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
        text_encoder: CLIPTextModel = None,
        decode_type: str = 'argmax', # rgb
        # text_encoder: CLIPTextModel = None,
        # segvae: SegAutoencoderKL = None,
        seg_head = None,
        tokenizer_2: CLIPTokenizer = None,
        text_encoder_2: CLIPTextModel = None,
        text_embeds: torch.Tensor = None,
        image_encoder: CLIPVisionModelWithProjection = None,
        image_projector: ImageProjModel = None,
        controlnet: ControlNetModel = None,
        image_mean_std = [0.5, 0.5],
        gt_mean_std = [0., 1.],
        noise_type = 'rgb',
        noise_factor = 0.3,
        use_cold_diffusion = False,
        denoiser_type='unet',
        use_vae_decoder_features = False,
        use_lora = False,
        lora_ckpt_path = None,
        sd_version = 'sd21'
    ):
        super().__init__()

        self.sd_version = sd_version
        self.use_vae_decoder_features = use_vae_decoder_features
        self.image_mean_std = image_mean_std
        self.gt_mean_std = gt_mean_std
        self.noise_type = noise_type
        self.noise_factor = noise_factor
        self.use_cold_diffusion = use_cold_diffusion
        self.denoiser_type = denoiser_type

        if seg_head is not None:
            self.register_modules(seg_head=seg_head)
            self.seg_head.eval()
        else:
            self.seg_head = seg_head

        self.decode_type = decode_type

        # if segvae is not None:
        #     self.register_modules(segvae=segvae)
        # else:
        #     self.segvae = segvae

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
                image_projector=image_projector,
            )

        if text_encoder is not None:
            self.register_modules(text_encoder=text_encoder)
        else:
            self.text_encoder = None

        if text_encoder_2 is not None:
            self.register_modules(text_encoder_2=text_encoder_2)
        else:
            self.text_encoder_2 = None

        if tokenizer_2 is not None:
            self.register_modules(tokenizer_2=tokenizer_2)
        else:
            self.tokenizer_2 = None

        if image_encoder is not None:
            self.register_modules(
                image_encoder=image_encoder,
                image_projector=image_projector,
                )
            # cls.config_name = "model_index_image_encoder.json"
        else:
            self.image_encoder = None
            self.image_projector = None 

        self.empty_text_embed = text_embeds
        self.use_lora = use_lora

        if self.use_lora:
            self.orig_attn_procs = self.unet.attn_processors
            self.define_and_load_lora(lora_ckpt_path, reset_first=False)
            self.has_prepared_infer = False

    def define_and_load_lora(self, ckpt, reset_first=False, self_attn_only=None):
        if self_attn_only is not None:
            self.self_attn_only = self_attn_only
        else:
            self_attn_only = self.self_attn_only
        if reset_first:
            set_attn_processors(self.unet, self.orig_attn_procs)

        if ckpt and os.path.isdir(ckpt): # automatically define lora from the state dict
            self.unet.load_attn_procs(ckpt)
        else: # add new LoRA weights to the attention layers
            # It's important to realize here how many attention weights will be added and of which sizes
            # The sizes of the attention layers consist only of two different variables:
            # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
            # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

            # Let's first see how many attention processors we will have to set.
            # For Stable Diffusion, it should be equal to:
            # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
            # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
            # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
            # => 32 layers

            # Set correct lora layers
            self.lora_attn_procs = {}
            for name in self.unet.attn_processors.keys():
                if not self_attn_only or name.endswith('attn1.processor'):
                    cross_attention_dim = None if name.endswith('attn1.processor') else self.unet.config.cross_attention_dim
                    if name.startswith('mid_block'):
                        hidden_size = self.unet.config.block_out_channels[-1]
                    elif name.startswith('up_blocks'):
                        block_id = int(name[len('up_blocks.')])
                        hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                    elif name.startswith('down_blocks'):
                        block_id = int(name[len('down_blocks.')])
                        hidden_size = self.unet.config.block_out_channels[block_id]

                    self.lora_attn_procs[name] = LoRAAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        rank=4,
                    )
            set_attn_processors(self.unet, self.lora_attn_procs)
            if ckpt is not None:
                miss, unexp = self.unet.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)
                if len(unexp):
                    print('Unexpected:', unexp)

        # Convert the lora attention processors to xformers
        if self.enable_xformers:
            self.unet.enable_xformers_memory_efficient_attention()

        self.unet.to(self.device)
        self.lora_attn_procs = self.unet.attn_processors

        self.lora_layers = AttnProcsLayers(
                {n: p for n, p in self.unet.attn_processors.items() \
                if not self_attn_only or n.endswith('attn1.processor')})


    # def trainable_parameters(self):
    #     return self.lora_layers

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
    def __call__(
        self,
        input_image: Image,
        input_prompt: str = "",
        denoising_steps: int = 5,
        # num_inference_steps: int = 10,
        ensemble_size: int = 1,
        processing_res: int = 768,
        resample_method: str = "bilinear",
        match_input_res: bool = True,
        batch_size: int = 0,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        mode: str = 'depth',
    ) -> GenPerceptDepthOutput:
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
            `GenPerceptDepthOutput`
        """

        device = self.device
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
        # duplicated_clip_rgb = torch.stack([clip_image] * ensemble_size)

        single_rgb_dataset = TensorDataset(duplicated_rgb)
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
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader


        for batch in iterable:
            # (batched_img, batch_clip_img) = batch
            batched_img = batch[0]
            batched_clip_img = batch[0]

            depth_pred_raw = self.single_infer(
                prompt=[input_prompt] * batched_img.shape[0],
                rgb_in=batched_img,
                clip_rgb_in=batched_clip_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                mode=mode,
            )
            depth_pred_ls.append(depth_pred_raw.detach().clone())

        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if mode == 'depth':
            depth_preds = torch.concat(depth_pred_ls, axis=0).squeeze(1)
            if ensemble_size > 1:
                depth_pred, pred_uncert = ensemble_depths(
                    depth_preds, **(ensemble_kwargs or {})
                )
            else:
                depth_pred = depth_preds.mean(dim=0).repeat(3,1,1).permute(1,2,0).mean(dim=2)
                pred_uncert = None

        # ----------------- Post processing -----------------

        if mode == 'normal':
            depth_preds = torch.concat(depth_pred_ls, axis=0).squeeze(1)
            if len(depth_preds.shape) == 4:
                depth_preds = depth_preds.mean(dim=0)
        
            normal = depth_preds.clip(-1, 1).cpu().numpy() # [-1, 1]
            normal_colored = norm_to_rgb(normal)
            normal_colored_hwc = chw2hwc(normal_colored)
            
            org_h, org_w = input_size[1], input_size[2]
            normal_colored_img = Image.fromarray(normal_colored_hwc) #.resize(input_size)

            if match_input_res:
                normal_colored_img = normal_colored_img.resize((org_w, org_h))
                # normal = np.asarray(Image.fromarray(normal).resize(input_size))
                normal = cv2.resize(chw2hwc(normal).astype(np.float32), (org_w, org_h), interpolation = cv2.INTER_NEAREST)

            return GenPerceptNormalOutput(
                normal_np=normal,
                normal_colored=normal_colored_img,
                uncertainty=None,
            )

        if mode == 'depth':
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
                    depth_pred, 0, 1, cmap=color_map
                ).squeeze()  # [3, H, W], value in (0, 1)
                depth_colored = (depth_colored * 255).astype(np.uint8)
                depth_colored_hwc = chw2hwc(depth_colored)
                depth_colored_img = Image.fromarray(depth_colored_hwc)
            else:
                depth_colored_img = None

            return GenPerceptDepthOutput(
                depth_np=depth_pred,
                depth_colored=depth_colored_img,
                uncertainty=pred_uncert,
            )

        elif mode == 'seg':
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
              
            return GenPerceptSegOutput(
                seg_colored=seg_colored_img,
                seg_colored_wo_remap=seg_colored_img_wo_remap,
                uncertainty=None,
                argmax_output=pred_seg if self.decode_type == 'argmax' else None
            )

        else:
            raise NotImplementedError

    def encode_clip_text_feature(self, batched_prompt):
        """
        Encode text embedding for empty prompt
        """
        if self.sd_version == 'sd21':
            assert self.text_encoder is not None
            text_inputs = self.tokenizer(
                batched_prompt,
                padding="do_not_pad",
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

    
    @torch.no_grad()
    def single_infer(
        self, 
        prompt: list,
        rgb_in: torch.Tensor, 
        clip_rgb_in: torch.Tensor, 
        num_inference_steps: int, 
        show_pbar: bool,
        mode: str = 'depth',
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

        bsz = rgb_in.shape[0]
        device = self.device
        rgb_in = rgb_in.to(device)
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]
        if self.use_cold_diffusion:
            timestep_interval = timesteps[0] - timesteps[1]

        if self.scheduler.config.beta_start == 1.0 and self.scheduler.config.beta_end == 1.0:
            timesteps = torch.ones_like(timesteps)

        # Encode image
        torch.manual_seed(0)
        rgb_latent = self.encode_rgb(rgb_in)
        torch.manual_seed(0)

        # Initial depth map (noise)
        # target_latent = torch.randn(
        #     rgb_latent.shape, device=device, dtype=self.dtype)  # [B, 4, h, w]
        if self.noise_type == 'rgb':
            target_latent = rgb_latent.clone()
            if self.use_cold_diffusion:
                noise_sampled_init = target_latent.clone()

        elif self.noise_type == 'fix_gaussian':
            target_latent = torch.from_numpy(
                        np.load('geobench/utils/fix_noise.npy', allow_pickle=True)
                    )
            size = (rgb_latent.shape[2], rgb_latent.shape[3])
            target_latent = F.interpolate(target_latent, size).repeat(
                rgb_latent.shape[0], 1, 1, 1).to(self.device, dtype=rgb_latent.dtype)
        elif self.noise_type == 'rgb_mix_gaussian':
            # gaussian_noise = torch.randn_like(rgb_latent)
            # target_latent = (rgb_latent + gaussian_noise) / 2
            # target_latent = rgb_latent
            # target_latent = gaussian_noise
            # noise_factor = 0.3
            # q_sample(rgb_latent, timesteps)
            target_latent = q_sample(rgb_latent, t=torch.ones_like(timesteps[[0]]) * 400)
            # target_latent = (1 - self.noise_factor) * rgb_latent + self.noise_factor * gaussian_noise

        elif self.noise_type == 'random_gaussian':
            target_latent = torch.randn_like(rgb_latent, device=device, dtype=self.dtype)
        else:
            raise NotImplementedError

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()

        # batch_empty_text_embed = self.empty_text_embed.repeat(
        #     (rgb_latent.shape[0], 1, 1)
        # ).to(device)  # [B, 2, 1024]
        encoder_hidden_states = self.empty_text_embed.repeat(
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

        if self.denoiser_type == 'dit':
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        else:
            added_cond_kwargs = None

        # if self.seg_head is not None and self.seg_head.has_time_embed:
        #     seg_time_embed = self.seg_head.get_time_embed
        # else:
        #     seg_time_embed = None

        for i, t in iterable:
            # print('timesteps', t)
            if self.controlnet is None:
                if self.unet.config.in_channels == 8:
                    unet_input = torch.cat(
                        [rgb_latent, target_latent], dim=1
                    )  # this order is important
                else:
                    unet_input = target_latent

                # predict the noise residual
                unet_pred = self.unet(
                    unet_input.float(), 
                    timestep=t[None], 
                    # seg_time_embed=seg_time_embed,
                    encoder_hidden_states=encoder_hidden_states.float(),
                    added_cond_kwargs=added_cond_kwargs,
                )  # [B, 4, h, w]
                noise_pred = unet_pred.sample # [B, 4, h, w]
            
                if self.sd_version == 'pixart' or isinstance(self.unet, Transformer2DModel):
                    noise_pred = noise_pred.chunk(2, 1)[0]
                    t = t[None]

            else:
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    target_latent, # 10, 4, 96, 72
                    timestep=t, # 
                    encoder_hidden_states=encoder_hidden_states, # 10, 2, 1024
                    controlnet_cond=rgb_in, # 10, 3, 768, 576
                    return_dict=False,
                )

                # Predict the noise residual
                unet_pred = self.unet(
                    target_latent,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=self.dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=self.dtype),
                )
                noise_pred = unet_pred.sample # [B, 4, h, w]
                if self.unet.config.out_channels / 2 == rgb_latent.shape[1]:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

            if self.scheduler.config.beta_start == 1.0 and self.scheduler.config.beta_end == 1.0:
                # target_latent = res.pred_original_sample
                if self.unet.config.out_channels / 2 == rgb_latent.shape[1]:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                
                target_latent = -noise_pred
                break

            
            res = self.scheduler.step(noise_pred, t, target_latent)
            if self.use_cold_diffusion:
                if (t - 1 ) != 0:
                    x0_pred_step_t = res.pred_original_sample # x0
                    batch_size = x0_pred_step_t.shape[0]
                    timesteps_improved_sampling = torch.full((batch_size,), t-1, dtype=torch.long, device=x0_pred_step_t.device)
                    xt_bar = self.scheduler.add_noise(x0_pred_step_t, noise_sampled_init, timesteps_improved_sampling)
                    xt_sub1_bar = self.scheduler.add_noise(x0_pred_step_t, noise_sampled_init, timesteps_improved_sampling - timestep_interval)
                    target_latent = target_latent - xt_bar + xt_sub1_bar
                else:
                    target_latent = res.prev_sample # [B, 4, h, w]
            else:
                target_latent = res.prev_sample
                # target_latent_org = res.pred_original_sample

        if mode == 'depth':
            depth = self.decode_depth(target_latent)
            # clip prediction
            if self.gt_mean_std[0] == 0.5 and self.gt_mean_std[1] == 0.5:
                depth = torch.clip(depth, -1.0, 1.0)
                # shift to [0, 1]
                depth = (depth * 0.5) + 0.5
            else:
                depth = torch.clip(depth, 0.0, 1.0)

            return depth

        elif mode == 'normal':
            normal = self.decode_normal(target_latent)
            return normal
            
        else:
            raise NotImplementedError


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
        
        posterior = self.vae.encode(rgb_in).latent_dist
        # mean, logvar =  posterior.mean, posterior.logvar
        # mean, logvar = torch.chunk(posteriormoments, 2, dim=1)
        # scale latent
        rgb_latent = posterior.mean * self.rgb_latent_scale_factor
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
        target_latent = target_latent / self.target_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(target_latent)
        stacked = self.vae.decoder(z)

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
        normal_latent = normal_latent / self.normal_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(normal_latent)
        seg = self.vae.decoder(z)

        return seg


class GenPerceptXLPipeline(GenPerceptPipeline):

    rgb_latent_scale_factor = 0.13025
    target_latent_scale_factor = 0.13025
    seg_latent_scale_factor = 0.13025
    normal_latent_scale_factor = 0.13025

    config_name = "model_index.json"
    sd_version = 'sdxl'

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        tokenizer_2: CLIPTokenizer,
        text_encoder_2: CLIPTextModel,
        seg_head = None,
        text_embeds: torch.Tensor = None,
        image_encoder: CLIPVisionModelWithProjection = None,
        image_projector: ImageProjModel = None,
        controlnet: ControlNetModel = None,
    ):
        super().__init__(
            unet,
            vae,
            scheduler,
            tokenizer,
            text_encoder,
            seg_head,
            tokenizer_2,
            text_encoder_2,
            text_embeds,
            image_encoder,
            image_projector,
            controlnet,
        )