import argparse
import os
from glob import glob
import logging

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import cv2

from geobench.pipelines.marigold_pipeline import MarigoldPipeline
from geobench.utils.seed_all import seed_all
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.schedulers import DDPMScheduler, LCMScheduler, DDIMScheduler
from diffusers import AutoencoderTiny

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Bingxin/Marigold",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--unet_ckpt_path",
        type=str,
        default=None,
        # default="./logs/logs_sd21_seg_pretrain/checkpoint-10000",
        help="Checkpoint path for unet.",
    )
    parser.add_argument(
        "--vae_ckpt_path",
        type=str,
        default=None,
        # default="./logs/logs_sd21_seg_pretrain/checkpoint-10000",
        help="Checkpoint path for unet.",
    )
    parser.add_argument(
        "--noise_scheduler_subfolder",
        type=str,
        default='scheduler',
        help="Path to scheduler.",
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['depth', 'seg', 'normal'],
        default="depth",
        help="inference mode.",
    )
    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=10,
        help="Diffusion denoising steps, more stepts results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--enable_lcm",
        action="store_true",
        help="Run with LCM.",
    )
    parser.add_argument(
        "--enable_log_normalization",
        action="store_true",
        help="Run with LCM.",
    )
    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )

    # other settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Flag of running on Apple Silicon.",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning(f"Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time
        seed = int(time.time())
        print('seed {}'.format(seed))

    seed_all(seed)

    # Output directories
    output_dir_color = os.path.join(output_dir, "{}_colored".format(args.mode))
    output_dir_tif = os.path.join(output_dir, "depth_bw")
    output_dir_npy = os.path.join(output_dir, "depth_npy")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_tif, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{input_rgb_dir}'")
        exit(1)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    if 'lcm' in args.noise_scheduler_subfolder or args.enable_lcm:
        scheduler = LCMScheduler.from_pretrained(
            checkpoint_path, 
            subfolder=args.noise_scheduler_subfolder,
        )
    else:
        scheduler = DDIMScheduler.from_pretrained(
            checkpoint_path, 
            subfolder=args.noise_scheduler_subfolder,
        )

    if args.vae_ckpt_path is not None:        
        vae = AutoencoderTiny.from_pretrained(args.vae_ckpt_path)
    else:
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=checkpoint_path,
            subfolder="vae",
            revision=None
        )


    if args.unet_ckpt_path is not None:
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_ckpt_path, 
            subfolder="unet", # "unet_lcm" if args.enable_lcm else 
            revision=args.non_ema_revision
        )
        pipe = MarigoldPipeline.from_pretrained(
            checkpoint_path, 
            vae=vae,
            unet=unet,
            text_encoder=None,
            torch_dtype=dtype,
            image_projector=None,
            image_encoder=None,
            scheduler=scheduler,
            # text_encoder=None,
            enable_log_normalization=args.enable_log_normalization,

            )
    else:
        pipe = MarigoldPipeline.from_pretrained(
            checkpoint_path, 
            vae=vae,
            torch_dtype=dtype,
            text_encoder=None,
            scheduler=scheduler,
            )     

    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    pipe = pipe.to(device)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for rgb_path in tqdm(rgb_filename_list, desc="Estimating {}".format(args.mode), leave=True):
            # Read input image
            input_image = Image.open(rgb_path)

            # Predict depth
            pipe_out = pipe(
                input_image,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                color_map=color_map,
                show_progress_bar=True,
                mode=args.mode,
            )

            if args.mode == 'depth':

                depth_pred: np.ndarray = pipe_out.depth_np
                depth_colored: Image.Image = pipe_out.depth_colored

                # Save as npy
                rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
                pred_name_base = rgb_name_base + "_depth_pred"
                npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
                if os.path.exists(npy_save_path):
                    logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
                np.save(npy_save_path, depth_pred)

                # Save as 16-bit uint png
                depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
                png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
                if os.path.exists(png_save_path):
                    logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
                Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

                # Colorize
                colored_save_path = os.path.join(
                    output_dir_color, f"{pred_name_base}_colored.png"
                )
                if os.path.exists(colored_save_path):
                    logging.warning(
                        f"Existing file: '{colored_save_path}' will be overwritten"
                    )
                depth_colored.save(colored_save_path)

            elif args.mode == 'seg':
                seg_colored: Image.Image = pipe_out.seg_colored
                rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
                pred_name_base = rgb_name_base + "_seg_pred"
                # Colorize
                colored_save_path = os.path.join(
                    output_dir_color, f"{pred_name_base}_steps_{denoise_steps}_colored.png"
                )
                if os.path.exists(colored_save_path):
                    logging.warning(
                        f"Existing file: '{colored_save_path}' will be overwritten"
                    )
                seg_res = Image.fromarray(np.concatenate([np.array(input_image), np.array(seg_colored)],axis=1))
                seg_res.save(colored_save_path)

            elif args.mode == 'normal':
                normal_colored: Image.Image = pipe_out.normal_colored
                rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
                pred_name_base = rgb_name_base + "_normal_pred_with_image"
                # Colorize
                colored_save_path = os.path.join(
                    output_dir_color, f"{pred_name_base}_colored.png"
                )
                if os.path.exists(colored_save_path):
                    logging.warning(
                        f"Existing file: '{colored_save_path}' will be overwritten"
                    )

                normal_colored.save(colored_save_path)
            else:
                raise NotImplementedError