import torch
import numpy as np
from PIL import Image
import requests
import os
import argparse
import logging
from glob import glob
from tqdm.auto import tqdm
import mediapy as media
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from geobench.utils.image_util import (
    chw2hwc,
    colorize_depth_maps,
)
from geobench.modeling.archs.dam2 import DepthAnythingV2
from easydict import EasyDict

def read_video(video_path):
    return media.read_video(video_path)

def export_to_video(video_frames, output_video_path, fps):
    media.write_video(output_video_path, video_frames, fps=fps)

EXTENSION_LIST = [".jpg", ".jpeg", ".png", ".mp4", ".mov"]

if "__main__" == __name__:

    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--input_video_dir",
        type=str,
        default='/models/gyt/code/GeoBench/data/videos/ferris_wheel.mp4',
        # required=True,
        help="Path to the input image folder.",
    )
    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        # required=True,
        help="Path to the input image folder.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Bingxin/Marigold",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['video', 'image'],
        default="image",
        help="inference mode.",
    )
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument(
        "--version",
        type=str,
        choices=['v1', 'v2'],
        default="image",
        help="inference mode.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of frames to infer per forward",
    )

    args = parser.parse_args()
    cfg = EasyDict(vars(args))

    input_rgb_dir = args.input_rgb_dir
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint

    color_map = "Spectral"
    # color_map = 'inferno'
    # color_map = None
    device = 'cuda'

    # -------------------- Model --------------------
    if 'depth-anything-large-hf' in checkpoint_path:
        image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)
        model = AutoModelForDepthEstimation.from_pretrained(checkpoint_path).to(device)
    else:
        if args.version == 'v1':
            model = DepthAnything.from_pretrained(checkpoint_path).to(device)
        else:
            image_processor = AutoImageProcessor.from_pretrained("data/weights/depth-anything-large-hf")
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }

            model = DepthAnythingV2(**model_configs[args.encoder])
            model.load_state_dict(torch.load(f'data/weights/depth-anything-v2/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
            model = model.to(device).eval()

    # -------------------- Data --------------------
    if args.mode == 'video':
        data_ls = []
        video_data = read_video(args.input_video_dir)
        video_name = args.input_video_dir.split('/')[-1].split('.')[0]
        fps = video_data.metadata.fps
        depth_colored_pred = []

        first_clip_idx = 0
        last_clip_idx = len(video_data) - cfg.batch_size

        for i in tqdm(range(0, len(video_data)-cfg.batch_size+1, cfg.batch_size)):
            # if is_first_clip or is_last_clip:
            if i <= last_clip_idx:
                rgb_inputs, pads, label_scale_factors = [], [], []
                images = np.array(video_data[i: i+cfg.batch_size])
                bs, h, w, c = images.shape
                inputs = image_processor(images=images, return_tensors="pt")
                inputs['pixel_values'] = inputs['pixel_values'].to(device)
                with torch.no_grad():
                    if 'depth-anything-large-hf' in checkpoint_path:
                        outputs = model(**inputs)
                        predicted_depth = outputs.predicted_depth[None]
                    else:
                        predicted_depth = model(inputs['pixel_values'].to(device))

                # import pdb;pdb.set_trace()

                if predicted_depth.dim() == 3:
                    predicted_depth = predicted_depth.unsqueeze(1)

                # interpolate to original size
                prediction = torch.nn.functional.interpolate(
                    predicted_depth,
                    size=images.shape[1:3],
                    # size=np.array(image).shape[:2],
                    # mode="bicubic",
                    mode='nearest',
                    # align_corners=False,
                )
                # visualize the prediction
                pred_disps = prediction.squeeze().cpu().numpy() # [h, w]
                for pred_disp in pred_disps:
                    pred_disp[pred_disp<=10] = pred_disp[pred_disp>10].min()
                    # import pdb;pdb.set_trace()
                    pred_disp = 1 / pred_disp
                    pred_disp = (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min())
                    pred_color = colorize_depth_maps(
                        pred_disp, 0, 1, cmap=color_map
                    ).squeeze()  # [3, H, W], value in (0, 1)

                    pred_color = (pred_color * 255).astype(np.uint8)
                    pred_color = chw2hwc(pred_color)
                    depth_colored_pred.append(pred_color)

                    # print('slen', len(depth_colored_pred))
                # intrinsic = np.array([fx, fy, w/2, h/2]).repeat(bs, 1)
        
        depth_colored_pred = np.stack(depth_colored_pred, axis=0)
        os.makedirs(args.output_dir, exist_ok=True)
        # import pdb;pdb.set_trace()
        export_to_video(
            depth_colored_pred, 
            os.path.join(
            args.output_dir, 
            f"{video_name}_depth_bs{cfg.batch_size}.mp4"
            ), 
            fps
        )

    else:
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

        output_dir_color = os.path.join(output_dir, "colored")
        output_dir_gray = os.path.join(output_dir, "gray")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir_color, exist_ok=True)
        os.makedirs(output_dir_gray, exist_ok=True)

        with torch.no_grad():
            
            for rgb_path in tqdm(rgb_filename_list, desc="Estimating {}".format(args.mode), leave=True):
        
                rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
                pred_name_base = rgb_name_base + "_depth_pred"
                image = Image.open(rgb_path).convert('RGB')

                # prepare image for the model
                inputs = image_processor(images=image, return_tensors="pt")
                inputs['pixel_values'] = inputs['pixel_values'].to(device)
                with torch.no_grad():
                    if 'depth-anything-large-hf' in checkpoint_path:
                        outputs = model(**inputs)
                        predicted_depth = outputs.predicted_depth[None]
                    else:
                        predicted_depth = model(inputs['pixel_values'].to(device))

                # interpolate to original size
                prediction = torch.nn.functional.interpolate(
                    predicted_depth,
                    size=image.size[::-1],
                    mode='nearest',
                    align_corners=False,
                )
                # visualize the prediction
                pred_disp = prediction.squeeze().cpu().numpy() # [h, w]

                if color_map is not None:
                    colored_save_path = os.path.join(
                        output_dir_color, f"{pred_name_base}_colored.png"
                    )
                    pred_disp = (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min())

                    depth_colored = colorize_depth_maps(
                        pred_disp, 0, 1, cmap=color_map
                    ).squeeze()  # [3, H, W], value in (0, 1)

                    depth_colored = (depth_colored * 255).astype(np.uint8)
                    depth_colored_hwc = chw2hwc(depth_colored)
                    depth_colored_img = Image.fromarray(depth_colored_hwc)
                    depth_colored_img.save(colored_save_path)

                else:
                    gray_save_path = os.path.join(
                        output_dir_gray, f"{pred_name_base}_gray.png"
                    )
                    formatted = (output * 255 / np.max(output)).astype("uint8")        
                    depth = Image.fromarray(formatted)
                    depth.save(gray_save_path)
