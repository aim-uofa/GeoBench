import cv2
import os
import torch
import logging
import requests
import argparse
import numpy as np

from glob import glob
from PIL import Image
from tqdm.auto import tqdm
from mmcv.utils import Config, DictAction

from geobench.models.metric3d.utils.do_test import transform_test_data_scalecano, get_prediction
from geobench.models.metric3d.model.monodepth_model import get_configured_monodepth_model
from geobench.models.metric3d.utils.running import load_ckpt
from geobench.models.metric3d.utils.transform import gray_to_colormap
from geobench.models.metric3d.utils.visualization import vis_surface_normal
from geobench.utils.image_util import chw2hwc, colorize_depth_maps

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


if "__main__" == __name__:

    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Bingxin/Marigold",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=['depth', 'seg', 'normal'],
        default="depth",
        help="inference mode.",
    )

    args = parser.parse_args()
    input_rgb_dir = args.input_rgb_dir
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint

    fx=1000.0
    fy=1000.0
    color_map = "Spectral"
    # color_map = None

    # -------------------- Model --------------------
    device = 'cuda'
    cfg = Config.fromfile('./geobench/models/metric3d/configs/HourglassDecoder/vit.raft5.large.py')
    model = get_configured_monodepth_model(cfg, )
    model, _,  _, _ = load_ckpt('./data/weights/metric3d/metric_depth_vit_large_800k.pth', model, strict_match=False)
    model.to(device)

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

    output_dir_color = os.path.join(output_dir, "colored")
    output_dir_gray = os.path.join(output_dir, "gray")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_gray, exist_ok=True)

    with torch.no_grad():
        
        for rgb_path in tqdm(rgb_filename_list, desc="Estimating {}".format(args.mode), leave=True):
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_depth_pred"
            rgb = Image.open(rgb_path).convert("RGB")
            img = np.array(rgb) 
            intrinsic = [fx, fy, img.shape[1]/2, img.shape[0]/2]
            rgb_input, cam_models_stacks, pad, label_scale_factor = \
                transform_test_data_scalecano(img, intrinsic, cfg.data_basic)
            rgb_input = rgb_input.to(device)
            cam_models_stacks = [cam_model.to(device) for cam_model in cam_models_stacks]

            with torch.no_grad():
                pred_depth, pred_depth_scale, scale, output, confidence = get_prediction(
                            model = model,
                            input = rgb_input,
                            cam_model = cam_models_stacks,
                            pad_info = pad,
                            scale_info = label_scale_factor,
                            gt_depth = None,
                            normalize_scale = cfg.data_basic.depth_range[1],
                            ori_shape=[img.shape[0], img.shape[1]],
                        )

            pred_depth = pred_depth.squeeze().cpu().numpy()
            pred_depth[pred_depth < 0] = 0

            if color_map is not None:
                colored_save_path = os.path.join(
                    output_dir_color, f"{pred_name_base}_depth.png"
                )
                pred_color = gray_to_colormap(pred_depth)
                Image.fromarray(pred_color).save(colored_save_path)

            else:
                gray_save_path = os.path.join(
                    output_dir_gray, f"{pred_name_base}_gray.png"
                )
                formatted = (depth_pred * 255 / np.max(depth_pred)).astype("uint8")        
                depth = Image.fromarray(formatted)
                depth.save(gray_save_path)

            if 'normal_out_list' in output:

                colored_save_path = os.path.join(
                    output_dir_color, f"{pred_name_base}_normal.png"
                )

                pred_normal = output['normal_out_list'][0][:, :3, :, :] 
                H, W = pred_normal.shape[2:]
                pred_normal = pred_normal[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]

                pred_normal = torch.nn.functional.interpolate(pred_normal, [img.shape[0], img.shape[1]], mode='bilinear').squeeze()
                pred_normal = pred_normal.permute(1, 2, 0)

                pred_normal[..., 0] = -pred_normal[..., 0]
                pred_normal[..., 1] = -pred_normal[..., 1]
                pred_normal[..., 2] = -pred_normal[..., 2]

                pred_color_normal = vis_surface_normal(pred_normal)
                pred_normal = pred_normal.cpu().numpy()
                Image.fromarray(pred_color_normal).save(colored_save_path)


