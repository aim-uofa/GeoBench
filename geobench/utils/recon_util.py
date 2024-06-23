# used for mesh reconstruction
import os
import torch
import trimesh
from rembg import remove
from PIL import Image
import numpy as np
try:
    from geobench.utils.bilateral_normal_integration_cupy import bilateral_normal_integration
except:
    from geobench.utils.bilateral_normal_integration_numpy import bilateral_normal_integration

def sam_init(device):
    from transformers import SamModel, SamProcessor
    sam_processor = SamProcessor.from_pretrained("./data/weights/sam-vit-large")
    sam_model = SamModel.from_pretrained("./data/weights/sam-vit-large").to(device)
    return sam_processor, sam_model


def scale_img(img):
    width, height = img.size

    if min(width, height) > 512:
        scale = 512 / min(width, height)
        img = img.resize((int(width*scale), int(scale*height)), Image.LANCZOS)
    
    return img


def seg_foreground(img, device='cuda'):
    # img = Image.open(image_file)
    img = img.convert("RGB")
    org_size = img.size
    img = scale_img(img)

    image_rem = img.convert('RGBA') #
    print("after resize ", image_rem.size)
    image_nobg = remove(image_rem, alpha_matting=True)
    arr = np.asarray(image_nobg)[:,:,-1]
    x_nonzero = np.nonzero(arr.sum(axis=0))
    y_nonzero = np.nonzero(arr.sum(axis=1))
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())

    # center, scale = npz_dict['center'], npz_dict['scale'] * 200
    # bbox = np.concatenate([center-scale / 2, center+scale / 2])    
    sam_processor, sam_model = sam_init(device)
    bbox = np.array([x_min, y_min, x_max, y_max])

    # bbox = np.clip(bbox, 0, img.size[0])
    # input_points = [[[450, 600]]]  # 2D location of a window in the image
    input_boxes = bbox.reshape(1, 1, 4).tolist() # batch_size, num_boxes, 4

    inputs = sam_processor(img, input_boxes=input_boxes, return_tensors="pt").to(device)
    inputs['multimask_output'] = False

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    pred_mask = masks[0].reshape(1, img.size[1], img.size[0]).to(torch.uint8) * 255
    # masked_image, mask = sam_segment(sam_predictor, img.convert('RGB'), x_min, y_min, x_max, y_max)
    # mask = Image.fromarray(np.array(mask[-1]).astype(np.uint8) * 255)

    image = np.asarray(img)
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    mask = np.array(pred_mask[-1]).astype(np.uint8)
    out_image_bbox[:, :, 3] = mask

    masked_image = Image.fromarray(out_image_bbox, mode='RGBA').resize(org_size)
    mask = Image.fromarray(mask).resize(org_size)
    

    return masked_image, mask


def reconstruction(rgb_img, normal_np, name_base, output_dir):

    torch.cuda.empty_cache()
    masked_image, mask = seg_foreground(rgb_img)
    mask = np.array(mask) > 0.5
    # depth_np = np.load(files[0])
    # normal_np = np.load(files[1])

    h, w, _ = np.shape(normal_np)
    # dir_name = os.path.dirname(os.path.realpath(files[0]))
    dir_name = 'mesh'
    mask_output_temp = mask

    
    # normal_pred = 
    # name_base = os.path.splitext(os.path.basename(files[0]))[0][:-6]
    normal_np[:, :, 0] *= -1

    _, surface, _,  _, _ = bilateral_normal_integration(normal_np, mask_output_temp, k=2, K=None, max_iter=100, tol=1e-4, cg_max_iter=5000, cg_tol=1e-3)
    ply_path = os.path.join(output_dir, dir_name)
    os.makedirs(ply_path, exist_ok=True)

    ply_name = os.path.join(ply_path, f"{name_base}_recon.ply")
    surface.save(ply_name, binary=False)

    obj_name = ply_name.replace('ply', 'obj')
    mesh = trimesh.load(ply_name)
    T2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    mesh.apply_transform(T2)
    mesh.export(obj_name)
    torch.cuda.empty_cache()
    
    return obj_name, [ply_name], masked_image