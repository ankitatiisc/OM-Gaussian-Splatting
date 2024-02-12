#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state,save_as_rgb_masks
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.system_utils import mkdir_p
import torch
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

colors = np.random.randint(0, 255, size=(300, 3))

def save_rgb_masks(file_paths,image_array):
    for i, file_path in enumerate(file_paths):
        image_data = image_array[i]  # Assuming the order in file_paths corresponds to image_array

        # Save the image using OpenCV
        cv2.imwrite(file_path, image_data*225)
    return

def get_rgb_masks(binary_tensor):
    decimal_tensor = torch.sum(binary_tensor * (2 ** torch.arange(binary_tensor.shape[-1])), dim=-1)

    # Generate unique colors for each unique binary code
    unique_codes,counts = torch.unique(decimal_tensor,return_counts=True)
    
    unique_codes = unique_codes[counts>500]# removing the noise by selecting the object id whose count is > 500
    num_unique_codes = len(unique_codes)
   
    # Convert the binary tensor to a NumPy array
    binary_np = binary_tensor.cpu().numpy()

    # Create an empty image to fill with colors
    image = np.zeros((binary_tensor.shape[0],binary_tensor.shape[1], 3), dtype=np.float32)
   
    # Fill the image with colors based on binary codes
    for i, code in enumerate(unique_codes):
        mask = decimal_tensor == code
        image[mask] = colors[int(code)]
    return image

def images_to_video(image_folder, video_name, fps=20):
    images =sorted( [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")])
    
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    
def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, f'ours_{iteration}',"renders")
    gts_path = os.path.join(model_path, name, f'ours_{iteration}', "gt")
    mask_path = os.path.join(model_path, name, f'ours_{iteration}', "mask")
    video_path = os.path.join(model_path,'rendered_videos',f'ours_{iteration}')
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)
    makedirs(video_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_output = render(view, gaussians, pipeline, background)
        rendering = render_output["render"]
        rendered_object = render_output['render_object'] # shape [O,H,W] O-no of objects

        rendered_object = rendered_object.permute(1,2,0).view(-1,8)

        # Use below line when you are applying sigmoid after Rasterization 
        # rendered_object = torch.sigmoid(rendered_object)
        
        # Use below line when you are appling sigmoid before Rasterization
        for idx in range(rendered_object.shape[0]):
            rendered_object[idx] = torch.clip(rendered_object[idx],0,1)
        
        dummy_rendered_object = torch.zeros_like(rendered_object)
        dummy_rendered_object[rendered_object>0.5] = 1
        dummy_rendered_object = dummy_rendered_object.to('cpu').view(rendering.shape[1],rendering.shape[2],8)
        # import pdb;pdb.set_trace()
        dummy_rendered_object = get_rgb_masks(dummy_rendered_object)
        gt = view.original_image[0:3, :, :]
       
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        cv2.imwrite(os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"),dummy_rendered_object)
    
    images_to_video(render_path,os.path.join(video_path, f'{name}_rgb.mp4') )
    images_to_video(mask_path,os.path.join(video_path,f'{name}_mask.mp4') )
    
# To render each object separately if you want to use this function you need to add --save_decomp while training the scene
def render_decomp_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, max_objects : int):
    save_path  = os.path.join(dataset.model_path,'rendered_decomp_objs') 
    mkdir_p(save_path)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, max_objects = max_objects)
        obj_plys_path = os.path.join(dataset.model_path ,'decomp_objs')
        obj_plys = os.listdir(obj_plys_path)
        obj_plys = sorted(obj_plys)
        for obj_ply in obj_plys:
            obj_ply_path  = os.path.join(obj_plys_path ,obj_ply)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,model_path = obj_ply_path)
            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
            render_set(save_path, obj_ply.split('.')[0], scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, max_objects : int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, max_objects = max_objects)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        gaussians.manipulation()
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--max_objects', type=int, default=50)
    parser.add_argument('--load_decomp',action='store_true',default=False)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    if args.load_decomp:
        render_decomp_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.max_objects)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.max_objects)