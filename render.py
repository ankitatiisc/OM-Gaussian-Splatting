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

def save_rgb_masks(file_paths,image_array):
    for i, file_path in enumerate(file_paths):
        image_data = image_array[i]  # Assuming the order in file_paths corresponds to image_array

        # Save the image using OpenCV
        cv2.imwrite(file_path, image_data*225)
    return

def get_rgb_masks(image_tensor,feature_tensor):
    num_images = image_tensor.shape[0]
    image_flat = image_tensor.view(num_images, -1, 3).cpu().numpy()
    feature_flat = feature_tensor.view(num_images, -1, 8).cpu().numpy()

    # Perform k-means clustering on the feature vectors for each image
    num_clusters = 6  # You can adjust this based on the number of clusters you want
    cluster_labels_all_images = []

    for i in range(num_images):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(feature_flat[i])
        cluster_labels_all_images.append(cluster_labels)

    # Create a colormap for visualization
    cmap = plt.get_cmap('tab10')
    
    # Assign consistent unique colors to each cluster across all images
    colored_images = np.zeros_like(image_flat, dtype=np.float32)
    for cluster_id in range(num_clusters):
        mask = np.array(cluster_labels_all_images) == cluster_id
        colored_images[ mask, :] = cmap(cluster_id)[:3]
   
    # Reshape the colored images back to the shape of the original images
    colored_images = colored_images.reshape(image_tensor.shape)
    return colored_images
def get_rgb_mask(image_tensor,feature_tensor):

    image_pixels = image_tensor.view(-1, 3)

    # Reshape the feature tensor to [num_pixels, num_features]
    feature_vector = feature_tensor.view(-1, 8)

    # Convert tensors to NumPy arrays
    image_pixels_np = image_pixels.cpu().numpy()
    feature_vector_np = feature_vector.cpu().numpy()

    # Perform k-means clustering on the feature vectors
    num_clusters = 6  # You can adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(feature_vector_np)
    # import pdb;pdb.set_trace()
    # Reshape the labels to match the original image shape
    cluster_labels = labels.reshape(image_pixels_np.shape[:1])

    # Create a color map for visualization
    cmap = plt.get_cmap('tab10')

    # Assign unique colors to each cluster
    colored_image = np.zeros_like(image_pixels_np, dtype=np.float32)
    for cluster_id in range(num_clusters):
        mask = (cluster_labels == cluster_id)
        colored_image[mask] = cmap(cluster_id)[:3]

    # Reshape the colored image back to the shape of the original image
    colored_image = colored_image.reshape(image_tensor.shape)
    return colored_image

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
    masks = []
    paths = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_output = render(view, gaussians, pipeline, background)
        rendering = render_output["render"]
        rendered_object = render_output['render_object'] # shape [O,H,W] O-no of objects
        # import pdb;pdb.set_trace()
        masks.append(rendered_object.permute(1,2,0))
        paths.append(os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))
        # rendered_object = get_rgb_mask(rendering.permute(1,2,0),rendered_object.permute(1,2,0))
        # rendered_object= torch.argmax(rendered_object,dim=0) # shape [H,W]
        
        gt = view.original_image[0:3, :, :]
       
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # save_as_rgb_masks(rendered_object, os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))
        # cv2.imwrite(os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"),rendered_object*225)
    import pdb;pdb.set_trace()
    x = torch.stack(masks,dim=0)
    rendered_objects = get_rgb_masks(x[...,:3],x)
    import pdb;pdb.set_trace()
    save_rgb_masks(paths,rendered_objects)
    images_to_video(render_path,os.path.join(video_path, f'{name}_rgb.mp4') )
    images_to_video(mask_path,os.path.join(video_path,f'{name}_mask.mp4') )
    

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