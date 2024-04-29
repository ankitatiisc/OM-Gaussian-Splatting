import torch
from scene import Scene
import os
import torch.nn.functional as F
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
# from hdbscan import HDBSCAN
from scipy.spatial.distance import cdist
colors = np.random.randint(0, 255, size=(8200, 3))

def do_cluster(rendered_object):
    # import pdb;pdb.set_trace()
    gt_ids = np.array([list(np.binary_repr(i, width=12)) for i in np.arange(256)], dtype=np.float32)
    pred_ids = rendered_object.cpu().numpy()
    distance_matrix = cdist(pred_ids, gt_ids, metric='euclidean')
    ids11 = np.argmin(distance_matrix,axis=1)
    t1,t2 = np.unique(ids11,return_counts =True)
    
    # ids = t1[t2>550]
    ids = t1
    gt_ids = np.array([list(np.binary_repr(i, width=12)) for i in ids], dtype=np.float32)
    distance_matrix = cdist(pred_ids, gt_ids, metric='euclidean')
    i_max = np.argmin(distance_matrix,axis=1)
    t1,t2 ,t3= np.unique(i_max,return_inverse=True,return_counts=True)
    
# pdb;pdb.set_trace()
    new_rendered_object = gt_ids[t2]
    return torch.from_numpy(new_rendered_object).to(rendered_object.device)
    
def save_rgb_masks(file_paths,image_array):
    for i, file_path in enumerate(file_paths):
        image_data = image_array[i]  # Assuming the order in file_paths corresponds to image_array

        # Save the image using OpenCV
        cv2.imwrite(file_path, image_data*225)
    return
def get_rgb_masks_temp(binary_tensor):
    decimal_tensor = torch.sum(binary_tensor * (2 ** torch.arange(binary_tensor.shape[-1])), dim=-1)
    # decimal_tensor = torch.sum(y * (2 ** torch.arange(y.shape[-1])), dim=-1)
    # Generate unique colors for each unique binary code
    unique_codes,counts = torch.unique(decimal_tensor,return_counts=True)
    
    unique_codes = unique_codes[counts>10]# removing the noise by selecting the object id whose count is > 500
    num_unique_codes = len(unique_codes)
    # print(counts)
    # Convert the binary tensor to a NumPy array
    binary_np = binary_tensor.cpu().numpy()
    # import pdb;pdb.set_trace()
    # Create an empty image to fill with colors
    image = np.zeros((binary_tensor.shape[0], 3), dtype=np.float32)
    # Fill the image with colors based on binary codes
    for i, code in enumerate(unique_codes):
        mask = decimal_tensor == code
        image[mask] = colors[int(code)]
    # import pdb;pdb.set_trace()
    return image
    
def get_rgb_masks(binary_tensor):
    decimal_tensor = torch.sum(binary_tensor * (2 ** torch.arange(binary_tensor.shape[-1])), dim=-1)
    # decimal_tensor = torch.sum(y * (2 ** torch.arange(y.shape[-1])), dim=-1)
    # Generate unique colors for each unique binary code
    unique_codes,counts = torch.unique(decimal_tensor,return_counts=True)
    
    unique_codes = unique_codes[counts>10]# removing the noise by selecting the object id whose count is > 500
    num_unique_codes = len(unique_codes)
    # print(counts)
    # Convert the binary tensor to a NumPy array
    binary_np = binary_tensor.cpu().numpy()
    # import pdb;pdb.set_trace()
    # Create an empty image to fill with colors
    image = np.zeros((binary_tensor.shape[0],binary_tensor.shape[1], 3), dtype=np.float32)
    image_instance = np.zeros((binary_tensor.shape[0],binary_tensor.shape[1]), dtype=np.int32)
    # Fill the image with colors based on binary codes
    for i, code in enumerate(unique_codes):
        mask = decimal_tensor == code
        image[mask] = colors[int(code)]
        image_instance[mask]= int(code)+1
    # import pdb;pdb.set_trace()
    return image,image_instance,decimal_tensor

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
    mask_instance_path_img = os.path.join(model_path, name, f'ours_{iteration}', "instancs_bw")
    mask_instance_path_1 = os.path.join(model_path, name, f'ours_{iteration}', "pred_semantics_org")
    mask_instance_path_2 = os.path.join(model_path, name, f'ours_{iteration}', "pred_surrogateid")
    video_path = os.path.join(model_path,'rendered_videos',f'ours_{iteration}')
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)
    makedirs(video_path, exist_ok=True)
    makedirs(mask_instance_path_1, exist_ok=True)
    makedirs(mask_instance_path_2, exist_ok=True)
    makedirs(mask_instance_path_img, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_output = render(view, gaussians, pipeline, background)
        rendering = render_output["render"]
        rendered_object = render_output['render_object'] # shape [O,H,W] O-no of objects
        rendered_object = rendered_object.permute(1,2,0).view(-1,12)
        
                # Use below line when you are applying sigmoid after Rasterization 
                # rendered_object = torch.sigmoid(rendered_object)
                # HDBSCAN(min_cluster_size=2000, min_samples=1, prediction_data=True, allow_single_cluster=True).fit(render_object.cpu())
                # Use below line when you are appling sigmoid before Rasterization
                # for idx1 in range(rendered_object.shape[0]): 
                #     rendered_object[idx] = torch.clip(rendered_object[idx1],0,1)
                # rendered_object = do_cluster(rendered_object)
                # dummy_rendered_object = torch.zeros_like(rendered_object)
                # dummy_rendered_object[rendered_object>0.5] = 1
                # # dummy_rendered_object[rendered_object<0] = 0
                # dummy_rendered_object = dummy_rendered_object.to('cpu').view(rendering.shape[1],rendering.shape[2],3)
                # # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        dummy_rendered_object = do_cluster(rendered_object.cpu()).view(rendering.shape[1],rendering.shape[2],12)
        dummy_rendered_object ,dummy_image_instance,decimal_tensor= get_rgb_masks(dummy_rendered_object)
        gt = view.original_image[0:3, :, :]
        # import pdb;pdb.set_trace()
        x,y = np.unique(decimal_tensor,return_counts =True)
        z = np.argmax(y)
        img = np.zeros_like(decimal_tensor)
        img[decimal_tensor!=x[z]] = 1
        # import pdb;pdb.set_trace()
        try:
            torchvision.utils.save_image(rendering, os.path.join(render_path,view.image_name + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path,view.image_name + ".png"))
            cv2.imwrite(os.path.join(mask_path,view.image_name + ".png"),dummy_rendered_object)
            cv2.imwrite(os.path.join(mask_instance_path_img,view.image_name + ".png"),np.array(img,dtype=np.int32)*255)
            np.save(os.path.join(mask_instance_path_1,view.image_name + ".npy"),dummy_image_instance)
            np.save(os.path.join(mask_instance_path_2,view.image_name + ".npy"),np.array(img,dtype=np.int32))
        except:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            cv2.imwrite(os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"),dummy_rendered_object)
            cv2.imwrite(os.path.join(mask_instance_path_img, '{0:05d}'.format(idx) + ".png"),np.array(img,dtype=np.int32)*255)
            np.save(os.path.join(mask_instance_path_1, '{0:05d}'.format(idx) + ".npy"),dummy_image_instance)
            np.save(os.path.join(mask_instance_path_2, '{0:05d}'.format(idx) + ".npy"),np.array(img,dtype=np.int32))
    
    images_to_video(render_path,os.path.join(video_path, f'{name}_rgb.mp4') )
    images_to_video(mask_path,os.path.join(video_path,f'{name}_mask.mp4') )
    images_to_video(mask_instance_path_img,os.path.join(video_path,f'{name}_bw.mp4') )
    
# To render each object separately if you want to use this function you need to add --save_decomp while training the scene
def render_decomp_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, max_objects : int):
    save_path  = os.path.join(dataset.model_path,'rendered_decomp_objs') 
    mkdir_p(save_path)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, max_objects = max_objects)
        obj_plys_path = os.path.join(dataset.model_path ,'decomp_objs','iteration_7000')
        obj_plys = os.listdir(obj_plys_path)
        obj_plys = sorted(obj_plys)
        for obj_ply in obj_plys:
            obj_ply_path  = os.path.join(obj_plys_path ,obj_ply)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,model_path = obj_ply_path)
            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
            render_set(save_path, obj_ply[:-4], scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            

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