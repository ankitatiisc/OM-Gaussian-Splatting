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
# import sys
# sys.path.append('/data/srinath/OMGS')

from threestudio.threestudio import find

import os
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import threestudio
import gc 
from torchvision import transforms
import time
import io
import matplotlib.pyplot as plt
# from ipywidgets import interact, IntSlider, Output
# from IPython.display import display, clear_output
from PIL import Image


import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def figure2image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def configure_other_guidance_params_manually(guidance, sds_config):
    # avoid reloading guidance every time change these params
    guidance.cfg.grad_clip = sds_config['guidance']['grad_clip']
    guidance.cfg.guidance_scale = sds_config['guidance']['guidance_scale']

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_latent(image_tensor, mask=True):
    # image_tensor = image_tensor.reshape([128, 128, 3]).permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)

    if mask==True:
        image_tensor[image_tensor>0] = 1
    return image_tensor

def get_latent_from_path(img_path,mask=True):
    # path = '/data/home/harshg/Person_Scene_new/threestudio/ bed1.jpg'
    
    init_image = Image.open(img_path).convert("RGB")    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize the image to 64x64
        transforms.ToTensor()         # Convert the resized image to tensor
    ])
    image_tensor = transform(init_image)
    # image_tensor = image_tensor.permute(1, 2, 0)
    image_tensor = image_tensor.unsqueeze(0)
    if mask==True:
        image_tensor[image_tensor>0] = 1
    return image_tensor, init_image

def get_latent_64(image_tensor, mask=True):
    # path = '/data/home/harshg/Person_Scene_new/threestudio/ bed1.jpg'
    
    # init_image = Image.open(img_path).convert("RGB")    

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize the image to 64x64
        # transforms.ToTensor()         # Convert the resized image to tensor
    ])
    # image_tensor = transform(image_tensor.permute(1, 2, 0))
    image_tensor = image_tensor.permute(1, 2, 0)
    image_tensor = torch.nn.functional.interpolate(image_tensor.reshape(1, 1, 512, 512), size=(64, 64), mode='bilinear', align_corners=False)[0]
    # image_tensor = image_tensor.permute(1, 2, 0)
    image_tensor = image_tensor.unsqueeze(0)
    if mask==True:
        image_tensor[image_tensor>0] = 1
    return image_tensor

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    prompt = "Cardboard box"
    negative_prompt = "distortion, blurry, incomplete, not blending with background"
    sds_config = {
            'max_iters': 1000,
            'seed': 42,
            'scheduler': 'cosine',
            'mode': 'latent',
            'prompt_processor_type': 'stable-diffusion-prompt-processor',
            'prompt_processor': {
                'prompt': prompt,
            },
            'guidance_type': 'stable-diffusion-guidance',
            'guidance': {
                'half_precision_weights': False,
                'guidance_scale': 100.,
                'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
                'grad_clip': None,
                'view_dependent_prompting': False,
            },
            'image': {
                'width': 64,
                'height': 64,
                # 'image': image_tensor,
                # 'mask' : mask_tensor[:,:1,:,:]
            },
            'title': None,
            'given_steps': 0,
            'x_center': 0.35,
            'y_center': 0.4,
        }


    seed_everything(sds_config['seed'])
    
    num_steps = sds_config['max_iters']
    scheduler = get_cosine_schedule_with_warmup(gaussians.optimizer, 100, int(num_steps*1.5)) if sds_config['scheduler'] == 'cosine' else None

    guidance = find(sds_config['guidance_type'])(sds_config['guidance'])
    prompt_processor = find(sds_config['prompt_processor_type'])(sds_config['prompt_processor'])
    prompt_processor.configure_text_encoder()

    configure_other_guidance_params_manually(guidance, sds_config)

    batch = {'elevation': torch.Tensor([0]), 'azimuth': torch.Tensor([0]), 'camera_distances': torch.Tensor([1])}

    mode = sds_config['mode']

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        img_1 = image.permute(1, 2, 0)
        img_1 = torch.nn.functional.interpolate(img_1.reshape(1, 3, 518, 800), size=(512, 512), mode='bilinear', align_corners=False)[0]

        image_tensor = get_latent(img_1, mask=False)

        # Loss        
        gt_image = viewpoint_cam.original_image.cuda()
        gt_mask = viewpoint_cam.original_mask.cuda()

        gt_mask = gt_mask.permute(1, 2, 0)
        _gt_mask = gt_mask
        gt_mask = torch.nn.functional.interpolate(gt_mask.reshape(1, 1, 518, 800), size=(512, 512), mode='bilinear', align_corners=False)[0]


        mask_tensor = get_latent(gt_mask)

        bg_target = image_tensor.to(guidance.device).permute(0,2,3,1)
        fg_target = image_tensor.to(guidance.device).permute(0,2,3,1)
        mask = mask_tensor[:,:1,:,:].to(guidance.device).permute(0,2,3,1)

        bg_target = guidance.encode_images(image_tensor.to(guidance.device)).permute(0,2,3,1)        
        fg_target = guidance.encode_images(image_tensor.to(guidance.device)).permute(0,2,3,1) 
        
        mask_tensor_8 = get_latent_64(gt_mask)

        mask_8 = mask_tensor_8[:,:1,:,:].to(guidance.device).permute(0,2,3,1) 
        target = (mask_8 * fg_target)
        
        sds_loss = 0
        Ll1 = l1_loss(image, gt_image) 
        
        mse_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image * (1- _gt_mask.permute(2,0,1)) , gt_image * (1- _gt_mask.permute(2,0,1))))

        if iteration > 100:
            sds_loss = guidance(target, prompt_processor(), **batch, rgb_as_latents=(mode != 'rgb')) 
        # sds_loss['loss_sds'].backward()
            loss = mse_loss.mean() * 100 + 0.0001 * sds_loss['loss_sds']
        else:
            loss = mse_loss.mean()

        loss.backward()

        iter_end.record()

        guidance.update_step(epoch=0, global_step=iteration)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
