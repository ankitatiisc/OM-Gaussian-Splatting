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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scipy.spatial.transform import Slerp, Rotation
import torch
import pyquaternion as pyquat
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    
    return np.array(c2w)
def read_cameras(meta):
    # K = np.array(meta["camera"]["K"]) # 3x3
    # K[0] *= W # multiplying first row by W
    # K[1] *= H # multiplying second row by H
    # K = np.abs(K) # a bit hacky... :/

    poses = []
    for i in range(len(meta["camera"]["positions"])):
        pose = np.eye(4)
        t = np.array(meta["camera"]["positions"][i])
        q = np.array(meta["camera"]["quaternions"][i])
        rot = pyquat.Quaternion(*q).rotation_matrix
        pose[:3, :3] = rot
        pose[:3, 3] = t
        # # we may need to convert blender convention to opencv convention
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pose = pose @ blender2opencv
        poses.append(pose)
    return  poses

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    mask:np.array
    mask_path:str
    image_name: str
    mask_flag:bool
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder,dataset_name):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        
        mask_path  = ''
        mask       = None
        mask_flag  = False
 
        if(dataset_name== 'mip360'):
            mask_path  = os.path.join(images_folder.rsplit('/', 1)[0],'masks',os.path.basename(extr.name))
        elif(dataset_name=='scannet'):
            mask_path  = os.path.join(images_folder.rsplit('/', 1)[0],'full', f"{extr.name.split('.')[0]}.instance-filt.png") 
        elif(dataset_name=='replica'):
            mask_path  = os.path.join(images_folder.rsplit('/', 1)[0],'masks', os.path.basename(extr.name))
        elif(dataset_name=='messy_room'):
            mask_path  = os.path.join(images_folder.rsplit('/', 1)[0],'instance',f"{extr.name.split('.')[0]}.npy")
            
        image_name = os.path.basename(image_path).split(".")[0]
        image      = Image.open(image_path)
        if os.path.isfile(mask_path):
            if dataset_name =='messy_room':
                mask = np.load(mask_path)
                # convert np array to PIL image
                mask = Image.fromarray(mask.astype(np.uint8))
            else:
                mask = Image.open(mask_path)
            mask_flag = True
        # import pdb;pdb.set_trace()
       
     
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path,mask = mask,mask_path = mask_path, image_name=image_name,mask_flag=mask_flag, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval,dataset_name, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir),dataset_name= dataset_name)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    #import pdb;pdb.set_trace()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background,dataset_name,load_360, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        #fovx = contents['fl_x']
        frames = contents["frames"]
        load_360=False
        if load_360:
            # To load 360 circle poses 
            poses = np.stack([pose_spherical(angle, -65.0, 7.0) for angle in np.linspace(0, 180, len(frames))], 0)

    
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            mask_flag =True
            mask_path =''
            if(dataset_name == 'dmnerf'):
                if(frame["file_path"][0] is 't'):
                    mask_path = os.path.join(path,'train/semantic_instance/semantic_instance_'+frame["file_path"][-4:]+extension)
                else:
                    mask_path = os.path.join(path,'val/semantic_instance/semantic_instance_'+frame["file_path"][-4:]+extension)
            elif(dataset_name == 'scannet'):
                mask_path = os.path.join(path,frame["file_path"]+'.instance-filt'+extension)

           
            if(load_360):# to generate circular translation of camera
                c2w = poses[idx]
            else:
                c2w = np.array(frame["transform_matrix"])
            
          
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            
            image_name =  Path(cam_name).stem
            
            image = Image.open(image_path)
            if os.path.isfile(mask_path):
                mask = Image.open(mask_path)   
            else:
                mask =None
                mask_flag = False
                
                
        
            im_data = np.array(image.convert("RGBA"))
            
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, mask=mask,mask_path =mask_path ,image_name=image_name,mask_flag=mask_flag, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval,dataset_name,load_360=False, extension=".png"):
    print("Reading Training Transforms")
    if(dataset_name=='replica'):
        train_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background,dataset_name, load_360,extension)
        test_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background,dataset_name, load_360,extension)
    else:
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background,dataset_name, load_360,extension)
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background,dataset_name, load_360,extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d1.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasOfMOSdataset(path, transformsfile, white_background,load_360, extension=".png"):
    random_train_val_ratio = 0.8
    subsample_frames=1
    world2scene = np.eye(4, dtype=np.float32)
    train_cam_infos = []
    test_cam_infos = []
    all_frame_names = sorted([x[:-4] for x in os.listdir(os.path.join(path , "color")) if x.endswith('.png')], key=lambda y: int(y) if y.isnumeric() else y)
    sample_indices = list(range(len(all_frame_names)))
    # poses = np.stack([pose_spherical(angle, -60.0, 50.0) for angle in np.linspace(0, 180, len(all_frame_names))], 0) 
        # use random_train_val_ratio to select last 20% as test set
        # this works because the frames were generated at random to begin with
        # also, always using last 20% means the test set is deterministic and fixed for all experiments
    test_indices = sample_indices[int(len(all_frame_names) * random_train_val_ratio):]
    train_indices = [sample_index for sample_index in sample_indices if sample_index not in test_indices]
    
    train_indices = train_indices[::subsample_frames]
    test_indices = test_indices[::subsample_frames]
    dims, intrinsics, cam2scene = [], [], []
    # img_h, img_w = np.array(Image.open(path / "color" / f"{all_frame_names[0]}.png")).shape[:2]
    metadata = json.load(open(os.path.join(path ,"metadata.json")))
    camera2world_list = read_cameras(metadata)
    # camera2world_list = poses 
    for idx,train_idx in enumerate(train_indices):
        c2w = camera2world_list[train_idx]
        # pose = world2scene @ cam2world
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        image_path = os.path.join(path , "color" , f"{all_frame_names[train_idx]}.png")
        image = Image.open(image_path)
        # if image.shape[-1] == 4:
        #     image = image[..., :3]
        mask_path = os.path.join(path , "instance" , f"{all_frame_names[train_idx]}.npy")
        mask = np.load(mask_path)
        # convert np array to PIL image
        mask = Image.fromarray(mask.astype(np.uint8))
        # import pdb;pdb.set_trace()
        W,H = image.size[0], image.size[1]
        K = np.array(metadata["camera"]["K"]) # 3x3
        K[0] *= W # multiplying first row by W
        K[1] *= H # multiplying second row by H
        K = np.abs(K) # a bit hacky... :/
        FovX = K[0][0]
        FovY=K[1][1]
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        train_cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, mask=mask,mask_path =mask_path ,image_name=all_frame_names[train_idx],
                        mask_flag=True, width=image.size[0], height=image.size[1]))
    for idx,test_idx in enumerate(test_indices):
        c2w = camera2world_list[test_idx]
        # pose = world2scene @ cam2world
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        image_path = os.path.join(path , "color" , f"{all_frame_names[test_idx]}.png")
        image = Image.open(image_path)
        # if image.shape[-1] == 4:
        #     image = image[..., :3]
        mask_path = os.path.join(path , "semantic" , f"{all_frame_names[test_idx]}.npy")
        mask = np.load(mask_path)
        # convert np array to PIL image
        mask = Image.fromarray(mask.astype(np.uint8))
        K = np.array(metadata["camera"]["K"]) # 3x3
        K[0] *= W # multiplying first row by W
        K[1] *= H # multiplying second row by H
        K = np.abs(K) # a bit hacky... :/
        FovX = K[0][0]
        FovY=K[1][1]
        
        test_cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, mask=mask,mask_path =mask_path ,image_name=all_frame_names[test_idx],
                        mask_flag=True, width=image.size[0], height=image.size[1]))
    return train_cam_infos,test_cam_infos
  

def readMOSdataset(path, white_background, eval,dataset_name,load_360=False, extension=".png"):
    train_cam_infos,test_cam_infos = readCamerasOfMOSdataset(path, "metadata.json", white_background,load_360,extension)
    
    
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d1.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info  

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    'MOS':readMOSdataset
}
