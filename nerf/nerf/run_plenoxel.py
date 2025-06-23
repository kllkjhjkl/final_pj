import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False
def evaluate_test_set(H, W, K, i_test, images, poses, render_kwargs_test, chunk):
    """评估测试集并返回平均loss和psnr"""
    test_losses = []
    test_psnrs = []
    
    with torch.no_grad():
        for img_idx in i_test:
            pose = poses[img_idx, :3, :4]
            gt_img = images[img_idx]
            
            rgb, _, _, _ = render(H, W, K, chunk=chunk, c2w=pose, **render_kwargs_test)
            
            # 计算loss和psnr
            img_loss = img2mse(rgb, gt_img)
            psnr = mse2psnr(img_loss)
            
            test_losses.append(img_loss.item())
            test_psnrs.append(psnr.item())
    
    return np.mean(test_losses), np.mean(test_psnrs)


def plot_and_save_curves(train_losses, train_psnrs, test_losses, test_psnrs, iterations, basedir, expname, current_iter):
    """绘制并保存loss和psnr曲线"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Train Loss
    ax1.plot(iterations, train_losses, 'b-', label='Train Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Test Loss
    ax2.plot(iterations, test_losses, 'r-', label='Test Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Test Loss')
    ax2.grid(True)
    ax2.legend()
    
    # Train PSNR
    ax3.plot(iterations, train_psnrs, 'b-', label='Train PSNR')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('PSNR')
    ax3.set_title('Training PSNR')
    ax3.grid(True)
    ax3.legend()
    
    # Test PSNR
    ax4.plot(iterations, test_psnrs, 'r-', label='Test PSNR')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('PSNR')
    ax4.set_title('Test PSNR')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    
    # 保存图片
    plot_dir = os.path.join(basedir, expname, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'curves_{current_iter:06d}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Curves saved to: {plot_dir}/curves_{current_iter:06d}.png")

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


class PlenoxelGrid(nn.Module):
    """Plenoxel voxel grid for storing density and color"""
    def __init__(self, grid_size=128, bbox_min=[-1,-1,-1], bbox_max=[1,1,1]):
        super().__init__()
        self.grid_size = grid_size
        self.bbox_min = torch.tensor(bbox_min, dtype=torch.float32)
        self.bbox_max = torch.tensor(bbox_max, dtype=torch.float32)
        
        # RGB + density (4 channels)
        self.voxel_grid = nn.Parameter(torch.zeros(grid_size, grid_size, grid_size, 4))
        # Initialize with small random values
        with torch.no_grad():
            self.voxel_grid.data.uniform_(-0.1, 0.1)
    
    def trilinear_interpolate(self, points):
        """Trilinear interpolation in the voxel grid"""
        # Normalize points to [0, 1]
        normalized_points = (points - self.bbox_min) / (self.bbox_max - self.bbox_min)
        
        # Scale to grid coordinates
        grid_coords = normalized_points * (self.grid_size - 1)
        
        # Get integer and fractional parts
        grid_coords_int = torch.floor(grid_coords).long()
        grid_coords_frac = grid_coords - grid_coords_int.float()
        
        # Clamp coordinates to valid range
        grid_coords_int = torch.clamp(grid_coords_int, 0, self.grid_size - 2)
        
        # Get 8 corner values
        x0, y0, z0 = grid_coords_int[..., 0], grid_coords_int[..., 1], grid_coords_int[..., 2]
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        
        dx, dy, dz = grid_coords_frac[..., 0:1], grid_coords_frac[..., 1:2], grid_coords_frac[..., 2:3]
        
        # Trilinear interpolation
        c000 = self.voxel_grid[x0, y0, z0]
        c001 = self.voxel_grid[x0, y0, z1]
        c010 = self.voxel_grid[x0, y1, z0]
        c011 = self.voxel_grid[x0, y1, z1]
        c100 = self.voxel_grid[x1, y0, z0]
        c101 = self.voxel_grid[x1, y0, z1]
        c110 = self.voxel_grid[x1, y1, z0]
        c111 = self.voxel_grid[x1, y1, z1]
        
        # Interpolate along x
        c00 = c000 * (1 - dx) + c100 * dx
        c01 = c001 * (1 - dx) + c101 * dx
        c10 = c010 * (1 - dx) + c110 * dx
        c11 = c011 * (1 - dx) + c111 * dx
        
        # Interpolate along y
        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy
        
        # Interpolate along z
        result = c0 * (1 - dz) + c1 * dz
        
        return result


def run_network_plenoxel(inputs, viewdirs, plenoxel_grid, embed_fn=None, embeddirs_fn=None, netchunk=1024*64):
    """Query the plenoxel grid instead of running a network"""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    
    # Query the voxel grid
    outputs_flat = plenoxel_grid.trilinear_interpolate(inputs_flat)
    
    # Apply sigmoid to RGB, keep density as is (will be processed in raw2outputs)
    rgb = torch.sigmoid(outputs_flat[..., :3])
    density = outputs_flat[..., 3:4]
    outputs_flat = torch.cat([rgb, density], -1)
    
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def get_3dgs_train_test_split(datadir, eval_mode=True, test_camera_indices=None):
    """
    Get the same train/test split as 3D Gaussian Splatting following their exact logic.
    
    Args:
        datadir: Path to dataset directory
        eval_mode: Whether in evaluation mode (matches 3DGS eval parameter)
        test_camera_indices: List of test camera indices if known (for manual override)
    
    Returns:
        i_train, i_test: Lists of training and testing indices
    """
    
    # Get total number of images first
    images_folder = os.path.join(datadir, 'images')
    if os.path.exists(images_folder):
        all_image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(all_image_files)
    else:
        print(f"Warning: Images folder not found at {images_folder}")
        total_images = 100  # Fallback
    
    # Method 1: Manual override if specific indices provided
    if test_camera_indices is not None:
        all_indices = set(range(total_images))
        i_test = test_camera_indices
        i_train = list(all_indices - set(i_test))
        return i_train, i_test
    
    if not eval_mode:
        # No test set in training mode
        i_train = list(range(total_images))
        i_test = []
        return i_train, i_test
    
    # Method 2: For LLFF datasets, use every 8th image as test
    llffhold = 8
    
    # Try to read from sparse/0/test.txt first (for datasets with explicit test specification)
    test_file_path = os.path.join(datadir, "sparse", "0", "test.txt")
    if os.path.exists(test_file_path):
        print(f"Loading test camera names from {test_file_path}")
        with open(test_file_path, 'r') as file:
            test_cam_names_list = [line.strip() for line in file]
        
        # Convert names to indices
        if os.path.exists(images_folder):
            all_image_files = sorted(all_image_files)
            name_to_index = {name: idx for idx, name in enumerate(all_image_files)}
            
            i_test = []
            for test_name in test_cam_names_list:
                found = False
                if test_name in name_to_index:
                    i_test.append(name_to_index[test_name])
                    found = True
                else:
                    # Try different extensions
                    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                        if test_name + ext in name_to_index:
                            i_test.append(name_to_index[test_name + ext])
                            found = True
                            break
                        elif test_name.replace(ext, '') in name_to_index:
                            i_test.append(name_to_index[test_name.replace(ext, '')])
                            found = True
                            break
                
                if not found:
                    print(f"Warning: Test camera name '{test_name}' not found in image files")
    else:
        # Default LLFF logic: every 8th image as test
        print(f"Using LLFF holdout logic: every {llffhold}th image as test")
        i_test = list(range(0, total_images, llffhold))
    
    # Ensure all test indices are within valid range
    i_test = [idx for idx in i_test if 0 <= idx < total_images]
    all_indices = set(range(total_images))
    i_train = list(all_indices - set(i_test))
    
    print(f"Found {len(i_test)} test images out of {total_images} total images")
    print(f"Test indices: {i_test}")
    
    return i_train, i_test


def save_test_results(basedir, expname, i_test, images, poses, hwf, K, render_kwargs_test, chunk):
    """
    Render and save test set results
    """
    print("Rendering and saving test set results...")
    
    # Create test results directory
    test_results_dir = os.path.join(basedir, expname, 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)
    
    # Create subdirectories
    rendered_dir = os.path.join(test_results_dir, 'rendered')
    gt_dir = os.path.join(test_results_dir, 'gt')
    os.makedirs(rendered_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    
    H, W, focal = hwf
    
    psnr_values = []
    ssim_values = []
    
    for idx, img_idx in enumerate(tqdm(i_test, desc="Rendering test images")):
        # Get pose and GT image
        pose = poses[img_idx, :3, :4]
        gt_img = images[img_idx]
        
        # Render image
        with torch.no_grad():
            rgb, disp, acc, extras = render(H, W, K, chunk=chunk, c2w=pose, **render_kwargs_test)
        
        # Convert to numpy and to 8-bit
        rendered_img = rgb.cpu().numpy()
        gt_img_np = gt_img.cpu().numpy() if torch.is_tensor(gt_img) else gt_img
        
        # Calculate metrics
        mse = np.mean((rendered_img - gt_img_np) ** 2)
        psnr = -10. * np.log10(mse)
        psnr_values.append(psnr)
        
        # Save images
        rendered_img_8bit = to8b(rendered_img)
        gt_img_8bit = to8b(gt_img_np)
        
        imageio.imwrite(os.path.join(rendered_dir, f'{img_idx:03d}.png'), rendered_img_8bit)
        imageio.imwrite(os.path.join(gt_dir, f'{img_idx:03d}.png'), gt_img_8bit)
        
        # Save depth map
        disp_np = disp.cpu().numpy()
        disp_normalized = (disp_np - disp_np.min()) / (disp_np.max() - disp_np.min())
        disp_8bit = to8b(disp_normalized)
        imageio.imwrite(os.path.join(rendered_dir, f'{img_idx:03d}_depth.png'), disp_8bit)
    
    # Calculate and save metrics
    mean_psnr = np.mean(psnr_values)
    std_psnr = np.std(psnr_values)
    
    print(f"Test set evaluation completed!")
    print(f"Mean PSNR: {mean_psnr:.2f} ± {std_psnr:.2f}")
    
    # Save metrics to file
    metrics = {
        'mean_psnr': float(mean_psnr),
        'std_psnr': float(std_psnr),
        'psnr_values': [float(p) for p in psnr_values],
        'test_indices': [int(i) for i in i_test]
    }
    
    with open(os.path.join(test_results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Test results saved to: {test_results_dir}")
    return mean_psnr, std_psnr


def create_plenoxel(args):
    """Instantiate Plenoxel model.
    """
    # Create plenoxel grid
    plenoxel_grid = PlenoxelGrid(
        grid_size=args.grid_size,
        bbox_min=[-args.scene_scale, -args.scene_scale, -args.scene_scale],
        bbox_max=[args.scene_scale, args.scene_scale, args.scene_scale]
    ).to(device)
    
    grad_vars = list(plenoxel_grid.parameters())

    # Network query function for plenoxels
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network_plenoxel(
        inputs, viewdirs, plenoxel_grid, netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        plenoxel_grid.load_state_dict(ckpt['plenoxel_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : 0,  # No hierarchical sampling for plenoxels
        'network_fine' : None,
        'N_samples' : args.N_samples,
        'network_fn' : plenoxel_grid,
        'use_viewdirs' : False,  # Simplified for plenoxels
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, plenoxel_grid


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = raw[...,:3]  # Already sigmoidized in run_network_plenoxel
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering for Plenoxels.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # Plenoxel specific options
    parser.add_argument("--grid_size", type=int, default=128, 
                        help='voxel grid size')
    parser.add_argument("--scene_scale", type=float, default=1.0, 
                        help='scene bounding box scale')

    # training options
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=1e-2, 
                        help='learning rate (higher for plenoxels)')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=5000, 
                        help='frequency of render_poses video saving')

    # New arguments for 3DGS compatibility
    parser.add_argument("--test_camera_indices", nargs="+", type=int, default=None,
                        help='specific test camera indices to match 3DGS split')
    parser.add_argument("--save_test_results", action='store_true',
                        help='save test set rendering results after training')
    parser.add_argument("--use_3dgs_split", action='store_true',
                        help='use exact 3DGS train/test split logic')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        # Use 3DGS compatible train/test split
        if args.test_camera_indices is not None:
            print(f'Using specified test camera indices: {args.test_camera_indices}')
            i_train, i_test = get_3dgs_train_test_split(
                args.datadir, 
                eval_mode=True,
                test_camera_indices=args.test_camera_indices
            )
        elif hasattr(args, 'use_3dgs_split') and args.use_3dgs_split:
            # Use exact 3DGS logic
            print('Using 3DGS train/test split logic')
            i_train, i_test = get_3dgs_train_test_split(
                args.datadir, 
                eval_mode=True
            )
        elif args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
            all_indices = set(range(images.shape[0]))
            i_train = list(all_indices - set(i_test))
        else:
            # Default: Use 3DGS split automatically
            print('Using 3DGS train/test split logic (default)')
            i_train, i_test = get_3dgs_train_test_split(
                args.datadir, 
                eval_mode=True
            )

        # Ensure i_test is a list and set validation set
        if not isinstance(i_test, list):
            i_test = list(i_test)
        i_val = i_test

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Print train/test split info
    print(f"Train images: {len(i_train)} (indices: {i_train})")
    print(f"Test images: {len(i_test)} (indices: {i_test})")
    
    # Save train/test split for reference
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    
    split_info = {
        'train_indices': [int(i) for i in i_train],
        'test_indices': [int(i) for i in i_test],
        'val_indices': [int(i) for i in i_val]
    }
    with open(os.path.join(basedir, expname, 'train_test_split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    # Create log dir and copy the config file
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create plenoxel model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, plenoxel_grid = create_plenoxel(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    # 初始化记录列表
    train_losses = []
    train_psnrs = []
    test_losses = []
    test_psnrs = []
    iterations = []

    N_iters = 30000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0

        # 每500轮记录数据并绘制曲线
        if i % 500 == 0 and i > 0:
            # 记录当前训练loss和psnr
            train_losses.append(loss.item())
            train_psnrs.append(psnr.item())
            iterations.append(i)
            
            # 计算测试集loss和psnr
            print(f"Evaluating test set at iteration {i}...")
            test_loss, test_psnr = evaluate_test_set(H, W, K, i_test, images, poses, render_kwargs_test, args.chunk)
            test_losses.append(test_loss)
            test_psnrs.append(test_psnr)
            
            # 绘制并保存曲线
            plot_and_save_curves(train_losses, train_psnrs, test_losses, test_psnrs, iterations, basedir, expname, i)
            
            print(f"[CURVES] Iter: {i}, Train Loss: {loss.item():.6f}, Test Loss: {test_loss:.6f}, Train PSNR: {psnr.item():.2f}, Test PSNR: {test_psnr:.2f}")

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'plenoxel_state_dict': plenoxel_grid.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1

    # Training completed - save test results if requested
    if args.save_test_results:
        print("\n" + "="*50)
        print("Training completed. Evaluating on test set...")
        print(f"Using test indices: {i_test}")
        print(f"Total test images: {len(i_test)}")
        mean_psnr, std_psnr = save_test_results(
            basedir, expname, i_test, images, poses, hwf, K, 
            render_kwargs_test, args.chunk
        )
        print("="*50)


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()