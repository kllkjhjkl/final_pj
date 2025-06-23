import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import imageio
import json
from utils.camera_utils import Camera

def debug_dataset_params(dataset):
    """Debug function to check dataset parameters"""
    print("Dataset parameters:")
    print("==================")
    print(f"Source path: {dataset.source_path}")
    print(f"Model path: {dataset.model_path}")
    print(f"Images: {dataset.images}")
    print(f"Resolution: {dataset.resolution}")
    print(f"White background: {dataset.white_background}")
    print(f"Data device: {dataset.data_device}")
    print(f"Eval: {dataset.eval}")
    
    # Check if source_path exists and what it contains
    if hasattr(dataset, 'source_path') and dataset.source_path:
        print(f"\nChecking source_path: {dataset.source_path}")
        if os.path.exists(dataset.source_path):
            print("✓ Source path exists")
            print(f"Contents: {os.listdir(dataset.source_path)}")
        else:
            print("✗ Source path does not exist")
    
    # Check model_path
    print(f"\nChecking model_path: {dataset.model_path}")
    if os.path.exists(dataset.model_path):
        print("✓ Model path exists")
        print(f"Contents: {os.listdir(dataset.model_path)}")
    else:
        print("✗ Model path does not exist")

def generate_circular_camera_path(cameras, num_frames=120):
    """
    Generate a circular camera path for video rendering
    Based on the training cameras' poses
    """
    print(f"Generating camera path with {num_frames} frames...")
    
    # Get camera positions and orientations
    positions = []
    rotations = []
    
    for camera in cameras:
        # Get world to camera matrix
        w2c = camera.world_view_transform.cpu().numpy()
        # Convert to camera to world
        c2w = np.linalg.inv(w2c)
        positions.append(c2w[:3, 3])
        rotations.append(c2w[:3, :3])
    
    positions = np.array(positions)
    
    # Calculate center and radius
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    radius = np.mean(distances) * 1.1  # Slightly larger for better view
    
    # Generate circular path
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
    
    # Use the first camera as reference for intrinsics
    ref_camera = cameras[0]
    
    spiral_cameras = []
    
    for i, angle in enumerate(angles):
        # Calculate new position on circle
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2] + (np.sin(angle * 2) * 0.3 * radius)  # Add some vertical motion
        
        # Create look-at matrix
        position = np.array([x, y, z])
        target = center
        up = np.array([0, 0, 1])  # Assuming Z is up
        
        # Calculate camera orientation
        forward = target - position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create rotation matrix
        rotation = np.column_stack([right, up, -forward])
        
        # Create transformation matrix
        c2w = np.eye(4)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = position
        
        # Convert to world-to-camera
        w2c = np.linalg.inv(c2w)
        
        # Create camera - try different constructor formats
        try:
            # Try format with data_device
            camera = Camera(
                colmap_id=i,
                R=w2c[:3, :3],
                T=w2c[:3, 3],
                FoVx=ref_camera.FoVx,
                FoVy=ref_camera.FoVy,
                image=torch.zeros_like(ref_camera.original_image),
                image_name=f"spiral_{i:03d}",
                uid=i,
                data_device=ref_camera.data_device
            )
        except TypeError:
            try:
                # Try format without data_device
                camera = Camera(
                    colmap_id=i,
                    R=w2c[:3, :3],
                    T=w2c[:3, 3],
                    FoVx=ref_camera.FoVx,
                    FoVy=ref_camera.FoVy,
                    image=torch.zeros_like(ref_camera.original_image),
                    image_name=f"spiral_{i:03d}",
                    uid=i
                )
            except TypeError:
                # Try minimal format
                camera = Camera(
                    colmap_id=i,
                    R=w2c[:3, :3],
                    T=w2c[:3, 3],
                    FoVx=ref_camera.FoVx,
                    FoVy=ref_camera.FoVy,
                    image=torch.zeros_like(ref_camera.original_image),
                    image_name=f"spiral_{i:03d}"
                )
        
        spiral_cameras.append(camera)
    
    print(f"Generated {len(spiral_cameras)} camera poses for spiral video")
    return spiral_cameras

def render_spiral_video(model_path, iteration, cameras, gaussians, pipeline, background):
    """
    Render a spiral video around the scene
    """
    print("Rendering spiral video...")
    
    # Generate spiral camera path
    spiral_cameras = generate_circular_camera_path(cameras, num_frames=120)
    
    # Create output directory
    video_dir = os.path.join(model_path, "video", f"ours_{iteration}")
    makedirs(video_dir, exist_ok=True)
    
    frames = []
    
    for idx, camera in enumerate(tqdm(spiral_cameras, desc="Rendering video frames")):
        try:
            # Render the view
            rendering = render(camera, gaussians, pipeline, background)["render"]
            
            # Convert to numpy and ensure correct format
            image = rendering.detach().cpu().numpy()
            
            # Convert from CHW to HWC format and scale to 0-255
            if image.shape[0] == 3:  # CHW format
                image = np.transpose(image, (1, 2, 0))
            
            # Ensure values are in [0, 1] range, then convert to [0, 255]
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
            
            frames.append(image)
            
            # Optionally save individual frames
            frame_path = os.path.join(video_dir, f"frame_{idx:04d}.png")
            imageio.imwrite(frame_path, image)
            
        except Exception as e:
            print(f"Error rendering frame {idx}: {e}")
            continue
    
    if len(frames) == 0:
        print("No frames were successfully rendered!")
        return
    
    # Save as video
    video_file = os.path.join(video_dir, "spiral_video.mp4")
    print(f"Saving video to: {video_file}")
    
    try:
        imageio.mimsave(video_file, frames, fps=30)
        print(f"Video saved successfully: {video_file}")
    except Exception as e:
        print(f"Error saving video: {e}")
        
        # Try saving as GIF as fallback
        gif_file = os.path.join(video_dir, "spiral_video.gif")
        try:
            imageio.mimsave(gif_file, frames, fps=10)
            print(f"Saved as GIF instead: {gif_file}")
        except Exception as e2:
            print(f"Failed to save as GIF too: {e2}")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    print(f"Saving renders to: {render_path}")
    print(f"Saving ground truth to: {gts_path}")
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]
        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, 
                skip_train: bool, skip_test: bool, separate_sh: bool, render_video: bool = False):
    
    print("==================================================")
    print(f"Rendering model: {dataset.model_path}")
    print(f"Source path: {dataset.source_path}")
    print(f"Iteration: {iteration}")
    print("==================================================")
    
    # Debug dataset parameters
    debug_dataset_params(dataset)
    
    # Fix the source path if it's not set correctly
    if not dataset.source_path or not os.path.exists(dataset.source_path):
        print(f"Warning: Source path '{dataset.source_path}' not found or not set")
        
        # Try using model_path as source_path if it contains the dataset structure
        if os.path.exists(os.path.join(dataset.model_path, "images")) and os.path.exists(os.path.join(dataset.model_path, "sparse")):
            print(f"Using model_path as source_path: {dataset.model_path}")
            dataset.source_path = dataset.model_path
        else:
            raise FileNotFoundError(f"Cannot find valid dataset at source_path: {dataset.source_path}")
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        
        print(f"Loading scene from: {dataset.source_path}")
        print(f"Loading model from: {dataset.model_path}")
        
        try:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            print("✓ Scene loaded successfully!")
        except Exception as e:
            print(f"Error loading scene: {e}")
            print("Trying alternative approach...")
            
            # Try creating a new dataset object with corrected paths
            class FixedDataset:
                def __init__(self, original_dataset):
                    # Copy all attributes
                    for attr in dir(original_dataset):
                        if not attr.startswith('_'):
                            setattr(self, attr, getattr(original_dataset, attr))
                    
                    # Ensure source_path is set correctly
                    if hasattr(self, 'model_path') and os.path.exists(os.path.join(self.model_path, "images")):
                        self.source_path = self.model_path
                        print(f"Fixed source_path to: {self.source_path}")
            
            fixed_dataset = FixedDataset(dataset)
            scene = Scene(fixed_dataset, gaussians, load_iteration=iteration, shuffle=False)
            print("✓ Scene loaded with fixed parameters!")
        
        # Print some info about the scene
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        print(f"✓ Training cameras: {len(train_cameras)}")
        print(f"✓ Test cameras: {len(test_cameras)}")
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            print("Rendering training views...")
            render_set(dataset.model_path, "train", scene.loaded_iter, train_cameras, 
                      gaussians, pipeline, background, dataset.train_test_exp, separate_sh)
        
        if not skip_test:
            print("Rendering test views...")
            render_set(dataset.model_path, "test", scene.loaded_iter, test_cameras, 
                      gaussians, pipeline, background, dataset.train_test_exp, separate_sh)
        
        if render_video:
            render_spiral_video(dataset.model_path, scene.loaded_iter, train_cameras,
                              gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_video", action="store_true", help="Render a spiral video")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), 
               args.skip_train, args.skip_test, False, args.render_video)