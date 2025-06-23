#!/usr/bin/env python3

import os
import sys

def debug_scene_structure(model_path):
    """
    Debug function to check what files exist and help determine scene type
    """
    print(f"Debugging scene structure for: {model_path}")
    print("=" * 50)
    
    # Check if path exists
    if not os.path.exists(model_path):
        print(f"ERROR: Path {model_path} does not exist!")
        return
    
    # List all files and directories
    print("Contents of model directory:")
    for item in sorted(os.listdir(model_path)):
        item_path = os.path.join(model_path, item)
        if os.path.isdir(item_path):
            print(f"📁 {item}/")
            # List contents of subdirectories
            try:
                subcontents = os.listdir(item_path)
                for subitem in sorted(subcontents)[:5]:  # Show first 5 items
                    print(f"   - {subitem}")
                if len(subcontents) > 5:
                    print(f"   ... and {len(subcontents) - 5} more items")
            except PermissionError:
                print("   (Permission denied)")
        else:
            print(f"📄 {item}")
    
    print("\n" + "=" * 50)
    
    # Check for common scene type indicators
    print("Scene type detection:")
    
    # COLMAP indicators
    sparse_dir = os.path.join(model_path, "sparse")
    images_dir = os.path.join(model_path, "images")
    
    if os.path.exists(sparse_dir):
        print("✓ Found 'sparse' directory (COLMAP indicator)")
        # Check for COLMAP files in sparse directory
        sparse_contents = os.listdir(sparse_dir)
        colmap_files = ['cameras.txt', 'images.txt', 'points3D.txt', 
                       'cameras.bin', 'images.bin', 'points3D.bin']
        found_colmap = [f for f in colmap_files if f in sparse_contents]
        if found_colmap:
            print(f"  COLMAP files found: {found_colmap}")
        else:
            print(f"  Sparse directory contents: {sparse_contents}")
    else:
        print("✗ No 'sparse' directory found")
    
    if os.path.exists(images_dir):
        print("✓ Found 'images' directory")
        try:
            image_count = len([f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  Contains {image_count} image files")
        except:
            print("  Could not count images")
    else:
        print("✗ No 'images' directory found")
    
    # Blender indicators
    transforms_json = os.path.join(model_path, "transforms_train.json")
    if os.path.exists(transforms_json):
        print("✓ Found 'transforms_train.json' (Blender indicator)")
    else:
        print("✗ No 'transforms_train.json' found")
    
    # Other indicators
    cameras_json = os.path.join(model_path, "cameras.json")
    if os.path.exists(cameras_json):
        print("✓ Found 'cameras.json'")
    else:
        print("✗ No 'cameras.json' found")
    
    # Check what the scene detection might be looking for
    print("\n" + "=" * 50)
    print("Recommendations:")
    
    if os.path.exists(sparse_dir) and os.path.exists(images_dir):
        print("• This looks like a COLMAP dataset structure")
        print("• Check if the sparse directory contains the required COLMAP files")
        print("• Make sure sparse/0/ subdirectory exists with cameras.bin, images.bin, points3D.bin")
    elif os.path.exists(transforms_json):
        print("• This looks like a Blender/NeRF dataset structure")
    else:
        print("• Unable to determine scene type from file structure")
        print("• You may need to check the Scene class implementation to see what it expects")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_scene.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    debug_scene_structure(model_path)