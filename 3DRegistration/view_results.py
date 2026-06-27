#!/usr/bin/env python3
"""
Simple viewer for the registered point cloud PLY files using Open3D
"""
import open3d as o3d
import sys
import os

def view_ply(filepath):
    """Load and visualize a PLY file"""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False
    
    print(f"Loading {filepath}...")
    pcd = o3d.io.read_point_cloud(filepath)
    
    num_points = len(pcd.points)
    print(f"Loaded {num_points:,} points")
    
    # Check if the point cloud has colors
    if pcd.has_colors():
        print("Point cloud has RGB colors")
    
    # Visualize
    print("Opening visualization window...")
    print("Controls:")
    print("  - Left mouse: Rotate")
    print("  - Middle mouse / Ctrl+Left: Pan")
    print("  - Mouse wheel: Zoom")
    print("  - Q or ESC: Close window")
    
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"Point Cloud: {os.path.basename(filepath)}",
        width=1280,
        height=720,
        left=50,
        top=50
    )
    
    return True

def main():
    # Available PLY files
    ply_files = {
        '1': 'output.ply',
        '2': 'palm_1_output.ply',
        '3': 'palm_2_output.ply',
        '4': 'palm_3_output.ply'
    }
    
    if len(sys.argv) > 1:
        # File path provided as argument
        filepath = sys.argv[1]
        view_ply(filepath)
    else:
        # Interactive menu
        print("\n=== Open3D Point Cloud Viewer ===")
        print("\nAvailable point clouds:")
        print("  1. Box Can (40 frames, 183 MB)")
        print("  2. Palm 1 (25 frames, 182 MB)")
        print("  3. Palm 2 (12 frames, 43 MB)")
        print("  4. Palm 3 (25 frames, 192 MB)")
        print("\nEnter number (1-4) or 'all' to view all sequentially: ", end='')
        
        choice = input().strip().lower()
        
        if choice == 'all':
            for key in sorted(ply_files.keys()):
                filepath = ply_files[key]
                if os.path.exists(filepath):
                    print(f"\n{'='*50}")
                    view_ply(filepath)
                else:
                    print(f"Skipping {filepath} (not found)")
        elif choice in ply_files:
            filepath = ply_files[choice]
            view_ply(filepath)
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
