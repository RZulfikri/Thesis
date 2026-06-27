import argparse
from lib.process3d import process3d

if __name__ == "__main__":
    class ArgumentParserWithDefaults(argparse.ArgumentParser):
        def add_argument(self, *args, help=None, default=None, **kwargs):
            if help is not None:
                kwargs["help"] = help
            if default is not None and args[0] != "-h":
                kwargs["default"] = default
                if help is not None:
                    kwargs["help"] += " (default: {})".format(default)
            super().add_argument(*args, **kwargs)

    parser = ArgumentParserWithDefaults(description="TrueDepth camera point cloud registration",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("folder", help="folder containing bins and camera calibration")
    parser.add_argument("--viz", type=int, default=1, help="visualize result")

    method = "Registration method\n"
    method += "0: sequential ICP\n"
    method += "1: sequential ICP with loop closure\n"
    method += "2: sequential vision based\n"
    method += "3: sequential vision based with loop closure\n"

    parser.add_argument("--method", type=int, default=1, help=method)
    parser.add_argument("--output", default=None, help="save PLY file (default: result/<scan_name>/output.ply)")
    parser.add_argument("--width", type=int, default=640, help="image width")
    parser.add_argument("--height", type=int, default=480, help="image height")
    parser.add_argument("--min_depth", type=float, default=0.1, help="min depth distance")
    parser.add_argument("--max_depth", type=float, default=0.5, help="max depth distance")
    parser.add_argument("--max_point_dist", type=float, default=0.02, help="max distance between points for ICP/vision methods")
    parser.add_argument("--normal_radius", type=float, default=0.01, help="max radius for normal calculation for ICP methods")
    parser.add_argument("--min_matches", type=int, default=15, help="min matches for vision based method")
    parser.add_argument("--loop_closure_range", type=int, default=10, help="search N images from the start to find a loop closure with the last image")
    parser.add_argument("--uniform_color", type=int, default=0, help="use uniform color for point instead of RGB image")
    parser.add_argument("--max_vision_rmse", type=float, default=0.04, help="max rmse when estimating pose using vision")
    parser.add_argument("--mesh", type=int, default=0, help="make a mesh instead of point cloud")
    parser.add_argument("--mesh_method", default="point_faces", choices=["point_faces", "bpa", "poisson"], help="mesh reconstruction method")
    parser.add_argument("--mesh_depth", type=int, default=8, help="Poisson reconstruction depth, higher results in more detail")
    parser.add_argument("--keep_largest_mesh", type=int, default=1, help="keep only the largest mesh, useful for filtering noise")
    parser.add_argument("--voxel_size", type=float, default=0.001, help="voxel downsample size in meters before meshing")
    parser.add_argument("--outlier_nb_neighbors", type=int, default=30, help="neighbors for statistical outlier removal")
    parser.add_argument("--outlier_std_ratio", type=float, default=1.2, help="std ratio for statistical outlier removal")
    parser.add_argument("--radius_outlier_nb_points", type=int, default=12, help="min neighbors for radius outlier removal")
    parser.add_argument("--radius_outlier_radius", type=float, default=0.01, help="search radius for radius outlier removal")
    parser.add_argument("--mesh_density_quantile", type=float, default=0.02, help="drop low density mesh vertices by quantile [0-1)")
    parser.add_argument("--target_triangles", type=int, default=70000, help="target triangle count after simplification")
    parser.add_argument("--target_vertices", type=int, default=69000, help="target vertex count for point_faces output")
    parser.add_argument("--keep_mesh_color", type=int, default=0, help="keep mesh vertex colors")
    parser.add_argument("--keep_nearest_cluster", type=int, default=1, help="keep nearest point cluster before meshing")
    parser.add_argument("--cluster_connectivity_eps", type=float, default=0.008, help="tight DBSCAN eps for spatial connectivity check (palm isolation)")
    parser.add_argument("--cluster_connectivity_min_points", type=int, default=20, help="min neighbor points for tight connectivity DBSCAN")
    parser.add_argument("--cluster_eps", type=float, default=0.015, help="fallback DBSCAN eps if tight connectivity finds no clusters")
    parser.add_argument("--cluster_min_points", type=int, default=80, help="dbscan min points for fallback cluster filtering")
    parser.add_argument("--cluster_z_tolerance", type=float, default=0.06, help="z tolerance when selecting nearest foreground cluster (fallback only)")
    parser.add_argument("--foreground_depth_range", type=float, default=0.05, help="crop to z_min + this range (meters) before clustering, 0=disabled")
    parser.add_argument("--xy_clip", type=float, default=3.0,
                        help="XY centroid crop: buang titik > N sigma dari centroid dalam bidang XY setelah DBSCAN isolation, 0=disabled (default: 3.0)")
    parser.add_argument("--view_only", type=int, default=0, help="view the data only. Hit any key to go to the next image. ESCAPE to exit.")

    args = parser.parse_args()

    if args.output is None:
        import os
        scan_name = os.path.basename(os.path.normpath(args.folder))
        output_dir = os.path.join("result", scan_name)
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, "output.ply")

    process3d(args)
