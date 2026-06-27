import open3d as o3d
import numpy as np
import cv2 as cv
import json
import struct
import base64
import time

class ImageDepth:
    def __init__(self,
        calibration_file,
        image_file,
        depth_file,
        width=640,
        height=480,
        min_depth=0.1,
        max_depth=0.5,
        normal_radius=0.1) :

        self.image_file = image_file if image_file is not None else depth_file
        self.calibration_file = calibration_file
        self.depth_file = depth_file
        self.width = width
        self.height = height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.normal_radius = normal_radius
        self.pose = np.eye(4, 4)

        self.load_calibration(calibration_file)
        self.create_undistortion_lookup()

        if image_file is not None:
            self.load_image(image_file)
        else:
            # Depth-only scan — image will be synthesized inside load_depth
            self.img = None
            self.img_undistort = None
            self.gray = None
            self.gray_undistort = None
        self.load_depth(depth_file)

    def load_calibration(self, file):
        with open(file) as f:
            data = json.load(f)

            if "lensDistortionLookup" in data:
                lensDistortionLookupBase64 = data["lensDistortionLookup"]
                inverseLensDistortionLookupBase64 = data["inverseLensDistortionLookup"]
                lensDistortionLookupByte = base64.decodebytes(lensDistortionLookupBase64.encode("ascii"))
                inverseLensDistortionLookupByte = base64.decodebytes(inverseLensDistortionLookupBase64.encode("ascii"))

                lensDistortionLookup = struct.unpack(f"<{len(lensDistortionLookupByte)//4}f",lensDistortionLookupByte)
                inverseLensDistortionLookup = struct.unpack(f"<{len(inverseLensDistortionLookupByte)//4}f",inverseLensDistortionLookupByte)

                self.lensDistortionLookup = lensDistortionLookup
                self.inverseLensDistortionLookup = inverseLensDistortionLookup
            else:
                self.lensDistortionLookup = None
                self.inverseLensDistortionLookup = None

            if "intrinsic" in data:
                self.intrinsic = np.array(data["intrinsic"]).reshape((3,3))
                self.intrinsic = self.intrinsic.transpose()
                
                ref_width = data.get("intrinsicReferenceDimensionWidth", data.get("width"))
            else:
                # Construct intrinsic from fx, fy, cx, cy
                fx = data["fx"]
                fy = data["fy"]
                cx = data["cx"]
                cy = data["cy"]
                self.intrinsic = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                ref_width = data["width"]

            self.scale = float(self.width) / ref_width
            self.intrinsic[0,0] *= self.scale
            self.intrinsic[1,1] *= self.scale
            self.intrinsic[0,2] *= self.scale
            self.intrinsic[1,2] *= self.scale

    def create_undistortion_lookup(self):
        if self.inverseLensDistortionLookup is None:
            # Identity mapping if no distortion info
            self.map_x, self.map_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            self.map_x = self.map_x.astype(np.float32)
            self.map_y = self.map_y.astype(np.float32)
            return

        xy_pos = [(x,y) for y in range(0, self.height) for x in range(0, self.width)]
        xy = np.array(xy_pos, dtype=np.float32).reshape(-1,2)

        # subtract center — must be [cx, cy] shape (2,) so broadcasting is correct
        center = np.array([self.intrinsic[0, 2], self.intrinsic[1, 2]], dtype=np.float32)
        xy -= center

        # calc radius from center
        r = np.sqrt(xy[:,0]**2 + xy[:,1]**2)

        # normalize radius
        max_r = np.max(r)
        norm_r = r / max_r

        # interpolate the scale
        table = self.inverseLensDistortionLookup
        num = len(table)
        scale = 1.0 + np.interp(norm_r*num, np.arange(0, num), table)

        new_xy = xy*np.expand_dims(scale, 1) + center

        self.map_x = new_xy[:,0].reshape((self.height, self.width)).astype(np.float32)
        self.map_y = new_xy[:,1].reshape((self.height, self.width)).astype(np.float32)

    def load_depth(self, file,):
        import os
        size = os.path.getsize(file)
        expected_pixels = self.width * self.height
        
        if size == expected_pixels * 4:
            depth = np.fromfile(file, dtype='float32')
        elif size == expected_pixels * 2:
            depth = np.fromfile(file, dtype='float16').astype(np.float32)
        else:
            # Fallback to original behavior but with warning
            print(f"Warning: Depth file size {size} does not match expected float16 or float32 size for {self.width}x{self.height}. Reading as float16.")
            depth = np.fromfile(file, dtype='float16').astype(np.float32)
        
        if depth.size > expected_pixels:
            depth = depth[:expected_pixels]


        # vectorize version, faster
        # all possible (x,y) position
        idx = np.arange(0, self.width*self.height)
        xy = np.zeros((self.width*self.height, 2), dtype=np.float32)

        xy[:,0] = np.mod(idx, self.width)
        xy[:,1] = idx // self.width

        # Synthesize grayscale from depth when no image was provided (depth-only scan)
        if self.img_undistort is None:
            d_vis = depth.reshape(self.height, self.width).clip(self.min_depth, self.max_depth)
            d_norm = ((d_vis - self.min_depth) / (self.max_depth - self.min_depth) * 255).astype(np.uint8)
            d_norm[depth.reshape(self.height, self.width) == 0] = 0
            self.img = d_norm
            self.gray = d_norm
            self.img_undistort = cv.remap(d_norm, self.map_x, self.map_y, cv.INTER_LINEAR)
            self.gray_undistort = self.img_undistort

        # remove bad values
        no_nan = np.invert(np.isnan(depth))
        depth1 = depth > self.min_depth
        depth2 = depth < self.max_depth
        idx = no_nan & depth1 & depth2
        xy = xy[np.where(idx)]
        if self.img_undistort.ndim == 2:
            # Grayscale input — replicate to 3 channels for Open3D compatibility
            gray_flat = self.img_undistort.flatten()[np.where(idx)] / 255.0
            rgb = np.stack([gray_flat, gray_flat, gray_flat], axis=1)
        else:
            rgb = self.img_undistort.reshape(-1, 3)[np.where(idx)] / 255.0

        self.mask = np.ones(self.height*self.width, dtype=np.uint8)*255
        self.mask[np.where(idx == False)] = 0
        self.mask = self.mask.reshape((self.height, self.width))

        # mask out depth buffer
        self.depth_map = depth
        self.depth_map[np.where(idx == False)] = -1000
        self.depth_map = self.depth_map.reshape((self.height, self.width, 1))
        self.depth_map_undistort = cv.remap(self.depth_map, self.map_x, self.map_y, cv.INTER_LINEAR)

        per = float(np.sum(idx==True))/len(depth)
        print(f"Processing {file}, keeping={np.sum(idx==True)}/{len(depth)} ({per:.3f}) points")

        depth = np.expand_dims(self.depth_map_undistort.flatten()[np.where(idx)],1)

        # project to 3D
        xyz, _, good_idx = self.project3d(xy)
        xyz = xyz[good_idx]
        rgb = rgb[good_idx]

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(xyz)
        self.pcd.colors = o3d.utility.Vector3dVector(rgb)

        # calc normal, required for ICP point-to-plane
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
        self.pcd.orient_normals_towards_camera_location()

    def load_image(self, file):
        print(f"Loading {file}")
        self.img = np.fromfile(file, dtype='uint8')

        num_pixels = self.width * self.height
        if self.img.size == num_pixels:
            # Grayscale (1 channel) — exported from iOS palm scanner
            self.img = self.img.reshape((self.height, self.width))
            self.gray = self.img
            self.img_undistort = cv.remap(self.img, self.map_x, self.map_y, cv.INTER_LINEAR)
            self.gray_undistort = self.img_undistort
        elif self.img.size == num_pixels * 4:
            self.img = self.img.reshape((self.height, self.width, 4))
            self.img = self.img[:, :, 0:3]
            # swap RB
            self.img = self.img[:, :, [2, 1, 0]]
            self.gray = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
            self.img_undistort = cv.remap(self.img, self.map_x, self.map_y, cv.INTER_LINEAR)
            self.gray_undistort = cv.remap(self.gray, self.map_x, self.map_y, cv.INTER_LINEAR)
        elif self.img.size == num_pixels * 3:
            self.img = self.img.reshape((self.height, self.width, 3))
            # swap RB
            self.img = self.img[:, :, [2, 1, 0]]
            self.gray = cv.cvtColor(self.img, cv.COLOR_RGB2GRAY)
            self.img_undistort = cv.remap(self.img, self.map_x, self.map_y, cv.INTER_LINEAR)
            self.gray_undistort = cv.remap(self.gray, self.map_x, self.map_y, cv.INTER_LINEAR)
        else:
            raise ValueError(f"Image size {self.img.size} does not match expected {self.width}x{self.height} with 1, 3, or 4 channels")

    def project3d(self, pts):
        # expect pts to be Nx2

        xy = np.round(pts).astype(int)

        fx = self.intrinsic[0,0]
        fy = self.intrinsic[1,1]
        cx = self.intrinsic[0,2]
        cy = self.intrinsic[1,2]

        depths = self.depth_map_undistort[xy[:,1], xy[:,0]]
        depths = np.expand_dims(depths, 1)
        good_idx = np.where((depths > self.min_depth) & (depths < self.max_depth))[0]

        pts -= np.array([cx, cy])
        pts /= np.array([fx, fy])
        pts *= depths
        pts = np.hstack((pts, depths))

        return pts, xy, good_idx
