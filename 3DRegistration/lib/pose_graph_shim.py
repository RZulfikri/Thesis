import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

def optimize_pose_graph_with_matches(poses_input, matches_input):
    # Handle inputs
    poses = np.array(poses_input)
    matches = np.array(matches_input)

    if poses.ndim == 1:
        num_poses = len(poses) // 7
        poses = poses.reshape((num_poses, 7))
    else:
        num_poses = poses.shape[0]
    
    if matches.ndim == 1:
        num_matches = len(matches) // 8
        matches = matches.reshape((num_matches, 8))
    
    # Check if we have enough poses
    if num_poses < 2:
        return poses
    
    # We fix the first pose (index 0)
    fixed_pose = poses[0].copy()
    initial_params = poses[1:].flatten()
    
    def get_full_poses(params):
        p = params.reshape((-1, 7))
        return np.vstack([fixed_pose, p])
    
    def cost_fun(params):
        current_poses = get_full_poses(params)
        
        # Pre-compute rotations and translations
        # Ceres uses [w, x, y, z], Scipy uses [x, y, z, w]
        # poses[:, :4] is [w, x, y, z]
        qs = current_poses[:, :4]
        # Convert to scipy format [x, y, z, w]
        qs_scipy = np.roll(qs, -1, axis=1)
        
        # Normalize quaternions to ensure validity
        norms = np.linalg.norm(qs_scipy, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms < 1e-9] = 1.0
        qs_scipy /= norms
        
        rotations = R.from_quat(qs_scipy)
        translations = current_poses[:, 4:]
        
        # matches: [id1, id2, x1, y1, z1, x2, y2, z2]
        id1s = matches[:, 0].astype(int)
        id2s = matches[:, 1].astype(int)
        pt1s = matches[:, 2:5]
        pt2s = matches[:, 5:8]
        
        rot_mats = rotations.as_matrix() # (N, 3, 3)
        
        R1s = rot_mats[id1s] # (M, 3, 3)
        t1s = translations[id1s] # (M, 3)
        
        R2s = rot_mats[id2s] # (M, 3, 3)
        t2s = translations[id2s] # (M, 3)
        
        # p1_world = (R1 * pt1) + t1
        # einsum for matrix-vector mult: 'ijk,ik->ij'
        p1_world = np.einsum('ijk,ik->ij', R1s, pt1s) + t1s
        p2_world = np.einsum('ijk,ik->ij', R2s, pt2s) + t2s
        
        diff = p1_world - p2_world
        return diff.flatten()

    res = least_squares(cost_fun, initial_params, verbose=2)
    
    final_poses = get_full_poses(res.x)
    
    # Normalize final quaternions
    qs = final_poses[:, :4]
    qs_scipy = np.roll(qs, -1, axis=1)
    norms = np.linalg.norm(qs_scipy, axis=1, keepdims=True)
    qs_scipy /= norms
    # Convert back to [w, x, y, z]
    final_poses[:, :4] = np.roll(qs_scipy, 1, axis=1)
    
    # Return structured array (N, 7)
    return final_poses

def optimize_pose_graph_with_odometry(poses_input, odometry_input):
    # Handle inputs
    poses = np.array(poses_input)
    odometry = np.array(odometry_input)

    if poses.ndim == 1:
        num_poses = len(poses) // 7
        poses = poses.reshape((num_poses, 7))
    else:
        num_poses = poses.shape[0]
    
    if odometry.ndim == 1:
        num_odo = len(odometry) // 9
        odometry = odometry.reshape((num_odo, 9))
    
    if num_poses < 2:
        return poses

    # Fix first pose
    fixed_pose = poses[0].copy()
    initial_params = poses[1:].flatten()
    
    def get_full_poses(params):
        p = params.reshape((-1, 7))
        return np.vstack([fixed_pose, p])
    
    def cost_fun(params):
        current_poses = get_full_poses(params)
        
        qs = current_poses[:, :4]
        # [w, x, y, z] -> [x, y, z, w]
        qs_scipy = np.roll(qs, -1, axis=1)
        norms = np.linalg.norm(qs_scipy, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        qs_scipy /= norms
        
        rotations = R.from_quat(qs_scipy)
        translations = current_poses[:, 4:]
        rot_mats = rotations.as_matrix() # (N, 3, 3)
        
        id1s = odometry[:, 0].astype(int)
        id2s = odometry[:, 1].astype(int)
        
        # Odometry relative pose: R_rel, t_rel
        # odom format: [id1, id2, qw, qx, qy, qz, tx, ty, tz]
        qs_rel = odometry[:, 2:6]
        qs_rel_scipy = np.roll(qs_rel, -1, axis=1) # [x, y, z, w]
        ts_rel = odometry[:, 6:9]
        
        Rs_rel = R.from_quat(qs_rel_scipy).as_matrix() # (K, 3, 3)
        
        R1s = rot_mats[id1s]
        t1s = translations[id1s]
        R2s = rot_mats[id2s]
        t2s = translations[id2s]
        
        # Translation error
        # t2_pred = R1 * t_rel + t1
        t2_pred = np.einsum('ijk,ik->ij', R1s, ts_rel) + t1s
        res_trans = t2s - t2_pred
        
        # Rotation error
        # R3 = R1 * R_rel
        # R_err = R3^T * R2
        # We want R_err to be identity
        R3s = np.matmul(R1s, Rs_rel)
        # Transpose R3s: swap axes 1 and 2
        R3s_T = np.transpose(R3s, axes=(0, 2, 1))
        R_errs = np.matmul(R3s_T, R2s)
        
        # Convert R_errs to axis-angle
        # Scipy as_rotvec returns axis-angle vector
        res_rot = R.from_matrix(R_errs).as_rotvec()
        
        return np.hstack([res_trans, res_rot]).flatten()

    res = least_squares(cost_fun, initial_params, verbose=2)
    
    final_poses = get_full_poses(res.x)
    
    # Normalize
    qs = final_poses[:, :4]
    qs_scipy = np.roll(qs, -1, axis=1)
    norms = np.linalg.norm(qs_scipy, axis=1, keepdims=True)
    qs_scipy /= norms
    final_poses[:, :4] = np.roll(qs_scipy, 1, axis=1)
    
    return final_poses
