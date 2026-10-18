import os
import torch
import numpy as np
import torch.nn as nn
import pytorch3d.transforms

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation



def load_pose_cl_from_file(file_path: str, calib_dir: str|None = None, cam_id: str = "00") -> np.ndarray:
    """
    Load an initial LiDAR-to-camera extrinsic matrix (T_cl, 4x4) from a file.
    Supports:
      1. Multi-camera calib_cam_to_pose.txt format (lines starting with 'image_XX: <12 floats>').
         Converts T_body_camXX -> T_cl using calib_cam_to_velo.txt and calib_cam_to_pose.txt in calib_dir.
      2. Single-matrix calib_cam_to_velo.txt format (12 floats for 3x4 or 16 floats for 4x4 T_lc),
         which is inverted to obtain T_cl = inv(T_lc).
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Initial calibration file not found: {file_path}")

    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Case 1: calib_cam_to_pose.txt format (contains 'image_XX:' prefixes)
    if any(line.startswith("image") for line in lines):
        target_prefix = f"image_{cam_id}:"
        T_body_cam_init = None
        T_body_cam00_from_file = None

        for line in lines:
            if line.startswith("image_00:"):
                parts = line.split()
                T_body_cam00_from_file = np.eye(4, dtype=np.float64)
                T_body_cam00_from_file[:3, :4] = np.array([float(x) for x in parts[1:]], dtype=np.float64).reshape(3, 4)
            if line.startswith(target_prefix):
                parts = line.split()
                T_body_cam_init = np.eye(4, dtype=np.float64)
                T_body_cam_init[:3, :4] = np.array([float(x) for x in parts[1:]], dtype=np.float64).reshape(3, 4)

        if T_body_cam_init is None:
            raise ValueError(f"Could not find entry '{target_prefix}' in {file_path}")

        # Resolve calib_dir to obtain T_body_velo
        if calib_dir is None:
            calib_dir = os.path.dirname(os.path.abspath(file_path))

        cam_to_velo_path = os.path.join(calib_dir, "calib_cam_to_velo.txt")
        cam_to_pose_path = os.path.join(calib_dir, "calib_cam_to_pose.txt")

        if not os.path.isfile(cam_to_velo_path):
            raise FileNotFoundError(
                f"Cannot convert {file_path} (cam-to-pose format) to T_cl without {cam_to_velo_path}"
            )

        with open(cam_to_velo_path, "r") as f:
            T_velo_cam00 = np.eye(4, dtype=np.float64)
            T_velo_cam00[:3, :4] = np.array([float(x) for x in f.read().split()], dtype=np.float64).reshape(3, 4)

        T_body_cam00 = T_body_cam00_from_file
        if os.path.isfile(cam_to_pose_path):
            with open(cam_to_pose_path, "r") as f:
                for line in f:
                    if line.startswith("image_00:"):
                        parts = line.split()
                        T_body_cam00 = np.eye(4, dtype=np.float64)
                        T_body_cam00[:3, :4] = np.array([float(x) for x in parts[1:]], dtype=np.float64).reshape(3, 4)
                        break

        if T_body_cam00 is None:
            raise ValueError(f"Could not find 'image_00:' reference pose to compute T_body_velo")

        T_body_velo = T_body_cam00 @ np.linalg.inv(T_velo_cam00)
        T_cl = np.linalg.inv(T_body_cam_init) @ T_body_velo
        return T_cl

    # Case 2: Raw 12-float (3x4) or 16-float (4x4) calib_cam_to_velo.txt format (T_lc)
    values = [float(x) for line in lines for x in line.split()]
    if len(values) == 12:
        T_lc = np.eye(4, dtype=np.float64)
        T_lc[:3, :4] = np.array(values, dtype=np.float64).reshape(3, 4)
    elif len(values) == 16:
        T_lc = np.array(values, dtype=np.float64).reshape(4, 4)
    else:
        raise ValueError(
            f"Expected 12 or 16 floats in {file_path}, but found {len(values)} values."
        )

    return np.linalg.inv(T_lc)


def build_initial_pose_cl(gt_pose_cl: np.ndarray, args, calib_dir: str|None = None, cam_id: str = "00") -> np.ndarray:
    """
    Construct the 4x4 initial LiDAR-to-camera extrinsic matrix (T_cl)
    from CLI options (explicit calibration file, GT + custom perturbation, or nominal default).
    """
    R_nom = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]], dtype=np.float64)
    has_perturbation = any([
        getattr(args, "init_from_gt", False),
        getattr(args, "rot_error_deg", None) is not None,
        getattr(args, "trans_error_m", None) is not None,
        getattr(args, "rand_rot_deg", None) is not None,
        getattr(args, "rand_trans_m", None) is not None,
    ])

    # 1. Determine base pose (Explicit File vs. GT vs. Nominal)
    if getattr(args, "init_calib_file", None) is not None:
        base_pose = load_pose_cl_from_file(args.init_calib_file, calib_dir=calib_dir, cam_id=cam_id)
    elif has_perturbation:
        base_pose = gt_pose_cl.astype(np.float64).copy()
    else:
        base_pose = np.eye(4, dtype=np.float64)
        base_pose[:3, :3] = R_nom

    # 2. Apply deterministic and/or random perturbations
    rng = np.random.default_rng(getattr(args, "calib_seed", None))
    total_rot_deg = np.zeros(3, dtype=np.float64)
    total_trans_m = np.zeros(3, dtype=np.float64)

    if getattr(args, "rot_error_deg", None) is not None:
        total_rot_deg += np.array(args.rot_error_deg, dtype=np.float64)
    if getattr(args, "rand_rot_deg", None) is not None:
        b = float(args.rand_rot_deg)
        total_rot_deg += rng.uniform(-b, b, size=3)

    if getattr(args, "trans_error_m", None) is not None:
        total_trans_m += np.array(args.trans_error_m, dtype=np.float64)
    if getattr(args, "rand_trans_m", None) is not None:
        b = float(args.rand_trans_m)
        total_trans_m += rng.uniform(-b, b, size=3)

    if np.any(total_rot_deg != 0):
        base_euler = pytorch3d.transforms.matrix_to_euler_angles(
            torch.tensor(base_pose[:3, :3], dtype=torch.float64), convention="XYZ"
        )
        perturbed_euler = base_euler + torch.tensor(np.deg2rad(total_rot_deg), dtype=torch.float64)
        base_pose[:3, :3] = pytorch3d.transforms.euler_angles_to_matrix(
            perturbed_euler, convention="XYZ"
        ).numpy()

    base_pose[:3, 3] = base_pose[:3, 3] + total_trans_m
    return base_pose


class LidarPose:
    def __init__(self, lidar_rotation, lidar_translation, timestamp):
        rotation_quat = pytorch3d.transforms.matrix_to_quaternion(torch.tensor(lidar_rotation)).float().to('cuda')
        self.lidar_rotation = torch.nn.Parameter(rotation_quat, requires_grad=False)
        self.lidar_translation = torch.nn.Parameter(torch.Tensor(lidar_translation).float().to('cuda'), requires_grad=False)
        self.timestamp = timestamp

class SensorTrajectories:
    def __init__(self, lidar_rotations, lidar_translations, lidar_timestamps, pose_cl, initialized_pose_cl = None):
        assert len(lidar_rotations) == len(lidar_translations)
        assert len(lidar_rotations) == len(lidar_timestamps)


        self.lidar_poses = []
        for i in range(len(lidar_rotations)):
            self.lidar_poses.append(LidarPose(lidar_rotations[i], lidar_translations[i], lidar_timestamps[i]))

        rotation_cl = pose_cl[:3,:3]
        translation_cl = pose_cl[:3,3]
        rotation_cl_quat = pytorch3d.transforms.matrix_to_quaternion(torch.tensor(rotation_cl))
        # rotation_disturbance = pytorch3d.transforms.axis_angle_to_quaternion(torch.rand(3) * 0.0)
        # translation_disturbance = np.random.rand(3) * 0.5

        self.gt_translation_cl = torch.Tensor(translation_cl)
        self.gt_rotation_cl = rotation_cl_quat


        self.gt_rotation_cl = rotation_cl_quat.float().to('cuda')
        self.gt_translation_cl = torch.Tensor(translation_cl).float().to('cuda')
        #TODO: Wrong in compute pose error
        # initial_rotation = torch.Tensor([[0., -1., 0.], [0., 0., -1.], [1., 0., 0,]]).float().to('cuda')
        # initial_translation = torch.Tensor([0., 0., 0.]).float().to('cuda')

        # initial_rotation = torch.Tensor([[4.27303574e-04, -9.99121606e-01,  4.19031493e-02], [-7.18921563e-03, -4.19051386e-02, -9.99095678e-01], [9.99974072e-01,  1.25666411e-04, -7.20080640e-03 ]]).float().to('cuda')


        rotation_disturbance = pytorch3d.transforms.matrix_to_quaternion(pytorch3d.transforms.euler_angles_to_matrix(torch.ones(3).float().to('cuda') * 3 / 180 * 1.57, convention='XYZ'))
        # initial_rotation_quat = pytorch3d.transforms.matrix_to_quaternion(initial_rotation)

        if initialized_pose_cl is None:
            initial_rotation_mat = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
            initial_rotation_mat = torch.Tensor(initial_rotation_mat).float().to('cuda')
            initial_translation = self.gt_translation_cl * 0.
        else:
            initial_rotation_mat = torch.Tensor(initialized_pose_cl[:3,:3]).float().to('cuda')
            initial_translation = torch.Tensor(initialized_pose_cl[:3,3]).float().to('cuda')
        # initial_rotation_quat = self.gt_rotation_cl
        # initial_translation = self.gt_translation_cl
        # initial_rotation_quat = pytorch3d.transforms.quaternion_multiply(initial_rotation_quat, rotation_disturbance)

        initial_rotation_quat = pytorch3d.transforms.matrix_to_quaternion(initial_rotation_mat)
        # initial_translation = self.gt_translation_cl * 0.

        self.rotation_cl = initial_rotation_quat
        self.translation_cl = initial_translation

        rotation_cl_delta = pytorch3d.transforms.matrix_to_quaternion(torch.eye(3)).float().to('cuda')
        translation_cl_delta = torch.zeros(3).float().to('cuda')
        self.rotation_cl_delta = torch.nn.Parameter(rotation_cl_delta, requires_grad=False)
        self.translation_cl_delta = torch.nn.Parameter(translation_cl_delta, requires_grad=False)



    def get_extrinsics(self):
        rotation_cl = pytorch3d.transforms.quaternion_multiply(self.rotation_cl_delta, self.rotation_cl)
        translation_cl = pytorch3d.transforms.quaternion_apply(self.rotation_cl_delta, self.translation_cl) + self.translation_cl_delta
        return rotation_cl, translation_cl

    def update_extrinsics(self):
        with torch.no_grad():
            self.rotation_cl = pytorch3d.transforms.quaternion_multiply(self.rotation_cl_delta, self.rotation_cl)
            self.translation_cl = pytorch3d.transforms.quaternion_apply(self.rotation_cl_delta, self.translation_cl) + self.translation_cl_delta


            self.rotation_cl_delta.data = pytorch3d.transforms.matrix_to_quaternion(torch.eye(3)).float().to('cuda')
            self.translation_cl_delta.data = torch.zeros(3).float().to('cuda')
        return True

    def get_length(self):
        return len(self.lidar_poses)

    def start_optimize_cl(self):
        if not self.rotation_cl_delta.requires_grad:
            self.rotation_cl_delta.requires_grad_(True)
        if not self.translation_cl_delta.requires_grad:
            self.translation_cl_delta.requires_grad_(True)


    def start_optimize_rotation_cl(self):
        if not self.rotation_cl_delta.requires_grad:
            self.rotation_cl_delta.requires_grad_(True)
        return

    def stop_optimize_rotation_cl(self):
        self.rotation_cl_delta.requires_grad_(False)
        self.rotation_cl_delta.grad = None
        return

    def stop_optimize_translation_cl(self):
        self.translation_cl_delta.requires_grad_(False)
        self.translation_cl_delta.grad = None
        return

    def start_optimize_translation_cl(self):
        if not self.translation_cl_delta.requires_grad:
            self.translation_cl_delta.requires_grad_(True)

    def quaternion_euler_error(
            self,
            pred_q: torch.Tensor,
            gt_q: torch.Tensor,
            rotation_order: str = 'XYZ',
            eps: float = 1e-7
    ) -> torch.Tensor:
        """
        Compute three-axis Euler angle error between quaternions.

        Args:
            pred_q (Tensor): Predicted quaternion [..., 4] (w, x, y, z)
            gt_q (Tensor): Ground truth quaternion [..., 4]
            rotation_order (str): Euler angle decomposition order, default is 'XYZ'
            eps (float): Numerical stability coefficient

        Returns:
            error (Tensor): Absolute Euler angle error on three axes [..., 3] (unit: degrees)
        """
        # Normalize quaternions (handle non-unit quaternions)
        pred_q = pred_q / (torch.norm(pred_q, dim=-1, keepdim=True) + eps)
        gt_q = gt_q / (torch.norm(gt_q, dim=-1, keepdim=True) + eps)

        # Convert to rotation matrices
        pred_rot = pytorch3d.transforms.quaternion_to_matrix(pred_q)  # [..., 3, 3]
        gt_rot = pytorch3d.transforms.quaternion_to_matrix(gt_q)  # [..., 3, 3]

        # Convert to Euler angles
        pred_euler = pytorch3d.transforms.matrix_to_euler_angles(pred_rot, rotation_order)  # [..., 3]
        gt_euler = pytorch3d.transforms.matrix_to_euler_angles(gt_rot, rotation_order)  # [..., 3]

        # Compute periodic angular error
        diff = pred_euler - gt_euler
        diff_rad = torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi  # [-π, π]
        # Convert to degrees and take absolute value
        diff_deg = torch.abs(diff_rad) * (180.0 / torch.pi)  # [..., 3] (degree)

        return diff_deg

    def get_relative_pose_error(self):
        frame_num = len(self.lidar_poses)

        i = 5
        rotation_wc, translation_wc = self.get_camera_pose(i)
        next_rotation_wc, next_translation_wc = self.get_camera_pose(i+1)
        relative_rotation = (rotation_wc).T @ (next_rotation_wc)

        gt_rotation_wc, gt_translation_wc = self.get_camera_pose_gt(i)
        next_gt_rotation_wc, next_gt_translation_wc = self.get_camera_pose_gt(i+1)
        relative_rotation_gt = (gt_rotation_wc).T @ (next_gt_rotation_wc)
        rotation_err = relative_rotation.T @ relative_rotation_gt
        # relative_rotation_euler = pytorch3d.transforms.matrix_to_euler_angles(relative_rotation, convention='XYZ')
        # relative_rotation_euler_gt = pytorch3d.transforms.matrix_to_euler_angles(relative_rotation_gt, convention='XYZ')
        return pytorch3d.transforms.matrix_to_axis_angle(rotation_err)



    def compute_extrinsic_error(self, rotation_cl, translation_cl):
        # gt_rotation_matrix_cl = pytorch3d.transforms.quaternion_to_matrix(self.gt_rotation_cl.cpu()).numpy()
        # curr_rotation_matrix_cl = pytorch3d.transforms.quaternion_to_matrix(self.rotation_cl.detach().cpu()).numpy()
        euler_error = self.quaternion_euler_error(rotation_cl.detach().cpu(), self.gt_rotation_cl.detach().cpu())
        gt_translation_cl = self.gt_translation_cl.cpu().numpy()
        curr_translation_cl = translation_cl.detach().cpu().numpy()
        # relative_rotation_error = gt_rotation_matrix_cl @ curr_rotation_matrix_cl.T
        relative_translation_error = gt_translation_cl - curr_translation_cl
        return euler_error, torch.Tensor(relative_translation_error)

    def get_extrinsic_error(self):
        # gt_rotation_matrix_cl = pytorch3d.transforms.quaternion_to_matrix(self.gt_rotation_cl.cpu()).numpy()
        # curr_rotation_matrix_cl = pytorch3d.transforms.quaternion_to_matrix(self.rotation_cl.detach().cpu()).numpy()
        euler_error = self.quaternion_euler_error(self.rotation_cl.detach().cpu(), self.gt_rotation_cl.detach().cpu())
        gt_translation_cl = self.gt_translation_cl.cpu().numpy()
        curr_translation_cl = self.translation_cl.detach().cpu().numpy()
        # relative_rotation_error = gt_rotation_matrix_cl @ curr_rotation_matrix_cl.T
        relative_translation_error = gt_translation_cl - curr_translation_cl
        return euler_error, torch.Tensor(relative_translation_error)

    def training_setup(self, training_args):
        l = [
            {'params': [lidar_pose.lidar_rotation for lidar_pose in self.lidar_poses], 'lr': training_args.pose_rotation_lr,
             "name": "rotation_lidar"},
            {'params': [lidar_pose.lidar_translation for lidar_pose in self.lidar_poses], 'lr': training_args.pose_translation_lr,
             "name": "translation_lidar"},
            {'params': [self.rotation_cl_delta], 'lr': training_args.pose_rotation_lr,
             "name": "rotation_cl"},
            {'params': [self.translation_cl_delta], 'lr': training_args.pose_translation_lr,
             "name": "translation_cl"},
        ]
        self.l = l
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)



    def update_extrinsic_in_search(self, rotation_cl_quat, translation_cl):
        self.rotation_cl.data.copy_(rotation_cl_quat)
        self.translation_cl.data.copy_(translation_cl)
        self.rotation_cl_delta.data = pytorch3d.transforms.matrix_to_quaternion(torch.eye(3)).float().to('cuda')
        self.translation_cl_delta.data = torch.zeros(3).float().to('cuda')


    def update_learning_rate(self, ratio):
        initial_rotation_lr = self.l[2]['lr']
        initial_translation_lr = self.l[3]['lr']
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "rotation_cl":
                param_group['lr'] = initial_rotation_lr * ratio
            if param_group["name"] == "translation_cl":
                param_group['lr'] = initial_rotation_lr * ratio

    def get_all_lidar_poses(self):
        rotation_param_list = []
        translation_param_list = []
        for lidar_pose in self.lidar_poses:
            rotation_param_list.append(lidar_pose.lidar_rotation)
            translation_param_list.append(lidar_pose.lidar_translation)
        return rotation_param_list, translation_param_list

    def get_lidar_pose(self, frame_num):
        lidar_pose = self.lidar_poses[frame_num]
        rotation = lidar_pose.lidar_rotation
        translation = lidar_pose.lidar_translation
        return rotation, translation

    def get_camera_pose(self, frame_num):
        rotation_cl, translation_cl = self.get_extrinsics()
        rotation_wl, translation_wl = self.get_lidar_pose(frame_num)
        rotation_wl = pytorch3d.transforms.quaternion_to_matrix(rotation_wl)
        rotation_wc = rotation_wl @ pytorch3d.transforms.quaternion_to_matrix(rotation_cl).T
        #twc = Rwl*tlc+twl = Rwl*(-Rcl.T * tcl) + twl
        translation_wc = -(rotation_wl @ pytorch3d.transforms.quaternion_to_matrix(rotation_cl).T) @ translation_cl + translation_wl
        return rotation_wc, translation_wc

    def get_camera_pose_gt(self, frame_num):
        rotation_wl, translation_wl = self.get_lidar_pose(frame_num)
        rotation_wl = pytorch3d.transforms.quaternion_to_matrix(rotation_wl)
        rotation_wc = rotation_wl @ pytorch3d.transforms.quaternion_to_matrix(self.gt_rotation_cl).T
        #twc = Rwl*tlc+twl = Rwl*(-Rcl.T * tcl) + twl
        translation_wc = -(rotation_wl @ pytorch3d.transforms.quaternion_to_matrix(self.gt_rotation_cl).T) @ self.gt_translation_cl + translation_wl
        return rotation_wc, translation_wc

    # def get_gt_camera_pose(self, frame_num):
    #     rotation_wl, translation_wl = self.get_lidar_pose(frame_num)
    #     rotation_wl = pytorch3d.transforms.quaternion_to_matrix(rotation_wl).to(self.gt_rotation_cl.device)
    #     translation_wl = translation_wl.to(self.gt_translation_cl.device)
    #
    #     rotation_wc = rotation_wl @ pytorch3d.transforms.quaternion_to_matrix(self.gt_rotation_cl).T
    #     # twc = Rwl*tlc+twl = Rwl*(-Rcl.T * tcl) + twl
    #     translation_wc = -(rotation_wl @ pytorch3d.transforms.quaternion_to_matrix(
    #         self.gt_rotation_cl).T) @ self.gt_translation_cl + translation_wl
    #     return rotation_wc, translation_wc

if __name__ == '__main__':
    lidar_rotations = [li.SO3.exp(torch.Tensor(np.random.rand(3))).matrix()[:3,:3].numpy()]
    lidar_translations = [torch.Tensor(np.random.rand(3)).numpy()]

    pose_cl = li.SE3.exp(torch.Tensor(np.random.rand(6))).matrix().numpy()

    traj = SensorTrajectories(lidar_rotations, lidar_translations, [0.], pose_cl)
    rotation_wc, translation_wc = traj.get_camera_pose(0)
    pred_pose_wc = torch.eye(4)
    pred_pose_wc[:3,:3] = rotation_wc
    pred_pose_wc[:3,3] = translation_wc

    pose_wl = np.eye(4)
    pose_wl[:3,:3] = lidar_rotations[0]
    pose_wl[:3,3] = lidar_translations[0]


    pose_lc = np.linalg.inv(pose_cl)
    pose_wc = pose_wl @ pose_lc

    print(pose_wc)
    print(pred_pose_wc)

