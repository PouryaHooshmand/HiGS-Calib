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

import torch
import sys
from datetime import datetime
import numpy as np
import random
import cv2
import torch.nn.functional as F

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution, image_cutt = 0):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    resized_image = resized_image[image_cutt:]
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def PILtoTorchUndistorted(pil_image, resolution,K, D, image_cutt = 0):
    resized_image_PIL = pil_image.resize(resolution)
    image_array = np.array(resized_image_PIL)
    image_array = cv2.undistort(image_array, K, D)
    resized_image = torch.from_numpy(image_array) / 255.0
    resized_image = resized_image[image_cutt:]
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def sample_uniform(min_tensor, max_tensor, M):
    """
    Uniformly sample randomly within the bounds defined by two given tensors.

    Args:
    min_tensor (Tensor): N-dimensional tensor containing minimum values for each dimension
    max_tensor (Tensor): N-dimensional tensor containing maximum values for each dimension
    M (int): Number of samples

    Returns:
    Tensor: Tensor of shape (M, N) containing randomly sampled points
    """
    # Validate inputs
    if min_tensor.dim() != max_tensor.dim():
        raise ValueError("min_tensor and max_tensor must have the same number of dimensions")
    if min_tensor.dim() != 1:
        raise ValueError("Input tensors should be 1D (N-dimensional vector)")
    if not torch.all(min_tensor <= max_tensor):
        raise ValueError("All min_tensor values must be less than or equal to max_tensor values")
    if M <= 0:
        raise ValueError("Sample count M must be a positive integer")

    # Get number of dimensions
    N = min_tensor.size(0)

    # Generate uniformly distributed random numbers in [0, 1)
    rand_samples = torch.rand(M, N).to(min_tensor.device)

    # Scale to specified range
    scaled_samples = min_tensor + rand_samples * (max_tensor - min_tensor)

    return scaled_samples

def create_gradient_mask(image_tensor, ratio = 0.1):
    """
    Check if color pixel gradient of an image is greater than the image mean using PyTorch, and generate a binary mask.

    Args:
        image_tensor (torch.Tensor): Input image tensor of shape [C, H, W] (C is channels, H is height, W is width).
                                     Image value range [0, 1] or normalized is recommended.

    Returns:
        torch.Tensor: Binary mask of shape [H, W]. Value 1 indicates pixel gradient is greater than image mean, otherwise 0.
    """
    # Ensure input is a float tensor to avoid type errors
    if image_tensor.dtype != torch.float32:
        image_tensor = image_tensor.float()

    # Step 1: Compute image gradient magnitude (using Sobel operator)
    # Define Sobel convolution kernels (horizontal and vertical directions)
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image_tensor.device)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image_tensor.device)

    # Expand convolution kernels to match the number of channels of the input image (suitable for color images)
    sobel_kernel_x = sobel_kernel_x.repeat(image_tensor.shape[0], 1, 1, 1)  # Shape [C, 1, 3, 3]
    sobel_kernel_y = sobel_kernel_y.repeat(image_tensor.shape[0], 1, 1, 1)

    # Apply convolution to compute horizontal and vertical gradient components
    pad_width = (1, 1, 1, 1)
    image_pad = F.pad(
        image_tensor.unsqueeze(0),
        pad_width,
        mode='replicate'  # Optional: 'constant' (default 0), 'reflect', 'replicate', 'circular'
    )
    gx = F.conv2d(image_pad, sobel_kernel_x, padding=0, groups=image_pad.shape[1]).squeeze(0)
    gy = F.conv2d(image_pad, sobel_kernel_y, padding=0, groups=image_pad.shape[1]).squeeze(0)

    # Compute gradient magnitude (computed independently per channel, then averaged): sqrt(gx^2 + gy^2)
    gradient_magnitude = torch.sqrt(gx.pow(2) + gy.pow(2)).mean(dim=0)  # Shape [H, W]

    # Step 2: Compute mean of gradients
    grad_mean = torch.mean(gradient_magnitude)  # Mean of entire image (scalar)

    # Step 3: Generate mask (pixels with gradient magnitude greater than mean are 1, otherwise 0)
    mask = (gradient_magnitude > grad_mean * ratio).float()  # Binarization, shape [H, W]


    c, h, w = image_tensor.shape
    h_grid_size = 3  # 38x38 grid
    w_grid_size = int(w/h * h_grid_size)

    # Adaptive block partitioning
    h_step = int(h / h_grid_size)
    w_step = int(w / w_grid_size)
    block_means = torch.zeros(h_grid_size, w_grid_size, device=image_tensor.device)

    # 5. Iterate through each block to compute average gradient
    for i in range(h_grid_size):
        for j in range(w_grid_size):
            # Compute block boundaries
            h_start = int(i * h_step)
            h_end = min(int((i + 1) * h_step), h)
            w_start = int(j * w_step)
            w_end = min(int((j + 1) * w_step), w)

            # Extract current block region
            block = gradient_magnitude[h_start:h_end, w_start:w_end]

            # Compute block average gradient
            block_means[i, j] = block.mean()

    # Mark low gradient blocks
    threshold = grad_mean * ratio
    for i in range(h_grid_size):
        for j in range(w_grid_size):
            h_start = int(i * h_step)
            h_end = min(int((i + 1) * h_step), h)
            w_start = int(j * w_step)
            w_end = min(int((j + 1) * w_step), w)

            if block_means[i, j] < threshold:
                mask[h_start:h_end, w_start:w_end] = 0.

    return gradient_magnitude/torch.max(gradient_magnitude), mask


