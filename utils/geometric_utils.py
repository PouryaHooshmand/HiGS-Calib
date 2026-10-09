import torch

def find_nearest_gaussian(points, gaussians):
    """
    Compute the index of the nearest Gaussian for each point in the point cloud.

    Args:
        points (torch.Tensor): Input point cloud of shape [N, 3]
        gaussians (torch.Tensor): Gaussian center coordinates of shape [M, 3]

    Returns:
        torch.Tensor: Nearest Gaussian index for each point of shape [N]
    """
    # Compute pairwise Euclidean distances between points and Gaussians
    distances = torch.cdist(points, gaussians)  # Output shape [N, M]

    # Get indices of minimum distance
    nearest_indices = torch.argmin(distances, dim=1)

    return nearest_indices

def batch_find_nearest(points, gaussians, batch_size=10000):
    nearest = []
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        dists = torch.cdist(batch, gaussians)
        nearest.append(torch.argmin(dists, dim=1))
    return torch.cat(nearest)