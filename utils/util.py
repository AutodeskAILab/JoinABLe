import sys
import logging
import warnings
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from trimesh import transformations


def pad_pts(pts):
    """
    Pad n,3 points with a 1 to become n,4
    suitable for use with a 4x4 matrix
    """
    return np.pad(
        pts,
        ((0, 0), (0, 1)),
        mode="constant",
        constant_values=1
    )


def pad_pts_torch(pts):
    """
    Pad n,3 points with a 1 to become n,4
    suitable for use with a 4x4 matrix
    """
    return F.pad(
        pts,
        (0, 1),
        mode="constant",
        value=1
    )


def transform_pts_by_matrix(pts, matrix, copy=False):
    """
    Transform an array of xyz pts (n, 3) by a 4x4 matrix using numpy
    """
    # Transpose first
    if pts.shape[1] == 3:
        # If necessary pad the points with a zero to be (n, 4)
        v = pad_pts(pts).T
    elif pts.shape[1] == 4:
        v = pts.T
    v = matrix @ v
    # Transpose and crop back to (n, 3)
    if copy:
        # Reorder to C-contiguous order
        return v.T[:, 0:3].copy()
    else:
        return v.T[:, 0:3]


def transform_pts_by_matrix_torch(pts, matrix, copy=False):
    """
    Transform an array of xyz pts (n, 3) by a 4x4 matrix using torch
    """
    # Transpose first
    if pts.shape[1] == 3:
        # If necessary pad the points with a zero to be (n, 4)
        v = pad_pts_torch(pts).T
    elif pts.shape[1] == 4:
        v = pts.T
    v = torch.matmul(matrix, v)
    # Transpose and crop back to (n, 3)
    if copy:
        # Reorder to C-contiguous order
        return v.T[:, 0:3].copy()
    else:
        return v.T[:, 0:3]


def matrix_to_trans_rot(aff_mat):
    """Get a translation point and rotation quaternion [w, x, y, z] from a 4x4 affine matrix"""
    quaternion = transformations.quaternion_from_matrix(aff_mat)
    quaternion_torch = torch.from_numpy(quaternion).float()
    # Right side column is the translation
    translation = aff_mat[:-1, -1]
    return translation, quaternion_torch


def trans_rot_to_matrix(translation, quaternion):
    """Get a 4x4 affine matrix from a translation point and rotation quaternion [w, x, y, z]"""
    aff_mat = transformations.quaternion_matrix(quaternion)
    aff_mat_torch = torch.from_numpy(aff_mat).float()
    aff_mat_torch[:-1, -1] = translation
    return aff_mat_torch


def vector_to_torch(vector):
    """
    Convert a vector3d dict into a numpy vector
    """
    x = vector["x"]
    y = vector["y"]
    z = vector["z"]
    h = 0.0
    return torch.tensor([x, y, z, h], dtype=torch.float)


def vector_to_np(vector):
    """
    Convert a vector3d dict into a numpy vector
    """
    x = vector["x"]
    y = vector["y"]
    z = vector["z"]
    h = 0.0
    return np.array([x, y, z, h])


def transform_to_torch(transform):
    """
    Convert a transform dict into a
    torch 4x4 affine transformation matrix
    """
    x_axis = vector_to_torch(transform["x_axis"])
    y_axis = vector_to_torch(transform["y_axis"])
    z_axis = vector_to_torch(transform["z_axis"])
    translation = vector_to_torch(transform["origin"])
    translation[3] = 1.0
    return torch.stack([x_axis, y_axis, z_axis, translation]).t().float()


def transform_to_np(transform):
    """
    Convert a transform dict into a
    numpy 4x4 affine transformation matrix
    """
    x_axis = vector_to_np(transform["x_axis"])
    y_axis = vector_to_np(transform["y_axis"])
    z_axis = vector_to_np(transform["z_axis"])
    translation = vector_to_np(transform["origin"])
    translation[3] = 1.0
    return np.transpose(np.stack([x_axis, y_axis, z_axis, translation]))


def intersect_ray_box(box_min, box_max, ray_origin, ray_dir_inv):
    """
    Intersect a ray with a box and return the offsets from the ray origin
    See: https://tavianator.com/2015/ray_box_nan.html
    """
    # This code intentionaly handles NaNs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t1 = (box_min[0] - ray_origin[0]) * ray_dir_inv[0]
        t2 = (box_max[0] - ray_origin[0]) * ray_dir_inv[0]
    tmin = min(t1, t2)
    tmax = max(t1, t2)

    for i in range(1, 3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t1 = (box_min[i] - ray_origin[i]) * ray_dir_inv[i]
            t2 = (box_max[i] - ray_origin[i]) * ray_dir_inv[i]
        tmin = max(tmin, min(min(t1, t2), tmax))
        tmax = min(tmax, max(max(t1, t2), tmin))

    if (tmax >= max(tmin, 0.0)):
        return tmin, tmax

    return None, None


def get_loggers(log_dir):
    """Get the loggers to use"""
    csv_logger = pl.loggers.CSVLogger(
        log_dir,
        name="log"
    )
    tb_logger = pl.loggers.TensorBoardLogger(
        log_dir,
        name="tb_log"
    )
    loggers = [csv_logger, tb_logger]
    return loggers