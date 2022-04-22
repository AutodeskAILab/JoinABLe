"""

Find the axis of B-Rep faces/edges
using the data provided as node features

"""

import torch
import numpy as np
import torch.nn.functional as F
from utils import util


def find_axis_line(entity, return_numpy=False):
    """
    Find an infinite line which passes through the
    middle of this entity
    """
    axis_line = None
    if "surface_type" in entity:
        axis_line = find_axis_line_from_face(entity)
    elif "curve_type" in entity:
        axis_line = find_axis_line_from_edge(entity)
    elif "is_degenerate" in entity and entity["is_degenerate"]:
        return None, None
    origin, direction = axis_line
    if axis_line is not None:
        if origin is not None and direction is not None and return_numpy:
            # Convert from dict to numpy
            origin = get_point(origin)
            direction = get_vector(direction)
            return origin, direction
        else:
            return axis_line
    print("Invalid entity for finding a joint axis")
    return None, None


def check_colinear_with_tolerance(axis_line1, axis_line2, angle_tol_degs=10.0, distance_tol=1e-2):
    """
    Similar to InfLine3D.IsColinearTo() but it allows us to give a tolerance
    for the angle and distance between the lines
    """
    if isinstance(axis_line1[0], dict):
        origin1_dict, direction1_dict = axis_line1
        origin2_dict, direction2_dict = axis_line2
        # Convert from dict to numpy
        origin1 = get_point(origin1_dict)
        origin2 = get_point(origin2_dict)
        direction1 = get_vector(direction1_dict)
        direction2 = get_vector(direction2_dict)
    else:
        origin1, direction1 = axis_line1
        origin2, direction2 = axis_line2
    # Find the angle between the two axis directions
    angle_rads = get_angle_between(direction1, direction2)
    reversed_direction2 = direction2 * -1.0
    reversed_angle_rads = get_angle_between(direction1, reversed_direction2)
    angle_rads = min(angle_rads, reversed_angle_rads)
    angle_degs = np.rad2deg(angle_rads)

    try:
        # Find the dist between a line and the locator point
        dist = dist_point_to_line(origin1, origin2, direction2)
    except Exception as ex:
        return False

    return angle_degs < angle_tol_degs and dist < distance_tol


def point_to_line_torch(points, line_start, line_direction):
    """Get the (non unit) vectors from multiple points to a single line using torch"""
    assert len(points.shape) == 2
    num_points = points.shape[0]
    # Repeat the line values to process in parallel
    line_start_r = line_start.repeat(num_points, 1)
    line_direction_r = line_direction.repeat(num_points, 1)
    line_end_r = line_start_r + 1.0 * line_direction_r
    x = line_start_r - line_end_r
    pt_end = points - line_end_r
    # dot product along dim -1
    t_1 = torch.sum(pt_end * x, dim=-1).unsqueeze(-1)
    t_2 = torch.sum(x * x, dim=-1).unsqueeze(-1)
    t = t_1 / t_2
    vectors = t * (line_start - line_end_r) + line_end_r - points
    return vectors


def projection_to_line_torch(points, line_start, line_direction):
    """Get the (non unit) vectors from multiple points to a single line using torch"""
    assert len(points.shape) == 2
    num_points = points.shape[0]
    # Repeat the line values to process in parallel
    line_start_r = line_start.repeat(num_points, 1)
    line_direction_r = line_direction.repeat(num_points, 1)
    line_end_r = line_start_r + 1.0 * line_direction_r
    x = line_start_r - line_end_r
    pt_end = points - line_end_r
    # dot product along dim -1
    t_1 = torch.sum(pt_end * x, dim=-1).unsqueeze(-1)
    t_2 = torch.sum(x * x, dim=-1).unsqueeze(-1)
    t = t_1 / t_2
    vectors = t * (line_start - line_end_r) + line_end_r - points

    norm_vector = F.normalize(vectors, dim=-1)
    dist = torch.linalg.norm(vectors, dim=-1)

    reg_vec = torch.cat([norm_vector, dist.unsqueeze(1)], dim=-1)
    return reg_vec


def dist_point_to_line(point, line_start, line_direction):
    """Get the distance from a single point to a line using numpy"""
    line_end = line_start + 1.0 * line_direction
    x = line_start - line_end
    pt_end = point - line_end
    t_1 = np.dot(pt_end, x)
    t_2 = np.dot(x, x)
    t = t_1 / t_2
    dist = np.linalg.norm(
        t * (line_start - line_end) + line_end - point
    )
    return dist


def dist_point_to_line_torch(points, line_start, line_direction):
    """Get the distance from multiple points to a single line using torch"""
    # Get the vectors from each point cloud point to the line
    vectors = point_to_line_torch(points, line_start, line_direction)
    dist = torch.linalg.norm(vectors, dim=-1)
    return dist


def axis_line_to_torch(axis_line):
    """
    Convert an axis line dict into a
    torch origin point and direction vector
    """
    origin = util.vector_to_torch(axis_line["origin"])[:3]
    direction = util.vector_to_torch(axis_line["direction"])[:3]
    length = torch.linalg.norm(direction)
    if length == 0:
        print("Warning: Joint axis direction of length 0")
    else:
        direction = direction / length
    return origin, direction


def align_vectors(a, b):
    """
    Calculate the rotation matrix to align two vectors
    Modified from: trimesh.geometry.align_vectors
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError('vectors must be (3,)!')

    # find the SVD of the two vectors
    au = np.linalg.svd(a.reshape((-1, 1)))[0]
    bu = np.linalg.svd(b.reshape((-1, 1)))[0]

    if np.linalg.det(au) < 0:
        au[:, -1] *= -1.0
    if np.linalg.det(bu) < 0:
        bu[:, -1] *= -1.0

    return bu.dot(au.T)


def align_vectors_torch(a, b, return_4x4=False):
    """
    Calculate the rotation matrix to align two batches of vectors
    Modified from: trimesh.geometry.align_vectors
    a and b contain batches of vectors wth shape (n, 3)
    """
    if a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError('vectors must be (n,3)!')
    assert a.shape[0] == b.shape[0]
    batch_size = a.shape[0]

    ar = a.reshape((-1, 3, 1))
    br = b.reshape((-1, 3, 1))

    # find the SVD of the two vectors
    au = torch.linalg.svd(ar)[0]
    bu = torch.linalg.svd(br)[0]

    au[torch.linalg.det(au) < 0, :, -1] *= -1.0
    bu[torch.linalg.det(bu) < 0, :, -1] *= -1.0

    # Transpose au along dim 1 and 2 (not batch dim 0)
    au_t = torch.transpose(au, 1, 2)
    # Batch matmul along dim 1 and 2 (not batch dim 0)?
    mat = torch.matmul(bu, au_t)

    if return_4x4:
        mat_4x4 = torch.tile(torch.eye(4), (batch_size, 1)).view((batch_size, 4, 4))
        mat_4x4[:, :3, :3] = mat
        return mat_4x4
    else:
        return mat


def get_transform_from_parameters(
    origin1, origin2, direction1, direction2,
    offset=0.0,
    rotation_in_radians=0.0,
    flip=False,
    align_mat=None
):
    # ALIGNMENT
    if align_mat is None:
        align_mat = get_joint_alignment_matrix(origin1, origin2, direction1, direction2)

    pred_mat = align_mat
    # ROTATION
    rot_mat = get_rotation_parameter_matrix(rotation_in_radians, origin2, direction2)
    pred_mat = torch.matmul(rot_mat, pred_mat)
    # OFFSET + FLIP
    offset_mat = get_offset_parameter_matrix(offset, origin2, direction2, flip)
    pred_mat = torch.matmul(offset_mat, pred_mat)
    return pred_mat


def get_joint_alignment_matrix(origin1, origin2, direction1, direction2):
    """
    Get the affine matrix (4x4) that aligns the axis of body one with the axis of body 2
    """
    # Currently we don't support batching
    assert origin1.shape == (3,)
    assert origin2.shape == (3,)
    assert direction1.shape == (3,)
    assert direction2.shape == (3,)
    # Align the directions to make a 4x4 rotation matrix
    # Expects a batch, so unsqueeze then squeeze
    align_mat = align_vectors_torch(direction1.unsqueeze(0), direction2.unsqueeze(0), return_4x4=True).squeeze(0)
    # rotate around the given origin
    align_mat[:3, 3] = origin1 - torch.matmul(align_mat[:3, :3], origin1)
    # translate from the origin of body 2's entity
    align_mat[:3, 3] += origin2 - origin1
    return align_mat


def get_rotation_parameter_matrix(rotation, origin, direction):
    """
    Get an affine matrix (4x4) to apply the rotation parameter about the provided joint axis
    """
    # We do this manually in torch so it is differentiable
    # the below code is similar to calling in scipy:
    # rot_mat = Rotation.from_rotvec(rotation_in_radians * direction)
    x, y, z = direction
    c = torch.cos(rotation)
    s = torch.sin(rotation)
    C = 1 - c
    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    rot_mat = torch.tensor([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c]
    ], dtype=torch.float, device=rotation.device)
    # rotation around the origin
    rot_point = origin - torch.matmul(rot_mat[:3, :3], origin)
    aff_mat = torch.eye(4, dtype=torch.float, device=rotation.device)
    aff_mat[:3, :3] = rot_mat
    aff_mat[:3, 3] = rot_point
    return aff_mat


def get_offset_parameter_matrix(offset, origin, direction, flip=0.0):
    """
    Get an affine matrix (4x4) to apply the offset parameter
    """
    # The offset gets applied first
    aff_mat = torch.eye(4, dtype=torch.float, device=offset.device)
    aff_mat[:3, 3] = direction * offset
    # Reflection matrix
    # Reflect at the origin normal to the direction
    # If flip is 0, this should be the identity matrix
    flip_direction = direction * flip
    aff_mat[:3, :3] = torch.eye(3, dtype=torch.float, device=offset.device) - 2 * torch.outer(flip_direction, flip_direction)
    # Normalized vector, vector divided by Euclidean (L2) norm
    normal = direction.squeeze()
    normal = normal / torch.linalg.norm(normal)
    # Flip at the origin point
    aff_mat[:3, 3] += ((2.0 * torch.dot(origin, normal)) * normal) * flip
    return aff_mat


def find_axis_line_from_face(face):
    """
    Find an infinite line which passes through the
    middle of this face
    """
    if face["surface_type"] == "PlaneSurfaceType":
        return find_axis_line_from_planar_face(face)
    elif face["surface_type"] == "CylinderSurfaceType":
        return find_axis_line_from_cylindrical_face(face)
    elif face["surface_type"] == "EllipticalCylinderSurfaceType":
        return find_axis_line_from_elliptical_cylindrical_face(face)
    elif face["surface_type"] == "ConeSurfaceType":
        return find_axis_line_from_conical_face(face)
    elif face["surface_type"] == "EllipticalConeSurfaceType":
        return find_axis_line_from_elliptical_conical_face(face)
    elif face["surface_type"] == "SphereSurfaceType":
        return find_axis_line_from_spherical_face(face)
    elif face["surface_type"] == "TorusSurfaceType":
        return find_axis_line_from_toroidal_face(face)
    # print(f"Joint axis not supported for {face['surface_type']}")
    return None, None


def find_axis_line_from_edge(edge):
    """
    Find an infinite line which passes through the
    middle of this edge
    """
    if "is_degenerate" in edge and edge["is_degenerate"]:
        print(f"Joint axis not supported for degenerate edges")
        return None, None
    if edge["curve_type"] == "Line3DCurveType":
        return find_axis_line_from_linear_edge(edge)
    elif edge["curve_type"] == "Arc3DCurveType":
        return find_axis_line_from_arc_edge(edge)
    elif edge["curve_type"] == "EllipticalArc3DCurveType":
        return find_axis_line_from_elliptical_arc_edge(edge)
    elif edge["curve_type"] == "Ellipse3DCurveType":
        return find_axis_line_from_elliptical_edge(edge)
    elif edge["curve_type"] == "Circle3DCurveType":
        return find_axis_line_from_circular_edge(edge)
    # print(f"Joint axis not supported for {edge['curve_type']}")
    return None, None


def find_axis_line_from_planar_face(face):
    centroid = get_point_data(face, "centroid")
    normal = get_vector_data(face, "normal")
    return centroid, normal


def find_axis_line_from_cylindrical_face(face):
    origin = get_point_data(face, "origin")
    axis = get_vector_data(face, "axis")
    return origin, axis


def find_axis_line_from_elliptical_cylindrical_face(face):
    origin = get_point_data(face, "origin")
    axis = get_vector_data(face, "axis")
    return origin, axis


def find_axis_line_from_conical_face(face):
    origin = get_point_data(face, "origin")
    axis = get_vector_data(face, "axis")
    return origin, axis


def find_axis_line_from_elliptical_conical_face(face):
    origin = get_point_data(face, "origin")
    axis = get_vector_data(face, "axis")
    return origin, axis


def find_axis_line_from_spherical_face(face):
    origin = get_point(face, "origin")
    direction = np.array([0.0, 0.0, 1.0], dtype=float)
    return get_point_data(origin), get_vector_data(direction)


def find_axis_line_from_toroidal_face(face):
    origin = get_point_data(face, "origin")
    axis = get_vector_data(face, "axis")
    return origin, axis


def find_axis_line_from_linear_edge(curve):
    start_point = get_point(curve, "start_point")
    end_point = get_point(curve, "end_point")
    direction = get_direction(start_point, end_point)
    return get_point_data(start_point), get_vector_data(direction)


def find_axis_line_from_arc_edge(curve):
    center = get_point_data(curve, "center")
    normal = get_vector_data(curve, "normal")
    return center, normal


def find_axis_line_from_elliptical_arc_edge(curve):
    center = get_point_data(curve, "center")
    normal = get_vector_data(curve, "normal")
    return center, normal


def find_axis_line_from_elliptical_edge(curve):
    center = get_point_data(curve, "center")
    normal = get_vector_data(curve, "normal")
    return center, normal


def find_axis_line_from_circular_edge(curve):
    center = get_point_data(curve, "center")
    normal = get_vector_data(curve, "normal")
    return center, normal


def get_vector(entity, name=None):
    """Get a Vector3D as numpy array"""
    if name is None:
        x = entity["x"]
        y = entity["y"]
        z = entity["z"]
    else:
        x = entity[f"{name}_x"]
        y = entity[f"{name}_y"]
        z = entity[f"{name}_z"]
    # Normalize the vector
    vector = np.array([x, y, z], dtype=float)
    dist = np.linalg.norm(vector)
    if dist == 0:
        # In some cases the vector is [0,0,0]
        # Return as is for now
        return vector
    else:
        return vector / dist


def get_point(entity, name=None):
    """Get a Point3D as numpy array"""
    if name is None:
        x = entity["x"]
        y = entity["y"]
        z = entity["z"]
    else:
        x = entity[f"{name}_x"]
        y = entity[f"{name}_y"]
        z = entity[f"{name}_z"]
    return np.array([x, y, z], dtype=float)


def get_vector_data(entity, name=None):
    """Get a Vector3D dict to export as json"""
    if isinstance(entity, dict) and name is not None:
        vector = get_vector(entity, name)
    else:
        vector = entity
    return {
        "type": "Vector3D",
        "x": vector[0],
        "y": vector[1],
        "z": vector[2],
        "length": 1.0
    }


def get_point_data(entity, name=None):
    """Get a Point3D dict to export as json"""
    if isinstance(entity, dict) and name is not None:
        point = get_point(entity, name)
    else:
        point = entity
    return {
        "type": "Point3D",
        "x": point[0],
        "y": point[1],
        "z": point[2]
    }


def get_direction(pt1, pt2):
    """Get the direction between two points"""
    delta = pt2 - pt1
    dist = np.linalg.norm(delta)
    if dist == 0:
        return delta
    direction = delta / dist
    return direction


def get_angle_between(v1, v2):
    """Get the angle between two vectors in radians"""
    dot_pr = v1.dot(v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    arccos_input = dot_pr / norms
    # Clamp arcos input to the [-1, 1] range
    if arccos_input < -1:
        arccos_input = -1
    if arccos_input > 1:
        arccos_input = 1
    return np.arccos(arccos_input)
