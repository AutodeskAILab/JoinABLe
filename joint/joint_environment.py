"""

Environment for interacting with joints

"""


import sys
import json
import math
from pathlib import Path
import warnings
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
import igl
import trimesh
import torch
import torch.nn as nn

from utils import util
from utils import plot_util
from joint.joint_prediction_set import JointPredictionSet


class JointEnvironment():

    @staticmethod
    def get_transform_from_parameters(
        jps,
        prediction_index=0,
        offset=0.0,
        rotation_in_degrees=0.0,
        flip=False,
        align_mat=None,
        origin2=None,
        direction2=None
    ):
        """Get a transform from a set of parameters"""
        # Start with an identity 4x4 affine matrix
        # which we update at each step
        aff_mat = np.eye(4)

        # ALIGN AXES
        if align_mat is None:
            align_mat, origin2, direction2 = JointEnvironment.get_joint_alignment_matrix(jps, prediction_index)
        aff_mat = align_mat @ aff_mat

        # ROTATION
        rot_mat = JointEnvironment.get_rotation_parameter_matrix(rotation_in_degrees, origin2, direction2)
        aff_mat = rot_mat @ aff_mat

        # OFFSET
        offset_mat = JointEnvironment.get_offset_parameter_matrix(offset, origin2, direction2, flip)
        aff_mat = offset_mat @ aff_mat
        return aff_mat

    @staticmethod
    def get_joint_alignment_matrix(jps, prediction_index=0):
        """
        Given a prediction index, get the affine matrix (4x4)
        that aligns the axis of body one with the axis of body 2
        """
        origin1, direction1 = jps.get_joint_prediction_axis(1, prediction_index)
        origin2, direction2 = jps.get_joint_prediction_axis(2, prediction_index)
        # The translation between the two axis origins
        translation = origin2 - origin1
        # The rotation between the two axis directions
        # Ignore "Optimal rotation is not uniquely or poorly defined" warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rotation, _ = Rotation.align_vectors(direction2.reshape(1, -1), direction1.reshape(1, -1))
        aff_mat = np.eye(4)
        aff_mat[:3, :3] = rotation.as_matrix()
        # rotate around the given origin
        aff_mat[:3, 3] = origin1 - np.dot(aff_mat[:3, :3], origin1)
        # translate from the origin of body 2's entity
        aff_mat[:3, 3] += translation
        return aff_mat, origin2, direction2

    @staticmethod
    def get_rotation_parameter_matrix(rotation_in_degrees, origin, direction):
        """
        Get an affine matrix (4x4) to apply the rotation parameter about the provided joint axis
        """
        rotation_in_radians = np.deg2rad(rotation_in_degrees)
        # We do this manually, in case we want to move to torch
        # later on to make this differentiable
        # the below code is similar to calling:
        # rot_mat = Rotation.from_rotvec(rotation_in_radians * direction)
        x, y, z = direction
        c = math.cos(rotation_in_radians)
        s = math.sin(rotation_in_radians)
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
        aff_mat = np.array([
            [x * xC + c, xyC - zs, zxC + ys, 0],
            [xyC + zs, y * yC + c, yzC - xs, 0],
            [zxC - ys, yzC + xs, z * zC + c, 0],
            [0, 0, 0, 1]
        ])
        # rotation around the origin
        aff_mat[:3, 3] = origin - np.dot(aff_mat[:3, :3], origin)
        return aff_mat

    @staticmethod
    def get_offset_parameter_matrix(offset, origin, direction, flip=False):
        """
        Get an affine matrix (4x4) to apply the offset parameter
        """
        # The offset gets applied first
        aff_mat = np.eye(4)
        aff_mat[:3, 3] = direction * offset
        # Then if we have flip selected
        # we reflect at the origin normal to the direction
        if flip:
            # Reflection matrix
            aff_mat[:3, :3] = np.eye(3) - 2 * np.outer(direction, direction)
            # Normalized vector, vector divided by Euclidean (L2) norm
            normal = direction.squeeze()
            normal = normal / math.sqrt((normal**2).sum())
            # Flip at the origin point
            aff_mat[:3, 3] += (2.0 * np.dot(origin, normal)) * normal
        return aff_mat

    @staticmethod
    def evaluate(jps, transform, eval_method=None):
        """Cost function to evalute the performance of a joint configuration"""
        # Apply the transform to the points
        volume_samples = util.transform_pts_by_matrix(jps.volume_samples, transform)
        surface_samples = util.transform_pts_by_matrix(jps.surface_samples, transform)
        volume1, volume2 = jps.prediction_data["body_one_properties"]["volume"], jps.prediction_data["body_two_properties"]["volume"]
        area1, area2 = jps.prediction_data["body_one_properties"]["area"], jps.prediction_data["body_two_properties"]["area"]

        # Default method
        if eval_method == "default" or eval_method is None:
            overlap1, _ = JointEnvironment.calculate_overlap(jps.sdf, volume_samples)
            contact_area1, _ = JointEnvironment.calculate_contact_area(jps.sdf, surface_samples, max_contact=1.0)
            overlap2 = overlap1 * volume1 / volume2
            contact_area2 = contact_area1 * area1 / area2
            overlap = np.clip(max(overlap1, overlap2), 0, 1)
            contact_area = np.clip(max(contact_area1, contact_area2), 0, 1)

            # Penalize overlap by zeroing out the contact area when we have overlap
            if overlap > 0.1:
                cost = overlap
            else:
                cost = overlap - 10 * contact_area

            return cost, overlap, contact_area

        # With Smooth distance
        elif eval_method == "smooth":
            return JointEnvironment.evaluate_smooth(jps, volume_samples, surface_samples)

    @staticmethod
    def evaluate_smooth(jps, volume_samples, surface_samples):
        """
        Smooth cost function
        """
        overlap_threshold = 0.01
        contact_area_threshold = 0.01

        overlap, sdf_results = JointEnvironment.calculate_overlap(
            jps.sdf,
            volume_samples,
            threshold=overlap_threshold
        )
        distance, _ = JointEnvironment.calculate_distance(sdf_results=sdf_results, samples=surface_samples)
        contact_area, _ = JointEnvironment.calculate_contact_area(
            jps.sdf,
            surface_samples,
            threshold=contact_area_threshold
        )
        # Only introduce a weight for distance when there is no overlap
        if contact_area == 0.0 and overlap == 0.0:
            closest_distance_to_surface = distance
        else:
            closest_distance_to_surface = 0.0
        # Penalize overlap by zeroing out the contact area when we have overlap
        if overlap > 0.1:
            contact_area = 0.0
        # Weight each of the different components
        overlap_weighted = overlap * 0.1
        contact_area_weighted = (1.0 - contact_area) * 0.6
        distance_weighted = closest_distance_to_surface * 0.3
        cost = overlap_weighted + contact_area_weighted + distance_weighted
        return cost, overlap

    @staticmethod
    def evaluate_vs_gt(jps, pred_transform, iou=True, cd=False, num_samples=4096):
        """
        Evaluate the given transform against the ground truth
        We do this for body one only as body two is static
        """
        if not iou and not cd:
            return None, None

        # Loop over all joints and check iou against each
        num_joints = len(jps.joint_data["joints"])
        gt_transforms = np.zeros((num_joints, 4, 4))
        for joint_index, joint in enumerate(jps.joint_data["joints"]):
            gt_transform = util.transform_to_np(joint["geometry_or_origin_one"]["transform"])
            gt_transforms[joint_index] = gt_transform

        best_iou = 0
        best_cd = sys.float_info.max
        if iou:
            best_iou = JointEnvironment.calculate_iou_batch(jps, pred_transform, gt_transforms, num_samples=num_samples)
        if cd:
            best_cd = JointEnvironment.calculate_cd_batch(jps, pred_transform, gt_transforms, num_samples=num_samples)

        if iou and cd:
            return best_iou, best_cd
        elif iou:
            return best_iou, None
        elif cd:
            return None, best_cd

    @staticmethod
    def calculate_overlap(
        sdf=None,
        samples=None,
        threshold=0.01,
        sdf_results=None
    ):
        """
        Calculate the overlap using samples and an sdf
        """
        num_samples = samples.shape[0]
        # Avoid recalculating if we can
        if sdf_results is None:
            sdf_results = sdf(samples)
        overlapping = (sdf_results > threshold).sum()
        return (overlapping / num_samples), sdf_results

    @staticmethod
    def calculate_contact_area(
        sdf=None,
        samples=None,
        threshold=0.01,
        max_contact=0.1,
        sdf_results=None
    ):
        """
        Calculate the contact area using samples and an sdf with a default tolerance in cm
        and the max contact area expected e.g. half (0.5) of all samples
        """
        num_samples = samples.shape[0]
        # Avoid recalculating if we can
        if sdf_results is None:
            sdf_results = sdf(samples)
        in_contact = (np.absolute(sdf_results) < threshold).sum()
        contact_percent = in_contact / (num_samples * max_contact)
        # Cap at 1.0
        if contact_percent > 1.0:
            contact_percent = 1.0
        return contact_percent, sdf_results

    @staticmethod
    def calculate_dofs(
        sdf=None,
        samples=None,
        translation=0.5,
        rotation_in_degrees=15,
    ):
        """
        Calculate the dofs available by translating and rotating
        the samples and checking for collision with the sdf
        """
        num_samples = samples.shape[0]
        overlap_threshold = 0.0
        translations = [
            [translation, 0, 0],
            [0, translation, 0],
            [0, 0, translation],
            [-translation, 0, 0],
            [0, -translation, 0],
            [0, 0, -translation],
        ]
        dof_count = 0
        for translation in translations:
            translated_samples = samples + translation
            overlap, _ = JointEnvironment.calculate_overlap(sdf, translated_samples, threshold=0.1)
            if overlap > overlap_threshold:
                dof_count += 1

        # Move to the origin before rotating
        centroid = np.mean(samples, axis=0)
        centered_samples = samples - centroid
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])
        rotation_in_radians = np.deg2rad(rotation_in_degrees)
        rotations = [
            Rotation.from_rotvec(rotation_in_radians * x_axis),
            Rotation.from_rotvec(-rotation_in_radians * x_axis),
            Rotation.from_rotvec(rotation_in_radians * y_axis),
            Rotation.from_rotvec(-rotation_in_radians * y_axis),
            Rotation.from_rotvec(rotation_in_radians * z_axis),
            Rotation.from_rotvec(-rotation_in_radians * z_axis)
        ]
        for rotation in rotations:
            rotated_samples = rotation.apply(centered_samples)
            rotated_samples += centroid
            overlap, _ = JointEnvironment.calculate_overlap(sdf, rotated_samples, threshold=0.1)
            if overlap > overlap_threshold:
                dof_count += 1

        all_dof_count = len(translations) + len(rotations)
        return dof_count / all_dof_count

    @staticmethod
    def calculate_distance(
        sdf=None,
        samples=None,
        sdf_results=None
    ):
        """
        Calculate the average distance between
        the point cloud samples and sdf
        """
        num_samples = samples.shape[0]
        # Avoid recalculating if we can
        if sdf_results is None:
            sdf_results = sdf(samples)
        smooth_dist = -(sdf_results.max())
        alpha = 1.0
        beta = 0.25
        return alpha - np.exp(-beta * smooth_dist), sdf_results

    @staticmethod
    def calculate_iou(
        mesh1,
        samples1,
        mesh2,
        samples2,
        threshold=0.01
    ):
        """
        Calculate the intersection over union
        between the ground truth sdf and
        the dynamic samples
        """
        wns1 = igl.fast_winding_number_for_meshes(mesh1.vertices, mesh1.faces, samples2)
        # Samples that are inside both meshes
        overlap = samples2[wns1 > threshold]
        overlap_count = len(overlap)
        # Samples only inside mesh2
        only_mesh2 = samples2[wns1 <= threshold]
        only_mesh2_count = len(only_mesh2)
        wns2 = igl.fast_winding_number_for_meshes(mesh2.vertices, mesh2.faces, samples1)
        # Samples only inside mesh1
        only_mesh1 = samples1[wns2 <= threshold]
        only_mesh1_count = len(only_mesh1)
        # Union of the samples in only mesh1/2 and overlapping
        union_count = overlap_count + only_mesh1_count + only_mesh2_count
        iou = overlap_count / union_count
        return iou

    @staticmethod
    def calculate_iou_batch(jps, pred_transform, gt_transforms, num_samples=4096):
        # Sample the points once
        # then we will translate them for each transform
        # For IoU we want samples inside of the volume
        gt_vol_pts = JointPredictionSet.sample_volume_points(
            jps.body_one_mesh,
            num_samples=num_samples,
            seed=jps.seed,
            sample_surface=False
        )
        # Store the samples as n,4 so they can be easily transformed with a 4x4 matrix
        gt_vol_pts = util.pad_pts(gt_vol_pts)

        # Copy and transform here as simply transforming the vertices
        # causes the face normals to invert if the flip parameter is used
        pred_mesh = jps.body_one_mesh.copy()
        pred_mesh.apply_transform(pred_transform)
        # Copy the verts so igl doesn't complain after we used transpose
        pred_vol_pts = util.transform_pts_by_matrix(gt_vol_pts, pred_transform, copy=True)

        best_iou = 0
        for gt_transform in gt_transforms:
            # Get a copy of the body one mesh
            # and apply the gt transform
            gt_mesh = trimesh.Trimesh(
                vertices=jps.body_one_mesh.vertices,
                faces=jps.body_one_mesh.faces
            )
            gt_mesh.apply_transform(gt_transform)
            # Tranform and copy the verts so igl doesn't complain after we used transpose
            gt_vol_pts_t = util.transform_pts_by_matrix(gt_vol_pts, gt_transform, copy=True)

            iou_result = JointEnvironment.calculate_iou(
                pred_mesh,
                pred_vol_pts,
                gt_mesh,
                gt_vol_pts_t
            )
            if iou_result > best_iou:
                best_iou = iou_result
        return best_iou

    @staticmethod
    def get_pc_scale(pc):
        return np.sqrt(np.max(np.sum((pc - np.mean(pc, axis=0))**2, axis=1)))

    @staticmethod
    def calculate_cd(pc1, pc2):
        dist = cdist(pc1, pc2)
        error = np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))
        scale = JointEnvironment.get_pc_scale(pc1) + JointEnvironment.get_pc_scale(pc2)
        return error / scale

    @staticmethod
    def calculate_cd_batch(jps, pred_transform, gt_transforms, num_samples=4096, debug_plot=False):
        # For chamfer distance we want samples on the surface
        gt_surf_pts1, _ = trimesh.sample.sample_surface(jps.body_one_mesh, num_samples)
        # Store the samples as n,4 so they can be easily transformed with a 4x4 matrix
        gt_surf_pts1 = util.pad_pts(gt_surf_pts1)
        # Transform the predicted samples
        pred_surf_pts1_t = util.transform_pts_by_matrix(gt_surf_pts1, pred_transform)
        # Transform the static body
        gt_surf_pts2, _ = trimesh.sample.sample_surface(jps.body_two_mesh, num_samples)
        gt_transform2 = util.transform_to_np(jps.joint_data["joints"][0]["geometry_or_origin_two"]["transform"])
        gt_surf_pts2_t = util.transform_pts_by_matrix(gt_surf_pts2, gt_transform2)
        # Merge the predicted samples
        pred_surf_pts_t = np.vstack([pred_surf_pts1_t, gt_surf_pts2_t])
        # Loop over each joint and take the lowest chamfer distance
        best_cd = sys.float_info.max
        for gt_transform in gt_transforms:
            gt_surf_pts1_t = util.transform_pts_by_matrix(gt_surf_pts1, gt_transform)
            gt_surf_pts_t = np.vstack([gt_surf_pts1_t, gt_surf_pts2_t])
            cd_result = JointEnvironment.calculate_cd(pred_surf_pts_t, gt_surf_pts_t)
            if cd_result < best_cd:
                best_cd = cd_result
            if debug_plot:
                plot_util.plot_point_cloud(
                    pred_surf_pts_t,
                    gt_surf_pts_t,
                )
        return best_cd
