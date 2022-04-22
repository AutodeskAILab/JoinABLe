"""

Class representing a Joint Prediction Set
containing ground truth joints between two bodies
and extended with prediction information

"""

import json
import math
import random
import warnings
from pathlib import Path

import torch
import numpy as np
import trimesh
import igl
from pysdf import SDF
import torch.nn.functional as F
from torch_geometric.data import Batch

from joint.joint_set import JointSet
from utils import util
from joint import joint_axis


class JointPredictionSet(JointSet):
    def __init__(
        self,
        joint_file, g1, g2, joint_graph, model,
        load_bodies=True,
        num_samples=4096,
        seed=None,
        prediction_limit=50
    ):
        super().__init__(joint_file, load_bodies=load_bodies)
        # Run inference and store the predictions in a cacheable data structure
        self.prediction_data = self.get_prediction_data(joint_file, g1, g2, joint_graph, model, prediction_limit)
        self.num_samples = num_samples
        self.seed = seed
        self.volume_samples = JointPredictionSet.sample_volume_points(self.body_one_mesh, self.num_samples, self.seed)
        self.surface_samples, _ = trimesh.sample.sample_surface(self.body_one_mesh, num_samples)
        # Store the samples as n,4 so they can be easily transformed with a 4x4 matrix
        self.volume_samples = util.pad_pts(self.volume_samples)
        self.surface_samples = util.pad_pts(self.surface_samples)
        # Apply the body two transform so we are in the correct coords
        # We take the first joint transform, assuming they will be constant
        transform2 = util.transform_to_np(self.joint_data["joints"][0]["geometry_or_origin_two"]["transform"])
        body_two_mesh_temp = self.body_two_mesh.copy()
        body_two_mesh_temp.apply_transform(transform2)
        self.sdf = SDF(body_two_mesh_temp.vertices, body_two_mesh_temp.faces)

    @staticmethod
    def sample_volume_points(mesh, num_samples=4096, seed=None, sample_surface=True):
        """Sample num_samples random points within the volume of the mesh
           We use fast winding numbers so we don't need a water tight mesh"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        all_samples = []
        all_sample_count = 0
        if sample_surface:
            # First lets make sure we get some samples on the surfaces
            surface_sample_count = int(num_samples * 0.15)
            surface_samples, surface_sample_face_indices = trimesh.sample.sample_surface(mesh, surface_sample_count)
            # Move the points back to inside of the surface
            surface_sample_face_normals = mesh.face_normals[surface_sample_face_indices]
            surface_samples -= surface_sample_face_normals * 0.01
            all_samples.append(surface_samples)
            all_sample_count += surface_samples.shape[0]
        # Next lets try trimesh for internal sample, which requires a watertight mesh
        if mesh.is_watertight:
            if sample_surface:
                trimesh_samples_count = num_samples - surface_samples.shape[0]
            else:
                trimesh_samples_count = num_samples
            trimesh_samples = trimesh.sample.volume_mesh(mesh, trimesh_samples_count)
            if trimesh_samples.shape[0] > 0:
                all_samples.append(trimesh_samples)
                all_sample_count += trimesh_samples.shape[0]
            if all_sample_count == num_samples:
                all_samples_np = np.concatenate(all_samples)
                return all_samples_np[:num_samples, :]

        # We have an open mesh, so fall back to using fast winding numbers
        box_min = mesh.bounds[0]
        box_max = mesh.bounds[1]
        # Loop until we have sufficient samples
        while all_sample_count < num_samples:
            xyz_samples = np.column_stack((
                np.random.uniform(box_min[0], box_max[0], num_samples),
                np.random.uniform(box_min[1], box_max[1], num_samples),
                np.random.uniform(box_min[2], box_max[2], num_samples)
            ))
            wns = igl.fast_winding_number_for_meshes(mesh.vertices, mesh.faces, xyz_samples)
            # Add a small threshold here rather then > 0
            # due to seeing some outside points included
            inside_samples = xyz_samples[wns > 0.01]
            # If we can't generate any volume samples
            # this may be a super thin geometry so break
            if len(inside_samples) == 0:
                break
            all_samples.append(inside_samples)
            all_sample_count += inside_samples.shape[0]

        # We should only need to add additional surface samples if
        # we failed to get volume samples due to a super thin geometry
        if all_sample_count < num_samples and (box_max - box_min).min() < 1e-10:
            # Sample again from the surface
            surface_sample_count = num_samples - all_sample_count
            surface_samples, surface_sample_face_indices = trimesh.sample.sample_surface(mesh, surface_sample_count)
            # Move the points back to inside of the surface
            surface_sample_face_normals = mesh.face_normals[surface_sample_face_indices]
            surface_samples -= surface_sample_face_normals * 0.01
            all_samples.append(surface_samples)
            all_sample_count += surface_samples.shape[0]

        # Concat and return num_samples only
        all_samples_np = np.concatenate(all_samples)
        return all_samples_np[:num_samples, :]

    @staticmethod
    def get_network_predictions(g1, g2, joint_graph, model):
        """Get the network predictions"""
        fake_batch = (
            Batch.from_data_list([g1]),
            Batch.from_data_list([g2]),
            Batch.from_data_list([joint_graph]),
        )
        x = model(fake_batch)
        prob = F.softmax(x, dim=0)
        return prob.view(g1.num_nodes, g2.num_nodes)

    def get_prediction_data(self, joint_file, g1, g2, joint_graph, model, prediction_limit):
        """
        Generate the prediction data structure
        Returns a dictionary that can be serialized to json for caching
        """
        # g1 and g2 are the B-Rep bodies represented as graphs with learning features
        # label_matrix has the user-selected joints
        label_matrix = joint_graph.edge_attr.view(joint_graph.num_nodes_graph1, joint_graph.num_nodes_graph2)
        preds = JointPredictionSet.get_network_predictions(g1, g2, joint_graph, model)

        # The graph files that we use to reference the original
        # graph, brep and mesh files
        g1_file = self.dataset_dir / f"{self.joint_data['body_one']}.json"
        g2_file = self.dataset_dir / f"{self.joint_data['body_two']}.json"
        # Get the graphs from the cache we preloaded
        # unlike the g1 and g2 graphs that contain only the learning features
        # these contain additional information for calculating a joint axis
        with open(g1_file, "r", encoding="utf-8") as f:
            g1_graph = json.load(f)
        with open(g2_file, "r", encoding="utf-8") as f:
            g2_graph = json.load(f)
        # Header information for our predictions file
        joint_data = {}
        joint_data["joint_set"] = joint_file.stem
        joint_data["body_one"] = g1_file.stem
        joint_data["body_two"] = g2_file.stem
        joint_data["body_one_face_count"] = g1_graph["properties"]["face_count"]
        joint_data["body_two_face_count"] = g2_graph["properties"]["face_count"]
        joint_data["representation"] = "graph"
        joint_data["prediction_method"] = "network"
        joint_data["predictions"] = []
        # Get the labels for each 1 in the label matrix
        # corresponding to a node in each graph
        label_indices = torch.nonzero(label_matrix)
        g1_labels = set(label_indices[:, 0].tolist())
        g2_labels = set(label_indices[:, 1].tolist())
        # Get the sorted list of 2D indices to the top-k predictions
        # where k is the prediction limit of how many entities we want to search over
        preds_np = preds.detach().numpy()
        g1_preds, g2_preds = np.unravel_index(np.argsort(-preds_np, axis=None), preds_np.shape)
        # Iterate over the top k predictions
        for g1_pred, g2_pred in zip(g1_preds[:prediction_limit], g2_preds[:prediction_limit]):
            # Prediction value
            pred_value = preds_np[g1_pred, g2_pred]
            # For each prediction
            # Get the node from the g1 and g2
            g1_node = g1_graph["nodes"][g1_pred]
            g2_node = g2_graph["nodes"][g2_pred]
            g1_type = "BRepFace" if "surface_type" in g1_node else "BRepEdge"
            g2_type = "BRepFace" if "surface_type" in g2_node else "BRepEdge"
            # Use the node features to find the axis
            pt1, dir1 = joint_axis.find_axis_line(g1_node)
            pt2, dir2 = joint_axis.find_axis_line(g2_node)
            # We don't support all entities e.g. Nurbs
            # So only report those we do support
            if pt1 is not None and pt2 is not None:
                # Cast to regular int for json serialization
                prediction = {
                    "value": float(pred_value),
                    "body_one": {
                        "index": int(g1_pred),
                        "type": g1_type,
                        "origin": pt1,
                        "direction": dir1,
                    },
                    "body_two": {
                        "index": int(g2_pred),
                        "type": g2_type,
                        "origin": pt2,
                        "direction": dir2,
                    }
                }
                joint_data["predictions"].append(prediction)
        # If we don't have any predictions we are out of luck
        if len(joint_data["predictions"]) == 0:
            return joint_file.stem, None
        # Properties
        joint_data["body_one_properties"] = g1_graph["properties"]
        joint_data["body_two_properties"] = g2_graph["properties"]
        return joint_data

    def get_joint_predictions(self, body=1, limit=None):
        """
        Get the predictions for joint entities
        Returns an array of per-triangle predictions
        """
        assert self.bodies_loaded, "Joint set bodies not loaded"
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        if body == 1:
            solid = self.body_one
            suffix = "one"
        elif body == 2:
            solid = self.body_two
            suffix = "two"
        num_faces = self.prediction_data[f"body_{suffix}_face_count"]
        values = {
            "BRepEdge": {},
            "BRepFace": {},
        }
        if limit is None:
            limit = len(self.prediction_data)
        # Store the max prediction values
        for prediction in self.prediction_data["predictions"][:limit]:
            value = prediction["value"]
            body_key = f"body_{suffix}"
            body_pred = prediction[body_key]
            index = body_pred["index"]
            # We need to subtract the number of faces
            # as faces are stored before edges
            # in the graph order used by the network
            entity_type = body_pred["type"]
            if entity_type == "BRepEdge":
                index -= num_faces
            if index not in values[entity_type]:
                values[entity_type][index] = 0
            if values[entity_type][index] < value:
                values[entity_type][index] = value
        # Face scalars
        tri_face_indices = solid.get_triangle_face_indices()
        triangles = np.zeros(len(tri_face_indices), dtype=float)
        for index, value in values["BRepFace"].items():
            mask = (tri_face_indices == index)
            triangles[mask] = value
        # Edge lines
        lines = []
        for index, value in values["BRepEdge"].items():
            lines.append(solid.get_polyline(index))
        if len(lines) > 0:
            lines = np.concatenate(lines)
        else:
            lines = None
        return triangles, lines

    def get_joint_prediction_axis_lines(self, body=1, limit=None, axis_length_scale_factor=0.35):
        """Get lines representing the predicted axis lines"""
        assert self.bodies_loaded, "Joint set bodies not loaded"
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        if body == 1:
            solid = self.body_one
        elif body == 2:
            solid = self.body_two
        num_preds = len(self.prediction_data["predictions"])
        if limit is not None:
            num_preds = limit
        start_points = np.zeros((num_preds, 3))
        end_points = np.zeros((num_preds, 3))
        for i, prediction in enumerate(self.prediction_data["predictions"][:num_preds]):
            origin, direction = self.get_joint_prediction_axis(body, i)
            # Get where the axis intersects the aabb
            # tmax will be the distance to the intersection in the positive direction
            # tmin the distance in the negative direction
            tmin, tmax = self.get_joint_prediction_axis_aabb_intersections(
                body=body,
                prediction_index=i,
                origin=origin,
                direction=direction,
                offset=0
            )
            if tmax is None or math.isinf(tmax) or math.isnan(tmax):
                tmax = 0
            if tmin is None or math.isinf(tmin) or math.isnan(tmin):
                tmin = 0
            # tmax will be very small if we are on a face and pointing away from it
            distance_to_aabb = abs(tmax)
            # So we want to calculate how far to extend beyond the aabb
            # Span across the aabb
            aabb_span = abs(tmin) + abs(tmax)
            # Span across the bounding box xyz
            v_min, v_max = solid.get_bounding_box()
            span = v_max - v_min
            # Add the axis span to the list of spans
            span_list = span.tolist()
            span_list.append(aabb_span)
            # Take the mean of the bbox spans and axis spans
            # To ensure a diagonal axis of really long part doesn't
            # skew the length
            mean_span = np.mean(span_list)
            # The distance beyond the aabb that we want to extend by
            # This is relative to the length of the axis inside the aabb
            distance_beyond_aabb = mean_span * axis_length_scale_factor
            # The total distance of the axis
            axis_length = distance_to_aabb + distance_beyond_aabb
            start_pt = origin
            end_pt = origin + direction * axis_length
            start_points[i] = start_pt
            end_points[i] = end_pt
        return start_points, end_points

    def get_joint_prediction_axis(self, body=1, prediction_index=0):
        """Get the joint axis (origin and direction) for the given prediction"""
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        if body == 1:
            suffix = "one"
        elif body == 2:
            suffix = "two"
        prediction = self.prediction_data["predictions"][prediction_index]
        origin = util.vector_to_np(prediction[f"body_{suffix}"]["origin"])[:3]
        direction = util.vector_to_np(prediction[f"body_{suffix}"]["direction"])[:3]
        length = np.linalg.norm(direction)
        if length < 0.00000001:
            return origin, None
        direction = direction / length
        return origin, direction

    def get_joint_prediction_axis_aabb_intersections(self, body=1, prediction_index=0, origin=None, direction=None, offset=None):
        """
        Get the distances from the origin along the joint prediction axis
        where the axis intersections with the axis aligned bounding box
        """
        if origin is None and direction is None:
            origin, direction = self.get_joint_prediction_axis(body, prediction_index)
        # Offset so the origin is outside the bounding box
        if offset is None:
            offset = np.max(self.body_one_mesh.extents) + np.max(self.body_two_mesh.extents)
        origin_offset = origin + offset * direction
        if direction is None:
            return None, None
        if body == 1:
            bbox = self.body_one_mesh.bounds
        elif body == 2:
            bbox = self.body_two_mesh.bounds
        # This will produce NaN's when the components of the direction are 0
        # But this is intentional and is handled downstream in intersect_ray_box()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direction_inverse = 1 / direction
        # Try find the intersection from either direction
        tmin, tmax = util.intersect_ray_box(bbox[0], bbox[1], origin_offset, direction_inverse)
        if tmin is None or tmax is None or math.isinf(tmin) or math.isinf(tmax):
            origin_offset_inverse = origin + offset * (direction * -1)
            tmin, tmax = util.intersect_ray_box(bbox[0], bbox[1], origin_offset_inverse, direction_inverse)
        return tmin, tmax

    def get_joint_prediction_axis_convex_hull_intersections(self, body=1, prediction_index=0):
        """
        Get the points along the joint prediction axis
        where the axis intersections with the convex hull of the given body
        """
        origin, direction = self.get_joint_prediction_axis(body, prediction_index)
        if body == 1:
            suffix = "one"
        elif body == 2:
            suffix = "two"

        # Cache the convex_hull
        convex_hull = getattr(self, f"body_{suffix}_convex_hull", None)
        if convex_hull is None:
            mesh = getattr(self, f"body_{suffix}_mesh", None)
            convex_hull = mesh.convex_hull
            setattr(self, f"body_{suffix}_convex_hull", convex_hull)

        # Cache the ray mesh intersector
        ray_mesh = getattr(self, f"body_{suffix}_ray_mesh", None)
        if ray_mesh is None:
            # Slower but more accurate than trimesh.ray.ray_pyembree.RayMeshIntersector
            ray_mesh = trimesh.ray.ray_triangle.RayMeshIntersector(convex_hull)
            setattr(self, f"body_{suffix}_ray_mesh", ray_mesh)

        offset = np.max(self.body_one_mesh.extents) + np.max(self.body_two_mesh.extents)
        origin_offset = origin + offset * direction
        locs, _, _ = ray_mesh.intersects_location([origin_offset], [direction])
        if len(locs) == 0:
            locs, _, _ = ray_mesh.intersects_location([origin_offset], [direction * -1])
        # Using a convex hul this should equal 2
        if len(locs) == 2:
            return locs
        return None

    def get_joint_prediction_indices(self, limit=None):
        """Get the indices of the joint predictions"""
        indices = np.arange(len(self.prediction_data["predictions"]))
        if limit is not None:
            return indices[:limit]
        return indices

    def get_joint_prediction_probabilities(self, limit=None):
        """Get the joint prediction probabilities"""
        probs = np.array([p["value"] for p in self.prediction_data["predictions"]])
        if limit is not None:
            probs = probs[:limit]
        probs /= probs.sum()
        return probs

    def get_joint_prediction_brep_indices(self):
        """
        Get a list of the predicted joint indices of the b-rep entities
        as tuples in the form:
        [ (body_one_index, body_two_index), ... ]
        """
        return [(p["body_one"]["index"], p["body_two"]["index"]) for p in self.prediction_data["predictions"]]

    def get_joint_prediction_brep_index(self, index):
        """
        Get the predicted joint indices of the b-rep entities as a tuple: (body_one_index, body_two_index)
        """
        p = self.prediction_data["predictions"][index]
        return (p["body_one"]["index"], p["body_two"]["index"])

    def get_joint_brep_indices(self):
        """
        Get a set of the user-selected joint indices of the b-rep entities
        (including joint equivalents) as tuples in the form:
        { (body_one_index, body_two_index), ... }
        """
        joints = self.joint_data["joints"]
        face_count1 = self.prediction_data["body_one_face_count"]
        face_count2 = self.prediction_data["body_two_face_count"]
        joint_indices = set()
        for joint in joints:
            entity1 = joint["geometry_or_origin_one"]["entity_one"]
            entity1_index = entity1["index"]
            entity1_type = entity1["type"]
            entity2 = joint["geometry_or_origin_two"]["entity_one"]
            entity2_index = entity2["index"]
            entity2_type = entity2["type"]
            # Offset the joint indices for use in the label matrix
            entity1_index = self.offset_joint_brep_index(entity1_index, entity1_type, face_count1)
            entity2_index = self.offset_joint_brep_index(entity2_index, entity2_type, face_count2)
            # Set the joint equivalent indices
            eq1_indices = self.get_joint_equivalents_brep_indices(joint["geometry_or_origin_one"], face_count1)
            eq2_indices = self.get_joint_equivalents_brep_indices(joint["geometry_or_origin_two"], face_count2)
            # Add the actual entities
            eq1_indices.append(entity1_index)
            eq2_indices.append(entity2_index)
            # For every pair we set a joint
            for eq1_index in eq1_indices:
                for eq2_index in eq2_indices:
                    joint_indices.add((eq1_index, eq2_index))
            # Set the user selected joint indices
            joint_indices.add((entity1_index, entity2_index))
        return joint_indices

    def get_joint_equivalents_brep_indices(self, geometry, face_count):
        """Get the joint equivalent brep indices from joint geometry data"""
        indices = []
        if "entity_one_equivalents" in geometry:
            for entity in geometry["entity_one_equivalents"]:
                index = self.offset_joint_brep_index(
                    entity["index"],
                    entity["type"],
                    face_count
                )
                indices.append(index)
        return indices

    def offset_joint_brep_index(self, entity_index, entity_type, face_count):
        """Offset the joint index"""
        joint_index = entity_index
        if entity_type == "BRepEdge":
            # If this is a brep edge we need to increment the index
            # to start past the number of faces as those are stored first
            joint_index += face_count
        # If we have a BRepFace life is simple...
        assert joint_index >= 0
        return joint_index

    def is_joint_body_rotationally_symmetric(self, body=1, prediction_index=0, joint_axis_direction=None):
        """
        Determine if a joint body has rotational symmetry about it's joint axis
        """
        if joint_axis_direction is None:
            _, joint_axis_direction = self.get_joint_prediction_axis(body, prediction_index)
        if body == 1:
            symmetry_axis = self.body_one_mesh.symmetry_axis
        elif body == 2:
            symmetry_axis = self.body_two_mesh.symmetry_axis
        if symmetry_axis is None:
            return False
        # Check both directions
        pairs = np.array([
            [symmetry_axis, joint_axis_direction],
            [symmetry_axis, joint_axis_direction * -1]
        ])
        # pairs = np.concatenate((symmetry_axis, joint_axis_direction), axis=1).reshape((, 2, 3))
        angles = trimesh.geometry.vector_angle(pairs)
        angle_threshold = np.deg2rad(1)
        aligned_mask = angles < angle_threshold
        is_aligned = aligned_mask.sum() > 0
        return is_aligned
