"""

Class representing a Joint Set
containing ground truth joints between two bodies

"""

import math
import warnings
import json
from pathlib import Path
import numpy as np
from utils import util
import trimesh
from geometry.obj_reader import OBJReader


class JointSet():
    def __init__(self, joint_file, load_bodies=True):
        if isinstance(joint_file, str):
            joint_file = Path(joint_file)
        if not joint_file.exists():
            print(f"Error: Joint file not found {joint_file}")
        self.joint_file = joint_file
        self.dataset_dir = joint_file.parent
        with open(joint_file, "r", encoding="utf-8") as f:
            self.joint_data = json.load(f)
        # Load the obj files into a datastructure with B-Rep information
        body_one_name = f"{self.joint_data['body_one']}.obj"
        body_two_name = f"{self.joint_data['body_two']}.obj"
        self.body_one_obj_file = self.dataset_dir / body_one_name
        self.body_two_obj_file = self.dataset_dir / body_two_name
        if load_bodies:
            # Load a data structure keeping track of the b-rep entities
            self.body_one = self.load_obj(self.body_one_obj_file)
            self.body_two = self.load_obj(self.body_two_obj_file)
            self.bodies_loaded = True
            # Store a trimesh as well
            self.body_one_mesh = self.body_one.get_mesh()
            self.body_two_mesh = self.body_two.get_mesh()
        else:
            self.body_one = None
            self.body_two = None
            self.body_one_mesh = trimesh.load(self.body_one_obj_file)
            self.body_two_mesh = trimesh.load(self.body_two_obj_file)
            self.bodies_loaded = False

    def get_meshes(
        self,
        joint_index=0,
        apply_transform=True,
        show_joint_entity_colors=True,
        show_joint_equivalent_colors=True,
        return_vertex_normals=False,
        body_one_transform=None
    ):
        """
        Load the meshes from the joint file and
        transform based on the joint index
        """
        assert self.bodies_loaded, "Joint set bodies not loaded"
        # Combine the triangles from both bodies
        # into a single list of vertices and face indices
        vertices_list = []
        faces_list = []
        colors_list = []
        edges_ent_list = []
        edges_eq_list = []
        normals_list = []
        normal_indices_list = []
        #
        # Mesh 1
        m1 = self.get_mesh(
            body=1,
            joint_index=joint_index,
            apply_transform=apply_transform,
            show_joint_entity_colors=show_joint_entity_colors,
            show_joint_equivalent_colors=show_joint_equivalent_colors,
            return_vertex_normals=return_vertex_normals,
            transform=body_one_transform
        )
        if return_vertex_normals:
            v1, f1, c1, e1, n1, ni1 = m1
            normals_list.append(n1)
            normal_indices_list.append(ni1)
        else:
            v1, f1, c1, e1 = m1
        vertices_list.append(v1)
        faces_list.append(f1)
        colors_list.append(c1)
        if e1 is not None and e1[0] is not None:
            edges_ent_list.append(e1[0])
        if e1 is not None and e1[1] is not None:
            edges_eq_list.append(e1[1])
        #
        # Mesh 2
        m2 = self.get_mesh(
            body=2,
            joint_index=joint_index,
            apply_transform=apply_transform,
            show_joint_entity_colors=show_joint_entity_colors,
            show_joint_equivalent_colors=show_joint_equivalent_colors,
            return_vertex_normals=return_vertex_normals
        )
        if return_vertex_normals:
            v2, f2, c2, e2, n2, ni2 = m2
            normals_list.append(n2)
            normal_indices_list.append(ni2 + n1.shape[0])
        else:
            v2, f2, c2, e2 = m2
        vertices_list.append(v2)
        faces_list.append(f2 + v1.shape[0])
        colors_list.append(c2)
        if e2 is not None and e2[0] is not None:
            edges_ent_list.append(e2[0] + v1.shape[0])
        if e2 is not None and e2[1] is not None:
            edges_eq_list.append(e2[1] + v1.shape[0])
        v = np.concatenate(vertices_list)
        f = np.concatenate(faces_list)
        c = np.concatenate(colors_list)
        e = None
        e_ent = None
        e_eq = None
        if len(edges_ent_list) > 0:
            e_ent = np.concatenate(edges_ent_list)
        if len(edges_eq_list) > 0:
            e_eq = np.concatenate(edges_eq_list)
        e = [e_ent, e_eq]
        if return_vertex_normals:
            n = np.concatenate(normals_list)
            ni = np.concatenate(normal_indices_list)
            return v, f, c, e, n, ni
        else:
            return v, f, c, e

    def get_mesh(
        self,
        body=1,
        joint_index=0,
        apply_transform=True,
        show_joint_entity_colors=True,
        show_joint_equivalent_colors=True,
        return_vertex_normals=False,
        transform=None
    ):
        """
        Get a single mesh with an optional transform based on the joint index
        The body value can be either 1 or 2
        """
        assert self.bodies_loaded, "Joint set bodies not loaded"
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        # There can be more than one joint in a joint set
        # each will give us different orientations
        joint = self.joint_data["joints"][joint_index]
        # The joint geometry contains the transform of each
        # body to assemble the joint
        if body == 1:
            geo = joint["geometry_or_origin_one"]
            solid = self.body_one
        elif body == 2:
            geo = joint["geometry_or_origin_two"]
            solid = self.body_two
        m = self.load_mesh_from_data(
            geo,
            solid,
            apply_transform,
            return_vertex_normals=return_vertex_normals,
            transform=transform
        )
        if return_vertex_normals:
            v, f, n, ni = m
        else:
            v, f = m
        c, e = self.get_mesh_colors(
            body,
            joint_index,
            show_joint_entity_colors,
            show_joint_equivalent_colors
        )
        if return_vertex_normals:
            return v, f, c, e, n, ni
        else:
            return v, f, c, e

    def get_joint_entity_indices(self, body=1, joint_index=0):
        """
        Get the indices for the joint entity selected by the user
        Returns an array of triangle indices and line segments

        """
        assert self.bodies_loaded, "Joint set bodies not loaded"
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        joint = self.joint_data["joints"][joint_index]
        if body == 1:
            geo = joint["geometry_or_origin_one"]
            solid = self.body_one
        elif body == 2:
            geo = joint["geometry_or_origin_two"]
            solid = self.body_two
        entity_type = geo["entity_one"]["type"]
        entity_index = geo["entity_one"]["index"]
        if entity_type == "BRepFace":
            tri_face_indices = solid.get_triangle_face_indices()
            entity_tris = tri_face_indices == entity_index
            return entity_tris.astype(int), None
        elif entity_type == "BRepEdge":
            return None, solid.get_polyline(entity_index)

    def get_joint_entity_equivalent_indices(self, body=1, joint_index=0):
        """
        Get the indices for the joint entity equivalents
        that define the same joint axis line set by the user
        Returns an array of triangle indices and line segments
        """
        assert self.bodies_loaded, "Joint set bodies not loaded"
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        joint = self.joint_data["joints"][joint_index]
        if body == 1:
            geo = joint["geometry_or_origin_one"]
            solid = self.body_one
        elif body == 2:
            geo = joint["geometry_or_origin_two"]
            solid = self.body_two
        if "entity_one_equivalents" not in geo:
            return None, None
        else:
            tri_face_indices = solid.get_triangle_face_indices()
            triangles = np.zeros(len(tri_face_indices), dtype=int)
            lines = []
            for equivalent in geo["entity_one_equivalents"]:
                entity_type = equivalent["type"]
                entity_index = equivalent["index"]
                if entity_type == "BRepFace":
                    entity_tris = tri_face_indices == entity_index
                    triangles = np.maximum(triangles, entity_tris.astype(int))
                elif entity_type == "BRepEdge":
                    lines.append(solid.get_polyline(entity_index))
            if len(lines) > 0:
                lines = np.concatenate(lines)
            else:
                lines = None
            return triangles, lines

    def get_joint_axis_line(
        self,
        body=1,
        joint_index=0,
        apply_transform=True,
        axis_length_scale_factor=0.35
    ):
        """
        Get a line representing the joint axis
        """
        assert self.bodies_loaded, "Joint set bodies not loaded"
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        joint = self.joint_data["joints"][joint_index]
        if body == 1:
            geo = joint["geometry_or_origin_one"]
            solid = self.body_one
        elif body == 2:
            geo = joint["geometry_or_origin_two"]
            solid = self.body_two
        assert "axis_line" in geo
        origin = util.vector_to_np(geo["axis_line"]["origin"])[:3]
        direction = util.vector_to_np(
            geo["axis_line"]["direction"]
        )[:3]
        # Get where the axis intersects the aabb
        # tmax will be the distance to the intersection in the positive direction
        # tmin the distance in the negative direction
        tmin, tmax = self.get_joint_axis_aabb_intersections(
            origin=origin,
            direction=direction,
            body=body,
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
        start_pt = np.reshape(start_pt, (1, 3))
        end_pt = np.reshape(end_pt, (1, 3))
        # The axis is in assembled coordinates
        # so if we don't want to apply the transform we need to
        # undo the transform that has been applied
        if apply_transform is False:
            transform_dict = geo["transform"]
            transform = util.transform_to_np(transform_dict)
            inv_transform = np.linalg.inv(transform)
            start_pt = util.transform_pts_by_matrix(start_pt, inv_transform)
            end_pt = util.transform_pts_by_matrix(end_pt, inv_transform)
        return start_pt, end_pt

    def get_joint_axis_aabb_intersections(self, origin, direction, body=1, offset=None):
        """
        Get the distances from the origin along the joint axis
        where the axis intersections with the axis aligned bounding box
        """
        # Offset so the origin is outside the bounding box
        if offset is None:
            offset = np.max(self.body_one_mesh.extents) + np.max(self.body_two_mesh.extents)
        origin_offset = origin + offset * direction
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

    def number_of_joints(self):
        """
        Get the number of joints in this joint set
        """
        return len(self.joint_data["joints"])

    def get_edge_indices(self, body=1):
        """
        Get the B-Rep edge indices for a single body
        with an optional transform based on the joint index
        The body value can be either 1 or 2
        """
        assert self.bodies_loaded, "Joint set bodies not loaded"
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        if body == 1:
            solid = self.body_one
        elif body == 2:
            solid = self.body_two
        return solid.get_polylines()

    def load_obj(self, obj_file):
        """
        Load the mesh into a data structure containing the B-Rep information
        """
        obj = OBJReader(obj_file)
        return obj.read()

    def load_mesh_from_data(
        self,
        geometry_or_origin,
        solid,
        apply_transform=True,
        return_vertex_normals=False,
        transform=None
    ):
        """
        Load the mesh and transform it
        according to the joint geometry_or_origin
        """
        assert self.bodies_loaded, "Joint set bodies not loaded"
        v = solid.get_vertices()
        f = solid.get_triangles()
        if return_vertex_normals:
            n = solid.get_normals()
            ni = solid.get_normal_indices()
        if apply_transform:
            if transform is None:
                transform_dict = geometry_or_origin["transform"]
                transform = util.transform_to_np(transform_dict)
            v = util.transform_pts_by_matrix(
                v,
                transform
            )
            if return_vertex_normals:
                # Rotation without translation
                rot_mat = np.eye(4)
                rot_mat[:3, :3] = transform[:3, :3]
                n = util.transform_pts_by_matrix(
                    n,
                    rot_mat
                )
        if return_vertex_normals:
            return v, f, n, ni
        else:
            return v, f

    def get_mesh_colors(
        self,
        body=1,
        joint_index=0,
        show_joint_entity_colors=True,
        show_joint_equivalent_colors=True
    ):
        """
        Get the list of colors for each triangle
        """
        assert self.bodies_loaded, "Joint set bodies not loaded"
        valid_bodies = {1, 2}
        assert body in valid_bodies, "Invalid body, please specify 1 or 2"
        # Color map to use for triangles
        color_map = np.array([
            [0.75, 1.00, 0.75],  # Body 1
            [0.75, 0.75, 1.00],  # Body 2
            [1.00, 0.00, 0.00],  # Red for joint entities
            [1.00, 1.00, 0.00],  # Yellow for joint equivalents
        ], dtype=float)

        f_ent = None
        e_ent = None
        f_eq = None
        e_eq = None
        if show_joint_entity_colors:
            f_ent, e_ent = self.get_joint_entity_indices(body, joint_index)
            if f_ent is not None:
                # Set the entities to color index 2
                f_ent[f_ent == 1] = 2
        if show_joint_equivalent_colors:
            f_eq, e_eq = self.get_joint_entity_equivalent_indices(
                body,
                joint_index
            )
            if f_eq is not None:
                # Set the equivalents to color index 3
                f_eq[f_eq == 1] = 3

        # Faces
        # Combine if we are showing both entities and equivalents
        # Entities override equivalents by taking the mimimum
        fc = None
        if f_ent is not None and f_eq is not None:
            # fc = np.minimum(f_ent, f_eq)
            mask = (f_ent == 0)
            fc = np.copy(f_ent)
            fc[mask] = f_eq[mask]
        elif f_ent is not None:
            fc = f_ent
        elif f_eq is not None:
            fc = f_eq
        # Face colors for the triangles
        # that aren't entities or equivalents
        if body == 1:
            tri_count = self.body_one.get_triangle_count()
            bc = np.zeros(tri_count, dtype=int)
        elif body == 2:
            tri_count = self.body_two.get_triangle_count()
            bc = np.ones(tri_count, dtype=int)
        # Combine together, giving the entities/equivalents priority
        if fc is not None:
            fc = np.maximum(fc, bc)
        else:
            fc = bc
        mesh_colors = color_map[fc]

        # Edges
        ec = None
        if e_ent is not None and e_eq is not None:
            ec = [e_ent, e_eq]
        elif e_ent is not None:
            ec = [e_ent, None]
        elif e_eq is not None:
            ec = [None, e_eq]
        return mesh_colors, ec

    def get_mesh_edges(self, body=2, joint_index=0, apply_transform=False):
        """Get the mesh edges to draw wireframes"""
        assert body in {1, 2}
        joint = self.joint_data["joints"][joint_index]
        if body == 1:
            mesh = self.body_one_mesh
            transform = util.transform_to_np(joint["geometry_or_origin_one"]["transform"])
        elif body == 2:
            mesh = self.body_two_mesh
            transform = util.transform_to_np(joint["geometry_or_origin_two"]["transform"])
        if apply_transform:
            mesh = mesh.copy()
            mesh.apply_transform(transform)
        f_roll = np.roll(mesh.faces, -1, axis=1)
        e = np.column_stack((mesh.faces, f_roll)).reshape(-1, 2)
        return mesh.vertices, e

    def calculate_gt_corners(self, joint_index=0):
        """Get the corners of the ground truth joint's bounding box"""
        joint = self.joint_data["joints"][joint_index]
        # Body one moves, so transform it
        transform = joint["geometry_or_origin_one"]["transform"]
        v1 = util.transform_pts_by_matrix(
            self.body_one_mesh.vertices,
            util.transform_to_np(transform)
        )
        v = np.concatenate([v1, self.body_two_mesh.vertices])
        bbox = util.calculate_bounding_box(v)
        corners = trimesh.bounds.corners(bbox)
        return corners
