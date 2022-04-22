"""

B-Rep data structures
referencing underlying mesh entities

"""


import numpy as np
import trimesh
from geometry.mesh import Vertex
from geometry.mesh import Normal
from geometry.mesh import PolyLine
from geometry.mesh import Triangle


class BRepFace():
    def __init__(self, triangles=None):
        self.triangles = []
        if triangles is not None:
            self.triangles = triangles

    def add_triangle(self, triangle: Triangle):
        self.triangles.append(triangle)


class BRepEdge():
    def __init__(self, line):
        self.line = line


class BRepBody():
    def __init__(self, file):
        # File
        self.file = file
        # Triangle vertices (xyz)
        self.vertices = []
        # Per vertex normals
        self.normals = []
        # B-Rep faces containing triangles
        self.faces = []
        # B-Rep edge containing a polyline
        self.edges = []
        # Indices for the B-Rep vertices
        self.vertex_indices = []

    def add_vertex(self, vertex: Vertex):
        self.vertices.append(vertex)

    def add_normal(self, normal: Normal):
        self.normals.append(normal)

    def add_face(self, face: BRepFace):
        self.faces.append(face)

    def add_edge(self, edge: BRepEdge):
        self.edges.append(edge)

    def add_vertex_index(self, index: int):
        self.vertex_indices.append(index)

    def get_vertices(self):
        """Return a numpy array of vertices"""
        vertices_np = np.zeros((len(self.vertices), 3))
        for index, vertex in enumerate(self.vertices):
            vertices_np[index][0] = vertex.x
            vertices_np[index][1] = vertex.y
            vertices_np[index][2] = vertex.z
        return vertices_np

    def get_triangles(self):
        """Return a numpy array of triangle indices"""
        num_tris = self.get_triangle_count()
        tris_np = np.zeros((num_tris, 3), dtype=np.uint64)
        index = 0
        for face in self.faces:
            for tri in face.triangles:
                tris_np[index] = tri.get_vertex_indices()
                index += 1
        return tris_np

    def get_triangle_count(self):
        """Return the total number of triangles"""
        return sum([len(face.triangles) for face in self.faces])

    def get_triangle_face_indices(self):
        """Return the face index of each triangle"""
        num_tris = self.get_triangle_count()
        tris_np = np.zeros((num_tris), dtype=np.uint64)
        tri_index = 0
        for face_index, face in enumerate(self.faces):
            for tri in face.triangles:
                tris_np[tri_index] = face_index
                tri_index += 1
        return tris_np

    def get_triangle_scalars(self, face_scalars):
        """Given scalar values for each face,
            return scalar values for each triangle"""
        assert len(face_scalars) == len(self.faces)
        tri_scalars = []
        for face_scalar, face in zip(face_scalars, self.faces):
            ts = [face_scalar] * len(face.triangles)
            tri_scalars.extend(ts)
        return np.array(tri_scalars)

    def get_normals(self):
        """Return a numpy array of vertex normals"""
        normals_np = np.zeros((len(self.normals), 3))
        for index, normal in enumerate(self.normals):
            normals_np[index][0] = normal.x
            normals_np[index][1] = normal.y
            normals_np[index][2] = normal.z
        return normals_np

    def get_normal_indices(self):
        """Return a numpy array of normal indices"""
        num_tris = self.get_triangle_count()
        normals_np = np.zeros((num_tris, 3), dtype=np.uint64)
        index = 0
        for face in self.faces:
            for tri in face.triangles:
                normals_np[index] = tri.get_normal_indices()
                index += 1
        return normals_np

    def get_polyline(self, index):
        """Return a numpy array of a polyline indices"""
        edge = self.edges[index]
        line_indices = edge.line.get_vertex_indices()
        repeat_indices = np.repeat(line_indices, 2)[1:-1]
        reshape_indices = np.reshape(repeat_indices, (-1, 2))
        return reshape_indices

    def get_polylines(self):
        """Return a numpy array of polylines indices"""
        lines = []
        for edge in self.edges:
            line_indices = edge.line.get_vertex_indices()
            repeat_indices = np.repeat(line_indices, 2)[1:-1]
            lines.append(repeat_indices)
        stacked_indices = np.hstack(lines)
        reshape_indices = np.reshape(stacked_indices, (-1, 2))
        return reshape_indices

    def get_polyline_scalars(self, edge_scalars):
        """Given scalar values for each edge,
            return scalar values for each polyline"""
        assert len(edge_scalars) == len(self.edges)
        polyline_scalars = []
        for face_scalar, face in zip(edge_scalars, self.edges):
            ts = [face_scalar] * len(face.triangles)
            polyline_scalars.extend(ts)
        return np.array(polyline_scalars)

    def get_bounding_box(self):
        """Get the bounding box (min and max point) of this B-Rep"""
        v = self.get_vertices()
        v_min = np.min(v, axis=0)
        v_max = np.max(v, axis=0)
        return v_min, v_max

    def get_mesh(self):
        """Get a trimesh mesh"""
        vertices = self.get_vertices()
        faces = self.get_triangles()
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
        )
        return mesh
