"""

Mesh data structures

"""


import numpy as np


class Vertex():
    def __init__(self, obj_line):
        # v 2.8 3.0 2.0
        assert obj_line[0] == "v"
        assert len(obj_line) == 4
        self.x = float(obj_line[1])
        self.y = float(obj_line[2])
        self.z = float(obj_line[3])


class Normal():
    def __init__(self, obj_line):
        # vn -0.0 -0.0 -1.0
        assert obj_line[0] == "vn"
        assert len(obj_line) == 4
        self.x = float(obj_line[1])
        self.y = float(obj_line[2])
        self.z = float(obj_line[3])


class PolyLine():
    def __init__(self, obj_line):
        # l 23 24 25 26 27 28
        assert obj_line[0] == "l"
        # .obj indices start from 1
        self.vertex_indices = [int(i) - 1 for i in obj_line[1:]]

    def get_vertex_indices(self):
        """Return a numpy array of vertex indices"""
        indices = np.array(self.vertex_indices, dtype=np.uint64)
        return indices


class Triangle():
    def __init__(self, obj_line):
        # f 2//1 44//1 1//2
        assert obj_line[0] == "f"
        self.vertex_indices = []
        self.normal_indices = []
        for vn in obj_line[1:]:
            vns = vn.split("//")
            assert len(vns) == 2, "Expected vertex//normal combination"
            # .obj indices start from 1
            self.vertex_indices.append(int(vns[0]) - 1)
            self.normal_indices.append(int(vns[1]) - 1)

    def get_vertex_indices(self):
        """Return a numpy array of vertex indices"""
        return np.array([
            self.vertex_indices[0],
            self.vertex_indices[1],
            self.vertex_indices[2]
        ], dtype=np.uint64)

    def get_normal_indices(self):
        """Return a numpy array of normal indices"""
        return np.array([
            self.normal_indices[0],
            self.normal_indices[1],
            self.normal_indices[2]
        ], dtype=np.uint64)
