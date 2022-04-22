import numpy as np

from geometry.brep import BRepBody, BRepEdge, BRepFace
from geometry.mesh import Vertex, PolyLine, Triangle, Normal


class OBJReader:
    """
    A class to read an OBJ file containing a mesh
    with B-Rep face and edge groups.
    """
    def __init__(self, pathname):
        self.pathname = pathname
        assert self.pathname.exists(), "No such file"

    def read(self):
        solid = BRepBody(self.pathname)
        group_name = None
        prev_edge_index = 0
        edge_index = -1
        brep_face = None
        brep_face_active = False
        # We assume that all vertices come before curves
        # to let us process the obj in a single pass
        with open(self.pathname) as fp:
            for line in fp:
                line = line.strip()
                splitted_line = line.split()
                if not splitted_line:
                    group_name = None
                    # If there is a break after adding triangles
                    # to a face, finish up the face
                    if brep_face_active and brep_face is not None:
                        solid.add_face(brep_face)
                        brep_face_active = False
                    continue
                # Vertex
                if splitted_line[0] == 'v':
                    vertex = Vertex(splitted_line)
                    solid.add_vertex(vertex)
                # Normal
                elif splitted_line[0] == 'vn':
                    normal = Normal(splitted_line)
                    solid.add_normal(normal)
                # Face
                elif splitted_line[0] == 'f':
                    triangle = Triangle(splitted_line)
                    brep_face.add_triangle(triangle)
                # PolyLine
                elif splitted_line[0] == 'l':
                    # Only take every second edge
                    # to ignore the half edges
                    if edge_index != prev_edge_index:
                        line = PolyLine(splitted_line)
                        edge = BRepEdge(line)
                        solid.add_edge(edge)
                # Point
                elif splitted_line[0] == 'p':
                    solid.add_vertex_index(int(splitted_line[1]))
                # Group
                elif splitted_line[0] == 'g':
                    assert len(splitted_line) >= 3
                    # g face 6
                    # g halfedge 0 edge 0
                    # g vertex 0
                    group_name = splitted_line[1]
                    # Polyline group
                    if len(splitted_line) == 5:
                        prev_edge_index = edge_index
                        edge_index = int(splitted_line[4])
                    # Face group
                    if group_name == "face":
                        if not brep_face_active:
                            brep_face = BRepFace()
                            brep_face_active = True

        return solid
