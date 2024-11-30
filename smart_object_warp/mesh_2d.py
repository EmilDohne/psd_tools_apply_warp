from typing import Tuple
import copy
import math
from collections import defaultdict

import numba

import numpy as np
import cv2

import smart_object_warp.point_2d as point_2d
import smart_object_warp.image_buffer as image_buffer
import smart_object_warp.util as util

class Vertex:
    def __init__(self, position, uv):
        self.position = position
        self.uv = uv

class Quad:
    def __init__(self, vertex_0, vertex_1, vertex_2, vertex_3):
        self.vertex_0 = vertex_0
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2
        self.vertex_3 = vertex_3


@numba.njit
def _barycentric_coordinates(p: point_2d.Point2d, a: point_2d.Point2d, b: point_2d.Point2d, c: point_2d.Point2d) -> Tuple[float, float, float]:
    def dot(p1, p2):
        return p1.x * p2.x + p1.y * p2.y

    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = dot(v0, v0)
    d01 = dot(v0, v1)
    d11 = dot(v1, v1)
    d20 = dot(v2, v0)
    d21 = dot(v2, v1)
    denom = d00 * d11 - d01 * d01

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w

@numba.njit
def _point_in_triangle(p: point_2d.Point2d, a: point_2d.Point2d, b: point_2d.Point2d, c: point_2d.Point2d) -> bool:
    u, v, w = _barycentric_coordinates(p, a, b, c)
    return u >= 0 and v >= 0 and w >= 0


class Mesh2d:
    
    def __init__(self, points: list[point_2d.Point2d], u_dims: int, v_dims: int):
        vertices: list[Vertex] = []

        for y in range(v_dims):
            v = float(y) / (v_dims - 1)
            for x in range(u_dims):
                u = float(x) / (u_dims - 1)
                idx = y * u_dims + x
                vertices.append(Vertex(points[idx], point_2d.Point2d(u, v)))


        self.faces: list[Quad] = []
        for y in range(v_dims - 1):
            for x in range(u_dims - 1):
                v0_idx = y * u_dims + x
                v1_idx = v0_idx + 1
                v2_idx = v0_idx + u_dims
                v3_idx = v2_idx + 1
                
                quad = Quad(
                    vertex_0=copy.copy(vertices[v0_idx]),  # Top left
                    vertex_1=copy.copy(vertices[v1_idx]),  # Top right
                    vertex_2=copy.copy(vertices[v2_idx]),  # Bot left
                    vertex_3=copy.copy(vertices[v3_idx])   # Bot right
                )
                self.faces.append(quad)

        # Initialize the spatial grid for acceleration
        self.grid_size = 20  # Number of cells along one axis
        self.grid = defaultdict(list)
        self._build_spatial_grid()

    @util.time_func
    def _build_spatial_grid(self):
        """Populates the spatial grid with quads."""
        # Calculate bounds of the mesh
        all_points = [vtx.position for face in self.faces for vtx in [face.vertex_0, face.vertex_1, face.vertex_2, face.vertex_3]]
        self.min_x = min(p.x for p in all_points)
        self.min_y = min(p.y for p in all_points)
        self.max_x = max(p.x for p in all_points)
        self.max_y = max(p.y for p in all_points)

        # Compute cell size
        self.cell_width = (self.max_x - self.min_x) / self.grid_size
        self.cell_height = (self.max_y - self.min_y) / self.grid_size

        # Assign each quad to grid cells
        for face in self.faces:
            quad_points = [face.vertex_0.position, face.vertex_1.position, face.vertex_2.position, face.vertex_3.position]
            min_quad_x = min(p.x for p in quad_points)
            max_quad_x = max(p.x for p in quad_points)
            min_quad_y = min(p.y for p in quad_points)
            max_quad_y = max(p.y for p in quad_points)

            # Determine cell range for the quad
            min_cell_x = math.floor((min_quad_x - self.min_x) / self.cell_width)
            max_cell_x = math.floor((max_quad_x - self.min_x) / self.cell_width)
            min_cell_y = math.floor((min_quad_y - self.min_y) / self.cell_height)
            max_cell_y = math.floor((max_quad_y - self.min_y) / self.cell_height)

            # Add the quad to all overlapping cells
            for cell_x in range(min_cell_x, max_cell_x + 1):
                for cell_y in range(min_cell_y, max_cell_y + 1):
                    self.grid[(cell_x, cell_y)].append(face)

    def _get_cell_indices(self, point: point_2d.Point2d):
        cell_x = math.floor((point.x - self.min_x) / self.cell_width)
        cell_y = math.floor((point.y - self.min_y) / self.cell_height)
        return cell_x, cell_y

    def uv(self, point: point_2d.Point2d) -> point_2d.Point2d:
        # Get the cell containing the point
        cell_x, cell_y = self._get_cell_indices(point)
        candidate_faces = self.grid.get((cell_x, cell_y), [])

        for face in candidate_faces:
            vtx_0 = face.vertex_0
            vtx_1 = face.vertex_1
            vtx_2 = face.vertex_2
            vtx_3 = face.vertex_3

            # Triangle 1: vertex_0, vertex_1, vertex_3
            if _point_in_triangle(point, vtx_0.position, vtx_1.position, vtx_3.position):
                u, v, w = _barycentric_coordinates(
                    point, vtx_0.position, vtx_1.position, vtx_3.position
                )
                return (
                    vtx_0.uv * u
                    + vtx_1.uv * v
                    + vtx_3.uv * w
                )
            # Triangle 2: vertex_0, vertex_2, vertex_3
            if _point_in_triangle(point, vtx_0.position, vtx_2.position, vtx_3.position):
                u, v, w = _barycentric_coordinates(
                    point, vtx_0.position, vtx_2.position, vtx_3.position
                )
                return (
                    vtx_0.uv * u
                    + vtx_2.uv * v
                    + vtx_3.uv * w
                )

        return point_2d.Point2d(-1.0, -1.0)

    def render_to_image(self, output_path: str):
        max_x = max(vertex.position.x for face in self.faces for vertex in [face.vertex_0, face.vertex_1, face.vertex_2, face.vertex_3])
        max_y = max(vertex.position.y for face in self.faces for vertex in [face.vertex_0, face.vertex_1, face.vertex_2, face.vertex_3])

        def get_point(p: point_2d.Point2d):
            return int(p.x), int(p.y)

        # Create a blank canvas
        img = np.zeros((int(max_y), int(max_x), 3), dtype=np.uint8)
        # Draw lines for each face
        for face in self.faces:
            vertices = [face.vertex_0, face.vertex_1, face.vertex_3, face.vertex_2]
            for i in range(4):
                p1 = get_point(vertices[i].position)
                p2 = get_point(vertices[(i + 1) % 4].position)  # Wrap around to close the quad
                cv2.line(img, p1, p2, (255, 255, 255), 1)  # Draw white lines

        # Save the image
        cv2.imwrite(output_path, img)
        print(f"Successfully wrote file {output_path}")

    @util.time_func
    def scale(self, factor: float):
        for quad in self.faces:
            quad.vertex_0.position *= factor
            quad.vertex_1.position *= factor
            quad.vertex_2.position *= factor
            quad.vertex_3.position *= factor
        self._build_spatial_grid()

    @property
    def width(self):
        all_points = [vtx.position for face in self.faces for vtx in [face.vertex_0, face.vertex_1, face.vertex_2, face.vertex_3]]
        min_x = min(p.x for p in all_points)
        max_x = max(p.x for p in all_points)
        return max_x - min_x
    
    @property
    def height(self):
        all_points = [vtx.position for face in self.faces for vtx in [face.vertex_0, face.vertex_1, face.vertex_2, face.vertex_3]]
        min_y = min(p.y for p in all_points)
        max_y = max(p.y for p in all_points)
        return max_y - min_y

    def generate_empty_buffer(self) -> image_buffer.ImageBuffer:
        max_x = max(vertex.position.x for face in self.faces for vertex in [face.vertex_0, face.vertex_1, face.vertex_2, face.vertex_3])
        max_y = max(vertex.position.y for face in self.faces for vertex in [face.vertex_0, face.vertex_1, face.vertex_2, face.vertex_3])
        return image_buffer.ImageBuffer(int(max_x), int(max_y))
    
    @util.time_func
    def apply_warp(self, buffer: image_buffer.ImageBuffer, image: image_buffer.ImageBuffer):
        for y in range(buffer.height):
            for x in range(buffer.width):

                position = point_2d.Point2d(x, y)
                uv = self.uv(position)
                # If the uv coordinate is outside of the image, skip it
                if uv.x == -1.0 and uv.y == -1.0:
                    continue

                buffer.buffer[y, x] = image_buffer.ImageBuffer.sample_bilinear_uv(image.buffer, image.width, image.height, uv)