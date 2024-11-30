import math

import smart_object_warp.point_2d as point_2d
import smart_object_warp.mesh_2d as mesh_2d


class BezierSurface:

    def __init__(self, points: list[point_2d.Point2d], u_dims: int, v_dims: int):
        self.points = points
        self.u_dims = u_dims
        self.v_dims = v_dims

        self.num_patches_u = self._calc_num_patches(self.u_dims)
        self.num_patches_v = self._calc_num_patches(self.v_dims)
        self.patches = self._make_patches()

    def _calc_num_patches(self, dims) -> int:
        if dims > 4:
            return int(1 + (dims - 4) / 3)
        return 1
    
    def _make_patches(self) -> list[list[point_2d.Point2d]]:
        patches = []
        for py in range(self.num_patches_v):
            for px in range(self.num_patches_u):
                patch = []

                # Fill the 4x4 patch in scanline order with overlap
                for y in range(4):
                    for x in range(4):
                        control_point_index = (py * 3 + y) * self.u_dims + (px * 3 + x)
                        patch.append(self.points[control_point_index])
                
                patches.append(patch)
        return patches
    

    def to_mesh(self, num_divisions: int = 5) -> mesh_2d.Mesh2d:
        points: list[point_2d.Point2d] = []

        for y in range(num_divisions):
            # Calculate the local v coordinate
            v = y / (num_divisions - 1)
            for x in range(num_divisions):
                # Calculate the local u coordinate
                u = x / (num_divisions - 1)

                points.append(self.evaluate(u, v))

        return mesh_2d.Mesh2d(points, num_divisions, num_divisions)


    def evaluate(self, u: float, v: float):
        patch_size_u = 1.0 / self.num_patches_u
        patch_size_v = 1.0 / self.num_patches_v

        # Determine which patch to use
        patch_index_x = min(int(math.floor(u / patch_size_u)), self.num_patches_u - 1)
        patch_index_y = min(int(math.floor(v / patch_size_v)), self.num_patches_v - 1)

        # Calculate base UV of the patch
        patch_base_u = patch_index_x * patch_size_u
        patch_base_v = patch_index_y * patch_size_v

        # Calculate local UV within the patch
        local_u = (u - patch_base_u) / patch_size_u
        local_v = (v - patch_base_v) / patch_size_v

        # Clamp to ensure valid range
        local_u = max(0.0, min(1.0, local_u))
        local_v = max(0.0, min(1.0, local_v))

        # Retrieve control points for this patch
        patch = self._get_patch_ctrl_points(patch_index_x, patch_index_y)
        return self.evaluate_bezier_patch(patch, local_u, local_v)


    def _get_patch_ctrl_points(self, patch_u: int, patch_v: int) -> list[point_2d.Point2d]:
        return self.patches[patch_v * self.num_patches_u + patch_u]
    
    def evaluate_bezier_patch(self, patch: list[point_2d.Point2d], u: float, v: float) -> point_2d.Point2d:
        curves = [
            self.evaluate_bezier_curve(patch[0:4], u),
            self.evaluate_bezier_curve(patch[4:8], u),
            self.evaluate_bezier_curve(patch[8:12], u),
            self.evaluate_bezier_curve(patch[12:16], u)
        ]
        return self.evaluate_bezier_curve(curves, v)

    def evaluate_bezier_curve(self, points: list[point_2d.Point2d], t: float) -> point_2d.Point2d:
        a = point_2d.Point2d.lerp(points[0], points[1], t)
        b = point_2d.Point2d.lerp(points[1], points[2], t)
        c = point_2d.Point2d.lerp(points[2], points[3], t)

        d = point_2d.Point2d.lerp(a, b, t)
        e = point_2d.Point2d.lerp(b, c, t)

        return point_2d.Point2d.lerp(d, e, t)
    
    def render_to_image(self, output_path: str):
        mesh = mesh_2d.Mesh2d(self.points, self.u_dims, self.v_dims)
        mesh.render_to_image(output_path)