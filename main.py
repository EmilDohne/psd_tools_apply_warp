import os

from psd_tools import PSDImage
import psd_tools.api.layers

import smart_object_warp.point_2d
import smart_object_warp.bezier_surface
import smart_object_warp.image_buffer


def extract_warp_information(layer:psd_tools.api.layers.SmartObjectLayer) -> smart_object_warp.bezier_surface.BezierSurface:
    # Get the warp, if it is not a default warp its a quilt warp with more than 4x4 dims
    if layer.smart_object._config.data.get(b"quiltWarp", None) is not None:
        warp = layer.smart_object._config.data.get(b"quiltWarp")
    else:
        warp = layer.smart_object.warp

    # Accumulate all the points and store them
    horizontal_pts = warp.get(b"customEnvelopeWarp").get(b"meshPoints").get(b"Hrzn")
    vertical_pts = warp.get(b"customEnvelopeWarp").get(b"meshPoints").get(b"Vrtc")

    points: list[smart_object_warp.point_2d.Point2d] = []
    for x, y in zip(horizontal_pts.values, vertical_pts.values):
        points.append(smart_object_warp.point_2d.Point2d(x, y))

    # construct the bezier surface
    if len(horizontal_pts.values) == 4 and len(vertical_pts.values) == 4:
        u_dims = 4
        v_dims = 4
    else:
        u_dims = warp.get(b"deformNumCols").value
        v_dims = warp.get(b"deformNumRows").value

    return smart_object_warp.bezier_surface.BezierSurface(points, u_dims, v_dims)


def apply_warp(layer: psd_tools.api.layers.SmartObjectLayer):
    bezier = extract_warp_information(layer)
    
    mesh = bezier.to_mesh(num_divisions=25)
    # Scale the mesh to make the width 500, scales the height accordingly
    mesh.scale(500 / mesh.width)

    # Render the mesh implementations
    mesh.render_to_image(os.path.join(os.path.dirname(__file__), "out", f"{layer.name}_mesh.jpg"))
    bezier.render_to_image(os.path.join(os.path.dirname(__file__), "out", f"{layer.name}_bezier.jpg"))

    image = smart_object_warp.image_buffer.ImageBuffer.load(os.path.join(os.path.dirname(__file__), "image_data", "image.png"))
    buffer = mesh.generate_empty_buffer()
    mesh.apply_warp(buffer, image)
    buffer.save(os.path.join(os.path.dirname(__file__), "out", "output_warped.jpg"))


def main():
    # Open the PSD and apply the warp to all layers
    psd = PSDImage.open(os.path.join(os.path.dirname(__file__), "psd_data", "input.psd"))
    for layer in psd:
        if isinstance(layer, psd_tools.api.layers.SmartObjectLayer):
            apply_warp(layer)

 
if __name__ == "__main__":
    main()