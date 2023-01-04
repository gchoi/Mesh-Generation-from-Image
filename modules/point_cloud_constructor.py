"""
@Description: Module for point cloud construction
@Date: 2022/12/28
"""

# %% Import packages
import open3d as o3d
from PIL import Image
import numpy as np

# %% Implementation
def point_cloud_construction(
    image: Image.Image,
    output: np.ndarray,
    viz_option: bool = False
) -> o3d.cpu.pybind.geometry.PointCloud:

    """ Construct point cloud
        Params
        --------
            image (PIL.Image.Image):                        Pillow image
            output (numpy.ndarray):                         Output of depth estimation
            viz_option (bool):                              Option for visualization of depth estimation
        Returns
        --------
            pcd (open3d.cpu.pybind.geometry.PointCloud):    Open3d point cloud
    """

    width, height = image.size

    depth_image = (output * 255 / np.max(output)).astype('uint8')
    image = np.array(image)

    # create RGBD image
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_o3d,
        depth_o3d,
        convert_rgb_to_intensity=False
    )

    # camera settings
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, 500, 500, width / 2, height / 2)

    # create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsic
    )

    if viz_option:
        o3d.visualization.draw_geometries([pcd])

    return pcd