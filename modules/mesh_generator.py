"""
@Description: Module for mesh generation
@Date: 2022/12/28
"""

# %% Import packages
import numpy as np
import open3d as o3d

# %% Implementation
def mesh_generation(
    pcd: o3d.cpu.pybind.geometry.PointCloud,
    mesh_depth: int,
    output_path: str,
    viz_option: bool = False
) -> None:

    """ Generate meshes
        Params
        --------
            pcd (open3d.cpu.pybind.geometry.PointCloud):    Open3d point cloud
            mesh_depth (int):                               Output of depth estimation
            output_path (str):                              Path to the output mesh object
            viz_option (bool):                              Option for visualization of depth estimation
        Returns
        --------
            None
    """

    # outliers removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
    pcd = pcd.select_by_index(ind)

    # estimate normals
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()

    # surface reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=mesh_depth,
        n_threads=1
    )[0]

    # rotate the mesh
    rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    mesh.rotate(rotation, center=(0, 0, 0))

    # save the mesh
    o3d.io.write_triangle_mesh(output_path, mesh)

    # visualize the mesh
    if viz_option:
        o3d.visualization.draw_geometries(
            [mesh],
            mesh_show_back_face=True
        )

    return