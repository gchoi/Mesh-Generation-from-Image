# %% Import packages
import os
import time
from logger import logging
import yaml

from modules import (
    depth_estimation,
    point_cloud_construction,
    mesh_generation
)


# %% Implementation
def main() -> None:
    ## 0. Configurations
    YAML_CONFIG_PATH = "./Configs/configs.yaml"
    assert (os.path.exists(YAML_CONFIG_PATH))

    with open(YAML_CONFIG_PATH) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    logging.info(
        f"\nMODEL:\n"
        f"\tFeature Extractor: {configs['model']['feature_extractor']}\n"
        f"\tDepth Estimator: {configs['model']['depth_estimator']}\n"
        f"\nPaths:\n"
        f"\tInput Image Path: {configs['image']}\n"
        f"\tOutput Mesh Path: {configs['output']}\n"
        f"Mesh Resolution:\n"
        f"\tMesh Depth: {configs['mesh_depth']}\n"
    )


    ## 1. Depth estimation
    logging.info("Started depth estimation...")
    start = time.time()
    image, output = depth_estimation(
        feature_extractor=configs['model']['feature_extractor'],
        depth_estimator=configs['model']['depth_estimator'],
        input_image=configs['image'],
        viz_option=configs['visualize']['depth_estimator']
    )
    end = time.time()
    logging.info("Elapsed time for depth estimation: {:10.5f} (seconds)".format(end - start))


    ## 2. Point cloud construction
    logging.info("Started point cloud construction...")
    start = time.time()
    pcd = point_cloud_construction(
        image=image,
        output=output,
        viz_option=configs['visualize']['point_cloud']
    )
    end = time.time()
    logging.info("Elapsed time for point cloud construction: {:10.5f} (seconds)".format(end - start))


    ## 3. Mesh generation
    logging.info("Started mesh generation...")
    start = time.time()
    mesh_generation(
        pcd=pcd,
        mesh_depth=configs['mesh_depth'],
        output_path=configs['output'],
        viz_option=configs['visualize']['mesh_generation']
    )
    end = time.time()
    logging.info("Elapsed time for mesh generation: {:10.5f} (seconds)".format(end - start))

    return


# %% Run main()
if __name__ == '__main__':
    main()