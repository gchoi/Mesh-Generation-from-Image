"""
@Description: Module for depth estimation
@Date: 2022/12/28
"""

# %% Import packages
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation

# %% Implementation
def depth_estimation(
    feature_extractor: str,
    depth_estimator: str,
    input_image: str,
    viz_option: bool = False
) -> (Image.Image, np.ndarray):

    """ Estimate depth from the input image
        Params
        --------
            feature_extractor (str):    Feature extractor pre-trained model
            depth_estimator (str):      Depth estimator pre-trained model
            input_image (str):          Path to the input image
            viz_option (bool):          Option for visualization of depth estimation
        Returns
        --------
            image (PIL.Image.Image):    Pillow image
            output (numpy.ndarray):     Output of depth estimation
    """

    feature_extractor = GLPNFeatureExtractor.from_pretrained(feature_extractor)
    model = GLPNForDepthEstimation.from_pretrained(depth_estimator)

    # load and resize the input image
    image = Image.open(input_image)
    new_height = 480 if image.height > 480 else image.height
    new_height -= (new_height % 32)
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    new_size = (new_width, new_height)
    image = image.resize(new_size)

    # prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    # get the prediction from the model
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # remove borders
    pad = 16
    output = predicted_depth.squeeze().cpu().numpy() * 1000.0
    output = output[pad:-pad, pad:-pad]
    image = image.crop((pad, pad, image.width - pad, image.height - pad))

    # visualize the prediction
    if viz_option:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax[1].imshow(output, cmap='plasma')
        ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.tight_layout()
        plt.pause(5)

    return image, output