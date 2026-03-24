import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from PIL import Image
def load_image(paths, plot=False, return_gray_scale=False, image_size=(1024, 1024)):
    """
    Loads multiple images into a single batched tensor.
    Args:
        paths(list[str]): list of image paths
        plot(bool): plots the first image into 4 channels
        return_gray_scale(bool): if True, returns grayscale images
        image_size(tuple[int, int]): resize target (height, width)
    return:
        torch.Tensor: tensor with shape (batch, channels, 640, 640)
    """
    preprocess = transforms.Compose([
        transforms.Resize(image_size),        # for SAM model
        transforms.ToTensor()
    ])
    image_tensors = []

    for path in paths:
        image = Image.open(path).convert("RGB")
        img_tensor = preprocess(image)
        if return_gray_scale:
            img_tensor = F.rgb_to_grayscale(img_tensor, num_output_channels=1)
        image_tensors.append(img_tensor)

    if len(image_tensors) == 0:
        raise ValueError("paths must contain at least one image path")

    first_shape = image_tensors[0].shape
    for img_tensor in image_tensors:
        if img_tensor.shape != first_shape:
            raise ValueError(
                "All images must have the same shape to form a batch tensor. "
                f"Found mismatched shape {img_tensor.shape} != {first_shape}"
            )

    batch_tensor = torch.stack(image_tensors, dim=0)

    # plotting the image
    if plot:
        fig, axs = plt.subplots(2, 2)
        rgb_for_plot = preprocess(Image.open(paths[0]).convert("RGB"))
        axs[0, 0].imshow(rgb_for_plot[:3].permute(1, 2, 0))
        axs[0, 0].set_title('RGB image')
        axs[0, 1].imshow(rgb_for_plot[0,:,:], cmap='Reds')
        axs[0, 1].set_title('Red channel')
        axs[1, 0].imshow(rgb_for_plot[1,:,:], cmap='Greens')
        axs[1, 0].set_title('Green channel')
        axs[1, 1].imshow(rgb_for_plot[2,:,:], cmap='Blues')
        axs[1, 1].set_title('Blue channel')
        plt.tight_layout()
        plt.show()
    return batch_tensor



import cv2
import numpy as np
def apply_blur(input_img,blur_type="bilateral", ksize=9, sigma1=5, sigma2=100, sigma3=75):
    """
    this fucntion applies the bluring to the input image.

    Args:
    imgut_img: (numpy.array) gray scale inout image with shape of (1,x,y).
    blur_type: (str) type of blur we want to apply. default: bilateral
    ksize: (int) size of the kernal or neighbourhood
    sigma1: (int) sigma value for guassian blur
    sigma2,sigma3: (int) sigma value for bilateral blur

    return:
    numpy.array
    """

    if blur_type == "bilateral":
        blur_image = cv2.bilateralFilter(np.array(input_img[0]), ksize, sigma2, sigma3)
    elif blur_type == "meidanblur":
        blur_image = cv2.medianBlur(np.array(input_img[0]), ksize)
    elif blur_type == "gaussianblur":
        blur_image = cv2.GaussianBlur(np.array(input_img[0]), (ksize,ksize),sigma1)

    return blur_image