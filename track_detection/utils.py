import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from PIL import Image
def load_image(path,  plot=False, return_gray_scale=False):
    """
    loads the image into tensors.
    Args:
        path(string): path of the image
        plot(bool): plots the image into 4 channels
        return_gray_scale(bool): return the gray scale tensor
    return:
        image_tensor(torch.tensor): tensor 
    """
    transform = transforms.Compose([transforms.PILToTensor()])
    img_tensor = transform(Image.open(path).convert("RGB"))

    if return_gray_scale:
        gray_tensor = F.rgb_to_grayscale(img_tensor, num_output_channels=1)



    # plotting the image
    if plot:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(img_tensor[:3].permute(1, 2, 0))
        axs[0, 0].set_title('RGB image')
        axs[0, 1].imshow(img_tensor[0,:,:], cmap='Reds')
        axs[0, 1].set_title('Red channel')
        axs[1, 0].imshow(img_tensor[1,:,:], cmap='Greens')
        axs[1, 0].set_title('Green channel')
        axs[1, 1].imshow(img_tensor[2,:,:], cmap='Blues')
        axs[1, 1].set_title('Blue channel')
        plt.tight_layout()
        plt.show()
    return img_tensor if not return_gray_scale else gray_tensor



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