# perform semantic segmentation.

'''
,
"images/AH/3_33_2_5_2_41_1_1_1_153.tif",
"images/AH/6_28_3_5_2_1_3_1_1_102.tif",
"images/AH/6_28_3_5_2_1_3_1_1_115.tif", ,
        "images/AH/4_19_2_5_2_34_3_1_1_128.tif", 
        "images/AH/4_19_2_5_2_48_2_1_1_125.tif",
         "images/AH/4_19_2_5_2_1_3_1_1_126.tif", 
        "images/AH/9_50_2_4_2_3_1_2_1_106.tif", 
        "images/AH/3_33_2_5_2_53_2_1_1_156.tif", 
        "images/AH/3_33_2_5_2_53_1_1_1_158.tif",
        "images/AH/4_19_2_5_2_53_3_1_1_148.tif", 
          "images/AH/3_33_2_5_2_40_4_1_1_149.tif" 
           "images/AH/3_33_2_5_1_41_1_1_1_152.tif",
           "images/AH/3_33_2_5_2_1_1_1_1_164.tif,
        "images/AH/4_19_2_5_3_1_1_1_1_114.tif",
        "images/AH/4_19_2_5_2_1_2_3_2_120.tif",
        "images/AH/3_33_2_5_2_40_4_1_1_149.tif",
        "images/AH/3_33_2_5_2_33_3_1_1_167.tif",
        "images/AH/3_33_2_5_2_53_3_1_1_160.tif",
        "images/AH/3_33_2_5_2_1_1_1_1_164.tif",
        "images/AH/3_33_2_5_2_53_1_1_1_155.tif",
        '''
#path = "images/AH/6_28_3_5_2_21_3_1_1_101.tif"  # hard
#path = "images/KS/1_4_1_5_2_1_1_1_2_507.tif"    # easy
#path ="images/AH/6_28_3_5_2_21_3_1_1_101.tif" 


from utils import load_image
import numpy as np
class Mask_builder():
    def __init__(self, path_list, model):
        self.path_list = [path_list] if isinstance(path_list, str) else path_list
        self.model = model
        self.model_results = None
        self.mask = None
    
    def segment(self):
        image_input = load_image(self.path_list, return_gray_scale=False )
        self.model_results = self.model(image_input)

    def _build_clipped_mask(self, result, threshold=0.1):
        if result.masks is None or result.masks.data is None or result.masks.data.shape[0] == 0:
            return np.zeros(result.orig_img.shape[:2], dtype=np.uint8)

        mask_data = result.masks.data
        height, width = tuple(mask_data.shape[1:])
        full_mask = np.zeros((height, width), dtype=np.float32)
        total_pixels = height * width

        for i in range(mask_data.shape[0]):
            mask_np = mask_data[i].cpu().numpy()
            if mask_np.sum() / total_pixels > threshold:
                continue
            full_mask += mask_np
        return (full_mask > 0).astype(np.uint8)


    def build_clipped_masks(self, threshold=0.1):
        if self.model_results is None:
            self.segment()

        self.mask = np.stack(
            [self._build_clipped_mask(result, threshold=threshold) for result in  self.model_results],
            axis=0
        )
        return self.mask

    def get_mask(self):
        return self.build_clipped_masks()

import matplotlib.pyplot as plt


def plot_masks(masks, max_images=None):
    total_masks = masks.shape[0]
    if max_images is None:
        max_images = total_masks
    max_images = min(max_images, total_masks)
    fig, axes = plt.subplots(1, max_images, figsize=(6 * max_images, 6))
    if max_images == 1:
        axes = [axes]
    for index in range(max_images):
        axes[index].imshow(masks[index], cmap="gray")
        axes[index].set_title(f"Mask {index}")
        axes[index].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from ultralytics import FastSAM

    path = ["images/AH/4_19_2_5_2_1_3_3_2_134.tif"]
    model = FastSAM("FastSAM-x.pt")
    segmentator = Mask_builder(path, model)
    mask = segmentator.get_mask()
    print("mask shape:", mask.shape)
    # plot_masks(mask)
