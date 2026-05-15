import glob
import os
import random
from pathlib import Path
from PIL import Image
import io
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T

def load_dataset(dataset_name: str = None):
    """Load images from `data/extracted_bands`.

    Args:
        dataset_name: Optional dataset folder name (e.g. 'KS', 'MH', 'SR'). If provided,
            only that dataset is loaded. Matching is case-insensitive.

    Returns:
        dict: mapping labels to data dicts: {label: {image_path, image_bytes, image_data, class, source_folder, label}}
    """
    data = {}
    base_dir = "./data/extracted_bands"
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    available = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    if dataset_name:
        # case-insensitive match
        matches = [d for d in available if d.lower() == dataset_name.lower()]
        if not matches:
            raise ValueError(f"Dataset '{dataset_name}' not found in {base_dir}. Available: {available}")
        target_names = matches
    else:
        target_names = available

    for class_name in target_names:
        class_dir = os.path.join(base_dir, class_name)
        folders = sorted([f for f in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, f))])

        for folder in folders:
            folder_path = os.path.join(class_dir, folder)
            files = sorted(glob.glob(os.path.join(folder_path, "*.png")))

            for file_path in files:
                label = f"{folder}/{Path(file_path).stem}"
                try:
                    with open(file_path, 'rb') as img_file:
                        img_bytes = img_file.read()
                        image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    data[label] = {
                        "image_path": file_path,
                        "image_data": T.ToTensor()(image_pil),
                        "class": class_name,
                        "source_folder": folder,
                        "label": label,
                    }
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    return data


from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
        transforms.Resize((512,128)),
    ])

class Custom_dataset(Dataset):
    def __init__(self, dataset:dict, transform= transform):
        self.length = len(dataset)
        self.data = dataset
        self.transform = transform

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        key_list = list(self.data.keys())
        if self.transform:
            sample = self.transform(self.data[key_list[idx]]["image_data"])
        label= self.data[key_list[idx]]["label"]        

        return sample, label


import matplotlib.pyplot as plt
import numpy as np
def plot_img(tensor):
    img = np.array(tensor.detach().numpy()).transpose(2,1,0)
    plt.imshow(img)
    plt.show()


def plot_random_batch(model, dataloader, device=None, batch_index=None, max_images=10, save_dir="figs"):
    """Select a random batch from a dataloader and plot up to `max_images` input/output pairs side by side.

    Args:
        model: A torch model that returns (reconstruction, latent) or reconstruction.
        dataloader: A torch DataLoader that yields (inputs, labels) batches.
        device: Optional torch device. Defaults to the model device or CPU.
        batch_index: Optional explicit batch index. If None, a random batch is chosen.
        max_images: Maximum number of images to plot from the batch.
        save_dir: Directory where the figure will be saved.
    """
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    if batch_index is None:
        batch_index = random.randrange(len(dataloader))
    elif batch_index < 0 or batch_index >= len(dataloader):
        raise IndexError(f"batch_index {batch_index} is out of range for {len(dataloader)} batches")

    was_training = model.training
    model.eval()

    batch_inputs = None
    for current_index, batch in enumerate(dataloader):
        if current_index == batch_index:
            batch_inputs = batch[0]
            break

    if batch_inputs is None:
        raise RuntimeError("Failed to fetch the selected batch from the dataloader")

    batch_inputs = batch_inputs.to(device)

    with torch.no_grad():
        model_output = model(batch_inputs)
        batch_outputs = model_output[0] if isinstance(model_output, (tuple, list)) else model_output

    input_batch = batch_inputs.detach().cpu()
    output_batch = batch_outputs.detach().cpu()

    batch_size = input_batch.shape[0]
    num_images = min(max_images, batch_size)
    selected_indices = random.sample(range(batch_size), num_images)
    fig, axes = plt.subplots(
        num_images,
        2,
        figsize=(16, 4 * num_images),
        dpi=140,
        squeeze=False,
        gridspec_kw={"wspace": 0.05, "hspace": 0.25},
    )

    for row_idx, sample_idx in enumerate(selected_indices):
        in_img = input_batch[sample_idx].permute(1, 2, 0).numpy()
        out_img = output_batch[sample_idx].permute(1, 2, 0).numpy()

        axes[row_idx, 0].imshow(in_img)
        axes[row_idx, 0].set_title(f"Input {sample_idx + 1}", fontsize=12)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(out_img)
        axes[row_idx, 1].set_title(f"Output {sample_idx + 1}", fontsize=12)
        axes[row_idx, 1].axis("off")

    fig.suptitle("Random Batch: Input vs Output", fontsize=16, y=1.01)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"random_batch_{batch_index}.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()


    return fig