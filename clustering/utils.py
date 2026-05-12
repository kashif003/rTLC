import glob
import os
from pathlib import Path
from PIL import Image
import io
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

