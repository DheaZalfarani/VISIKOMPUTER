import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class BrainTumorDataset(Dataset):
    """
    Dataset kustom untuk data MRI tumor otak.
    Dataset ini akan memuat gambar MRI dari direktori yang ditentukan
    dan melakukan preprocessing sesuai kebutuhan.
    """
    def __init__(self, data_dir, mode='Training', transform=None):
        """
        Inisialisasi dataset.
        
        Args:
            data_dir: Direktori root yang berisi folder training dan testing
            mode: 'Training' atau 'Testing' untuk menentukan subset data yang akan digunakan
            transform: Transformasi yang akan diterapkan pada gambar
        """
        self.data_dir = os.path.join(data_dir, mode)
        self.mode = mode
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.classes = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
        self.image_paths = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """
        Memuat path gambar dan label dari direktori data.
        """
        
        print(f"Looking for data in: {self.data_dir}")
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist. Skipping.")
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith('.jpg'):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
        
        print(f"Loaded {len(self.image_paths)} images for {self.mode}")
        print(f"Class distribution:")
        for i, class_name in enumerate(self.classes):
            count = self.labels.count(i)
            print(f"  {class_name}: {count} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            dummy_image = torch.zeros((1, 224, 224)) if self.transform else Image.new('RGB', (224, 224))
            return dummy_image, label

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Membuat data loader untuk training dan testing.
    """
    train_dataset = BrainTumorDataset(data_dir, mode='Training')
    test_dataset = BrainTumorDataset(data_dir, mode='Testing')
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Please check the data directory.")
    
    if len(test_dataset) == 0:
        raise ValueError("Testing dataset is empty. Please check the data directory.")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, test_loader

# Contoh penggunaan
if __name__ == "__main__":
    project_dir = "/mnt/extended-home/dheazalfarani/visikomputer_dhea"
    data_dir = os.path.join(project_dir, "data")
    
    print(f"Project directory: {project_dir}")
    print(f"Data directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist.")
        exit(1)
    
    try:
        train_loader, test_loader = get_data_loaders(data_dir, batch_size=16)
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Test loader: {len(test_loader)} batches")
    except ValueError as e:
        print(f"Error: {e}")
