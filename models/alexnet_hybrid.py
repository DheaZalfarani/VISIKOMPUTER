import torch
import torch.nn as nn
import torch.nn.functional as F

# Import hybrid pooling dengan memperbaiki path
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from models.pooling import HybridPool2d

class AlexNetHybrid(nn.Module):
    """
    Implementasi arsitektur AlexNet dengan hybrid pooling untuk klasifikasi tumor otak.
    """
    def __init__(self, num_classes=4, grayscale=True, max_weight=0.7, avg_weight=0.3):
        super(AlexNetHybrid, self).__init__()
        
        # Menentukan jumlah channel input berdasarkan jenis gambar (grayscale atau RGB)
        self.input_channels = 1 if grayscale else 3
        self.max_weight = max_weight
        self.avg_weight = avg_weight
        
        # Lapisan ekstraksi fitur (Feature Extraction)
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            HybridPool2d(kernel_size=3, stride=2, max_weight=max_weight, avg_weight=avg_weight),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            HybridPool2d(kernel_size=3, stride=2, max_weight=max_weight, avg_weight=avg_weight),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            HybridPool2d(kernel_size=3, stride=2, max_weight=max_weight, avg_weight=avg_weight),
        )
        
        # Menghitung ukuran output dari feature extractor
        self.feature_size = self._get_feature_size((1, self.input_channels, 224, 224))
        
        # Lapisan klasifikasi (Fully Connected Layers)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.feature_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        # Inisialisasi bobot untuk konvergensi yang lebih baik
        self._initialize_weights()
    
    def _get_feature_size(self, input_shape):
        # Menghitung ukuran output dari feature extractor
        with torch.no_grad():
            x = torch.zeros(input_shape)
            x = self.features(x)
        # Menghitung total elemen (flattened size)
        return x.view(1, -1).size(1)
        
    def forward(self, x):
        # Ekstraksi fitur melalui lapisan konvolusi
        x = self.features(x)
        # Flatten tensor untuk input ke fully connected layer
        x = x.view(x.size(0), -1)
        # Klasifikasi melalui fully connected layers
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        # Inisialisasi bobot jaringan menggunakan distribusi normal
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# if __name__ == "__main__":
#     # Membuat instance model AlexNet dengan hybrid pooling
#     model = AlexNetHybrid(num_classes=4, grayscale=True, max_weight=0.7, avg_weight=0.3)
    
#     # Membuat tensor input dummy untuk testing
#     dummy_input = torch.randn(1, 1, 224, 224)
    
#     # Mencetak ukuran input untuk verifikasi
#     print(f"Input shape: {dummy_input.shape}")
    
#     # Forward pass
#     output = model(dummy_input)
    
#     # Mencetak ukuran output untuk verifikasi
#     print(f"Output shape: {output.shape}")
    
#     # Mencetak ukuran feature size yang dihitung otomatis
#     print(f"Computed feature size: {model.feature_size}")
    
#     # Mencetak bobot hybrid pooling yang digunakan
#     print(f"Hybrid pooling weights - Max: {model.max_weight}, Avg: {model.avg_weight}")
