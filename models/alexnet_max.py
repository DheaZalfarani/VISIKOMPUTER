import torch
import torch.nn as nn

class AlexNetMax(nn.Module):
    """
    Implementasi arsitektur AlexNet dengan max pooling untuk klasifikasi tumor otak.
    Arsitektur ini mengikuti paper "Brain Tumor Classification Based On MRI Image Processing With Alexnet Architecture"
    dengan 5 layer konvolusi dan 3 fully connected layer.
    """
    def __init__(self, num_classes=4, grayscale=True):
        """
        Inisialisasi model AlexNet dengan max pooling.
        
        Args:
            num_classes: Jumlah kelas output (default: 4 untuk no_tumor, glioma, meningioma, pituitary)
            grayscale: Boolean yang menunjukkan apakah input adalah gambar grayscale (1 channel) atau RGB (3 channel)
        """
        super(AlexNetMax, self).__init__()
        
        # Menentukan jumlah channel input berdasarkan jenis gambar (grayscale atau RGB)
        self.input_channels = 1 if grayscale else 3
        
        # Lapisan ekstraksi fitur (Feature Extraction)
        self.features = nn.Sequential(
            # Layer Konvolusi 1
            # Input: 224x224x1, Output: 55x55x96
            # Konvolusi dengan 96 filter berukuran 11x11, stride 4, tanpa padding
            nn.Conv2d(self.input_channels, 96, kernel_size=11, stride=4, padding=0),
            # Normalisasi batch untuk stabilitas training
            nn.BatchNorm2d(96),
            # Fungsi aktivasi ReLU untuk menambahkan non-linearitas
            nn.ReLU(inplace=True),
            # Max pooling dengan kernel 3x3 dan stride 2, Output: 27x27x96
            # Max pooling mengambil nilai maksimum dari setiap area kernel
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer Konvolusi 2
            # Input: 27x27x96, Output: 27x27x256
            # Konvolusi dengan 256 filter berukuran 5x5, padding 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Max pooling, Output: 13x13x256
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer Konvolusi 3
            # Input: 13x13x256, Output: 13x13x384
            # Konvolusi dengan 384 filter berukuran 3x3, padding 1
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Layer Konvolusi 4
            # Input: 13x13x384, Output: 13x13x384
            # Konvolusi dengan 384 filter berukuran 3x3, padding 1
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Layer Konvolusi 5
            # Input: 13x13x384, Output: 13x13x256
            # Konvolusi dengan 256 filter berukuran 3x3, padding 1
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Max pooling, Output: 6x6x256
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Menghitung ukuran output dari feature extractor untuk menghindari error dimensi
        # Ini akan digunakan untuk menentukan ukuran input layer fully connected pertama
        self.feature_size = self._get_feature_size((1, self.input_channels, 224, 224))
        
        # Lapisan klasifikasi (Fully Connected Layers)
        self.classifier = nn.Sequential(
            # Dropout untuk mengurangi overfitting dengan rate 0.4 sesuai paper
            # Dropout secara acak mematikan neuron selama training
            nn.Dropout(0.4),
            
            # Fully Connected Layer 1
            # Input: ukuran output dari feature extractor, Output: 4096
            # Transformasi linear dari feature map yang di-flatten ke 4096 neuron
            nn.Linear(self.feature_size, 4096),
            nn.ReLU(inplace=True),
            
            # Dropout untuk mengurangi overfitting
            nn.Dropout(0.4),
            
            # Fully Connected Layer 2
            # Input: 4096, Output: 4096
            # Transformasi linear dari 4096 ke 4096 neuron
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            # Fully Connected Layer 3 (Output Layer)
            # Input: 4096, Output: num_classes (4)
            # Transformasi linear dari 4096 ke jumlah kelas (4)
            nn.Linear(4096, num_classes),
        )
        
        # Inisialisasi bobot untuk konvergensi yang lebih baik
        self._initialize_weights()
    
    def _get_feature_size(self, input_shape):
        """
        Menghitung ukuran output dari feature extractor.
        Ini membantu menentukan ukuran input yang tepat untuk fully connected layer pertama.
        
        Args:
            input_shape: Bentuk tensor input [batch_size, channels, height, width]
            
        Returns:
            Ukuran output yang di-flatten dari feature extractor
        """
        # Membuat tensor dummy dengan ukuran input yang diberikan
        dummy_input = torch.zeros(input_shape)
        
        # Forward pass melalui feature extractor
        with torch.no_grad():
            output = self.features(dummy_input)
        
        # Menghitung ukuran output yang di-flatten
        return output.numel() // output.shape[0]
        
    def forward(self, x):
        """
        Forward pass melalui jaringan.
        
        Args:
            x: Tensor input dengan shape [batch_size, channels, height, width]
            
        Returns:
            Tensor output dengan shape [batch_size, num_classes]
        """
        # Ekstraksi fitur melalui lapisan konvolusi
        x = self.features(x)
        
        # Mencetak ukuran output dari feature extractor untuk debugging
        # print(f"Feature extractor output shape: {x.shape}")
        
        # Flatten tensor untuk input ke fully connected layer
        # Menggunakan -1 untuk menghitung ukuran yang tepat secara otomatis
        # Ini menghindari error dimensi jika ukuran output feature extractor berubah
        x = x.view(x.size(0), -1)
        
        # Mencetak ukuran setelah flatten untuk debugging
        # print(f"After flatten shape: {x.shape}")
        
        # Klasifikasi melalui fully connected layers
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        """
        Inisialisasi bobot jaringan menggunakan distribusi normal.
        Ini membantu konvergensi yang lebih baik selama training.
        """
        for m in self.modules():
            # Inisialisasi layer konvolusi dengan Kaiming initialization
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Inisialisasi batch normalization dengan konstanta
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # Inisialisasi fully connected layer dengan distribusi normal
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# if __name__ == "__main__":
#     # Membuat instance model AlexNet dengan max pooling
#     model = AlexNetMax(num_classes=4, grayscale=True)
    
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
    
#     # Menghitung jumlah parameter
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"\nTotal parameters: {total_params:,}")

