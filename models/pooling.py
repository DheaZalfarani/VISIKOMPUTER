import torch
import torch.nn as nn
import matplotlib.pyplot as plt  # Pastikan matplotlib diimpor
import os

class HybridPool2d(nn.Module):
    """
    Implementasi hybrid pooling yang menggabungkan max pooling dan average pooling.
    Hybrid pooling mengambil keuntungan dari kedua metode:
    - Max pooling baik dalam mendeteksi fitur yang menonjol
    - Average pooling baik dalam mempertahankan informasi latar belakang
    """
    def __init__(self, kernel_size=3, stride=2, padding=0, max_weight=0.7, avg_weight=0.3):
        super(HybridPool2d, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride, padding)
        self.max_weight = max_weight
        self.avg_weight = avg_weight

    def forward(self, x):
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        return self.max_weight * max_pooled + self.avg_weight * avg_pooled

# Kode untuk menguji implementasi hybrid pooling
if __name__ == "__main__":
    # Membuat instance hybrid pooling
    hybrid_pool = HybridPool2d(kernel_size=3, stride=2, max_weight=0.7, avg_weight=0.3)
    
    # Membuat tensor input dummy untuk testing
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Menerapkan hybrid pooling
    output = hybrid_pool(dummy_input)
    
    # Menerapkan max pooling dan average pooling standar untuk perbandingan
    max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
    avg_pool = nn.AvgPool2d(kernel_size=3, stride=2)
    max_pooled = max_pool(dummy_input)
    avg_pooled = avg_pool(dummy_input)
    
    # Mencetak ukuran output untuk verifikasi
    print(f"Input shape: {dummy_input.shape}")
    print(f"Hybrid pooling output shape: {output.shape}")
    print(f"Max pooling output shape: {max_pooled.shape}")
    print(f"Avg pooling output shape: {avg_pooled.shape}")
    
    # Visualisasi perbedaan antara max pooling, average pooling, dan hybrid pooling
    try:
        # Menampilkan direktori kerja saat ini
        print(f"Current working directory: {os.getcwd()}")
        
        # Mengambil channel pertama dari batch pertama untuk visualisasi
        channel_idx = 0
        
        # Membuat figure dengan subplot
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        # Menampilkan input
        axs[0, 0].imshow(dummy_input[0, channel_idx].detach().numpy(), cmap='viridis')
        axs[0, 0].set_title('Input')
        
        # Menampilkan hasil max pooling
        axs[0, 1].imshow(max_pooled[0, channel_idx].detach().numpy(), cmap='viridis')
        axs[0, 1].set_title('Max Pooling')
        
        # Menampilkan hasil average pooling
        axs[0, 2].imshow(avg_pooled[0, channel_idx].detach().numpy(), cmap='viridis')
        axs[0, 2].set_title('Average Pooling')
        
        # Menampilkan hasil hybrid pooling
        axs[1, 0].imshow(output[0, channel_idx].detach().numpy(), cmap='viridis')
        axs[1, 0].set_title('Hybrid Pooling')
        
        # Menghapus subplot yang tidak digunakan
        axs[1, 1].axis('off')
        axs[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('pooling_comparison.png')  # Menyimpan gambar
        print("\nVisualization saved as 'pooling_comparison.png'")
    except ImportError:
        print("\nMatplotlib not installed. Skipping visualization.")
    except Exception as e:
        print(f"An error occurred: {e}")
