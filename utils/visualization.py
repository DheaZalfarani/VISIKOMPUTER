import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import torch
from sklearn.metrics import confusion_matrix
from PIL import Image
import torchvision.transforms as transforms
import itertools

# Mengatur style plot untuk konsistensi visual
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Memvisualisasikan history training model (loss dan akurasi).
    
    Args:
        train_losses: List berisi nilai loss training per epoch
        val_losses: List berisi nilai loss validasi per epoch
        train_accs: List berisi nilai akurasi training per epoch
        val_accs: List berisi nilai akurasi validasi per epoch
        save_path: Path untuk menyimpan plot (opsional)
    """
    # Membuat figure dengan 2 subplot (loss dan akurasi)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot untuk loss
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot untuk akurasi
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    # Menyesuaikan layout
    plt.tight_layout()
    
    # Menyimpan plot jika path disediakan
    if save_path:
        # Membuat direktori jika belum ada
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=False, title='Confusion Matrix'):
    """
    Memvisualisasikan confusion matrix.
    
    Args:
        y_true: Label sebenarnya (ground truth)
        y_pred: Label hasil prediksi model
        class_names: List nama kelas
        save_path: Path untuk menyimpan plot (opsional)
        normalize: Boolean untuk menormalisasi nilai confusion matrix
        title: Judul plot
    """
    # Menghitung confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalisasi confusion matrix jika diminta
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Membuat plot
    plt.figure(figsize=(10, 8))
    
    # Menggunakan seaborn untuk heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    # Menambahkan label dan judul
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    
    # Menyesuaikan layout
    plt.tight_layout()
    
    # Menyimpan plot jika path disediakan
    if save_path:
        # Membuat direktori jika belum ada
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()

def plot_metrics_comparison(max_metrics, hybrid_metrics, metrics_names=None, save_path=None):
    """
    Membuat plot perbandingan metrik antara model max pooling dan hybrid pooling.
    
    Args:
        max_metrics: Dictionary atau list berisi nilai metrik untuk model max pooling
        hybrid_metrics: Dictionary atau list berisi nilai metrik untuk model hybrid pooling
        metrics_names: List nama metrik (opsional, default: ['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        save_path: Path untuk menyimpan plot (opsional)
    """
    # Mengatur nama metrik default jika tidak disediakan
    if metrics_names is None:
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Mengkonversi dictionary ke list jika input berupa dictionary
    if isinstance(max_metrics, dict):
        metrics_keys = ['accuracy', 'precision', 'recall', 'f1_score']
        max_values = [max_metrics[k] for k in metrics_keys]
        hybrid_values = [hybrid_metrics[k] for k in metrics_keys]
    else:
        max_values = max_metrics
        hybrid_values = hybrid_metrics
    
    # Membuat plot bar
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, max_values, width, label='Max Pooling')
    rects2 = ax.bar(x + width/2, hybrid_values, width, label='Hybrid Pooling')
    
    # Menambahkan label dan judul
    ax.set_ylabel('Score (%)')
    ax.set_title('Performance Comparison: Max Pooling vs Hybrid Pooling')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    
    # Menambahkan nilai di atas bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    # Menambahkan grid untuk memudahkan pembacaan
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Menyesuaikan layout
    fig.tight_layout()
    
    # Menyimpan plot jika path disediakan
    if save_path:
        # Membuat direktori jika belum ada
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to {save_path}")
    
    plt.show()

def visualize_model_predictions(model, data_loader, class_names, device, num_images=8, save_path=None):
    """
    Memvisualisasikan prediksi model pada beberapa gambar dari data loader.
    
    Args:
        model: Model PyTorch yang akan digunakan untuk prediksi
        data_loader: DataLoader berisi data yang akan divisualisasikan
        class_names: List nama kelas
        device: Device untuk menjalankan model (CPU atau GPU)
        num_images: Jumlah gambar yang akan divisualisasikan
        save_path: Path untuk menyimpan plot (opsional)
    """
    # Mengatur model ke mode evaluasi
    model.eval()
    
    # Mendapatkan batch gambar dari data loader
    images, labels = next(iter(data_loader))
    
    # Membatasi jumlah gambar yang akan divisualisasikan
    num_images = min(num_images, len(images))
    
    # Membuat prediksi
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Memindahkan tensor ke CPU dan mengkonversi ke numpy
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    
    # Membuat grid untuk visualisasi
    fig, axes = plt.subplots(2, num_images // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    # Menampilkan setiap gambar dengan label sebenarnya dan prediksi
    for i in range(num_images):
        # Mengambil gambar dan mengubah format untuk visualisasi
        img = np.transpose(images[i], (1, 2, 0))  # CHW -> HWC
        
        # Denormalisasi gambar jika perlu
        img = img * 0.5 + 0.5  # Asumsi normalisasi [-1, 1] -> [0, 1]
        
        # Menampilkan gambar
        axes[i].imshow(img.squeeze(), cmap='gray')
        
        # Menambahkan label
        title_color = 'green' if preds[i] == labels[i] else 'red'
        axes[i].set_title(f'True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}', 
                         color=title_color)
        
        # Menyembunyikan axis
        axes[i].axis('off')
    
    # Menyesuaikan layout
    plt.tight_layout()
    
    # Menyimpan plot jika path disediakan
    if save_path:
        # Membuat direktori jika belum ada
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model predictions visualization saved to {save_path}")
    
    plt.show()

def visualize_feature_maps(model, image, layer_name, num_features=8, save_path=None):
    """
    Memvisualisasikan feature maps dari layer tertentu dalam model.
    
    Args:
        model: Model PyTorch yang akan dianalisis
        image: Tensor gambar input (1, C, H, W)
        layer_name: Nama layer yang feature map-nya akan divisualisasikan
        num_features: Jumlah feature maps yang akan ditampilkan
        save_path: Path untuk menyimpan plot (opsional)
    """
    # Mengatur model ke mode evaluasi
    model.eval()
    
    # Dictionary untuk menyimpan output dari layer yang dipilih
    activation = {}
    
    # Fungsi hook untuk mendapatkan output dari layer
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Mendaftarkan hook untuk layer yang dipilih
    for name, layer in model.named_modules():
        if name == layer_name:
            layer.register_forward_hook(get_activation(layer_name))
            break
    
    # Forward pass untuk mendapatkan aktivasi
    with torch.no_grad():
        output = model(image)
    
    # Memeriksa apakah layer ditemukan
    if layer_name not in activation:
        print(f"Layer '{layer_name}' not found in the model.")
        print("Available layers:")
        for name, _ in model.named_modules():
            print(f"  {name}")
        return
    
    # Mendapatkan feature maps
    feature_maps = activation[layer_name][0].cpu().numpy()
    
    # Membatasi jumlah feature maps yang akan ditampilkan
    num_features = min(num_features, feature_maps.shape[0])
    
    # Menghitung ukuran grid
    grid_size = int(np.ceil(np.sqrt(num_features)))
    
    # Membuat grid untuk visualisasi
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    # Menampilkan setiap feature map
    for i in range(grid_size * grid_size):
        if i < num_features:
            # Menampilkan feature map
            axes[i].imshow(feature_maps[i], cmap='viridis')
            axes[i].set_title(f'Feature Map {i+1}')
        
        # Menyembunyikan axis
        axes[i].axis('off')
    
    # Menambahkan judul
    plt.suptitle(f'Feature Maps from {layer_name}', fontsize=16)
    
    # Menyesuaikan layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Menyimpan plot jika path disediakan
    if save_path:
        # Membuat direktori jika belum ada
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature maps visualization saved to {save_path}")
    
    plt.show()

def visualize_pooling_comparison(image, kernel_size=3, stride=2, save_path=None):
    """
    Memvisualisasikan perbandingan antara max pooling, average pooling, dan hybrid pooling.
    
    Args:
        image: Tensor gambar input (1, C, H, W) atau path ke file gambar
        kernel_size: Ukuran kernel untuk pooling
        stride: Stride untuk pooling
        save_path: Path untuk menyimpan plot (opsional)
    """
    # Memuat gambar jika input adalah path
    if isinstance(image, str):
        # Memuat gambar dan mengkonversi ke tensor
        img = Image.open(image).convert('L')  # Konversi ke grayscale
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        image = transform(img).unsqueeze(0)  # Menambahkan dimensi batch
    
    # Memastikan image adalah tensor dengan dimensi yang benar
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a PyTorch tensor or a path to an image file")
    
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Menambahkan dimensi batch jika belum ada
    
    # Mengambil channel pertama jika gambar berwarna
    if image.size(1) > 1:
        image_display = image[0, 0].unsqueeze(0).unsqueeze(0)
    else:
        image_display = image
    
    # Menerapkan max pooling
    max_pooled = torch.nn.functional.max_pool2d(image_display, kernel_size=kernel_size, stride=stride)
    
    # Menerapkan average pooling
    avg_pooled = torch.nn.functional.avg_pool2d(image_display, kernel_size=kernel_size, stride=stride)
    
    # Menerapkan hybrid pooling (50% max, 50% avg)
    hybrid_pooled_balanced = 0.5 * max_pooled + 0.5 * avg_pooled
    
    # Menerapkan hybrid pooling (70% max, 30% avg)
    hybrid_pooled_max = 0.7 * max_pooled + 0.3 * avg_pooled
    
    # Menerapkan hybrid pooling (30% max, 70% avg)
    hybrid_pooled_avg = 0.3 * max_pooled + 0.7 * avg_pooled
    
    # Mengkonversi tensor ke numpy untuk visualisasi
    image_np = image_display.squeeze().cpu().numpy()
    max_pooled_np = max_pooled.squeeze().cpu().numpy()
    avg_pooled_np = avg_pooled.squeeze().cpu().numpy()
    hybrid_balanced_np = hybrid_pooled_balanced.squeeze().cpu().numpy()
    hybrid_max_np = hybrid_pooled_max.squeeze().cpu().numpy()
    hybrid_avg_np = hybrid_pooled_avg.squeeze().cpu().numpy()
    
    # Membuat grid untuk visualisasi
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Menampilkan gambar asli
    axes[0, 0].imshow(image_np, cmap='gray')
    axes[0, 0].set_title('Original Image')
    
    # Menampilkan hasil max pooling
    axes[0, 1].imshow(max_pooled_np, cmap='gray')
    axes[0, 1].set_title('Max Pooling')
    
    # Menampilkan hasil average pooling
    axes[0, 2].imshow(avg_pooled_np, cmap='gray')
    axes[0, 2].set_title('Average Pooling')
    
    # Menampilkan hasil hybrid pooling (balanced)
    axes[1, 0].imshow(hybrid_balanced_np, cmap='gray')
    axes[1, 0].set_title('Hybrid Pooling (50% Max, 50% Avg)')
    
    # Menampilkan hasil hybrid pooling (max dominant)
    axes[1, 1].imshow(hybrid_max_np, cmap='gray')
    axes[1, 1].set_title('Hybrid Pooling (70% Max, 30% Avg)')
    
    # Menampilkan hasil hybrid pooling (avg dominant)
    axes[1, 2].imshow(hybrid_avg_np, cmap='gray')
    axes[1, 2].set_title('Hybrid Pooling (30% Max, 70% Avg)')
    
    # Menyembunyikan axis untuk semua subplot
    for ax in axes.flatten():
        ax.axis('off')
    
    # Menambahkan judul
    plt.suptitle(f'Comparison of Pooling Methods (Kernel: {kernel_size}x{kernel_size}, Stride: {stride})', 
                fontsize=16)
    
    # Menyesuaikan layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Menyimpan plot jika path disediakan
    if save_path:
        # Membuat direktori jika belum ada
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pooling comparison visualization saved to {save_path}")
    
    plt.show()

def plot_learning_rate_comparison(lr_results, save_path=None):
    """
    Memvisualisasikan perbandingan akurasi untuk berbagai learning rate.
    
    Args:
        lr_results: Dictionary dengan learning rate sebagai key dan akurasi sebagai value
        save_path: Path untuk menyimpan plot (opsional)
    """
    # Mengekstrak learning rates dan akurasi
    learning_rates = list(lr_results.keys())
    accuracies = list(lr_results.values())
    
    # Membuat plot
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, accuracies, 'o-', linewidth=2, markersize=8)
    
    # Menambahkan label dan judul
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Learning Rate vs. Validation Accuracy')
    
    # Menggunakan skala logaritmik untuk sumbu x
    plt.xscale('log')
    
    # Menambahkan grid
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Menambahkan nilai di setiap titik
    for i, (lr, acc) in enumerate(zip(learning_rates, accuracies)):
        plt.annotate(f'{acc:.2f}%', 
                    (lr, acc),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
    
    # Menyesuaikan layout
    plt.tight_layout()
    
    # Menyimpan plot jika path disediakan
    if save_path:
        # Membuat direktori jika belum ada
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate comparison plot saved to {save_path}")
    
    plt.show()

def plot_class_distribution(data_loader, class_names, save_path=None):
    """
    Memvisualisasikan distribusi kelas dalam dataset.
    
    Args:
        data_loader: DataLoader berisi data yang akan divisualisasikan
        class_names: List nama kelas
        save_path: Path untuk menyimpan plot (opsional)
    """
    # Menghitung jumlah sampel per kelas
    class_counts = {}
    for _, labels in data_loader:
        for label in labels:
            label = label.item()
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1
    
    # Mengkonversi ke format yang sesuai untuk plotting
    classes = []
    counts = []
    for i in range(len(class_names)):
        classes.append(class_names[i])
        counts.append(class_counts.get(i, 0))
    
    # Membuat plot bar
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color=sns.color_palette('deep'))
    
    # Menambahkan label dan judul
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Dataset')
    
    # Menambahkan nilai di atas bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}',
                ha='center', va='bottom')
    
    # Menyesuaikan layout
    plt.tight_layout()
    
    # Menyimpan plot jika path disediakan
    if save_path:
        # Membuat direktori jika belum ada
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()

# Contoh penggunaan
if __name__ == "__main__":
    # Contoh data untuk demonstrasi
    
    # Contoh history training
    train_losses = [0.8, 0.6, 0.4, 0.3, 0.2]
    val_losses = [0.9, 0.7, 0.5, 0.4, 0.35]
    train_accs = [70, 80, 85, 90, 95]
    val_accs = [65, 75, 80, 85, 87]
    
    # Visualisasi history training
    plot_training_history(train_losses, val_losses, train_accs, val_accs, 
                         save_path='results/training_history.png')
    
    # Contoh data untuk confusion matrix
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 0, 3, 1, 1])
    class_names = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
    
    # Visualisasi confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, 
                         save_path='results/confusion_matrix.png')
    
    # Contoh data untuk perbandingan metrik
    max_metrics = [93.0, 92.75, 92.75, 92.75]
    hybrid_metrics = [94.5, 94.25, 94.25, 94.25]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Visualisasi perbandingan metrik
    plot_metrics_comparison(max_metrics, hybrid_metrics, metrics_names, 
                           save_path='results/metrics_comparison.png')
    
    # Contoh data untuk perbandingan learning rate
    lr_results = {
        0.0001: 85.5,
        0.001: 92.3,
        0.01: 88.7,
        0.1: 75.2
    }
    
    # Visualisasi perbandingan learning rate
    plot_learning_rate_comparison(lr_results, 
                                 save_path='results/learning_rate_comparison.png')
