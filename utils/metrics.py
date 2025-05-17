import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import csv
import json
from tabulate import tabulate

def calculate_accuracy(y_true, y_pred):
    """
    Menghitung akurasi dari prediksi.
    
    Args:
        y_true: Label sebenarnya (ground truth)
        y_pred: Label hasil prediksi model
        
    Returns:
        Nilai akurasi dalam persentase
    """
    # Menghitung akurasi menggunakan sklearn
    # Akurasi adalah rasio jumlah prediksi yang benar dibanding total prediksi
    accuracy = accuracy_score(y_true, y_pred) * 100
    return accuracy

def calculate_precision(y_true, y_pred, average='weighted'):
    """
    Menghitung presisi dari prediksi.
    
    Args:
        y_true: Label sebenarnya (ground truth)
        y_pred: Label hasil prediksi model
        average: Metode untuk menghitung rata-rata presisi (default: 'weighted')
                 'weighted' mempertimbangkan jumlah sampel di setiap kelas
        
    Returns:
        Nilai presisi dalam persentase
    """
    # Menghitung presisi menggunakan sklearn
    # Presisi adalah rasio true positive dibanding jumlah prediksi positif
    precision = precision_score(y_true, y_pred, average=average, zero_division=0) * 100
    return precision

def calculate_recall(y_true, y_pred, average='weighted'):
    """
    Menghitung recall dari prediksi.
    
    Args:
        y_true: Label sebenarnya (ground truth)
        y_pred: Label hasil prediksi model
        average: Metode untuk menghitung rata-rata recall (default: 'weighted')
                 'weighted' mempertimbangkan jumlah sampel di setiap kelas
        
    Returns:
        Nilai recall dalam persentase
    """
    # Menghitung recall menggunakan sklearn
    # Recall adalah rasio true positive dibanding jumlah sampel positif sebenarnya
    recall = recall_score(y_true, y_pred, average=average, zero_division=0) * 100
    return recall

def calculate_f1(y_true, y_pred, average='weighted'):
    """
    Menghitung F1-score dari prediksi.
    
    Args:
        y_true: Label sebenarnya (ground truth)
        y_pred: Label hasil prediksi model
        average: Metode untuk menghitung rata-rata F1-score (default: 'weighted')
                 'weighted' mempertimbangkan jumlah sampel di setiap kelas
        
    Returns:
        Nilai F1-score dalam persentase
    """
    # Menghitung F1-score menggunakan sklearn
    # F1-score adalah rata-rata harmonik dari presisi dan recall
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0) * 100
    return f1

def calculate_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Menghitung confusion matrix dari prediksi.
    
    Args:
        y_true: Label sebenarnya (ground truth)
        y_pred: Label hasil prediksi model
        class_names: Nama-nama kelas (opsional)
        
    Returns:
        Confusion matrix
    """
    # Menghitung confusion matrix menggunakan sklearn
    # Confusion matrix menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas
    cm = confusion_matrix(y_true, y_pred)
    return cm

def plot_confusion_matrix(cm, class_names=None, save_path=None, title='Confusion Matrix'):
    """
    Memvisualisasikan confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Nama-nama kelas
        save_path: Path untuk menyimpan gambar (opsional)
        title: Judul plot
    """
    # Jika nama kelas tidak diberikan, gunakan indeks sebagai nama kelas
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    # Membuat plot confusion matrix menggunakan seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    # Menyimpan gambar jika path diberikan
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def calculate_metrics(y_true, y_pred, class_names=None):
    """
    Menghitung semua metrik evaluasi (akurasi, presisi, recall, F1-score).
    
    Args:
        y_true: Label sebenarnya (ground truth)
        y_pred: Label hasil prediksi model
        class_names: Nama-nama kelas (opsional)
        
    Returns:
        Dictionary berisi semua metrik
    """
    # Menghitung semua metrik
    accuracy = calculate_accuracy(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    f1 = calculate_f1(y_true, y_pred)
    
    # Membuat dictionary hasil
    metrics = {
        'accuracy': round(accuracy, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'f1_score': round(f1, 2)
    }
    
    # Menghitung metrik per kelas jika nama kelas diberikan
    if class_names:
        # Menggunakan classification_report untuk mendapatkan metrik per kelas
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Menambahkan metrik per kelas ke dictionary hasil
        metrics['per_class'] = {}
        for class_name in class_names:
            metrics['per_class'][class_name] = {
                'precision': round(report[class_name]['precision'] * 100, 2),
                'recall': round(report[class_name]['recall'] * 100, 2),
                'f1_score': round(report[class_name]['f1-score'] * 100, 2)
            }
    
    return metrics

def save_metrics_to_csv(metrics, save_path):
    """
    Menyimpan metrik ke file CSV.
    
    Args:
        metrics: Dictionary berisi metrik
        save_path: Path untuk menyimpan file CSV
    """
    # Membuat direktori jika belum ada
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Menyimpan metrik ke file CSV
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        
        # Menulis metrik global
        for metric, value in metrics.items():
            if metric != 'per_class':
                writer.writerow([metric, value])
        
        # Menulis metrik per kelas jika ada
        if 'per_class' in metrics:
            for class_name, class_metrics in metrics['per_class'].items():
                for metric, value in class_metrics.items():
                    writer.writerow([f"{class_name}_{metric}", value])
    
    print(f"Metrics saved to {save_path}")

def save_metrics_to_json(metrics, save_path):
    """
    Menyimpan metrik ke file JSON.
    
    Args:
        metrics: Dictionary berisi metrik
        save_path: Path untuk menyimpan file JSON
    """
    # Membuat direktori jika belum ada
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Menyimpan metrik ke file JSON
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {save_path}")

def create_comparison_table(max_metrics, hybrid_metrics, save_dir):
    """
    Membuat tabel perbandingan antara metrik max pooling dan hybrid pooling.
    
    Args:
        max_metrics: Dictionary berisi metrik untuk model max pooling
        hybrid_metrics: Dictionary berisi metrik untuk model hybrid pooling
        save_dir: Direktori untuk menyimpan hasil perbandingan
    """
    # Membuat direktori jika belum ada
    os.makedirs(save_dir, exist_ok=True)
    
    # Membuat data untuk tabel perbandingan
    comparison_data = {
        'Pooling Method': ['Max Pooling (Original)', 'Hybrid Pooling'],
        'Accuracy': [max_metrics['accuracy'], hybrid_metrics['accuracy']],
        'Precision': [max_metrics['precision'], hybrid_metrics['precision']],
        'Recall': [max_metrics['recall'], hybrid_metrics['recall']],
        'F1-Score': [max_metrics['f1_score'], hybrid_metrics['f1_score']]
    }
    
    # Membuat DataFrame pandas
    df = pd.DataFrame(comparison_data)
    
    # Menyimpan sebagai CSV
    csv_path = os.path.join(save_dir, 'metrics_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"Comparison table saved to {csv_path}")
    
    # Menyimpan sebagai Markdown
    md_path = os.path.join(save_dir, 'metrics_table.md')
    with open(md_path, 'w') as f:
        f.write(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
    print(f"Markdown table saved to {md_path}")
    
    # Menghitung peningkatan performa
    improvements = {
        'accuracy': hybrid_metrics['accuracy'] - max_metrics['accuracy'],
        'precision': hybrid_metrics['precision'] - max_metrics['precision'],
        'recall': hybrid_metrics['recall'] - max_metrics['recall'],
        'f1_score': hybrid_metrics['f1_score'] - max_metrics['f1_score']
    }
    
    # Menyimpan peningkatan performa sebagai JSON
    improvements_path = os.path.join(save_dir, 'improvements.json')
    with open(improvements_path, 'w') as f:
        json.dump(improvements, f, indent=4)
    print(f"Performance improvements saved to {improvements_path}")
    
    return df

def plot_comparison_chart(max_metrics, hybrid_metrics, save_path=None):
    """
    Membuat grafik perbandingan antara metrik max pooling dan hybrid pooling.
    
    Args:
        max_metrics: Dictionary berisi metrik untuk model max pooling
        hybrid_metrics: Dictionary berisi metrik untuk model hybrid pooling
        save_path: Path untuk menyimpan gambar (opsional)
    """
    # Metrik yang akan dibandingkan
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Nilai metrik untuk kedua model
    max_values = [max_metrics[m] for m in metrics]
    hybrid_values = [hybrid_metrics[m] for m in metrics]
    
    # Membuat grafik bar
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, max_values, width, label='Max Pooling')
    rects2 = ax.bar(x + width/2, hybrid_values, width, label='Hybrid Pooling')
    
    # Menambahkan label dan judul
    ax.set_ylabel('Score (%)')
    ax.set_title('Performance Comparison: Max Pooling vs Hybrid Pooling')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
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
    
    # Menyimpan gambar jika path diberikan
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison chart saved to {save_path}")
    
    plt.show()

def evaluate_model(model, data_loader, device, class_names=None):
    """
    Mengevaluasi model pada data loader dan menghitung metrik.
    
    Args:
        model: Model PyTorch yang akan dievaluasi
        data_loader: DataLoader untuk data evaluasi
        device: Device untuk menjalankan model (CPU atau GPU)
        class_names: Nama-nama kelas (opsional)
        
    Returns:
        y_true: Label sebenarnya
        y_pred: Label hasil prediksi
        metrics: Dictionary berisi metrik evaluasi
    """
    # Mengatur model ke mode evaluasi
    model.eval()
    
    # Inisialisasi list untuk menyimpan label sebenarnya dan prediksi
    y_true = []
    y_pred = []
    
    # Menonaktifkan gradient untuk evaluasi
    with torch.no_grad():
        for inputs, labels in data_loader:
            # Memindahkan input dan label ke device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Menyimpan label sebenarnya dan prediksi
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Mengkonversi list ke array numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Menghitung metrik
    metrics = calculate_metrics(y_true, y_pred, class_names)
    
    return y_true, y_pred, metrics

# Contoh penggunaan
if __name__ == "__main__":
    # Contoh data untuk demonstrasi
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 0, 3, 1, 1])
    
    # Nama kelas
    class_names = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
    
    # Menghitung metrik
    metrics = calculate_metrics(y_true, y_pred, class_names)
    print("Metrics:")
    print(json.dumps(metrics, indent=4))
    
    # Menghitung dan memvisualisasikan confusion matrix
    cm = calculate_confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, title='Example Confusion Matrix')
    
    # Contoh perbandingan metrik
    max_metrics = {
        'accuracy': 93.00,
        'precision': 92.75,
        'recall': 92.75,
        'f1_score': 92.75
    }
    
    hybrid_metrics = {
        'accuracy': 94.50,
        'precision': 94.25,
        'recall': 94.25,
        'f1_score': 94.25
    }
    
    # Membuat tabel perbandingan
    comparison_df = create_comparison_table(max_metrics, hybrid_metrics, 'results/comparison')
    print("\nComparison Table:")
    print(comparison_df)
    
    # Membuat grafik perbandingan
    plot_comparison_chart(max_metrics, hybrid_metrics)
