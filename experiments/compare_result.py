import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Menentukan path untuk log hasil training
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_POOLING_LOG = os.path.join(ROOT_DIR, "results", "max_pooling", "training_log.csv")
HYBRID_POOLING_LOG = os.path.join(ROOT_DIR, "results", "hybrid_pooling", "training_log.csv")
COMPARISON_DIR = os.path.join(ROOT_DIR, "results", "comparison")
os.makedirs(COMPARISON_DIR, exist_ok=True)

# Menentukan file untuk menyimpan hasil perbandingan
METRICS_CSV = os.path.join(COMPARISON_DIR, "metrics_table.csv")
METRICS_MD = os.path.join(COMPARISON_DIR, "metrics_table.md")
COMPARISON_CHART = os.path.join(COMPARISON_DIR, "comparison_chart.png")


def load_training_logs(log_path):
    """
    Memuat log training dari file CSV.
    """
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        print(f"Loaded training log from {log_path}")
        return df
    else:
        print(f"Error: File not found - {log_path}")
        return None


def calculate_final_metrics(df):
    """
    Menghitung metrik akhir (akurasi dan loss) dari hasil training.
    """
    final_metrics = {
        "Final Train Loss": df["train_loss"].values[-1],
        "Final Train Acc": df["train_acc"].values[-1],
        "Final Val Loss": df["val_loss"].values[-1],
        "Final Val Acc": df["val_acc"].values[-1],
        "Best Val Acc": df["val_acc"].max(),
        "Best Train Acc": df["train_acc"].max(),
        "Epochs": len(df)
    }
    return final_metrics


def save_metrics_table(metrics, csv_path, md_path):
    """
    Menyimpan metrik akhir ke file CSV dan Markdown.
    """
    # Mengonversi metrik ke dataframe untuk kemudahan penyimpanan
    df = pd.DataFrame(metrics).transpose()
    df.index.name = "Model"
    df.to_csv(csv_path)
    
    # Membuat tabel dalam format Markdown
    with open(md_path, "w") as md_file:
        md_file.write(df.to_markdown())
    
    print(f"Metrics saved to {csv_path} and {md_path}")


def plot_comparison_chart(max_df, hybrid_df, save_path):
    """
    Membuat grafik perbandingan performa model.
    """
    plt.figure(figsize=(14, 10))

    # Membuat grafik untuk akurasi training
    plt.subplot(2, 2, 1)
    plt.plot(max_df["train_acc"], label="Max Pooling - Train Acc", color="blue")
    plt.plot(hybrid_df["train_acc"], label="Hybrid Pooling - Train Acc", color="green")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Membuat grafik untuk akurasi validasi
    plt.subplot(2, 2, 2)
    plt.plot(max_df["val_acc"], label="Max Pooling - Val Acc", color="blue")
    plt.plot(hybrid_df["val_acc"], label="Hybrid Pooling - Val Acc", color="green")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Membuat grafik untuk loss training
    plt.subplot(2, 2, 3)
    plt.plot(max_df["train_loss"], label="Max Pooling - Train Loss", color="blue")
    plt.plot(hybrid_df["train_loss"], label="Hybrid Pooling - Train Loss", color="green")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Membuat grafik untuk loss validasi
    plt.subplot(2, 2, 4)
    plt.plot(max_df["val_loss"], label="Max Pooling - Val Loss", color="blue")
    plt.plot(hybrid_df["val_loss"], label="Hybrid Pooling - Val Loss", color="green")
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Comparison chart saved to {save_path}")


def main():
    # Memuat log training
    max_df = load_training_logs(MAX_POOLING_LOG)
    hybrid_df = load_training_logs(HYBRID_POOLING_LOG)

    # Memastikan kedua file log ditemukan
    if max_df is None or hybrid_df is None:
        print("Error: One or both training logs not found.")
        return

    # Menghitung metrik akhir untuk kedua model
    max_metrics = calculate_final_metrics(max_df)
    hybrid_metrics = calculate_final_metrics(hybrid_df)

    # Menyimpan hasil perbandingan ke file
    metrics = {
        "Max Pooling": max_metrics,
        "Hybrid Pooling": hybrid_metrics
    }
    save_metrics_table(metrics, METRICS_CSV, METRICS_MD)

    # Membuat grafik perbandingan
    plot_comparison_chart(max_df, hybrid_df, COMPARISON_CHART)


# if __name__ == "__main__":
#     main()
