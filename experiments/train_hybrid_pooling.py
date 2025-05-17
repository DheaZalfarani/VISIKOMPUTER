#!/usr/bin/env python3

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Menambahkan root direktori ke sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Import model hybrid, data loader, dan utilitas
from models.alexnet_hybrid import AlexNetHybrid
from utils.data_loader import get_data_loaders


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25, save_dir='results/hybrid_pooling'):
    # Membuat direktori untuk menyimpan hasil jika belum ada
    os.makedirs(save_dir, exist_ok=True)

    # Inisialisasi untuk menyimpan model terbaik
    best_model_wts = model.state_dict()
    best_acc = 0.0

    # Membuka file log untuk menyimpan hasil training
    log_file_path = os.path.join(save_dir, "training_log.csv")
    with open(log_file_path, "w") as log_file:
        log_file.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    # Menyimpan riwayat training untuk plotting
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Loop untuk fase training dan testing
        for phase in ['Training', 'Testing']:
            if phase == 'Training':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            all_preds = []
            all_labels = []

            # Iterasi batch
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward pass dan backward pass untuk training
                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()

                # Mengumpulkan statistik
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

                # Simpan prediksi dan label untuk confusion matrix
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Menghitung loss dan akurasi epoch
            epoch_loss = running_loss / total_samples
            epoch_acc = (running_corrects / total_samples) * 100

            # Menyimpan hasil ke history
            if phase == 'Training':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')

            # Menyimpan model terbaik
            if phase == 'Testing' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                
                # Simpan model terbaik
                model_save_path = os.path.join(save_dir, 'model_weights.pth')
                torch.save(best_model_wts, model_save_path)
                print(f'New best model saved to {model_save_path} with accuracy: {best_acc:.2f}%')

                # Buat confusion matrix
                cm = confusion_matrix(all_labels, all_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"])
                disp.plot(cmap="Blues")
                cm_save_path = os.path.join(save_dir, "confusion_matrix.png")
                plt.savefig(cm_save_path)
                plt.close()
                print(f'Confusion matrix saved to {cm_save_path}')

        # Simpan hasil epoch ke file log
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{epoch+1},{history['train_loss'][-1]},{history['train_acc'][-1]},{history['val_loss'][-1]},{history['val_acc'][-1]}\n")

        # Update scheduler setelah setiap epoch
        scheduler.step()
        print()

    # Memuat model terbaik sebelum mengembalikan
    model.load_state_dict(best_model_wts)
    return model, history


# Fungsi utama untuk memulai pelatihan
def main():
    data_dir = os.path.join(ROOT_DIR, "data")
    results_dir = os.path.join(ROOT_DIR, "results/hybrid_pooling")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_data_loaders(
        data_dir, 
        batch_size=16,
        num_workers=4
    )

    dataloaders = {'Training': train_loader, 'Testing': val_loader}

    model = AlexNetHybrid(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model, history = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer, 
        scheduler,
        device, 
        num_epochs=25,
        save_dir=results_dir
    )

    print(f"All results saved to {results_dir}")


# Blok utama untuk menjalankan skrip tanpa argumen
# if __name__ == "__main__":
#     main()
