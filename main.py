#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import shutil

# Menambahkan root direktori ke sys.path untuk memastikan import dari models/ dan utils/
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Import modul yang diperlukan untuk training dan evaluasi
from experiments.train_max_pooling import main as train_max_pooling
from experiments.train_hybrid_pooling import main as train_hybrid_pooling
from experiments.compare_result import main as compare_results


def clear_results():
    """
    Membersihkan direktori hasil sebelumnya untuk memastikan eksperimen baru dimulai dari awal.
    """
    results_dir = os.path.join(ROOT_DIR, "results")
    if os.path.exists(results_dir):
        print(f"Menghapus direktori hasil sebelumnya: {results_dir}")
        shutil.rmtree(results_dir)
    os.makedirs(os.path.join(results_dir, "max_pooling"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "hybrid_pooling"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "comparison"), exist_ok=True)
    print("Direktori hasil berhasil dibuat ulang.")


def main(args=None):
    """
    Fungsi utama untuk menjalankan eksperimen lengkap.
    """
    # Parsing argumen baris perintah
    parser = argparse.ArgumentParser(description="Run full brain tumor classification experiment")
    parser.add_argument("--clear_results", action="store_true", help="Hapus hasil eksperimen sebelumnya")
    parser.add_argument("--train_max", action="store_true", help="Jalankan training untuk model max pooling")
    parser.add_argument("--train_hybrid", action="store_true", help="Jalankan training untuk model hybrid pooling")
    parser.add_argument("--compare", action="store_true", help="Bandingkan hasil model max pooling dan hybrid pooling")
    args = parser.parse_args()

    # Hapus hasil eksperimen sebelumnya jika diminta
    if args.clear_results:
        clear_results()

    # Jalankan training untuk model max pooling
    if args.train_max:
        print("\n==== Memulai Training Max Pooling Model ====")
        train_max_pooling()
        print("\n==== Selesai Training Max Pooling Model ====")

    # Jalankan training untuk model hybrid pooling
    if args.train_hybrid:
        print("\n==== Memulai Training Hybrid Pooling Model ====")
        train_hybrid_pooling()
        print("\n==== Selesai Training Hybrid Pooling Model ====")

    # Bandingkan hasil kedua model
    if args.compare:
        print("\n==== Memulai Perbandingan Hasil ====")
        compare_results()
        print("\n==== Selesai Perbandingan Hasil ====")

    # Jika tidak ada argumen yang diberikan, jalankan eksperimen lengkap
    if not (args.train_max or args.train_hybrid or args.compare):
        print("\n==== Memulai Eksperimen Lengkap ====")
        clear_results()
        train_max_pooling()
        train_hybrid_pooling()
        compare_results()
        print("\n==== Selesai Eksperimen Lengkap ====")


if __name__ == "__main__":
    main()
