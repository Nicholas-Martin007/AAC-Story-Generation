# Fine-Tuned LLM – Python

Repository ini digunakan untuk proses fine-tuning Large Language Model (LLM) menggunakan Python dengan pendekatan LoRA / PEFT, serta mendukung proses preprocessing, training, inference, dan synthetic data generation.

Untuk mendapatkan full experience aplikasi AAC, silakan gunakan repository berikut:

Backend (Django + LLM API): https://github.com/Nicholas-Martin007/Backend-LLM-Django

Frontend (React): https://github.com/Nicholas-Martin007/AAC-React

Struktur proyek dirancang modular agar mudah dipahami, dikembangkan, dan direproduksi.

## 🤖 Setup Model

1. Unduh model dasar secara manual melalui Hugging Face:

   * **LLaMA 3.2 3B Instruct**
     [https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

2. Untuk model yang telah dilakukan *fine-tuning*, silakan unduh file model melalui tautan berikut:

   * **LLaMA 3.2 3B (Fine-Tuned)**
     `https://drive.google.com/drive/folders/1x5uzL9SEvdquhbx9rctEY9DxijIjtnwA`

---

## 📊 Dataset

Dataset yang digunakan untuk training dapat dilihat pada folder:

```text
dataset/
```

Dataset ini merupakan hasil preprocessing dan/atau synthetic data generation yang telah disesuaikan dengan kebutuhan model AAC.

---

## ⚡ Inference

Untuk melakukan inference setelah model selesai di-training, tersedia beberapa opsi:

```text
inference/inference_llama.py
```

* Digunakan untuk inference standar dengan model fine-tuned

```text
inference/inference_quantized.py
```

* Digunakan untuk inference lebih cepat menggunakan model ter-quantization

---

## 🧹 Pre-processing

Tahapan preprocessing data dibagi menjadi beberapa bagian:

```text
pre-processing/clean-filter/
```

* Digunakan untuk **filtering** dan **cleaning data teks**

```text
pre-processing/named-entity-recognition/
```

* Digunakan untuk **transformasi entitas teks** menggunakan model **IndoBERT** agar sesuai dengan format training

---

## 🧪 Synthetic Data Generation

Untuk menghasilkan dataset tambahan secara otomatis menggunakan AI, gunakan folder berikut:

```text
synthetic_data_generation/main.py
```

Proses ini berguna untuk memperkaya variasi data training dan meningkatkan generalisasi model.

---


## 🧠 Training Pipeline

Proses training utama dapat dijalankan melalui file berikut:

```text
fine_tuning/main.py
```

Di dalam folder `fine_tuning` terdapat komponen utama:

* Pembuatan format dataset sesuai standar **Hugging Face**
* Konfigurasi **model**, **tokenizer**, dan **LoRA setup**
* Training menggunakan **Trainer API**

## 📦 Requirements

Seluruh dependency Python yang dibutuhkan untuk proses fine-tuning, preprocessing, dan inference dapat dilihat pada file:

```text
requirements.txt
```



## 📘 Manual Book

Panduan penggunaan sistem secara lengkap dapat dilihat pada file berikut:

- **Manual Book**: `535220027_Manual Book.pdf`

