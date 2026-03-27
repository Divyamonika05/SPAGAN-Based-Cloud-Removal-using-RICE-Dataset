# GAN-Baesd Cloud Removal for Enhancing Satellite Imagery

## 📌 Overview

Clouds in satellite imagery can obstruct important details required for agricultural monitoring.
This project presents a deep learning-based solution to remove clouds from rice field images using a GAN-based architecture.

The model learns to reconstruct cloud-free images from cloudy inputs, improving the quality and usability of satellite data.

---

## 🎯 Objectives

- Remove clouds from satellite images
- Improve visibility of rice field data
- Apply deep learning (GAN-based architecture)
- Evaluate performance using image quality metrics

---

## 🧠 Methodology

The project uses a **Generator-Discriminator framework**:

- **Generator (SPANet)**
  Generates cloud-free images from cloudy inputs

- **Discriminator**
  Distinguishes between real and generated images

- **Training Process**
  - Adversarial loss
  - Reconstruction loss (MSE)
  - Performance metrics: PSNR, SSIM

---

## 📂 Project Structure

```
riceproject/
│
├── app.py                     # Main application script
├── frontend.ipynb            # UI / demo notebook
├── model_training.ipynb      # Training workflow
├── requirements.txt
│
├── src/
│   ├── train.py              # Model training
│   ├── predict.py            # Prediction script
│   ├── eval.py               # Evaluation metrics
│   ├── log_report.py         # Logging and reporting
│   │
│   ├── models/
│   │   ├── layers.py
│   │   ├── models_utils.py
│   │   ├── dis/dis.py        # Discriminator
│   │   └── gen/SPANet.py     # Generator
│   │
│   └── utils/
│       ├── data_manager.py
│       └── utils.py
│
├── configs/
│   └── config.yml

---

##  Dataset

The dataset is not included in this repository due to size limitations.

You can download it from Kaggle:
👉 https://www.kaggle.com/datasets/shubhank001/rice-remote-sensing-images-for-cloud-removal

### 📥 Setup Dataset

1. Download dataset from Kaggle
2. Extract it
3. Place it in:
   dataset/

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Train the model

```bash
python src/train.py
```

### 🔹 Run prediction

```bash
python src/predict.py
```

### 🔹 Evaluate model

```bash
python src/eval.py
```

---

## 📈 Evaluation Metrics

- **MSE (Mean Squared Error)**
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**

---

## 🖼️ Results

- Successfully removes cloud regions from images
- Improves visual clarity of satellite data
- Produces realistic reconstructed outputs

_(Add sample output images here for better presentation)_

---

## 🚀 Future Work

- Improve model generalization
- Use larger datasets
- Optimize training time
- Deploy as web application

---

## 📜 License

This project is for academic and research purposes.
