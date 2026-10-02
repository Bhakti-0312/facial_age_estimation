# Facial Age Estimation Using Deep Learning

A deep learning-based facial age estimation system that predicts a person's approximate age from a facial image. The project treats age estimation as a regression problem and compares three different neural network architectures.

## 📌 Overview

Facial age estimation is a challenging computer vision task because facial appearance varies with factors such as expression, lighting, facial structure, and individual aging patterns.

This project implements and compares:

- CNN — Conventional Convolutional Neural Network
- ConvNeXt-Tiny — ImageNet-pretrained modern CNN
- ConvNeXt-Transformer — ConvNeXt feature extraction combined with Transformer encoder layers

All three models are trained and evaluated using the same dataset split and experimental procedure.

## 📊 Dataset

The experiments use the **UTKFace Aligned & Cropped** dataset.

- **Total images:** 23,708
- **Age range:** 0–116 years
- **Image type:** Aligned and cropped facial images

### Dataset Split

| Split | Images |
|---|---:|
| Training | 16,595 |
| Validation | 3,556 |
| Testing | 3,557 |
| **Total** | **23,708** |

The dataset is not included in this repository.

## 🖼️ Preprocessing

All images are resized to **224 × 224** pixels.

Training images use:

- Random horizontal flip
- Color jitter
- Random rotation
- ImageNet normalization

Validation and test images use resizing and ImageNet normalization without random augmentation.

## 🧠 Model Architectures

### 1. CNN

A conventional CNN consisting of four convolutional blocks followed by adaptive global average pooling and fully connected regression layers.

**Parameters:** 422,401

### 2. ConvNeXt-Tiny

An ImageNet-pretrained ConvNeXt-Tiny backbone with the original classification head replaced by a single-output regression head.

**Parameters:** 27,820,897

### 3. ConvNeXt-Transformer

A pretrained ConvNeXt-Tiny backbone is used for spatial feature extraction. The resulting feature map is converted into tokens and projected before being processed by four Transformer encoder layers.

The processed features are aggregated and passed through a regression head to predict age.

**Parameters:** 30,156,897

## ⚙️ Experimental Setup

| Configuration | Value |
|---|---|
| Framework | PyTorch |
| Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Input Size | 224 × 224 |
| Batch Size | 64 |
| Epochs | 5 |
| Loss Function | MSE Loss |
| Optimizer | AdamW |
| Weight Decay | 1e-4 |
| CNN Learning Rate | 1e-4 |
| ConvNeXt Learning Rate | 1e-5 |

## 📈 Evaluation Metrics

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between the actual and predicted ages.

**Lower MAE indicates lower average prediction error.**

### Root Mean Squared Error (RMSE)

RMSE measures prediction error while giving greater weight to larger errors.

**Lower RMSE indicates lower prediction error.**

## 🏆 Results

| Model | Test MAE ↓ | Test RMSE ↓ | Parameters | Training Time | Inference Time |
|---|---:|---:|---:|---:|---:|
| **CNN** | **11.4446** | **15.0564** | **0.422 M** | **7.84 min** | **10.08 s** |
| ConvNeXt-Tiny | 17.4991 | 23.2055 | 27.821 M | 25.11 min | 16.86 s |
| ConvNeXt-Transformer | 22.1395 | 27.6782 | 30.157 M | 25.39 min | 15.74 s |

Under the five-epoch experimental configuration, the CNN achieved the lowest test MAE and RMSE among the three evaluated models.

The results are specific to the dataset, preprocessing, optimization settings, and training configuration used in this experiment.

## 📓 Project Notebook

The complete implementation, training process, evaluation, and experiment outputs are available in:

**`Facial_Age_Estimation_Project.ipynb`**

The notebook contains the code used for:

- Dataset loading
- Image preprocessing
- Model implementation
- Model training
- Validation
- Test evaluation
- Metric calculation
- Prediction analysis
- Experimental results

## 📁 Repository Structure

```text
facial-age-estimation/
│
├── Facial_Age_Estimation_Project.ipynb
└── README.md
