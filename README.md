# PneuVision

PneuVision is my final year project for detecting pneumonia from chest X-ray images using deep learning. The project compares several pretrained convolutional neural network models and uses Grad-CAM to provide a visual explanation of the model prediction.

The final system uses DenseNet169 and provides a simple Gradio-based interface where a user can upload a chest X-ray image and receive a NORMAL or PNEUMONIA prediction together with confidence scores and a Grad-CAM heatmap.

> This project is developed for academic demonstration purposes only. It is not intended for clinical diagnosis or as a replacement for professional medical judgment.

---

## Project Objective

The main objectives of this project are:

- To classify chest X-ray images into NORMAL and PNEUMONIA.
- To compare the performance of multiple pretrained deep learning models.
- To evaluate the models using different performance metrics.
- To use Grad-CAM for visual explanation of model decisions.
- To develop a simple user interface for pneumonia prediction.

---

## Dataset

The project uses the public Chest X-Ray Images (Pneumonia) dataset available on Kaggle.

The original dataset contains NORMAL and PNEUMONIA chest X-ray images.

For this project, the original training set was divided into training and validation sets while the original test set was kept unchanged.

Final data distribution:

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1139 | 3293 | 4432 |
| Validation | 202 | 582 | 784 |
| Test | 234 | 390 | 624 |

---

## Preprocessing

All images are resized to:

```text
224 × 224