# PneuVision

## An Explainable Deep Learning Framework for Chest X-ray Pneumonia Detection

PneuVision is my final year project. The main purpose of this project is to detect pneumonia from chest X-ray images using deep learning.

The system will classify an input chest X-ray into two classes:

- NORMAL
- PNEUMONIA

Along with the prediction, I also want to understand which part of the X-ray image is influencing the model's decision. For this reason, I will use Grad-CAM to generate a visual heatmap.

---

## About the Project

Pneumonia can be detected from chest X-ray images, but reading X-rays properly requires medical expertise.

In this project, I am trying to build a simple deep learning-based system that can classify chest X-ray images.

Instead of training a deep learning model completely from scratch, I will use transfer learning with pretrained models.

I will test different models on the same dataset and compare their performance.

After comparing the models, I will select the best-performing model and use Grad-CAM to visualize the important regions of the X-ray image.

At the final stage, I plan to build a simple Gradio web application so that an X-ray image can be uploaded and the prediction can be viewed easily.

---

## Project Workflow

The basic workflow of the project is:

Chest X-ray Dataset  
↓  
Data Preprocessing  
↓  
Transfer Learning  
↓  
Model Training  
↓  
Model Evaluation  
↓  
Model Comparison  
↓  
Best Model Selection  
↓  
Grad-CAM Visualization  
↓  
Gradio Web Application

---

## Dataset

For this project, I am using the **Chest X-Ray Images (Pneumonia)** dataset available on Kaggle.

The dataset contains two classes:

- NORMAL
- PNEUMONIA

Dataset link:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The full dataset will not be uploaded to this repository.

---

## Data Preprocessing

Before training the models, the chest X-ray images will go through some basic preprocessing steps.

These include:

- Image resizing
- Normalization
- Rotation
- Horizontal flipping
- Brightness adjustment
- Basic data augmentation

The purpose of these steps is to prepare the images properly for model training and reduce overfitting.

---

## Models

I plan to experiment with several pretrained deep learning models.

The models may include:

- ResNet
- DenseNet
- EfficientNet
- MobileNet
- ResNeXt
- Wide ResNet
- Vision Transformer (ViT)

I will start with simpler models first and then add other models depending on the training time and project requirements.

---

## Model Evaluation

After training, the models will be evaluated using several common classification metrics.

These include:

- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- ROC-AUC
- Confusion Matrix

The results of the models will be compared to identify the most suitable model for the final system.

---

## Grad-CAM

Grad-CAM will be used to make the model prediction easier to understand.

Instead of only showing whether an image is NORMAL or PNEUMONIA, Grad-CAM will generate a heatmap showing which regions of the chest X-ray had more influence on the prediction.

The final output will roughly follow this format:

Chest X-ray  
↓  
Prediction: NORMAL / PNEUMONIA  
↓  
Confidence Score  
↓  
Grad-CAM Heatmap

---

## Web Application

After completing the model training and Grad-CAM part, I plan to develop a simple web interface using Gradio.

The application will allow a user to:

1. Upload a chest X-ray image.
2. Get the predicted class.
3. View the confidence score.
4. See the Grad-CAM visualization.

---

## Tools and Technologies

The main tools I plan to use are:

- Python
- PyTorch
- Torchvision
- OpenCV
- Pillow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Grad-CAM
- Gradio
- Google Colab

---

## Current Progress

The project is currently under development.

Current status:

- Project proposal completed
- GitHub repository created
- Dataset selected
- Dataset preparation started
- Model training pending
- Model comparison pending
- Grad-CAM implementation pending
- Gradio application pending

I will keep updating this repository as I complete different parts of the project.

---

## Disclaimer

This project is being developed for academic and educational purposes.

It is not intended to replace professional medical diagnosis or clinical judgment.
