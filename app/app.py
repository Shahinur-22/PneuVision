from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "densenet169_pneumonia.pth"

DEVICE = torch.device("cpu")

# Current project threshold
FINAL_THRESHOLD = 0.70


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Load DenseNet169
# --------------------------------------------------

def load_model():
    model = models.densenet169(weights=None)

    model.classifier = nn.Linear(
        model.classifier.in_features,
        2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # Supports both a direct state_dict and checkpoint dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


# --------------------------------------------------
# Prediction + Grad-CAM
# --------------------------------------------------

def predict_xray(image):
    if image is None:
        return (
            "Please upload a chest X-ray image.",
            None
        )

    image = image.convert("RGB")

    # Original image for Grad-CAM overlay
    rgb_image = np.array(
        image.resize((224, 224))
    ).astype(np.float32) / 255.0

    # Model input
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    normal_probability = probabilities[0].item()
    pneumonia_probability = probabilities[1].item()

    # Threshold-based prediction
    if pneumonia_probability >= FINAL_THRESHOLD:
        prediction = "PNEUMONIA"
        predicted_class = 1
        confidence = pneumonia_probability
    else:
        prediction = "NORMAL"
        predicted_class = 0
        confidence = normal_probability

    # --------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------

    target_layers = [model.features.norm5]

    targets = [
        ClassifierOutputTarget(predicted_class)
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    visualization = np.uint8(visualization)

    # --------------------------------------------------
    # Result text
    # --------------------------------------------------

    result_text = (
        f"Prediction: {prediction}\n\n"
        f"Confidence: {confidence * 100:.2f}%\n\n"
        f"NORMAL Probability: {normal_probability * 100:.2f}%\n"
        f"PNEUMONIA Probability: {pneumonia_probability * 100:.2f}%\n\n"
        f"Decision Threshold: {FINAL_THRESHOLD:.2f}"
    )

    return result_text, visualization


# --------------------------------------------------
# Gradio Interface
# --------------------------------------------------

demo = gr.Interface(
    fn=predict_xray,

    inputs=gr.Image(
        type="pil",
        label="Upload Chest X-ray"
    ),

    outputs=[
        gr.Textbox(
            label="Prediction Result",
            lines=8
        ),

        gr.Image(
            label="Grad-CAM Explanation"
        )
    ],

    title="PneuVision",

    description=(
        "An Explainable Deep Learning Framework for "
        "Chest X-ray Pneumonia Detection using DenseNet169.\n\n"
        "Academic demonstration only. "
        "This system is not intended for clinical diagnosis."
    )
)


if __name__ == "__main__":
    print("Starting PneuVision...")
    print(f"Model path: {MODEL_PATH}")

    demo.launch(
        inbrowser=True,
        share=False
    )