import os
import io
import gc
import base64
import cv2
import numpy as np

# Optimize memory footprint
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

app = FastAPI(title="AquaPath API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ID = "abhimishra001/fish-disease-weights"
WEIGHT_FILE = "weights_efficientnetb0.weights.h5"

CLASSES = [
    'Bacterial Red disease',
    'Bacterial diseases - Aeromoniasis',
    'Bacterial gill disease',
    'Fungal diseases - Saprolegniasis',
    'Healthy Fish',
    'Parasitic diseases',
    'Viral diseases'
]

# Model Setup
def build_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    base = EfficientNetB0(include_top=False, weights=None, input_tensor=inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs), base

model, base_model = build_model()
weights_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHT_FILE)
model.load_weights(weights_path)

target_layer = base_model.get_layer("top_conv")
grad_model = tf.keras.models.Model(inputs=[model.inputs], outputs=[target_layer.output, model.output])

gc.collect()

def run_inference(orig_np):
    img_resized = cv2.resize(orig_np, (224, 224), interpolation=cv2.INTER_AREA)
    x = preprocess_input(np.expand_dims(img_resized.astype(np.float32), axis=0))

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        top_idx = tf.argmax(predictions[0])
        loss = predictions[:, top_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_outputs = conv_outputs[0].numpy()
    del tape, grads

    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
    for i in range(len(pooled_grads)):
        cam += pooled_grads[i] * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    max_val = np.max(cam)
    if max_val > 0:
        cam /= max_val

    cam_resized = cv2.resize(cam, (orig_np.shape[1], orig_np.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = np.uint8(orig_np * 0.55 + cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) * 0.45)

    raw_probs = predictions[0].numpy()
    logits = np.log(np.clip(raw_probs, 1e-7, 1.0))
    calibrated = np.exp(logits / 0.45)
    calibrated_probs = calibrated / np.sum(calibrated)

    confidences = {CLASSES[i]: float(calibrated_probs[i]) for i in range(len(CLASSES))}
    winner = CLASSES[int(np.argmax(calibrated_probs))]

    return overlay, confidences, winner

@app.get("/")
def health_check():
    return {"status": "AquaPath Workstation Active"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    orig_np = np.array(pil_img)
    del contents

    overlay, confidences, winner = run_inference(orig_np)

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64_cam = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    del orig_np, overlay, buffer
    gc.collect()

    return JSONResponse(content={
        "cam_image": b64_cam,
        "confidences": confidences,
        "primary_class": winner
    })
