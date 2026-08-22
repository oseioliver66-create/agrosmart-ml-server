"""Disease prediction using AgroSmart's local ONNX models."""

import io
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_SIZE = 300
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONFIDENCE_THRESHOLDS = {'cassava': 0.30, 'maize': 0.65, 'tomato': 0.75}

ONNX_MODEL_CONFIGS = {
    crop: {
        'model_path': os.path.join(BASE_DIR, 'model', 'onnx', f'best_{crop}.onnx'),
        'labels_path': os.path.join(BASE_DIR, 'model', 'onnx', f'class_indices_{crop}.json'),
    } for crop in ('cassava', 'maize', 'tomato')
}

CANONICAL_LABELS = {
    'Cassava___bacterial_blight': 'Cassava___Bacterial_Blight',
    'Cassava___brown_streak_disease': 'Cassava___Brown_Streak_Disease',
    'Cassava___green_mottle': 'Cassava___Green_Mottle',
    'Cassava___healthy': 'Cassava___Healthy',
    'Cassava___mosaic_disease': 'Cassava___Mosaic_Disease',
    'Tomato_Bacterial_spot': 'Tomato___Bacterial_Spot',
    'Tomato_Early_blight': 'Tomato___Early_Blight',
    'Tomato_Late_blight': 'Tomato___Late_Blight',
    'Tomato_Leaf_Mold': 'Tomato___Leaf_Mold',
    'Tomato_Septoria_leaf_spot': 'Tomato___Septoria_Leaf_Spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato___Spider_Mites',
    'Tomato__Target_Spot': 'Tomato___Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato___Yellow_Leaf_Curl_Virus',
    'Tomato__Tomato_mosaic_virus': 'Tomato___Tomato_Mosaic_Virus',
    'Tomato_healthy': 'Tomato___Healthy',
}

SEVERITY_MAP = {
    'Cassava___Bacterial_Blight': 'high', 'Cassava___Brown_Streak_Disease': 'high',
    'Cassava___Green_Mottle': 'medium', 'Cassava___Healthy': 'low', 'Cassava___Mosaic_Disease': 'high',
    'Maize___Blight': 'high', 'Maize___Common_Rust': 'medium', 'Maize___Gray_Leaf_Spot': 'medium', 'Maize___Healthy': 'low',
    'Tomato___Bacterial_Spot': 'medium', 'Tomato___Early_Blight': 'medium', 'Tomato___Healthy': 'low',
    'Tomato___Late_Blight': 'high', 'Tomato___Leaf_Mold': 'medium', 'Tomato___Septoria_Leaf_Spot': 'medium',
    'Tomato___Spider_Mites': 'low', 'Tomato___Target_Spot': 'medium',
    'Tomato___Tomato_Mosaic_Virus': 'high', 'Tomato___Yellow_Leaf_Curl_Virus': 'high',
}

_loaded_models = {}
_loaded_labels = {}


def _get_disease_model(crop_key):
    if crop_key in _loaded_models:
        return _loaded_models[crop_key], _loaded_labels[crop_key]
    if crop_key not in ONNX_MODEL_CONFIGS:
        raise ValueError(f'Unknown crop_type: {crop_key}')
    config = ONNX_MODEL_CONFIGS[crop_key]
    if not os.path.isfile(config['model_path']) or not os.path.isfile(config['labels_path']):
        raise FileNotFoundError(f'Missing local ONNX model or labels for {crop_key}.')
    with open(config['labels_path'], encoding='utf-8') as label_file:
        label_map = json.load(label_file)
    labels = [label_map[str(index)] for index in range(len(label_map))]
    session = ort.InferenceSession(config['model_path'], providers=['CPUExecutionProvider'])
    _loaded_models[crop_key], _loaded_labels[crop_key] = session, labels
    return session, labels


def _onnx_probabilities(session, image_array):
    batch = image_array / 255.0
    batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
    batch = np.transpose(batch, (0, 3, 1, 2)).astype(np.float32)
    logits = session.run(None, {session.get_inputs()[0].name: batch})[0][0]
    shifted = logits - np.max(logits)
    return np.exp(shifted) / np.exp(shifted).sum()


def format_disease_name(label):
    parts = label.split('___', 1)
    return parts[1].replace('_', ' ') if len(parts) == 2 else label.replace('_', ' ')


def predict_disease(image_bytes, crop_type):
    try:
        crop_key = crop_type.strip().lower()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        image_array = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
        session, labels = _get_disease_model(crop_key)
        probabilities = _onnx_probabilities(session, image_array)
        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])
        if confidence < CONFIDENCE_THRESHOLDS[crop_key]:
            return {'disease_label': 'Unknown', 'disease_name': 'Unrecognised image', 'confidence': round(confidence, 4), 'severity': 'low', 'crop_type': crop_type, 'warning': f'This image does not appear to be a clear {crop_type} leaf. Please upload a clear leaf photo.'}
        label = CANONICAL_LABELS.get(labels[best_idx], labels[best_idx])
        return {'disease_label': label, 'disease_name': format_disease_name(label), 'confidence': round(confidence, 4), 'severity': SEVERITY_MAP.get(label, 'medium'), 'crop_type': crop_type}
    except Exception as error:
        raise Exception(f'Inference failed: {error}')
