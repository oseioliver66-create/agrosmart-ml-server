import json
import numpy as np
from PIL import Image, ImageFile
import tensorflow as tf
import io
import os
import requests

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_model.h5')
INDEX_PATH = os.path.join(BASE_DIR, 'model', 'class_indices.json')

GDRIVE_FILE_ID = '1fa-Esn0w3JVqZwZTzvvPKPWQsgmJamyc'

def download_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print('Downloading model from Google Drive...')

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        # Also check response content for confirmation token
        return None

    session = requests.Session()
    url = 'https://drive.google.com/uc?export=download'
    params = {'id': GDRIVE_FILE_ID, 'confirm': 't'}

    response = session.get(url, params=params, stream=True)

    # Check if we got the virus scan warning page
    token = get_confirm_token(response)
    if token:
        params['confirm'] = token
        response = session.get(url, params=params, stream=True)

    total = 0
    with open(MODEL_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    size_mb = total / (1024 * 1024)
    print(f'Model downloaded: {size_mb:.1f} MB')

    if size_mb < 10:
        os.remove(MODEL_PATH)
        raise ValueError(f'Download failed — only got {size_mb:.1f} MB. Check Google Drive sharing settings.')

if not os.path.exists(MODEL_PATH):
    download_model()
else:
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    if size_mb < 10:
        print(f'Corrupted model found ({size_mb:.1f} MB), re-downloading...')
        os.remove(MODEL_PATH)
        download_model()
    else:
        print(f'Model found: {MODEL_PATH} ({size_mb:.1f} MB)')

print('Loading model...')
model    = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = 300
print('Model loaded successfully!')

with open(INDEX_PATH, 'r') as f:
    idx_to_class = json.load(f)

SEVERITY_MAP = {
    'Cassava___Bacterial_Blight':     'high',
    'Cassava___Brown_Streak_Disease': 'high',
    'Cassava___Green_Mottle':         'medium',
    'Cassava___Healthy':              'low',
    'Cassava___Mosaic_Disease':       'high',
    'Maize___Blight':                 'high',
    'Maize___Common_Rust':            'medium',
    'Maize___Gray_Leaf_Spot':         'medium',
    'Maize___Healthy':                'low',
    'Tomato___Bacterial_Spot':        'medium',
    'Tomato___Early_Blight':          'medium',
    'Tomato___Healthy':               'low',
    'Tomato___Late_Blight':           'high',
    'Tomato___Leaf_Mold':             'medium',
    'Tomato___Septoria_Leaf_Spot':    'medium',
    'Tomato___Spider_Mites':          'low',
    'Tomato___Target_Spot':           'medium',
    'Tomato___Tomato_Mosaic_Virus':   'high',
    'Tomato___Yellow_Leaf_Curl_Virus':'high',
}

def format_disease_name(label: str) -> str:
    parts = label.split('___')
    return parts[1].replace('_', ' ') if len(parts) == 2 else label.replace('_', ' ')

def predict_disease(image_bytes: bytes, crop_type: str) -> dict:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        output = model.predict(arr, verbose=0)[0]

        crop_prefix   = crop_type.capitalize() + '___'
        valid_indices = [int(k) for k, v in idx_to_class.items() if v.startswith(crop_prefix)]

        if valid_indices:
            crop_scores = {i: float(output[i]) for i in valid_indices}
            best_idx    = max(crop_scores, key=crop_scores.get)
            confidence  = crop_scores[best_idx]
        else:
            best_idx   = int(np.argmax(output))
            confidence = float(output[best_idx])

        label        = idx_to_class[str(best_idx)]
        disease_name = format_disease_name(label)
        severity     = SEVERITY_MAP.get(label, 'medium')

        return {
            'disease_label': label,
            'disease_name':  disease_name,
            'confidence':    round(confidence, 4),
            'severity':      severity,
            'crop_type':     crop_type,
        }
    except Exception as e:
        raise Exception(f'Inference failed: {str(e)}')
