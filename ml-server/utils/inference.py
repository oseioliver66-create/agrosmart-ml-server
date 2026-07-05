import numpy as np
from PIL import Image, ImageFile
import tensorflow as tf
import io
import os
import zipfile
import gdown

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_SIZE = 300
CONFIDENCE_THRESHOLD = 0.3  # 30% is sufficient for crop leaves

# ---------------------------------------------------------
# Per-crop model configuration
# 'format' is either 'h5' (single Keras .h5 file) or
# 'saved_model' (zipped TensorFlow SavedModel folder)
# ---------------------------------------------------------
MODEL_CONFIGS = {
    'cassava': {
        'format': 'saved_model',
        'zip_filename': 'cassava_savedmodel.zip',
        'folder_name': 'cassava_savedmodel',
        'gdrive_file_id': '1MjdVEpVQs6Y0ApAZacNOx3JimMt3JHsm',
        'classes': {
            0: 'Cassava___Bacterial_Blight',
            1: 'Cassava___Brown_Streak_Disease',
            2: 'Cassava___Green_Mottle',
            3: 'Cassava___Healthy',
            4: 'Cassava___Mosaic_Disease',
        },
    },
    'maize': {
        'format': 'h5',
        'filename': 'best_maize.h5',
        'gdrive_file_id': '1hht1qJOxAe6wZX0qdrIGQ_wAMUI7cEKr',
        'classes': {
            0: 'Maize___Gray_Leaf_Spot',
            1: 'Maize___Healthy',
            2: 'Maize___Common_Rust',
            3: 'Maize___Blight',
        },
    },
    'tomato': {
        'format': 'saved_model',
        'zip_filename': 'tomato_savedmodel.zip',
        'folder_name': 'tomato_savedmodel',
        'gdrive_file_id': '1UVu8TB_KE-ZPbTK-y-3OA-ttSetnIm1I',
        'classes': {
            0: 'Tomato___Bacterial_Spot',
            1: 'Tomato___Early_Blight',
            2: 'Tomato___Healthy',
            3: 'Tomato___Late_Blight',
            4: 'Tomato___Leaf_Mold',
            5: 'Tomato___Septoria_Leaf_Spot',
            6: 'Tomato___Spider_Mites',
            7: 'Tomato___Target_Spot',
            8: 'Tomato___Tomato_Mosaic_Virus',
            9: 'Tomato___Yellow_Leaf_Curl_Virus',
        },
    },
}

SEVERITY_MAP = {
    'Cassava___Bacterial_Blight':       'high',
    'Cassava___Brown_Streak_Disease':   'high',
    'Cassava___Green_Mottle':           'medium',
    'Cassava___Healthy':                'low',
    'Cassava___Mosaic_Disease':         'high',
    'Maize___Blight':                   'high',
    'Maize___Common_Rust':              'medium',
    'Maize___Gray_Leaf_Spot':           'medium',
    'Maize___Healthy':                  'low',
    'Tomato___Bacterial_Spot':          'medium',
    'Tomato___Early_Blight':            'medium',
    'Tomato___Healthy':                 'low',
    'Tomato___Late_Blight':             'high',
    'Tomato___Leaf_Mold':               'medium',
    'Tomato___Septoria_Leaf_Spot':      'medium',
    'Tomato___Spider_Mites':            'low',
    'Tomato___Target_Spot':             'medium',
    'Tomato___Tomato_Mosaic_Virus':     'high',
    'Tomato___Yellow_Leaf_Curl_Virus':  'high',
}

# In-memory cache so each model only loads once per server run
# For h5 models this stores the Keras model directly.
# For saved_model models this stores the callable inference signature.
_loaded_models = {}
_loaded_formats = {}


def format_disease_name(label: str) -> str:
    parts = label.split('___')
    return parts[1].replace('_', ' ') if len(parts) == 2 else label.replace('_', ' ')


def _download_h5_model(model_path: str, gdrive_file_id: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f'Downloading model from Google Drive using gdown... ({model_path})')
    gdown.download(id=gdrive_file_id, output=model_path, quiet=False)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f'Model downloaded: {size_mb:.1f} MB')
    if size_mb < 10:
        os.remove(model_path)
        raise ValueError(f'Download failed — only {size_mb:.1f} MB received.')


def _download_saved_model(crop_dir: str, zip_filename: str, folder_name: str, gdrive_file_id: str):
    os.makedirs(crop_dir, exist_ok=True)
    zip_path = os.path.join(crop_dir, zip_filename)
    extract_path = os.path.join(crop_dir, folder_name)

    print(f'Downloading SavedModel zip from Google Drive using gdown... ({zip_path})')
    gdown.download(id=gdrive_file_id, output=zip_path, quiet=False)
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f'Zip downloaded: {size_mb:.1f} MB')
    if size_mb < 1:
        os.remove(zip_path)
        raise ValueError(f'Download failed — only {size_mb:.1f} MB received.')

    print(f'Extracting SavedModel to {extract_path}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    os.remove(zip_path)
    print('Extraction complete, zip removed.')
    return extract_path


def _get_model(crop_type: str):
    crop_key = crop_type.lower()
    if crop_key not in MODEL_CONFIGS:
        raise ValueError(f'Unknown crop_type: {crop_type}')

    if crop_key in _loaded_models:
        return _loaded_models[crop_key]

    config = MODEL_CONFIGS[crop_key]
    crop_dir = os.path.join(BASE_DIR, 'model', crop_key)

    if config['format'] == 'h5':
        model_path = os.path.join(crop_dir, config['filename'])

        if not os.path.exists(model_path):
            _download_h5_model(model_path, config['gdrive_file_id'])
        else:
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if size_mb < 10:
                print(f'Corrupted model ({size_mb:.1f} MB), re-downloading...')
                os.remove(model_path)
                _download_h5_model(model_path, config['gdrive_file_id'])
            else:
                print(f'Model found: {model_path} ({size_mb:.1f} MB)')

        print(f'Loading {crop_key} model (.h5)...')
        model = tf.keras.models.load_model(model_path)
        print(f'{crop_key.capitalize()} model loaded successfully!')

        _loaded_models[crop_key] = model
        _loaded_formats[crop_key] = 'h5'
        return model

    else:  # saved_model
        extract_path = os.path.join(crop_dir, config['folder_name'])

        # A valid extracted SavedModel folder contains a saved_model.pb
        # somewhere inside it (possibly nested one level from the zip).
        def _find_saved_model_dir(root):
            for dirpath, _, filenames in os.walk(root):
                if 'saved_model.pb' in filenames:
                    return dirpath
            return None

        saved_model_dir = _find_saved_model_dir(extract_path) if os.path.exists(extract_path) else None

        if not saved_model_dir:
            _download_saved_model(crop_dir, config['zip_filename'], config['folder_name'], config['gdrive_file_id'])
            saved_model_dir = _find_saved_model_dir(extract_path)

        if not saved_model_dir:
            raise ValueError(f'Could not locate saved_model.pb after extracting {crop_key} SavedModel.')

        print(f'Loading {crop_key} model (SavedModel) from {saved_model_dir}...')
        loaded = tf.saved_model.load(saved_model_dir)
        infer = loaded.signatures['serving_default']
        print(f'{crop_key.capitalize()} model loaded successfully!')

        _loaded_models[crop_key] = infer
        _loaded_formats[crop_key] = 'saved_model'
        return infer


def _run_inference(model, arr, crop_key: str) -> np.ndarray:
    fmt = _loaded_formats.get(crop_key, 'h5')

    if fmt == 'h5':
        return model.predict(arr, verbose=0)[0]

    # saved_model: call the serving signature directly
    input_tensor = tf.constant(arr)
    input_key = list(model.structured_input_signature[1].keys())[0]
    output = model(**{input_key: input_tensor})
    output_key = list(output.keys())[0]
    return output[output_key].numpy()[0]


def predict_disease(image_bytes: bytes, crop_type: str) -> dict:
    try:
        crop_key = crop_type.lower()
        model = _get_model(crop_key)
        classes = MODEL_CONFIGS[crop_key]['classes']

        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        output = _run_inference(model, arr, crop_key)

        best_idx = int(np.argmax(output))
        confidence = float(output[best_idx])

        if confidence < CONFIDENCE_THRESHOLD:
            return {
                'disease_label': 'Unknown',
                'disease_name': 'Unrecognised image',
                'confidence': round(confidence, 4),
                'severity': 'low',
                'crop_type': crop_type,
                'warning': f'This image does not appear to be a recognised {crop_type} leaf. Please upload a clear photo of a {crop_type} leaf.',
            }

        label = classes[best_idx]
        disease_name = format_disease_name(label)
        severity = SEVERITY_MAP.get(label, 'medium')

        return {
            'disease_label': label,
            'disease_name': disease_name,
            'confidence': round(confidence, 4),
            'severity': severity,
            'crop_type': crop_type,
        }
    except Exception as e:
        raise Exception(f'Inference failed: {str(e)}')