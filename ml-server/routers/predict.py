from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from utils.inference import predict_disease

router = APIRouter()

@router.post('/predict')
async def predict(
    file: UploadFile = File(...),
    crop_type: str = Form(...)
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='File must be an image')
    image_bytes = await file.read()
    result = predict_disease(image_bytes, crop_type)
    return result
