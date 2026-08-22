from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import predict

app = FastAPI(title='AgroSmart ML Server', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(predict.router)

@app.get('/')
def root():
    return {'status': 'AgroSmart ML Server running', 'model': 'EfficientNetB3'}

@app.get('/health')
def health():
    return {'status': 'ok'}
