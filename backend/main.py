from fastapi import FastAPI, File, UploadFile, HTTPException
from predict import predict_image
import io
from PIL import Image, UnidentifiedImageError

app = FastAPI(
    title="Knee X-Ray Classifier",
    description="API for knee X-ray classification",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "Knee X-Ray Classifier API is running"
    }
    
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )
    try:
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        result = predict_image(image)

        return {
            "filename" : file.filename,
            **result
        }
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )