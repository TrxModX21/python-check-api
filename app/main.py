from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

from app.predict import predict_image

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Stunting Detection API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        prediction = predict_image(image)
        return {"result": prediction}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})