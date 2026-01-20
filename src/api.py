import argparse
import io

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="YOLO11 Number Detection API")

# Load model once
model = YOLO(model="models/best.onnx", task="detect")
labels = model.names


def predict_image(image: Image.Image) -> str:
    results = model.predict(image, verbose=False)

    preds = []

    for result in results:
        boxes = result.boxes.xyxy
        classes = result.boxes.cls.reshape(-1, 1)
        confs = result.boxes.conf.reshape(-1, 1)

        for j in range(boxes.shape[0]):
            if confs[j] < 0.3:
                continue

            preds.append((boxes[j], classes[j], confs[j]))

    # Sort left to right
    preds.sort(key=lambda x: x[0][0])

    final_pred = ""
    for p in preds:
        final_pred += str(labels[p[1].item()])

    return final_pred


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prediction = predict_image(image)

    return {
        "prediction": prediction,
        "length": len(prediction),
    }


def main():
    parser = argparse.ArgumentParser(description="YOLO11 FastAPI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
