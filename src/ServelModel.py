import uvicorn
import cv2
import numpy as np
import pickle
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from ultralytics.engine.results import Results

from Config import ZHUOZI_MODEL_PATH
from YoloModel import YoloModel

app = FastAPI()
server_model = YoloModel(ZHUOZI_MODEL_PATH, task="detect")


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = server_model.predict(img)
    res: Results = results[0]

    res.orig_img = None

    return Response(content=pickle.dumps(res), media_type="application/octet-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
