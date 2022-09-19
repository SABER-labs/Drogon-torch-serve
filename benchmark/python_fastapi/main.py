from fastapi import FastAPI, File, UploadFile

import utils

app = FastAPI()


@app.get("/")
def home():
    return 'OK'


@app.post("/classify")
async def classify(image: UploadFile = File(...)):
    return utils.get_result(image_file=image, is_api=True)
