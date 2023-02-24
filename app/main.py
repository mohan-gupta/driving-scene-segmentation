from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from predict import get_results, encode_image, decode_base64

class InpImage(BaseModel):
    image: str

app = FastAPI()

@app.get("/")
def home():
    return {"Data": "Welcome Home"}

@app.post("/predict")
def predict(inp_img: InpImage):
    if(len(inp_img.image)==0):
        return {"Error": "Image Not Found"}

    #converting base64 string to numpy array
    np_arr = decode_base64(inp_img.image)

    #predicting the result
    result = get_results(np_arr)

    #converting result to base64 string
    img_string = encode_image(result, np_arr.shape[:-1][::-1])
    
    return {"output": img_string}

if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()