import tensorflow  as tf
from PIL import Image
from io import BytesIO
import numpy as np

from fastapi import FastAPI
import uvicorn
from fastapi import File, UploadFile


INPUT_SHAPE = (244,244)

def get_prediction(image: Image.Image):
    class_names = ['borowik szlachetny', 
                    'czernikdlak kolpakowaty', 
                    'czubajka kania', 
                    'kozlarz babka', 
                    'maślak zwyczajny', 
                    'muchomor czerwony', 
                    'opienka miodowa', 
                    'pieczarka biaława', 
                    'pieczarka okazala', 
                    'pieprznik jadalny']

    loaded_model = tf.keras.models.load_model(r'C:\Users\pbirylo\Desktop\grzyby_projekt\trening0801\model1')

    img = image.resize(INPUT_SHAPE)

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    pred = (
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    return pred

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

app = FastAPI()

@app.get('/index')
def hello_world():
    return "Hellow world"

@app.post('/api/predict')
async def predict_image(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    preds = get_prediction(image)
    print(preds)
    
    return preds

if __name__ =="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
