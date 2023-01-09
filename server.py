import tensorflow  as tf
from PIL import Image
from io import BytesIO

from fastapi import FastAPI
import uvicorn
from fastapi import File, UploadFile

INPUT_SHAPE = (244,244)

def get_prediction(image: Image.Image, model):
    class_names = ['borowik szlachetny', #0
                    'czernikdlak kolpakowaty', #1
                    'czubajka kania', #2
                    'kozlarz babka', #3
                    'maślak zwyczajny',#4 
                    'muchomor czerwony', #5
                    'opienka miodowa', #6
                    'pieczarka biaława', #7
                    'pieczarka okazala', #8
                    'pieprznik jadalny']#9

    img = image.resize(INPUT_SHAPE)

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)

    score = tf.nn.softmax(predictions[0])
    
    result = []
    for probability, name in zip(score.numpy(), class_names):
        result.append((probability, name))
    result.sort(key=lambda y: y[0])

    # print("Predictions:")
    # print("{} with a {:.2f}% confidence."
    #     .format(result[-1][1], 100 * result[-1][0]))
    # print("{} with a {:.2f}% confidence."
    #     .format(result[-2][1], 100 * result[-2][0]))
    # print("{} with a {:.2f}% confidence."
    #     .format(result[-3][1], 100 * result[-3][0]))


    preds = [
        "{}, {:.2f}%".format(result[-1][1], 100 * result[-1][0]),
        "{}, {:.2f}%".format(result[-2][1], 100 * result[-2][0]),
        "{}, {:.2f}%".format(result[-3][1], 100 * result[-3][0])
    ]


    return preds

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

app = FastAPI()

@app.get('/index')
def hello_world():
    return "Hellow world"

@app.post('/api/predict')
async def predict_image(file: UploadFile = File(...)):
    LOADED_MODEL = tf.keras.models.load_model(r'C:\Users\pbirylo\Desktop\grzyby_projekt\trening0801\model1')
    image = read_imagefile(await file.read())
    preds = get_prediction(image, model=LOADED_MODEL)
    print(preds)
    
    return preds

if __name__ =="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
