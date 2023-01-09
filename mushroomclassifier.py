import tensorflow as tf
import keras.utils
from tensorflow import keras
from keras import layers
import sys
import splitfolders
import os
import numpy as np

DATA_PATH = r'C:\Users\pbirylo\Desktop\grzyby_projekt\images\train'
SPLITTED_PATH =r'C:\Users\pbirylo\Desktop\grzyby_projekt\output'
SAVE_PATH = r'C:\Users\pbirylo\Desktop\grzyby_projekt\saved\model_1'

IMG_SHAPE = (244, 244)
BATCH_SIZE = 1
EPOCHS = 5


""" Check versions & if GPU is visible """
def check():
    print ("Python version: ",sys.version)
    print("Tensorflow version: ",tf.__version__)
    print("GPU name: ",tf.test.gpu_device_name())


""" Split all data into train, test and validation dirs """
def split_data(data=DATA_PATH):
    splitfolders.ratio(data, output=SPLITTED_PATH, seed=2137, ratio=(0.8, 0.1,0.1)) 

    train = os.path.join(SPLITTED_PATH,'train')
    test = os.path.join(SPLITTED_PATH,'test')
    val = os.path.join(SPLITTED_PATH,'val')

    return train, test, val


""" Create keras image datasets """
def make_ds(train_path,test_path,val_path):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        label_mode = 'categorical',
        image_size = IMG_SHAPE,
        batch_size = BATCH_SIZE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        label_mode = 'categorical',
        image_size = IMG_SHAPE,
        batch_size = BATCH_SIZE)

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        val_path,
        label_mode = 'categorical',
        image_size = IMG_SHAPE,
        batch_size = BATCH_SIZE)

    class_names = train_ds.class_names
    train_ds = train_ds.cache()
    test_ds = test_ds.cache()
    valid_ds = valid_ds.cache()

    num_classes = len(class_names)

    return train_ds, test_ds, valid_ds, num_classes, class_names


""" Train the classifier """
def train_classifier(train_ds, valid_ds, num_classes, epochs=EPOCHS, save_path=SAVE_PATH):
    tf.keras.backend.clear_session()

    data_augmentation = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.2,fill_mode = 'nearest')
    ])

    base_model = tf.keras.applications.EfficientNetB7(include_top=False)

    inputs = tf.keras.Input(shape = (244,244,3))
    x = data_augmentation(inputs)
    x = base_model(x, training = False)
    x = layers.GlobalAveragePooling2D(name = 'Global_Average_Pool_2D')(x)

    outputs = layers.Dense(num_classes, activation = 'softmax')(x)


    model = keras.Model(inputs, outputs, name = "model")
    print("MODEL SUMMARY:\n", model.summary())


    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9, nesterov = True),
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])

    print(history = model.fit( 
        train_ds, 
        steps_per_epoch = len(train_ds), 
        epochs = epochs,
        validation_data = valid_ds,
        validation_steps = len(valid_ds),
    ))

    model.save(save_path)
    print("Model saved at:\n", save_path)


""" Load trained model and predict on image """
def load_predict(img_path, class_names, saved_path=SAVE_PATH):
    loaded_model = tf.keras.models.load_model(saved_path)

    img = tf.keras.utils.load_img(
        img_path, target_size=IMG_SHAPE
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


if __name__ == "__main__":
    # check if everything is ok
    check()
    # split train data into train, test, valid (0.8, 0.1, 0.1)
    train, test, val = split_data()
    # prepare datasets
    train_ds, test_ds, valid_ds, num_classes, class_names = make_ds(train, test, val)
    # train
    train_classifier(train_ds, test_ds, valid_ds, num_classes)
    # load model and provide path for image to predict
    img_path = r'C:\Users\pbirylo\Desktop\grzyby_projekt\output\test\muchomorczerwony-Amanita_muscaria\Amanita muscaria_33.jpg'
    load_predict(img_path, class_names)