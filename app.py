from flask import Flask,render_template,request
import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array
# import cv2 as cv
import pathlib
import numpy as np
import base64
import io

app  = Flask(__name__)
@app.route('/',methods=['GET'])
def home():
  return render_template('index.html')
@app.route('/',methods=['POST'])
def predict():
    imageFile = request.files['upload']
    imagePath = "./images/" + imageFile.filename
    imageFile.save(imagePath)
    # Image = load_img(imagePath,grayscale=True,target_size=(28,28))
    # Image = img_to_array(Image)
    # Image = 255 - Image
    # Image = np.array([Image])
    # model = tf.keras.models.load_model('my_model')
    # print(model.summary())
    # prediction = model.predict(Image)
    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # y_hat = class_names[np.argmax(prediction)]
    # surity = np.argmax(prediction)
    # return render_template('index.html',prediction=y_hat,surity = surity)



    # img = cv.imread(imagePath,0)
    # new_img = cv.resize(img, (28, 28))
    # new_img = np.array(new_img,dtype="float32")
    # new_img = 255 - new_img
    Image = load_img(imagePath,color_mode="grayscale",target_size=(28,28))
    Image = np.array(Image,dtype="float32")
    # Image = Image/255.0
    Image = 255 - Image
    print(Image)
    # cv.imwrite("./images/xyz.png",Image)
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="fmnist.tflite")
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], [Image])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    y_hat = class_names[np.argmax(output_data)]
    surity = np.argmax(output_data)
    im = load_img(imagePath)
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return render_template('index.html',prediction=y_hat,img_data=encoded_img_data.decode('utf-8'))

app.run()
