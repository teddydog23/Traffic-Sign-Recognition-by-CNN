import tkinter as tk
from tkinter import filedialog
import numpy as np
from tkinter import*
from PIL import ImageTk, Image
import pickle
import matplotlib
import matplotlib.pyplot as plt
import random
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load pre-trained model
from tensorflow.keras.models import load_model
model = load_model('traffic_sign_classification.h5')

# Load images from test set
with open("./test.p", mode="rb") as f:
    test = pickle.load(f)
testX = test["features"]
testY = test["labels"]
testX = testX.astype("float") / 255.0


# Define traffic signs class
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }


# Init main
window = tk.Tk()
window.geometry('800x600')
window.title('Traffic sign classification')


matplotlib.use("TkAgg")
figure = Figure(figsize=(7, 5), dpi = 100)


# Functions
def display_image(path_file, sign):
    figure.clear()

    im = Image.open(path_file)
    image = figure.add_subplot(1, 1, 1)
    image.imshow(im)
    image.axis("off")
    image.set_title("Predict result: " + sign)

    canvas.draw_idle()


def random_images():
    figure.clear()
    for i in range(9):
        #prediction
        rand = random.randint(1, 10000)
        result = model.predict([testX[rand : rand+1]])
        result = np.argmax(result)

        color = "red"
        if result == testY[rand]:
           color = "green"


        image = figure.add_subplot(3, 3, i+1)
        image.imshow(testX[rand])
        image.axis("off")
        image.set_title(classes[result+1], color = color)
    canvas.draw_idle()


def predict_an_image(path_file):
    #preprocesing images
    img = Image.open(path_file)
    img = img.resize((32, 32))
    img = np.expand_dims(img, axis = 0)
    img = np.array(img)
    img = img / 255.0

    #prediction
    result = model.predict([img])
    result = np.argmax(result)
    sign = classes[result+1]
    
    display_image(path_file, sign)




def upload_image():
	try:
		path_file =filedialog.askopenfilename()
		uploaded = Image.open(path_file)
		uploaded.thumbnail(((window.winfo_width()/2.25),(window.winfo_height()/2.25)))
		im = ImageTk.PhotoImage(uploaded)
		predict_an_image(path_file)

	except:
		pass
    


canvas = FigureCanvasTkAgg(figure, window)
canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    

# Button "Upload an image" and "Upload an image"
upload = Button(window, text = "Upload an image", command = upload_image)
upload.pack()

random_images = Button(window, text = "Random images", command = random_images)
random_images.pack()


window.mainloop()
