import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("digit_model.h5")

window = tk.Tk()
window.title("Digit Recognizer")

canvas = tk.Canvas(window, width=280, height=280, bg='white')
canvas.pack()

image = Image.new("L", (280, 280), 'white')
draw = ImageDraw.Draw(image)

def draw_lines(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
    draw.ellipse([x-r, y-r, x+r, y+r], fill='black')

canvas.bind("<B1-Motion>", draw_lines)

def predict():
    img = image.resize((28, 28))
    img = np.array(img)
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1, 28, 28)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    result_label.config(text=f"Prediction: {digit}")

def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, 280, 280], fill='white')
    result_label.config(text="")

tk.Button(window, text="Predict", command=predict).pack()
tk.Button(window, text="Clear", command=clear).pack()

result_label = tk.Label(window, text="", font=("Arial", 20))
result_label.pack()

window.mainloop()