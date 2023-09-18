import platform
import pathlib

if platform.system() == "Windows":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

from fastai.vision.all import *
import gradio as gr

learn = load_learner(Path("C:\KMIDS 1104\Python\Chicken_or_Bear\export.pkl"))

def classify_image(img):
    pred, idx, probs = learn.predict(img)

    return f"This is {pred}.\nConfidence Level: {float(max(probs))}"


image = gr.Image(shape=(192, 192))
examples = ['C:\KMIDS 1104\Python\Chicken_or_Bear\images\grizzlybear.jpg', 'C:\KMIDS 1104\Python\Chicken_or_Bear\images\chicken.jpg']

app = gr.Interface(fn=classify_image,
                   inputs=image,
                   outputs=gr.Textbox(label="Output"),
                   examples=examples)
app.launch()