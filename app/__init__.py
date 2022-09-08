from fileinput import filename
from genericpath import exists
from flask import Flask
from pathlib import Path
import base64

app=Flask(__name__,static_folder='static')
current_dir = Path(__file__)
APP_ROOT = [p for p in current_dir.parents if p.parts[-1]=='app'][0]


import os

from flask import render_template, request
from werkzeug.utils import redirect

from app.process import predict_img

@app.route('/')
def home():
    print("got here")
    return render_template('index.html',title='Home')

@app.route('/about')
def about():
    return render_template('about.html',title='About',name='Passed by variable')

@app.route("/predict")
def predict():
    return render_template("predict.html",title="Predict")

@app.route("/take")
def take():
    return render_template("take.html",title="Take")

@app.route("/upload",methods=["GET","POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/')
    if request.method == 'POST':
        if request.form:
            data = request.form['file'].replace('data:image/png;base64,', '')
            imgdata = base64.b64decode(data)
            with open("".join([target, 'img.jpg']),'wb') as f:
                f.write(imgdata)
                return redirect('/prediction/{}'.format('img.jpg'))
        file = request.files['img'] # 'img' is the id passed in input file form field
        filename = file.filename
        # filename = filename(filename)
        file.save("".join([target, filename])) #saving file in temp folder
        print("upload Completed") #printing on terminal
        return redirect('/prediction/{}'.format(filename))

@app.route("/prediction/<filename>",methods=["GET","POST"])
def prediction(filename):
    x=predict_img(filename) #imported from process file
    return render_template('output.html',results=x,filename=filename)


@app.route("/app")
def webapp():
    return render_template('app.html',title='Home')