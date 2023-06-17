from flask import Flask ,request, render_template
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ["POST"])
def predict():
    if request.method == "POST":
        age = float(request.form['age'])
        education = float(request.form['education'])
        sex = float(request.form['sex'])
        smoking = float(request.form['smoking'])
        cigsPerDay = float(request.form['cigsPerDay'])
        BPMeds = float(request.form['BPMeds'])
        prevalentStroke = float(request.form['prevalentStroke'])
        prevalentHyp = float(request.form['prevalentHyp'])
        diabetes = float(request.form['diabetes'])
        totChol = float(request.form['totChol'])
        sysBP = float(request.form['sysBP'])
        diaBP = float(request.form['diaBP'])
        BMI = float(request.form['BMI'])
        heartRate = float(request.form['heartRate'])
        glucose = float(request.form['glucose'])
        prediction = model.predict([[age,education,sex,smoking,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose]])
        pred = prediction[0]
        out = 'error'
        if pred ==[1] : out = "Cardiovascular Risk"
        else : out = "No Cardiovascular Risk"

        return render_template('index.html',results = out)
    

if __name__ == "__main__":
    app.run(debug=True)