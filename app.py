import pickle
import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


scaler_model = pickle.load(open("/Users/mdshalique/Downloads/practicePythonOP/Algerian_Forest_Fire_Prediction/model/scaler.pkl","rb"))
ridge_model  = pickle.load(open("/Users/mdshalique/Downloads/practicePythonOP/Algerian_Forest_Fire_Prediction/model/ridge.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods = ['POST' , 'GET'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        region = float(request.form.get("region"))
        
        
        new_data_scaled  = scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,region]])

        result = ridge_model.predict(new_data_scaled)

        return (render_template("home.html", result = result[0]))
    
    else:
        return render_template("home.html")
    
if __name__ == '__main__':
    app.run(host = "0.0.0.0")
        