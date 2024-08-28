from flask import Flask, render_template
import pandas as pd
from flask_cors import CORS,cross_origin
import pickle
import numpy as np

app = Flask(__name__)
data=pd.read_csv("Crop_recommendation (1).csv")

cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv('Crop_recommendation (1).csv')


@app.route('/')
def index():
    Nitrogens=sorted(data['N'].unique())
    Potassium = sorted(data['P'].unique())
    Kalcium = sorted(data['K'].unique())
    Temperature = sorted(data['temperature'].unique())
    Humidity = sorted(data['humidity'].unique())
    PH = sorted(data['ph'].unique(), reverse=True)
    Rainfall = sorted(data['rainfall'].unique())
    return render_template('index.html', ph=PH)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    Nitrogen=request.form.get('N')
    

    Potassium=request.form.get('P')
    Kalcium=request.form.get('K')
    Temperature=request.form.get('temperature')
    Humidity=request.form.get('humidity')
    PH = request.form.get('ph')
    Rainfall = request.form.get('rainfall')

    prediction=model.predict(pd.DataFrame(columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
                              data=np.array([N,P,K,temperature,humidity,ph,rainfall]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__ == '__main__':
    app.run(debug=True)

