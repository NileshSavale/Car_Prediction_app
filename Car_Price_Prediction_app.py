# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:40:54 2022

@author: Nilesh
"""

from flask import Flask,render_template,request,redirect
#from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np


app=Flask(__name__)
#cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv('Cleaned_Car_Data.csv')

@app.route("/")
def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique())
    kms_driven=sorted(car['kms_driven'].unique(),reverse=True)
    fuel_type=sorted(car['fuel_type'].unique())

    companies.insert(0,'Select Company')
    return render_template("indexhtml.html",companies=companies,car_models=car_models,year=year,kms_driven=kms_driven,fuel_type=fuel_type)

@app.route("/Predict",methods=['POST'])
def Predict():
    company=request.form.get('companies')
    car_model=request.form.get('car_model')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    driven=request.form.get('kilo_driven')

    prediction=model.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)
    return str(np.round(prediction[0],2))

if __name__=="__main__":
    app.run(debug=True)
