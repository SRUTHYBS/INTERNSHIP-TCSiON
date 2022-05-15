# -*- coding: utf-8 -*-
"""
Created on Mon May  9 20:31:15 2022

@author: User
"""

from flask import Flask,render_template,request

import pickle
import numpy as np
app=Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        easeofUse    = request.values['EaseofUse']
        satisfaction = request.values['Satisfaction']
        age          = request.form['Age']
        condition    = request.form['Condition']
        sex          = request.form['Sex']
    
        index_dict = pickle.load(open('cat','rb'))
        cat_vector = np.zeros(len(index_dict))
        
        cat_vector[index_dict['EaseofUse']] = easeofUse
        cat_vector[index_dict['Satisfaction']] = satisfaction
        
        try:
            cat_vector[index_dict['Age_'+str(age)]] = 1
        except:
            pass
        try:
            cat_vector[index_dict['Condition_'+str(condition)]] = 1
        except:
            pass
        try:
            cat_vector[index_dict['Sex_'+str(sex)]] = 1
        except:
            pass
    
    
    cat_vector=np.reshape(cat_vector,(1,-1))
    
    model=pickle.load(open('quality.pkl','rb'))
    result_prediction = model.predict(cat_vector)
    
    return render_template('result.html', prediction_text="RESULT: {}".format(result_prediction[0]))
if __name__=='__main__':
    app.run(port=8000)