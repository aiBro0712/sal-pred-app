#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 20:20:40 2020

@author: snk
"""
from flask import Flask, jsonify,request
import numpy as np
import pickle as cPick

my_sal_predict_model = cPick.load(open('sal_pred.pkl','rb'))
print(my_sal_predict_model)


app=Flask('__name__')

@app.route('/api', methods=["POST"])
def sal_predict():
    data = request.get_json(force=True)
    predict_request = [data["exp"]]
    predict_request = np.array(predict_request)
    y_hat = my_sal_predict_model.predict([predict_request])
    output= [y_hat[0]]
    
    return jsonify(results= output)


if __name__ == '__main__':
    app.run()