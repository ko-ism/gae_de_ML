from io import BytesIO
from google.cloud import storage

from flask import Flask, render_template, request, redirect, url_for
from wtforms import Form, validators, ValidationError, SelectField

import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.externals import joblib
import pickle
import numpy as np


PROJECT_ID = '<your project>'
CLOUD_STORAGE_BUCKET = '<your bucket>'

app = Flask(__name__)

def download_blob(source_blob_name):
    """Downloads a blob from the bucket."""
    #storage_client = storage.Client()
    storage_client= storage.Client(project=PROJECT_ID)
    bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
    blob = bucket.blob(source_blob_name)
    file = blob.download_as_string()
    # <class 'bytes'>
    return file


# [START upload_file]
def upload_file(file_stream, filename, content_type):
    storage_client= storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(CLOUD_STORAGE_BUCKET)
    blob = bucket.blob(filename)

    blob.upload_from_string(
        file_stream,
        content_type=content_type)

    url = 'gs://{}/{}'.format(CLOUD_STORAGE_BUCKET, filename)

    return url
# [END upload_file]


def LR(X_train, X_test, Y_train, Y_test):
    l_reg = LinearRegression()
    l_reg.fit(X_train, Y_train)
    model = pickle.dumps(l_reg)
    url = upload_file(model, '20181019_ml_model/LinearRegModel.pkl', 'application/octet-stream')

    return l_reg.score(X_train,Y_train), l_reg.score(X_test,Y_test), url


def RFR(X_train, X_test, Y_train, Y_test):
    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train, Y_train)
    model = pickle.dumps(rf_reg)
    url = upload_file(model, '20181019_ml_model/RandomForestModel.pkl', 'application/octet-stream')
    return rf_reg.score(X_train,Y_train), rf_reg.score(X_test,Y_test), url


def GBR(X_train, X_test, Y_train, Y_test):
    gb_reg = GradientBoostingRegressor()
    gb_reg.fit(X_train, Y_train)
    model = pickle.dumps(gb_reg)
    url = upload_file(model, '20181019_ml_model/GradientBoostingModel.pkl', 'application/octet-stream')
    return gb_reg.score(X_train,Y_train), gb_reg.score(X_test,Y_test), url


@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/multi_ml", methods=['POST'])
def multi_ml():
    try:
        lr_train_score=0
        lr_test_score=0
        rfr_train_score=0
        rfr_test_score=0
        gbr_train_score=0
        gbr_test_score=0
        
        body = request.get_data(as_text=True)
        source_filename = request.form['source_file']
        #print(source_file)
        #print(type(source_file))
        source_data = download_blob(source_filename)
        df = pd.read_csv(BytesIO(source_data))
        #print(df)
        #print(len(df.columns))
        X = df.drop(df.columns.values[len(df.columns)-1], axis=1)
        #print(X)
        Y = df.iloc[:, [len(df.columns)-1]]
        #print(Y)
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y)

        lr_train_score, lr_test_score , lr_model_url = LR(X_train, X_test, Y_train, Y_test)
        rfr_train_score, rfr_test_score, rfr_model_url = RFR(X_train, X_test, Y_train, Y_test)
        gbr_train_score, gbr_test_score, gbr_model_url = GBR(X_train, X_test, Y_train, Y_test)

        message = "success!"

        return render_template('multi_ml_kekka.html', lr_train_score=lr_train_score, lr_test_score=lr_test_score, lr_model_url=lr_model_url, rfr_train_score=rfr_train_score, rfr_test_score=rfr_test_score, rfr_model_url=rfr_model_url, gbr_train_score=gbr_train_score, gbr_test_score=gbr_test_score, gbr_model_url=gbr_model_url, message=message)

    except Exception as e:
        message = "false"+str(e)
        return render_template('multi_ml_kekka.html', message=message)

    return 'OK'


@app.route("/predict_index", methods=['GET'])
def predict_index():
    return render_template('predict_index.html')


@app.route("/predict", methods=['POST'])
def predict():
    try:
        predict_kekka = 0
        input_data = request.form['input']
        input_list = input_data.split(',')
        input_array = np.array([input_list]).astype(np.float64)
        print(input_array)
        model_uri = request.form['model']
        model_data = download_blob(model_uri)
        #model = pickle.loads(BytesIO(model_data))
        model = pickle.loads(model_data)
        print(type(model))

        predict_kekka = model.predict(input_array)
        message = "success!"
        
        return render_template('predict_kekka.html', predict_kekka=predict_kekka, message=message)
    except Exception as e:
        message = "false"+str(e)
        return render_template('predict_kekka.html', message=message)

    return 'OK'
