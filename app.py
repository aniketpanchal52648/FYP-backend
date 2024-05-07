import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask,render_template,request,jsonify,make_response
import csv
import io
# import os
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
# import keras as kr
from tensorflow import keras
# from numpyencoder import NumpyEncoder
# from keras.models import Sequential
# from tensorflow.python.keras.models import LSTM
# from tensorflow.python.keras.models import Dense
# from tensorflow.python.keras.models import Bidirectional
# # from tensorflow.python.keras.models import Dropout
# from tensorflow.python.keras.models import model_from_json

# from tensorflow.python.keras.models import Dropout, Flatten

# from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
import joblib
pd.options.mode.chained_assignment = None
# scaler = scaler.fit(df_for_training)
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
def load_model():
    try:
    
        custom_objects = {'mse': keras.losses.MeanSquaredError()}
        model = keras.models.load_model('STLF2.h5',custom_objects=custom_objects)
        
        
    
        print(model.summary()) 
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return jsonify({'Error':e})

def load_model_MT():
    try:
    
        custom_objects = {'mse': keras.losses.MeanSquaredError()}
        model_MT = keras.models.load_model('medium_term_LF.h5',custom_objects=custom_objects)
        
        
    
        print(model_MT.summary()) 
        return model_MT
    except Exception as e:
        print(f"Error loading model: {e}")
        return jsonify({'Error':e})

# Function to preprocess input data (adapt based on your model's input format)
def add_features(df):
    df['T2M_toc_s']=df['T2M_toc'].shift(-1).fillna(0)
    df['QV2M_toc_s']=df['QV2M_toc'].shift(-1).fillna(0)
    df['TQL_toc_s']=df['TQL_toc'].shift(-1).fillna(0)
    df['W2M_toc_s']=df['W2M_toc'].shift(-1).fillna(0)
    df['T2M_toc_s']=df['T2M_san'].shift(-1).fillna(0)
    df['QV2M_san_s']=df['QV2M_san'].shift(-1).fillna(0)
    df['TQL_san_s']=df['TQL_san'].shift(-1).fillna(0)
    df['W2M_san_s']=df['W2M_san'].shift(-1).fillna(0)
    df['T2M_dav_s']=df['T2M_dav'].shift(-1).fillna(0)
    df['QV2M_dav_s']=df['QV2M_dav'].shift(-1).fillna(0)
    df['TQL_dav_s']=df['TQL_dav'].shift(-1).fillna(0)
    df['W2M_dav_s']=df['W2M_dav'].shift(-1).fillna(0)
    df['Holiday_ID_s']=df['Holiday_ID'].shift(-1).fillna(0)
    df['holiday_s']=df['holiday'].shift(-1).fillna(0)
    df['school_s']=df['school'].shift(-1).fillna(0)

    df['T2M_toc_s1']=df['T2M_toc'].shift(-2).fillna(0)
    df['QV2M_toc_s1']=df['QV2M_toc'].shift(-2).fillna(0)
    df['TQL_toc_s1']=df['TQL_toc'].shift(-2).fillna(0)
    df['W2M_toc_s1']=df['W2M_toc'].shift(-2).fillna(0)
    df['T2M_toc_s1']=df['T2M_san'].shift(-2).fillna(0)
    df['QV2M_san_s1']=df['QV2M_san'].shift(-2).fillna(0)
    df['TQL_san_s1']=df['TQL_san'].shift(-2).fillna(0)
    df['W2M_san_s1']=df['W2M_san'].shift(-2).fillna(0)
    df['T2M_dav_s1']=df['T2M_dav'].shift(-2).fillna(0)
    df['QV2M_dav_s1']=df['QV2M_dav'].shift(-2).fillna(0)
    df['TQL_dav_s1']=df['TQL_dav'].shift(-2).fillna(0)
    df['W2M_dav_s1']=df['W2M_dav'].shift(-2).fillna(0)

    df['nat_demand3']=df['nat_demand'].shift(3).fillna(0)
    df['nat_demand4']=df['nat_demand'].shift(4).fillna(0)
    df['nat_demand5']=df['nat_demand'].shift(5).fillna(0)
    df['nat_demand6']=df['nat_demand'].shift(6).fillna(0)
    df['nat_demand7']=df['nat_demand'].shift(7).fillna(0)
    df['nat_demand8']=df['nat_demand'].shift(8).fillna(0)
    df['nat_demand9']=df['nat_demand'].shift(9).fillna(0)
    df['nat_demand10']=df['nat_demand'].shift(10).fillna(0)
    df['nat_demand11']=df['nat_demand'].shift(11).fillna(0)
    df['nat_demand12']=df['nat_demand'].shift(12).fillna(0)
    df['nat_demand13']=df['nat_demand'].shift(13).fillna(0)
    df['nat_demand14']=df['nat_demand'].shift(14).fillna(0)
    df['nat_demand_n']=df['nat_demand']
    #df = pd.get_dummies(df)
    return df

def predicted_short_term(data,model):
    scaler = joblib.load('scaler.joblib')
    predicted_y=[]
    dates=[]
    previousData=[]
    time=[]
    for i in range(24):
        df_test1=data.iloc[i:i+48]
        df_test1['datetime']=pd.to_datetime(df_test1['datetime'],format='%d-%m-%Y %H:%M')

        df_test1['week_day']=df_test1['datetime'].dt.dayofweek
        df_test1['date']=df_test1['datetime'].dt.day
        df_test1['month']=df_test1['datetime'].dt.month
        df_test1['hour']=df_test1['datetime'].dt.hour
        # df.head()
        date=data.iloc[i+48,0]
        date=pd.to_datetime(date,format='%d-%m-%Y %H:%M')
        # time.append(date.dt.time)


        previousData.append(data.iloc[i+48,1])
        df_test2 = add_features(df_test1)

        # df_test1 = add_features(df_test1)
        # print(df_test1.head())
        col=['datetime']
        new_df_test2= df_test2.drop(columns=col)
        df_for_testing1 = new_df_test2.astype(float)
        # print(df_for_testing1)
        # df_for_pred1=df_test1.drop(columns=['datetime'])
        # df_test1=df_for_pred1.astype(float)
        df_pred_scaled1=scaler.transform(df_for_testing1)

        # print(df_pred_scaled1)
        # print(df_test.sha)
        X_pred1 = []
        

        X_pred1.append(df_pred_scaled1[0:48, 0:df_for_testing1.shape[1]])


        X_pred1= np.array(X_pred1)
        
        prediction1 = model.predict(X_pred1)
        
        prediction1_copies = np.repeat(prediction1, df_for_testing1.shape[1], axis=-1)
       
        y_pred_future1 = scaler.inverse_transform(prediction1_copies)[:,0]
        # print(y_pred_future1)
        predicted_y.append(y_pred_future1)
        dates.append(date)
        data.loc[i+48,'nat_demand']=y_pred_future1

  
    return predicted_y,dates



def predicted_medium_term(data,model):
    scaler=joblib.load('scaler_MT.joblib')
    # data= pd.read_csv('test_load_forecast.csv')
    predicted_y=[]
    dates=[]
    previousData=[]
    time=[]
    for i in range(24):
        df_test1=data.iloc[i:i+48]
        df_test1['datetime']=pd.to_datetime(df_test1['datetime'],format='%d-%m-%Y')

        df_test1['week_day']=df_test1['datetime'].dt.dayofweek
        df_test1['date']=df_test1['datetime'].dt.day
        df_test1['month']=df_test1['datetime'].dt.month
        # df_test1['hour']=df_test1['datetime'].dt.hour
        # df.head()
        date=data.iloc[i+48,0]
        date=pd.to_datetime(date,format='%d-%m-%Y')
        # time.append(date.dt.time)


        previousData.append(data.iloc[i+48,1])
        df_test2 = (df_test1)

        # df_test1 = add_features(df_test1)
        # print(df_test1.head())
        col=['datetime']
        new_df_test2= df_test2.drop(columns=col)
        df_for_testing1 = new_df_test2.astype(float)
        # print(df_for_testing1)
        # df_for_pred1=df_test1.drop(columns=['datetime'])
        # df_test1=df_for_pred1.astype(float)
        df_pred_scaled1=scaler.transform(df_for_testing1)

        # print(df_pred_scaled1)
        # print(df_test.sha)
        X_pred1 = []
        # for i in range(n_past, len(df_pred_scaled1) - n_future +1):

        X_pred1.append(df_pred_scaled1[0:48, 0:df_for_testing1.shape[1]])


        X_pred1= np.array(X_pred1)
        # print(X_pred1)

        print('X for prediction shape == {}.'.format(X_pred1.shape))
        prediction1 = model.predict(X_pred1)
        print(prediction1.shape)
        prediction1_copies = np.repeat(prediction1, df_for_testing1.shape[1], axis=-1)
        # print(prediction1_copies)
        y_pred_future1 = scaler.inverse_transform(prediction1_copies)[:,0]
        # print(y_pred_future1)
        predicted_y.append(y_pred_future1)
        dates.append(date)
        data.loc[i+48,'nat_demand']=y_pred_future1



    # print(predicted_y)
    return predicted_y,dates





app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')
    # return 'Hello, World!'

@app.route('/route1')
def getRoute():
    model =load_model()
    return "this is route 1"


@app.route('/get_short_term', methods=['POST'])
def get_short_term():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    # Check if the file has a name and is a valid CSV file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Uploaded file is not a CSV'}), 400
    try:
        # Read the CSV file into a list of dictionaries
        csv_data = []
        csv_file = io.StringIO(file.stream.read().decode("utf-8"))
        csv_reader = csv.DictReader(csv_file)
        data= pd.read_csv(csv_file,encoding="latin-1")
        # print(df_test)
        model =load_model()

        y_predicted,dates=predicted_short_term(data,model)

        
        # print((dates))
        
        pred = pd.Series(y_predicted).to_json(orient='values')
    
        
        


    

        response= jsonify({'message': 'Data Fetched Succefully', "dates":dates,
            "predicted_demand":pred
            })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print(e)
        return make_response(str(e)+'sent proper CSV data', 404) 



@app.route('/get_medium_term', methods=['POST'])
def get_medium_term():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    # Check if the file has a name and is a valid CSV file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Uploaded file is not a CSV'}), 400
    try:
        # Read the CSV file into a list of dictionaries
        
        csv_file = io.StringIO(file.stream.read().decode("utf-8"))
        # csv_reader = csv.DictReader(csv_file)
        data= pd.read_csv(csv_file,encoding="latin-1")
        # print(df_test)
        model =load_model_MT()

        y_predicted,dates=predicted_medium_term(data,model)

        
        # print((dates))
        
        pred = pd.Series(y_predicted).to_json(orient='values')
    
        
        


    

        response= jsonify({'message': 'Data Fetched Succefully', "dates":dates,
            "predicted_demand":pred
            })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print(e)
        return make_response(str(e)+'sent proper CSV data', 404) 




if __name__=="__main__":
    app.run(debug=True)    