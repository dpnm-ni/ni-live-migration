import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from keras import backend as K
from keras.models import Model,Sequential,load_model 
from keras.layers import Input, Dense, concatenate, Flatten, Reshape, Lambda, Embedding
from tensorflow.keras.optimizers import Adam
from tensorboard import main as tb
#from tensorflow.keras.models import load_model

LABEL_DATA_COL_INDEX = [-3, -2, -1]

# Mean Absolute Percentage Error
# https://brunch.co.kr/@chris-song/34
def MAPE(y_test, y_hat):
    return np.mean(np.abs((y_test - y_hat) / y_test)) * 100

def RMSE(y_test, y_hat):
    return K.sqrt(K.mean(K.square(y_test - y_hat),axis=-1))


def ml(trained=False):
    # refer to sample_dataset_columns.txt for column numbers.
    df = pd.read_csv("dataset.csv", header=None)

    X = df.iloc[:, :-len(LABEL_DATA_COL_INDEX)].values
    y = df.iloc[:, -1:].values
    
    X_size = X.shape[-1]
    
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    y = y.ravel()

    X_train, X_val1, y_train, y_val1 = train_test_split(X, y, test_size=50, random_state=42)
    X_train, X_val2, y_train, y_val2 = train_test_split(X_train, y_train, test_size=50, random_state=42)
    X_train, X_val3, y_train, y_val3 = train_test_split(X_train, y_train, test_size=50, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=150, random_state=42)

            
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    X_size = X_train_scaled.shape[-1]
    Y_size = y_train.shape[-1]
    '''

    X_in = Input(shape=(X_size))

    layer1 = Dense(300)(X_in)
    layer2 = Dense(300)(layer1)
    layer3 = Dense(30)(layer2)
    output = Dense(1)(layer3)
    l_size = 0.0001
    e_size = 1000
    b_size = 10000

    model = Model(inputs=[X_in],outputs=output)
    optimizer = Adam(lr=l_size)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.MeanAbsoluteError(),metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

    if trained == False:
        model.fit([X_train],y_train,batch_size=b_size,validation_data=([X_test],y_test),epochs=e_size)
        model.save('preidiction_model.h5')
        
    else :
        model = load_model('preidiction_model.h5')
        

    # 각 validation set에 대한 정확도를 계산하고 출력
    y_pred_val1 = model.predict([X_val1]).ravel()


    val1_results = pd.DataFrame({'Feature Data': [x.tolist() for x in X_val1], 'Actual Data': y_val1, 'Predicted Data': y_pred_val1})
    val1_results.to_csv('validation_set1_results.csv', index=False)
    
    val1_accuracy = 100 - MAPE(y_val1, y_pred_val1)
    print("val1_accuracy : {}".format(val1_accuracy))

    y_pred_val2 = model.predict([X_val2]).ravel()
    val2_results = pd.DataFrame({'Feature Data': [x.tolist() for x in X_val2], 'Actual Data': y_val2, 'Predicted Data': y_pred_val2})
    val2_results.to_csv('validation_set2_results.csv', index=False)
    val2_accuracy = 100 - MAPE(y_val2, y_pred_val2)    
    
    print("val2_accuracy : {}".format(val2_accuracy))

    y_pred_val3 = model.predict([X_val3]).ravel()
    val3_results = pd.DataFrame({'Feature Data': [x.tolist() for x in X_val3], 'Actual Data': y_val3, 'Predicted Data': y_pred_val3})
    val3_results.to_csv('validation_set3_results.csv', index=False)
    val3_accuracy = 100 - MAPE(y_val3, y_pred_val3)
    
    print("val3_accuracy : {}".format(val3_accuracy))

    return (val1_accuracy + val2_accuracy + val3_accuracy)/3
    


