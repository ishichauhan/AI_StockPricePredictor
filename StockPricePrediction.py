#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and loading Functions

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (15,6)

import json
import csv
import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras import regularizers
from keras.layers.core import Dense, Activation
import io
import collections
from sklearn import preprocessing
import matplotlib.pyplot as plt
import shutil
import os
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
import requests
import tensorflow as tf
from scipy import sparse
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.wrappers import TimeDistributed


# In[2]:


# Regression chart.
def chart_regression(pred,y,sort=True):
    t = pd.DataFrame({'pred' : pred, 'y' : y.flatten()})
    if sort:
        t.sort_values(by=['y'],inplace=True)
    a = plt.plot(t['y'].tolist(),label='expected')
    b = plt.plot(t['pred'].tolist(),label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()


# In[3]:


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
import collections
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column. 
    target_type = df[target].dtypes
    target_type = target_type[0] if isinstance(target_type, collections.Sequence) else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    else:
        # Regression
        return df[result].values.astype(np.float32), df[target].values.astype(np.float32)


# In[4]:


#Function to normalize columns
def normalize_numeric_minmax(df, name):
        df[name] = ((df[name] - df[name].min()) / (df[name].max() - df[name].min())).astype(np.float32)
    


# # Data Pre Processing

# In[5]:


#Reading the file and loading dataset into stock_df dataframe 
stock_df= pd.read_csv('CSC215_Project4_Stock_Price.csv')
stock_df[0:5]


# In[6]:


stock_df.columns.isnull().sum()


# In[7]:


stock_df = stock_df.drop(['Date', 'Adj_Close'], axis = 1)
stock_df[0:5]


# In[8]:


normalize_numeric_minmax(stock_df,"Open")
normalize_numeric_minmax(stock_df,"High") 
normalize_numeric_minmax(stock_df,"Low") 
normalize_numeric_minmax(stock_df,"Volume") 
stock_df[0:5]


# In[32]:


# to xy to convert pandas to tensor flow
x,y=to_xy(stock_df,"Close")


# In[33]:


print(x.shape)
print(y.shape)


# In[34]:


x


# In[35]:


y


# # Task 1  -  Fully Connected/Dense Neural Network

# In[36]:


#Split for train and test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)


# In[37]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ##  ReLu - Activation Function

# ###  Adam - Optimizer

# #### 4 Hidden Layers  

# In[38]:


# set up Model CHeck Point
checkpoint_relu = ModelCheckpoint(filepath="./best_weights_relu_adam1.hdf5", verbose=1, save_best_only=True)


# In[39]:


for i in range(10):
    print(i)
    
    # Build network
    model_relu_adam = Sequential()

    model_relu_adam.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))  
    model_relu_adam.add(Dense(32, activation='relu')) # Hidden 2
    model_relu_adam.add(Dense(16, activation='relu')) # Hidden 3
    model_relu_adam.add(Dense(8, activation='relu')) # Hidden 4
    model_relu_adam.add(Dense(1)) # Output
    
    model_relu_adam.compile(loss='mean_squared_error', optimizer='adam')
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    
    model_relu_adam.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpoint_relu],verbose=2,epochs=100) 
    
print('Training Completed..') 
print('Loading the best model')
print()
model_relu_adam.load_weights('./best_weights_relu_adam1.hdf5')


# In[40]:


pred_relu_adam = model_relu_adam.predict(x_test)


# In[41]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(y_test[i],pred_relu_adam[i]))


# In[42]:


# Measure RMSE error.  RMSE is common for regression.
score_relu_adam = np.sqrt(mean_squared_error(y_test,pred_relu_adam))
print("Final score (RMSE): {}".format(score_relu_adam))
print('R2 score: %.2f' % r2_score(y_test, pred_relu_adam))


# In[43]:


#Regression lift Chart
chart_regression(pred_relu_adam.flatten(),y_test)


# ####    5 Hidden Layers  +  2  Dropout Layers 

# In[44]:


# set up Model CHeck Point
checkpoint_relu = ModelCheckpoint(filepath="./best_weights_relu_adam2.hdf5", verbose=1, save_best_only=True)


# In[45]:


for i in range(10):
    print(i)
    
    # Build network
    model_relu_adam = Sequential()

    model_relu_adam.add(Dense(128, input_dim=x_train.shape[1], activation='relu')) 
    model_relu_adam.add(Dropout(0.2))
    model_relu_adam.add(Dense(64, activation='relu')) # Hidden 2
    model_relu_adam.add(Dropout(0.3))
    model_relu_adam.add(Dense(32, activation='relu')) # Hidden 3
    model_relu_adam.add(Dense(16, activation='relu')) # Hidden 4
    
    model_relu_adam.add(Dense(8, activation='relu')) # Hidden 5
    model_relu_adam.add(Dense(1)) # Output
    
    model_relu_adam.compile(loss='mean_squared_error', optimizer='adam')
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    
    model_relu_adam.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpoint_relu],verbose=2,epochs=100) 
    
print('Training Completed..') 
print('Loading the best model')
print()
model_relu_adam.load_weights('./best_weights_relu_adam2.hdf5')


# In[46]:


pred_relu_adam = model_relu_adam.predict(x_test)


# In[47]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(y_test[i],pred_relu_adam[i]))


# In[48]:


# Measure RMSE error.  RMSE is common for regression.
score_relu_adam = np.sqrt(mean_squared_error(y_test,pred_relu_adam))
print("Final score (RMSE): {}".format(score_relu_adam))
print('R2 score: %.2f' % r2_score(y_test, pred_relu_adam))


# In[49]:


#Regression lift Chart
chart_regression(pred_relu_adam.flatten(),y_test)


# ####   6 Hidden Layers  +  3 Dropout Layers 

# In[43]:


# set up Model CHeck Point
checkpoint_relu = ModelCheckpoint(filepath="./best_weights_relu_adam3.hdf5", verbose=1, save_best_only=True)


# In[44]:


for i in range(10):
    print(i)
    
    # Build network
    model_relu_adam = Sequential()

    model_relu_adam.add(Dense(200, input_dim=x_train.shape[1], activation='relu')) 
    model_relu_adam.add(Dropout(0.5))
    model_relu_adam.add(Dense(100, activation='relu')) # Hidden 2
    model_relu_adam.add(Dropout(0.3))
    model_relu_adam.add(Dense(80, activation='relu')) # Hidden 3
    model_relu_adam.add(Dropout(0.4))
    model_relu_adam.add(Dense(50, activation='relu')) # Hidden 4
    model_relu_adam.add(Dense(30, activation='relu')) # Hidden 5
    model_relu_adam.add(Dense(15, activation='relu')) # Hidden 6
    model_relu_adam.add(Dense(1)) # Output
    
    model_relu_adam.compile(loss='mean_squared_error', optimizer='adam')
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    
    model_relu_adam.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpoint_relu],verbose=2,epochs=100) 
    
print('Training Completed..') 
print('Loading the best model')
print()
model_relu_adam.load_weights('./best_weights_relu_adam3.hdf5')


# In[49]:


pred_relu_adam = model_relu_adam.predict(x_test)


# In[50]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(y_test[i],pred_relu_adam[i]))


# In[51]:


# Measure RMSE error.  RMSE is common for regression.
score_relu_adam = np.sqrt(mean_squared_error(y_test,pred_relu_adam))
print("Final score (RMSE): {}".format(score_relu_adam))
print('R2 score: %.2f' % r2_score(y_test, pred_relu_adam))


# In[48]:


# Measure RMSE error.  RMSE is common for regression.
score_relu_adam = np.sqrt(mean_squared_error(y_test,pred_relu_adam))
print("Final score (RMSE): {}".format(score_relu_adam))
print('R2 score: %.2f' % r2_score(y_test, pred_relu_adam))


# In[92]:


#Regression lift Chart
chart_regression(pred_relu_adam.flatten(),y_test)


# ### Experimenting with Splitting Data Year wise (Sequential Order)

# #### Data Pre Processing

# In[5]:


#Reading the file and loading dataset into stock_df dataframe 
stock_df1= pd.read_csv('CSC215_Project4_Stock_Price.csv')
stock_df1[0:5]


# In[6]:


stock_df1_raw = stock_df1.copy()


# In[7]:


stock_df1_raw.head(2)


# In[8]:


stock_df1 = stock_df1.drop(['Date', 'Adj_Close'], axis = 1)
stock_df1[0:5]


# In[9]:


normalize_numeric_minmax(stock_df1,"Open")
normalize_numeric_minmax(stock_df1,"High") 
normalize_numeric_minmax(stock_df1,"Low") 
normalize_numeric_minmax(stock_df1,"Volume") 
stock_df1[0:5]


# In[10]:


# to xy to convert pandas to tensor flow
x,y=to_xy(stock_df1,"Close")


# In[11]:


print(x.shape)
print(y.shape)


# In[12]:


x


# In[13]:


y


# In[14]:


x_train1 = x[:3000]
y_train1 = y[:3000]
x_test1 = x[3000:]
y_test1 = y[3000:]


# In[15]:


print(x_train1.shape)
print(y_train1.shape)
print(x_test1.shape)
print(y_test1.shape)


# ####  Model Train Test

# In[17]:


# set up Model CHeck Point
checkpoint_relu = ModelCheckpoint(filepath="./best_weights_relu_adam_year.hdf5", verbose=1, save_best_only=True)


# In[18]:


for i in range(10):
    print(i)
    
    # Build network
    model_relu_adam = Sequential()

    model_relu_adam.add(Dense(128, input_dim=x_train1.shape[1], activation='relu')) 
    model_relu_adam.add(Dropout(0.2))
    model_relu_adam.add(Dense(64, activation='relu')) # Hidden 2
    model_relu_adam.add(Dropout(0.3))
    model_relu_adam.add(Dense(32, activation='relu')) # Hidden 3
    model_relu_adam.add(Dense(16, activation='relu')) # Hidden 4
    
    model_relu_adam.add(Dense(8, activation='relu')) # Hidden 5
    model_relu_adam.add(Dense(1)) # Output
    
    model_relu_adam.compile(loss='mean_squared_error', optimizer='adam')
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    
    model_relu_adam.fit(x_train1,y_train1,validation_data=(x_test1,y_test1),callbacks=[monitor,checkpoint_relu],verbose=2,epochs=100) 
    
print('Training Completed..') 
print('Loading the best model')
print()
model_relu_adam.load_weights('./best_weights_relu_adam_year.hdf5')


# In[19]:


pred_relu_adam1 = model_relu_adam.predict(x_test1)


# In[63]:


for i in range(10):
    print("Date: {}, Actual Value : {} , predicted Value :  {}".format(stock_df1_raw['Date'][3000+i],
                                                                      y_test1[i],pred_relu_adam1[i]))


# In[58]:


for i in range(10):
    print("Date: {}, Original:  {} Actual Value:{} , predicted Value:{}".format(stock_df1_raw['Date'][3000+1],
                                                                      stock_df1_raw['Close'][3000+i],y_test1[i],pred_relu_adam1[i]))


# In[64]:


# Measure RMSE error.  RMSE is common for regression.
score_relu_adam = np.sqrt(mean_squared_error(y_test1,pred_relu_adam1))
print("Final score (RMSE): {}".format(score_relu_adam))
print('R2 score: %.2f' % r2_score(y_test1, pred_relu_adam1))


# In[91]:


# Test Train Split
# Measure RMSE error.  RMSE is common for regression.
score_relu_adam = np.sqrt(mean_squared_error(y_test,pred_relu_adam))
print("Final score (RMSE): {}".format(score_relu_adam))
print('R2 score: %.2f' % r2_score(y_test, pred_relu_adam))


# In[27]:


#Regression lift Chart
chart_regression(pred_relu_adam1.flatten(),y_test1)


# In[49]:


#Regression lift Chart
chart_regression(pred_relu_adam.flatten(),y_test)


# ### Experimenting with the output Feature 

# #### Data Processing

# In[68]:


y.shape


# In[69]:


out_scaled = y.copy()


# In[71]:


out_scaled.shape


# In[75]:


df = stock_df1_raw.copy()


# In[76]:


normalize_numeric_minmax(df,"Close")


# In[77]:


df = df.drop(['Date', 'Adj_Close'], axis = 1)
df[0:5]


# In[79]:


normalize_numeric_minmax(df,"Open")
normalize_numeric_minmax(df,"High") 
normalize_numeric_minmax(df,"Low") 
normalize_numeric_minmax(df,"Volume") 
df[0:5]


# In[80]:


# to xy to convert pandas to tensor flow
x1,y1=to_xy(df,"Close")


# In[81]:


print(x1.shape)
print(y1.shape)


# In[82]:


#Split for train and test
x_train2, x_test2, y_train2, y_test2 = train_test_split(x1,y1, test_size=0.3, random_state=42)


# In[83]:


print(x_train2.shape)
print(x_test2.shape)
print(y_train2.shape)
print(y_test2.shape)


# In[84]:


y_test2


# #### Model Train Test

# In[86]:


# set up Model Check Point
checkpoint_relu3 = ModelCheckpoint(filepath="./best_weights_relu_adam_scale.hdf5", verbose=1, save_best_only=True)


# In[87]:


for i in range(10):
    print(i)
    
    # Build network
    model_relu_adam = Sequential()

    model_relu_adam.add(Dense(128, input_dim=x_train2.shape[1], activation='relu')) 
    model_relu_adam.add(Dropout(0.2))
    model_relu_adam.add(Dense(64, activation='relu')) # Hidden 2
    model_relu_adam.add(Dropout(0.3))
    model_relu_adam.add(Dense(32, activation='relu')) # Hidden 3
    model_relu_adam.add(Dense(16, activation='relu')) # Hidden 4
    
    model_relu_adam.add(Dense(8, activation='relu')) # Hidden 5
    model_relu_adam.add(Dense(1)) # Output
    
    model_relu_adam.compile(loss='mean_squared_error', optimizer='adam')
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    
    model_relu_adam.fit(x_train2,y_train2,validation_data=(x_test2,y_test2),callbacks=[monitor,checkpoint_relu3],verbose=2,epochs=100) 
    
print('Training Completed..') 
print('Loading the best model')
print()
model_relu_adam.load_weights('./best_weights_relu_adam_scale.hdf5')


# In[88]:


pred_relu_adam2 = model_relu_adam.predict(x_test2)


# In[89]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(y_test2[i],pred_relu_adam2[i]))


# In[90]:


# Measure RMSE error.  RMSE is common for regression.
score_relu_adam = np.sqrt(mean_squared_error(y_test2,pred_relu_adam2))
print("Final score (RMSE): {}".format(score_relu_adam))
print('R2 score: %.2f' % r2_score(y_test2, pred_relu_adam2))


# In[48]:


# Not Scaled
# Measure RMSE error.  RMSE is common for regression.
score_relu_adam = np.sqrt(mean_squared_error(y_test,pred_relu_adam))
print("Final score (RMSE): {}".format(score_relu_adam))
print('R2 score: %.2f' % r2_score(y_test, pred_relu_adam))


# In[49]:


#Regression lift Chart
chart_regression(pred_relu_adam.flatten(),y_test)


# ### SGD - Optimizer 

# #### 4 Hidden Layers  

# In[53]:


# set up Model CHeck Point
checkpoint_sgd = ModelCheckpoint(filepath="./best_weights_relu_sgd1.hdf5", verbose=1, save_best_only=True)


# In[54]:


for i in range(10):
    print(i)
    
    # Build network
    model_relu_sgd = Sequential()

    model_relu_sgd.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))
    model_relu_sgd.add(Dense(64, activation='relu')) # Hidden 2
    model_relu_sgd.add(Dense(32, activation='relu')) # Hidden 3
    model_relu_sgd.add(Dense(16, activation='relu')) # Hidden 4
    model_relu_sgd.add(Dense(1)) # Output
    
    model_relu_sgd.compile(loss='mean_squared_error', optimizer='sgd')
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    
    model_relu_sgd.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpoint_sgd],verbose=2,epochs=100) 
    
print('Training Completed..') 
print('Loading the best model')
print()
model_relu_sgd.load_weights('./best_weights_relu_sgd1.hdf5')


# In[55]:


pred_relu_sgd = model_relu_sgd.predict(x_test)


# In[56]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(y_test[i],pred_relu_sgd[i]))


# In[57]:


# Measure RMSE error.  RMSE is common for regression.
score_relu_sgd = np.sqrt(mean_squared_error(y_test,pred_relu_sgd))
print("Final score (RMSE): {}".format(score_relu_sgd))
print('R2 score: %.2f' % r2_score(y_test, pred_relu_sgd))


# In[58]:


#Regression lift Chart
chart_regression(pred_relu_sgd.flatten(),y_test)


# In[159]:


plt.plot(pred_relu_sgd[:50], color = 'red', label = 'Stock Price')
plt.plot(y_test[:50], color = 'green', label = 'Predicted Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# ## Tanh - Activation Function

# ### Adam - Optimizer

# #### 4 Hidden Layers  

# In[104]:


# set up Model CHeck Point
checkpoint_tanh_adam = ModelCheckpoint(filepath="./best_weights_tanh_adam1.hdf5", verbose=1, save_best_only=True)


# In[105]:


for i in range(10):
    print(i)
    
    # Build network
    model_tanh_adam = Sequential()

    model_tanh_adam.add(Dense(128, input_dim=x_train.shape[1], activation='tanh'))
    model_tanh_adam.add(Dense(64, activation='tanh')) # Hidden 2
    model_tanh_adam.add(Dense(32, activation='tanh')) # Hidden 3
    model_tanh_adam.add(Dense(16, activation='tanh')) # Hidden 4
    model_tanh_adam.add(Dense(1)) # Output
    
    model_tanh_adam.compile(loss='mean_squared_error', optimizer='adam')
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    
    model_tanh_adam.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpoint_tanh_adam],verbose=2,epochs=100) 
    
print('Training Completed..') 
print('Loading the best model')
print()
model_tanh_adam.load_weights('./best_weights_tanh_adam1.hdf5')


# In[106]:


pred_tanh_adam = model_tanh_adam.predict(x_test)


# In[107]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(y_test[i],pred_tanh_adam[i]))


# In[108]:


# Measure RMSE error.  RMSE is common for regression.
score_tanh_adam = np.sqrt(mean_squared_error(y_test,pred_tanh_adam))
print("Final score (RMSE): {}".format(score_tanh_adam))
print('R2 score: %.2f' % r2_score(y_test, pred_tanh_adam))


# In[121]:


# Regression lift Chart
chart_regression(pred_tanh_adam.flatten(),y_test)


# #### 5 Hidden Layers  +  2 Dropout Layers

# In[66]:


# set up Model CHeck Point
checkpoint_tanh_adam = ModelCheckpoint(filepath="./best_weights_tanh_adam2.hdf5", verbose=1, save_best_only=True)


# In[67]:


for i in range(10):
    print(i)
    
    # Build network
    model_tanh_adam = Sequential()

    model_tanh_adam.add(Dense(128, input_dim=x_train.shape[1], activation='tanh'))
    model_tanh_adam.add(Dropout(0.2))
    model_tanh_adam.add(Dense(64, activation='tanh')) # Hidden 2
    model_tanh_adam.add(Dropout(0.3))
    model_tanh_adam.add(Dense(32, activation='tanh')) # Hidden 3
    model_tanh_adam.add(Dense(16, activation='tanh')) # Hidden 4
    model_tanh_adam.add(Dense(8, activation='tanh')) # Hidden 4
    model_tanh_adam.add(Dense(1)) # Output
    
    model_tanh_adam.compile(loss='mean_squared_error', optimizer='adam')
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    
    model_tanh_adam.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpoint_tanh_adam],verbose=2,epochs=100) 
    
print('Training Completed..') 
print('Loading the best model')
print()
model_tanh_adam.load_weights('./best_weights_tanh_adam2.hdf5')


# In[68]:


pred_tanh_adam = model_tanh_adam.predict(x_test)


# In[69]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(y_test[i],pred_tanh_adam[i]))


# In[70]:


# Measure RMSE error.  RMSE is common for regression.
score_tanh_adam = np.sqrt(mean_squared_error(y_test,pred_tanh_adam))
print("Final score (RMSE): {}".format(score_tanh_adam))
print('R2 score: %.2f' % r2_score(y_test, pred_tanh_adam))


# In[79]:


#Regression lift Chart
chart_regression(pred_tanh_adam.flatten(),y_test)


# ### SGD - Optimizer 

# ####  4  Hidden Layers  

# In[72]:


# set up Model CHeck Point
checkpoint_tanh_sgd = ModelCheckpoint(filepath="./best_weights_tanh_sgd.hdf5", verbose=1, save_best_only=True)


# In[73]:


for i in range(10):
    print(i)
    
    # Build network
    model_tanh_sgd = Sequential()

    model_tanh_sgd.add(Dense(128, input_dim=x_train.shape[1], activation='tanh'))
    model_tanh_sgd.add(Dense(64, activation='tanh')) # Hidden 2
    model_tanh_sgd.add(Dense(32, activation='tanh')) # Hidden 3
    model_tanh_sgd.add(Dense(16, activation='tanh')) # Hidden 4
    model_tanh_sgd.add(Dense(1)) # Output
    
    model_tanh_sgd.compile(loss='mean_squared_error', optimizer='sgd')
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    
    model_tanh_sgd.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpoint_tanh_sgd],verbose=2,epochs=100) 
    
print('Training Completed..') 
print('Loading the best model')
print()
model_tanh_sgd.load_weights('./best_weights_tanh_sgd.hdf5')


# In[74]:


pred_tanh_sgd = model_tanh_sgd.predict(x_test)


# In[75]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(y_test[i],pred_tanh_adam[i]))


# In[76]:


# Measure RMSE error.  RMSE is common for regression.
score_tanh_sgd = np.sqrt(mean_squared_error(y_test,pred_tanh_sgd))
print("Final score (RMSE): {}".format(score_tanh_sgd))
print('R2 score: %.2f' % r2_score(y_test, pred_tanh_sgd))


# In[80]:


#Regression lift Chart
chart_regression(pred_tanh_sgd.flatten(),y_test)


# ## Sigmoid  - Activation Function

# ### Adam - Optimizer

# ####  4  Hidden Layers  

# In[91]:


# set up Model CHeck Point
checkpoint_sigmoid = ModelCheckpoint(filepath="./best_weights_sigmoid_adam1.hdf5", verbose=1, save_best_only=True)


# In[92]:


for i in range(10):
    print(i)
    
    # Build network
    model_sigmoid_adam = Sequential()

    model_sigmoid_adam.add(Dense(64, input_dim=x_train.shape[1], activation='sigmoid'))  
    model_sigmoid_adam.add(Dense(32, activation='sigmoid')) # Hidden 2
    model_sigmoid_adam.add(Dense(16, activation='sigmoid')) # Hidden 3
    model_sigmoid_adam.add(Dense(8,  activation='sigmoid')) # Hidden 4
    model_sigmoid_adam.add(Dense(1)) # Output
    
    model_sigmoid_adam.compile(loss='mean_squared_error', optimizer='adam')
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    
    model_sigmoid_adam.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpoint_sigmoid],verbose=2,epochs=100) 
    
print('Training Completed..') 
print('Loading the best model')
print()
model_sigmoid_adam.load_weights('./best_weights_sigmoid_adam1.hdf5')


# In[94]:


pred_sigmoid_adam = model_sigmoid_adam.predict(x_test)


# In[95]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(y_test[i],pred_sigmoid_adam[i]))


# In[96]:


# Measure RMSE error.  RMSE is common for regression.
score_sigmoid_adam = np.sqrt(mean_squared_error(y_test,pred_sigmoid_adam))
print("Final score (RMSE): {}".format(score_sigmoid_adam))
print('R2 score: %.2f' % r2_score(y_test, pred_sigmoid_adam))


# In[97]:


#Regression lift Chart
chart_regression(pred_sigmoid_adam.flatten(),y_test)


# ####  5 Hidden Layers  +  2 Dropout Layer

# In[93]:


# set up Model CHeck Point
checkpoint_sigmoid = ModelCheckpoint(filepath="./best_weights_sigmoid_adam2.hdf5", verbose=1, save_best_only=True)


# In[98]:


for i in range(10):
    print(i)
    
    # Build network
    model_sigmoid_adam = Sequential()

    model_sigmoid_adam.add(Dense(64, input_dim=x_train.shape[1], activation='sigmoid'))  
    model_sigmoid_adam.add(Dropout(0.2))
    model_sigmoid_adam.add(Dense(32, activation='sigmoid')) # Hidden 2
    model_sigmoid_adam.add(Dropout(0.3))
    model_sigmoid_adam.add(Dense(16, activation='sigmoid')) # Hidden 3
    model_sigmoid_adam.add(Dense(8,  activation='sigmoid')) # Hidden 4
    model_sigmoid_adam.add(Dense(1)) # Output
    
    model_sigmoid_adam.compile(loss='mean_squared_error', optimizer='adam')
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    
    model_sigmoid_adam.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpoint_sigmoid],verbose=2,epochs=100) 
    
print('Training Completed..') 
print('Loading the best model')
print()
model_sigmoid_adam.load_weights('./best_weights_sigmoid_adam2.hdf5')


# In[99]:


pred_sigmoid_adam = model_sigmoid_adam.predict(x_test)


# In[100]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(y_test[i],pred_sigmoid_adam[i]))


# In[101]:


# Measure RMSE error.  RMSE is common for regression.
score_sigmoid_adam = np.sqrt(mean_squared_error(y_test,pred_sigmoid_adam))
print("Final score (RMSE): {}".format(score_sigmoid_adam))
print('R2 score: %.2f' % r2_score(y_test, pred_sigmoid_adam))


# In[102]:


#Regression lift Chart
chart_regression(pred_sigmoid_adam.flatten(),y_test)


# ####  4 Hidden Layer + 1 dropout Layer

# In[110]:


# set up Model CHeck Point
checkpoint_sigmoid = ModelCheckpoint(filepath="./best_weights_sigmoid_adam3.hdf5", verbose=1, save_best_only=True)


# In[111]:


for i in range(10):
    print(i)
    
    # Build network
    model_sig_adam = Sequential()
    model_sig_adam.add(Dense(80, input_dim=x_train.shape[1]))  
    model_sig_adam.add(Dropout(0.1))
    model_sig_adam.add(Dense(60, activation='sigmoid')) # Hidden 2
    model_sig_adam.add(Dense(20, activation='sigmoid')) # Hidden 3
    model_sig_adam.add(Dense(10, activation='sigmoid')) # Hidden 4
    model_sig_adam.add(Dense(1)) # Output
    model_sig_adam.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    model_sig_adam.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor, checkpoint_sigmoid],verbose=2,epochs=100) 
    
print('Training Complete..') 
print('Loading the best model')
model_sig_adam.load_weights('./best_weights_sigmoid_adam3.hdf5')


# In[122]:


pred_sig_adam = model_sig_adam.predict(x_test)


# In[123]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(y_test[i],pred_sig_adam[i]))


# In[125]:


# Measure RMSE error.  RMSE is common for regression.
score_sig_adam = np.sqrt(mean_squared_error(y_test,pred_sig_adam))
print("Final score (RMSE): {}".format(score_sig_adam))
print('R2 score: %.2f' % r2_score(y_test, pred_sig_adam))


# In[127]:


#Regression lift Chart
chart_regression(pred_sig_adam.flatten(),y_test)


# In[ ]:


#Regression lift Chart
chart_regression(pred_sig_adam.flatten(),y_test)


# ### SGD- Optimizer 

# ####  4  Hidden Layers  

# In[115]:


# set up Model CHeck Point
checkpoint_sigmoid_sgd = ModelCheckpoint(filepath="./best_weights_sigmoid_sgd.hdf5", verbose=1, save_best_only=True)


# In[116]:


for i in range(10):
    print(i)
    
    # Build network
    model_sigmoid_sgd = Sequential()

    model_sigmoid_sgd.add(Dense(128, input_dim=x_train.shape[1], activation='sigmoid'))
    model_sigmoid_sgd.add(Dense(64, activation='sigmoid')) # Hidden 2
    model_sigmoid_sgd.add(Dense(32, activation='sigmoid')) # Hidden 3
    model_sigmoid_sgd.add(Dense(16, activation='sigmoid')) # Hidden 4
    model_sigmoid_sgd.add(Dense(1)) # Output
    
    model_sigmoid_sgd.compile(loss='mean_squared_error', optimizer='sgd')
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    
    model_sigmoid_sgd.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor,checkpoint_sigmoid_sgd],verbose=2,epochs=100) 
    
print('Training Completed..') 
print('Loading the best model')
print()
model_sigmoid_sgd.load_weights('./best_weights_sigmoid_sgd.hdf5')


# In[117]:


pred_sigmoid_sgd = model_sigmoid_sgd.predict(x_test)


# In[118]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(y_test[i],pred_tanh_adam[i]))


# In[119]:


# Measure RMSE error.  RMSE is common for regression.
score_sigmoid_sgd = np.sqrt(mean_squared_error(y_test,pred_sigmoid_sgd))
print("Final score (RMSE): {}".format(score_sigmoid_sgd))
print('R2 score: %.2f' % r2_score(y_test, pred_sigmoid_sgd))


# In[120]:


#Regression lift Chart
chart_regression(pred_sigmoid_sgd.flatten(),y_test)


# ## Comparing all Models for Task1 ( Fully Connected)

# In[94]:


Relu_adam_RMSE = 0.44610342383384705
Relu_Sgd_RMSE = 29.908370971679688
Tanh_adam1_RMSE = 2.475679636001587
Tanh_adam2_RMSE = 10.592331886291504
Tanh_Sgd_RMSE = 29.803329467773438
Sigmoid_adam1_RMSE =  29.908071517944336
Sigmoid_adam2_RMSE =  11.40207290649414
Sigmoid_Sgd_RMSE =  0.9340808987617493


# In[96]:


score_list_RMSE= [Relu_adam_RMSE, Relu_Sgd_RMSE, Tanh_adam1_RMSE ,
               Tanh_adam2_RMSE, Tanh_Sgd_RMSE , Sigmoid_adam1_RMSE ,Sigmoid_adam2_RMSE ,Sigmoid_Sgd_RMSE  ]
names =['Relu_adam_RMSE','Relu_Sgd_RMSE','Tanh_adam1_RMSE','Tanh_adam2_RMSE','Tanh_Sgd_RMSE','Sigmoid_adam1_RMSE',
        'Sigmoid_adam2_RMSE','Sigmoid_Sgd_RMSE']
tick_marks = np.arange(len(names))
plt.bar(range(len(score_list_RMSE)), score_list_RMSE)
plt.xticks(tick_marks, names, rotation=45)
plt.show()


# # Task 2   -  LSTM Model

# In[166]:


#Reading the file and loading dataset into stock_df dataframe 
stock_df_rnn= pd.read_csv('CSC215_Project4_Stock_Price.csv')
stock_df_rnn[0:5]


# In[167]:


stock_df_rnn = stock_df_rnn.drop(['Date', 'Adj_Close'], axis = 1)
stock_close_df_rnn = stock_df_rnn['Close']


# In[168]:


type(stock_df_rnn)


# In[169]:


# Normalize the columns

normalize_numeric_minmax(stock_df_rnn,"Low") 
normalize_numeric_minmax(stock_df_rnn,"Volume") 
normalize_numeric_minmax(stock_df_rnn,"Close")    
normalize_numeric_minmax(stock_df_rnn,"Open")
normalize_numeric_minmax(stock_df_rnn,"High") 
 


# In[170]:


print(stock_df_rnn[0:10])
print(stock_close_df_rnn[0:10])


# In[171]:


# From Lab
import numpy as np

def to_sequences(seq_size, data):
    x = []
    y = []

    for i in range(len(data)-SEQUENCE_SIZE-1):
        #print(i)
        window = data[i:(i+SEQUENCE_SIZE)]
        after_window = data[i+SEQUENCE_SIZE]
        window = [[x] for x in window]
        #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)
        
    return np.array(x),np.array(y)


# In[15]:


# Little Modified
def to_sequences_new(seq_size, data, OutComelabel):
    x = []
    y = []

    for i in range(len(data)-seq_size-1):
        print(i)
        window = data[i:(i+seq_size)].values
        after_window = OutComelabel[i+seq_size]
        print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)
        
    return np.array(x),np.array(y)


# In[113]:


x_,y_ = to_sequences_new(7,stock_df_rnn,stock_close_df_rnn)


# In[126]:


print("Shape of x_rnn_lstm: {}".format(x_.shape))
print("Shape of y_rnn_lstm: {}".format(y_.shape))


# In[127]:


print(x_[0:5])
print(y_[0:5])


# In[128]:


x_.shape[1:3]


# In[129]:


#Split for train and test
x_train_rnn, x_test_rnn, y_train_rnn, y_test_rnn = train_test_split(x_,y_, test_size=0.3, random_state=42)


# In[130]:


print(x_train_rnn.shape)
print(x_test_rnn.shape)
print(y_train_rnn.shape)
print(y_test_rnn.shape)
print(x_train_rnn.shape[1:3])


# ## LSTM  and Adam Optimizer

# ###  1- LSTM layer + 1 Dense Layer + adam

# In[148]:


# set up checkpointer
checkpoint_lstm = ModelCheckpoint(filepath="./best_weights_lstm1.hdf5", verbose=1, save_best_only=True)


# In[149]:


for i in range(10):
    print(i)
    
    print('Build model...')
    model_lstm = Sequential()

    model_lstm.add(LSTM(64, input_shape=x_train_rnn.shape[1:3]))
    model_lstm.add(Dense(32))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    print('Train...')
    model_lstm.fit(x_train_rnn,y_train_rnn,validation_data=(x_test_rnn,y_test_rnn),callbacks=[monitor,checkpoint_lstm],verbose=2, epochs=10)  

print('Training finished...Loading the best model') 
print()
model_lstm.load_weights('./best_weights_lstm1.hdf5')


# In[150]:


pred_rnn = model_lstm.predict(x_test_rnn)


# In[161]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(pred_rnn[i],y_test_rnn[i]))


# In[175]:


chart_regression(pred_rnn.flatten(),y_test_rnn)


# In[181]:


get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (15,6)
plt.plot(pred_rnn[1300:], color = 'red', label = 'Stock Price')
plt.plot(y_test_rnn[1300:], color = 'green', label = 'Predicted Stock Price')
plt.title(' Stock Price Prediction')
#plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[22]:


get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (15,6)


# In[152]:


score_rnn= np.sqrt(mean_squared_error(pred_rnn,y_test_rnn))
print("Score (RMSE): {}".format(score_rnn))
score_rnn_r2= r2_score(pred_rnn,y_test_rnn)
print("Score (R2): {}".format(score_rnn_r2))


# In[179]:


y_test_rnn.shape


# In[186]:


y_test_rnn[1300:]


# In[187]:


pred_rnn[1300:]


# In[165]:


pred_rnn[:5]


# ###       2- LSTM layers  + adam  + 2 Dropout Layers

# In[131]:


# set up checkpointer
checkpoint_lstm_3L = ModelCheckpoint(filepath="./best_weights_lstm_2L.hdf5", verbose=1, save_best_only=True)


# In[134]:


for i in range(4):
    print(i)
    
    print('Build model...')
    model_lstm = Sequential()
    
    model_lstm.add(LSTM(units = 80,recurrent_dropout=0.1, input_shape=x_train_rnn.shape[1:3],return_sequences=True))
    model_lstm.add(Dropout(0.2))
    
    model_lstm.add(LSTM(units = 50))
    model_lstm.add(Dropout(0.2))
    
    model_lstm.add(Dense(32))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    print('Train...')
    model_lstm.fit(x_train_rnn,y_train_rnn,validation_data=(x_test_rnn,y_test_rnn),callbacks=[monitor,checkpoint_lstm_3L],verbose=2, epochs=10)  

print('Loading the best model') 
print()
model_lstm.load_weights('./best_weights_lstm_2L.hdf5')


# In[150]:


pred_rnn_3L = model_lstm.predict(x_test_rnn)


# In[136]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(y_test_rnn[i],pred_rnn_3L[i]))


# In[137]:


score_rnn = np.sqrt(mean_squared_error(pred_rnn_3L,y_test_rnn))
print("Score (RMSE): {}".format(score_rnn))
score_rnn_r2= r2_score(pred_rnn_3L,y_test_rnn)
print("Score (R2): {}".format(score_rnn_r2))


# In[138]:


chart_regression(pred_rnn_3L.flatten(),y_test_rnn)


# # Task 3   -   CNN Model

# ## Data Preprocessing

# In[188]:


#Readinf the file and loading dataset into stock_df dataframe 
stock_df_cnn= pd.read_csv('CSC215_Project4_Stock_Price.csv')
stock_df_cnn[0:5]


# In[189]:


stock_df_cnn = stock_df_cnn.drop(['Date', 'Adj_Close'], axis = 1)


# In[190]:


close_df_cnn = stock_df_cnn[['Close']]


# In[191]:


# Normalize the input columns   
normalize_numeric_minmax(stock_df_cnn,"Open")
normalize_numeric_minmax(stock_df_cnn,"High") 
normalize_numeric_minmax(stock_df_cnn,"Low") 
normalize_numeric_minmax(stock_df_cnn,"Volume") 
normalize_numeric_minmax(stock_df_cnn,"Close") 


# In[194]:


print(stock_df_cnn[0:5])
print(close_df_cnn[0:5])


# In[195]:


#Create a sliding Window
x_cnn,y_cnn = to_sequences_new(7,stock_df_cnn,stock_close_df_rnn)


# In[196]:


print(x_cnn.shape)
print(y_cnn.shape)


# In[197]:


#Split for train and test
x_train_cnn, x_test_cnn, y_train_cnn, y_test_cnn = train_test_split(x_cnn,y_cnn, test_size=0.3, random_state=42)


# In[200]:


print(x_train_cnn.shape)
print(x_test_cnn.shape)
print(y_train_cnn.shape)
print(y_test_cnn.shape)


# In[201]:


# Reshaping
x_train_cnn = x_train_cnn.reshape(x_train_cnn.shape[0], 1,  x_train_cnn.shape[1],  x_train_cnn.shape[2])
x_test_cnn = x_test_cnn.reshape(x_test_cnn.shape[0], 1, x_test_cnn.shape[1],x_test_cnn.shape[2])


# In[202]:


print(x_train_cnn.shape)
print(y_train_cnn.shape)
print(x_test_cnn.shape)
print(y_test_cnn.shape)


# ## CNN - 1 Conv2D layer + Dropout layer + 1 FC Layer

# In[212]:


#input_shape: should be the shape of 1 sample i.e [rows,cols,1]
input_shape = x_train_cnn[0].shape
input_shape


# In[203]:


checkpoint_cnn = ModelCheckpoint(filepath="./best_weights_cnn.hdf5", verbose=1, save_best_only=True)


# In[205]:


# CNN 2D

for i in range(3):
    print(i)

    cnn = Sequential()

# Building Conv2D layer 1
    cnn.add(Conv2D(32, kernel_size=(1,3), strides=(1,1), padding='valid',
                     activation='relu',
                     input_shape=input_shape))
# Max pool layer
    cnn.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))    
    

# Drop out layer 
    cnn.add(Dropout(0.10))
    
# Flatten
    cnn.add(Flatten())
    
# Fully Connected layer 1

    cnn.add(Dense(16, activation='relu'))
    
# Drop out layer
    cnn.add(Dropout(0.20))
    

# output Layer
    cnn.add(Dense(1))

# Compile

    cnn.compile(loss='mean_squared_error', optimizer='adam')

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    
    cnn.fit(x_train_cnn, y_train_cnn, 
            batch_size=64, 
            callbacks=[monitor,checkpoint_cnn], 
            epochs=10, 
            verbose=2, 
            validation_data=(x_test_cnn, y_test_cnn))

print('Training finished') 
print()
cnn.load_weights('./best_weights_cnn.hdf5')


# In[206]:


cnn.summary()


# In[207]:


pred_cnn = cnn.predict(x_test_cnn)
pred_cnn.flatten()


# In[208]:


score_cnn= np.sqrt(mean_squared_error(pred_cnn,y_test_cnn))
print("Score (RMSE): {}".format(score_cnn))
score_cnn_r2= r2_score(pred_cnn,y_test_cnn)
print("Score (R2): {}".format(score_cnn_r2))


# In[228]:


chart_regression(pred_cnn.flatten(),y_test_cnn)


# ## CNN - 3 Conv2D layer + Dropout Layers + Max pool layers + 3 Fully Connected Layers

# In[209]:


checkpoint_cnn_2L = ModelCheckpoint(filepath="./best_weights_cnn_2L.hdf5", verbose=1, save_best_only=True)


# In[223]:


# CNN 2D

for i in range(5):
    print(i)

    cnn_2L = Sequential()

# Building Conv2D layer 1
    cnn_2L.add(Conv2D(128, kernel_size=(1,3), strides=(1,1), padding='valid',
                     activation='relu',
                     input_shape=input_shape))
# Max pool layer
    cnn_2L.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))  
#Drop out layer 
    cnn_2L.add(Dropout(0.10))   
    
# Building Conv2D layer 2
    cnn_2L.add(Conv2D(64, kernel_size= (1, 1), strides=(1,1), activation='relu'))
# Max pool layer
    cnn_2L.add(MaxPooling2D(pool_size=(1,1), strides=None))  
# Drop out layer 
    cnn_2L.add(Dropout(0.20))
    
# Building Conv2D layer 3
    cnn_2L.add(Conv2D(32 ,kernel_size= (1, 1), strides=(1, 1), activation='relu'))
# Max pool layer
    cnn_2L.add(MaxPooling2D(pool_size=(1,1), strides=None))  
# Drop out layer 
    cnn_2L.add(Dropout(0.20))
    
# Flatten
    cnn_2L.add(Flatten())
    
# Fully Connected layer 1

    cnn_2L.add(Dense(128, activation='relu'))
# Drop out layer
    cnn_2L.add(Dropout(0.20))
    
    
# Fully Connected layer 2
    cnn_2L.add(Dense(64, activation='relu'))
# Drop out layer
    cnn_2L.add(Dropout(0.20))
    
    
# Fully Connected layer 3
    cnn_2L.add(Dense(16, activation='relu'))
# Drop out layer
    cnn_2L.add(Dropout(0.10))

# output Layer
    cnn_2L.add(Dense(1))

# Compile

    cnn_2L.compile(loss='mean_squared_error', optimizer='adam')

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    
    cnn_2L.fit(x_train_cnn, y_train_cnn, 
            batch_size=64, 
            callbacks=[monitor,checkpoint_cnn_2L], 
            epochs=20, 
            verbose=2, 
            validation_data=(x_test_cnn, y_test_cnn))

print('Training finished') 
print()
cnn_2L.load_weights('./best_weights_cnn_2L.hdf5')


# In[224]:


cnn_2L.summary()


# In[225]:


pred_cnn_2L = cnn_2L.predict(x_test_cnn)
pred_cnn_2L.flatten()


# In[226]:


score_cnn_2L= np.sqrt(mean_squared_error(pred_cnn_2L,y_test_cnn))
print("Score (RMSE): {}".format(score_cnn_2L))
score_cnn_r2_2L= r2_score(pred_cnn_2L,y_test_cnn)
print("Score (R2): {}".format(score_cnn_r2_2L))


# In[229]:


chart_regression(pred_cnn_2L.flatten(),y_test_cnn)


# # Comparing Best of All - Fully Connected, LSTM, CNN 

# In[139]:


Relu_adam_RMSE = 0.44610342383384705
LSTM_RMSE = 1.4032532490615899
CNN_RMSE = 2.0277379697960916


# In[141]:


score_list_RMSE= [Relu_adam_RMSE, LSTM_RMSE, CNN_RMSE ]
names =['Relu_adam_RMSE','LSTM_RMSE','CNN_RMSE']
tick_marks = np.arange(len(names))
plt.bar(range(len(score_list_RMSE)), score_list_RMSE)
plt.xticks(tick_marks, names, rotation=45)
plt.show()


# # Additional Features

# ##  Experimenting N value 

# ###  Experimenting N value - LSTM

# In[230]:


x_rnn,y_rnn = to_sequences(60,stock_df_rnn,stock_close_df_rnn)


# In[231]:


print("Shape of x_rnn_lstm: {}".format(x_rnn.shape))
print("Shape of y_rnn_lstm: {}".format(y_rnn.shape))


# In[232]:


print(x_rnn[0:5])
print(y_rnn[0:5])


# In[233]:


x_rnn.shape[1:3]


# In[234]:


print(x_rnn.shape)
print(y_rnn.shape)


# In[239]:


#Split for train and test
x_train_rnn_n60, x_test_rnn_n60, y_train_rnn_n60, y_test_rnn_n60 = train_test_split(x_rnn,y_rnn, test_size=0.3, random_state=42)


# In[240]:


print(x_train_rnn_n60.shape)
print(x_test_rnn_n60.shape)
print(y_train_rnn_n60.shape)
print(y_test_rnn_n60.shape)
print(x_train_rnn_n60.shape[1:3])


# In[241]:


# set up checkpointer
checkpoint_rnn_n60 = ModelCheckpoint(filepath="./best_weights_rnn_n60.hdf5", verbose=1, save_best_only=True)


# In[243]:


for i in range(5):
    print(i)
    
    print('Build model...')
    model_rnn_n60 = Sequential()

    model_rnn_n60.add(LSTM(units = 150,recurrent_dropout=0.1, input_shape=x_train_rnn_n60.shape[1:3],return_sequences=True))
    model_rnn_n60.add(Dropout(0.2))
    
    model_rnn_n60.add(LSTM(units = 150, return_sequences = True))
    model_rnn_n60.add(Dropout(0.2))
    
    model_rnn_n60.add(LSTM(units = 80, return_sequences = True ))
    model_rnn_n60.add(Dropout(0.2))
    
    model_rnn_n60.add(LSTM(units = 80, return_sequences = True))
    model_rnn_n60.add(Dropout(0.2))
    
    model_rnn_n60.add(LSTM(units = 80))
    
    model_rnn_n60.add(Dense(128) ) 
    model_rnn_n60.add(Dense(64)) 
    model_rnn_n60.add(Dense(32)) 
    
    model_rnn_n60.add(Dense(1))
    model_rnn_n60.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    print('Train...')
    model_rnn_n60.fit(x_train_rnn_n60,y_train_rnn_n60,validation_data=(x_test_rnn_n60,y_test_rnn_n60),callbacks=[monitor,checkpoint_rnn_n60],verbose=2, epochs=20)  

print('Training finished...') 
print('Loading the best model')
model_rnn_n60.load_weights('./best_weights_rnn_n60.hdf5')


# In[244]:


model_rnn_n60.load_weights('./best_weights_rnn_n60.hdf5')


# In[245]:


pred_rnn_n60 = model_rnn_n60.predict(x_test_rnn_n60)


# In[246]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(pred_rnn_n60[i],y_test_rnn_n60[i]))


# In[247]:


chart_regression(pred_rnn_n60.flatten(),y_test_rnn_n60)


# In[248]:


get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (15,6)
plt.plot(pred_rnn_n60[1250:], color = 'red', label = 'Stock Price')
plt.plot(y_test_rnn_n60[1250:], color = 'green', label = 'Predicted Stock Price')
plt.title(' Stock Price Prediction')
#plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[252]:


score_rnn_n60= np.sqrt(mean_squared_error(pred_rnn_n60,y_test_rnn_n60))
print("Score (RMSE): {}".format(score_rnn_n60))
score_rnn_r2_n60= r2_score(pred_rnn_n60,y_test_rnn_n60)
print("Score (R2): {}".format(score_rnn_r2_n60))


# ### 2 Months vs 2 Week

# In[17]:


x14,y14 = to_sequences_new(14,stock_df_rnn,stock_close_df_rnn)


# In[160]:


print("Shape of x_rnn_lstm: {}".format(x14.shape))
print("Shape of y_rnn_lstm: {}".format(y14.shape))


# In[164]:


x_train14 =x14[:3000]
y_train14 =y14[:3000]
x_test14 =x14[3000::]
y_test14 =y14[3000::]


# In[165]:


print(x_train14.shape)
print(y_train14.shape)
print(x_test14.shape)
print(y_test14.shape)


# In[172]:


x_test14[1376]


# In[174]:


x14[4376]


# In[177]:


# set up checkpointer
checkpoint14 = ModelCheckpoint(filepath="./best_weights_lstm14.hdf5", verbose=1, save_best_only=True)


# In[178]:


for i in range(4):
    print(i)
    
    print('Build model...')
    model_lstm = Sequential()
    
    model_lstm.add(LSTM(units = 80,recurrent_dropout=0.1, input_shape=x_train14.shape[1:3],return_sequences=True))
    model_lstm.add(Dropout(0.2))
    
    model_lstm.add(LSTM(units = 50))
    model_lstm.add(Dropout(0.2))
    
    model_lstm.add(Dense(32))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    print('Train...')
    model_lstm.fit(x_train14,y_train14,validation_data=(x_test14,y_test14),callbacks=[monitor,checkpoint14],verbose=2, epochs=10)  

print('Loading the best model') 
print()
model_lstm.load_weights('./best_weights_lstm14.hdf5')


# In[179]:


pred_rnn14 = model_lstm.predict(x_test14)


# In[180]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(pred_rnn14[i],y_test14[i]))


# In[181]:


chart_regression(pred_rnn14.flatten(),y_test14)


# In[184]:


score_rnn= np.sqrt(mean_squared_error(pred_rnn14,y_test14))
print("Score (RMSE): {}".format(score_rnn))
score_rnn_r2= r2_score(pred_rnn14,y_test14)
print("Score (R2): {}".format(score_rnn_r2))


# ### 2 Weeks

# In[18]:


#Split for train and test
x_train14, x_test14, y_train14, y_test14 = train_test_split(x14,y14, test_size=0.3, random_state=42)


# In[19]:


# set up checkpoint
checkpoint14 = ModelCheckpoint(filepath="./best_weights_lstm14.hdf5", verbose=1, save_best_only=True)


# In[20]:


x_train14.shape[1:3]


# In[21]:


x_train14


# In[22]:


for i in range(4):
    print(i)
    
    print('Build model...')
    model_lstm5 = Sequential()
    
    model_lstm5.add(LSTM(units = 90,recurrent_dropout=0.1, input_shape=x_train14.shape[1:3],return_sequences=True))
    model_lstm5.add(Dropout(0.2))
    
    model_lstm5.add(LSTM(units = 90))
    model_lstm5.add(Dropout(0.2))
    
    model_lstm5.add(Dense(32))
    model_lstm5.add(Dense(1))
    model_lstm5.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    print('Train...')
    model_lstm5.fit(x_train14,y_train14,validation_data=(x_test14,y_test14),callbacks=[monitor,checkpoint14],verbose=2, epochs=10)  

print('Loading the best model') 
print()
model_lstm5.load_weights('./best_weights_lstm14.hdf5')


# In[24]:


pred_rnn14 = model_lstm5.predict(x_test14)


# In[25]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(pred_rnn14[i],y_test14[i]))


# In[26]:


chart_regression(pred_rnn14.flatten(),y_test14)


# In[27]:


score_rnn= np.sqrt(mean_squared_error(pred_rnn14,y_test14))
print("Score (RMSE): {}".format(score_rnn))
score_rnn_r2= r2_score(pred_rnn14,y_test14)
print("Score (R2): {}".format(score_rnn_r2))


# ### 2 Months - 60 Days 

# In[172]:


x60,y60 = to_sequences_new(60,stock_df_rnn,stock_close_df_rnn)


# In[173]:


#Split for train and test
x_train60, x_test60, y_train60, y_test60 = train_test_split(x60,y60, test_size=0.3, random_state=42)


# In[174]:


# set up checkpointer
checkpoint60 = ModelCheckpoint(filepath="./best_weights_lstm60.hdf5", verbose=1, save_best_only=True)


# In[175]:


x_train60.shape[1:3]


# In[176]:


for i in range(10):
    print(i)
    
    print('Build model...')
    model_lstm = Sequential()
    
    model_lstm.add(LSTM(units = 90,recurrent_dropout=0.1, input_shape=x_train60.shape[1:3],return_sequences=True))
    model_lstm.add(Dropout(0.2))
    
    model_lstm.add(LSTM(units = 50))
    model_lstm.add(Dropout(0.2))
    
    model_lstm.add(Dense(32))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    print('Train...')
    model_lstm.fit(x_train60,y_train60,validation_data=(x_test60,y_test60),callbacks=[monitor,checkpoint60],verbose=2, epochs=20)  

print('Loading the best model') 
print()
model_lstm.load_weights('./best_weights_lstm60.hdf5')


# In[181]:


pred_rnn60 = model_lstm.predict(x_test60)


# In[34]:


for i in range(10):
    print(" Actual Value: {}, predicted Value: {}".format(pred_rnn60[i],y_test60[i]))


# In[182]:


chart_regression(pred_rnn60.flatten(),y_test60)


# In[183]:


score_rnn= np.sqrt(mean_squared_error(pred_rnn60,y_test60))
print("Score (RMSE): {}".format(score_rnn))
score_rnn_r2= r2_score(pred_rnn60,y_test60)
print("Score (R2): {}".format(score_rnn_r2))


# ### Comparison For Different N Values

# #### RMSE Score

# In[2]:


RMSE_7_Days=1.4680909946360896
RMSE_14_Days = 1.5034853051806818
RMSE_60_Days = 1.4458206118168693


# In[3]:


score_list_RMSE= [RMSE_7_Days, RMSE_14_Days, RMSE_60_Days]
names =['RMSE_7_Days','RMSE_14_Days','RMSE_60_Days']
tick_marks = np.arange(len(names))
plt.bar(range(len(score_list_RMSE)), score_list_RMSE)
plt.xticks(tick_marks, names, rotation=45)
plt.show()


# #### R2 Score 

# In[4]:


R2_7_Days=0.9974678447605176
R2_14_Days = 0.997448769710897
R2_60_Days =  0.9975989687634519


# In[5]:


score_list_RMSE= [R2_7_Days, R2_14_Days, R2_60_Days]
names =['R2_7_Days','R2_14_Days','R2_60_Days']
tick_marks = np.arange(len(names))
plt.bar(range(len(score_list_RMSE)), score_list_RMSE)
plt.xticks(tick_marks, names, rotation=45)
plt.show()


# ## LSTM - Continous Time Period

# In[47]:


def to_Sequences_continous(seq_size, output_seq_size, InputFeatutes, OutputFeatures):
    x = []
    y = []

    for i in range(len(InputFeatutes)-seq_size-output_seq_size+1):
        print(i)
        window = InputFeatutes[i:(i+seq_size)].values
        after_window = np.array(OutputFeatures[i+seq_size:(i+seq_size+output_seq_size)].values)
        print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)
        
    return np.array(x),np.array(y)


# In[48]:


x_rnn_check,y_rnn_check = to_Sequences_continous(15,5,stock_df_rnn,stock_close_df_rnn)


# In[49]:


print(x_rnn_check.shape)
print(y_rnn_check.shape)


# In[50]:


y_rnn_check


# In[55]:


#Split for train and test
x_rnn_check_train, x_rnn_check_test, y_rnn_check_train, y_rnn_check_test = train_test_split(x_rnn_check,y_rnn_check, test_size=0.3, random_state=42)


# In[56]:


print(x_rnn_check_train.shape)
print(x_rnn_check_test.shape)
print(y_rnn_check_train.shape)
print(y_rnn_check_test.shape)


# In[57]:


# set up checkpointer
checkpoint_bi = ModelCheckpoint(filepath="./best_weights_continous.hdf5", verbose=1, save_best_only=True)


# In[58]:


for i in range(10):
    print(i)
    
    print('Build model...')
    model_lstm_con = Sequential()

    model_lstm_con.add(LSTM(64, input_shape=x_rnn_check_train.shape[1:3]))
    model_lstm_con.add(Dense(32))
    model_lstm_con.add(Dense(5))
    model_lstm_con.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    print('Train...')
    model_lstm_con.fit(x_rnn_check_train,y_rnn_check_train,validation_data=(x_rnn_check_test,y_rnn_check_test),callbacks=[monitor,checkpoint_bi],verbose=2, epochs=10)  

print('Loading the best model') 
print()
model_lstm_con.load_weights('./best_weights_continous.hdf5')


# In[60]:


pred_continous = model_lstm_con.predict(x_rnn_check_test)


# In[64]:


for i in range(7):
    print("Day {}".format(i+1))
    print(" Actual Value: {}, \n predicted Value: {}".format(pred_continous[i],y_rnn_check_test[i]))


# In[65]:


chart_regression(pred_continous.flatten(),y_rnn_check_test)


# In[72]:


get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (15,6)
plt.plot(pred_continous[1280:], color = 'red', label = 'Stock Price')
plt.plot(y_rnn_check_test[1280:], color = 'green', label = 'Predicted Stock Price')
plt.title(' Stock Price Prediction')
#plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[70]:


score_rnn_continous= np.sqrt(mean_squared_error(pred_continous,y_rnn_check_test))
print("Score (RMSE): {}".format(score_rnn_continous))
score_rnn_r2_continous= r2_score(pred_continous,y_rnn_check_test)
print("Score (R2): {}".format(score_rnn_r2_continous))


# In[421]:


# Tried to implement Bisirectional LSTM. Not sure about if implemented correctly

for i in range(1):
    print(i)
    
    print('Build model...')
    model_lstm_bidirect = Sequential()

    model_lstm_bidirect.add(Bidirectional(LSTM(10, input_shape=x_rnn_check_train.shape[1:3],return_state =True,return_sequences =True,go_backwards=True)))
    model_lstm_bidirect.add(Bidirectional(LSTM(10)))
    model_lstm_bidirect.add(Dense(7))
    model_lstm_bidirect.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    print('Train...')
    model_lstm_bidirect.fit(x_rnn_check_train,y_rnn_check_train,validation_data=(x_rnn_check_test,y_rnn_check_test),callbacks=[monitor,checkpoint_lstm_bidirect],verbose=2, epochs=2)  

print('Training finished...Loading the best model') 
print()
model_lstm_bidirect.load_weights('./best_weights_bidirectional.hdf5')


# In[422]:


pred = model_lstm_bidirect.predict(x_rnn_check_test)


# In[423]:


pred


# ##  Other Companies

# ###  Google

# In[73]:


google_df= pd.read_csv('GOOG.csv')


# In[74]:


google_df.shape


# In[75]:


google_df.head(5)


# In[76]:


google_df = google_df.drop(['Date', 'Adj Close'], axis = 1)
google_close_df = google_df[['Close']]


# In[77]:



normalize_numeric_minmax(google_df,"Open")
normalize_numeric_minmax(google_df,"High") 
normalize_numeric_minmax(google_df,"Low") 
normalize_numeric_minmax(google_df,"Volume") 
normalize_numeric_minmax(google_df,"Close") 


# In[80]:


x_google,y_google = to_sequences_new(7,google_df, google_close_df['Close'])


# In[81]:


x_train_google, x_test_google, y_train_google, y_test_google = train_test_split(x_google,y_google, test_size=0.3, random_state=42)


# In[84]:


print(x_train_google.shape)
print(x_test_google.shape)
print(y_train_google.shape)
print(y_test_google.shape)


# In[82]:


# set up checkpoint
checkpoint_google = ModelCheckpoint(filepath="./best_weights_lstm_google.hdf5", verbose=1, save_best_only=True)


# In[85]:


for i in range(8):
    print(i)
    
    print('Build model...')
    model_google = Sequential()

    model_google.add(LSTM(64,recurrent_dropout=0.1, input_shape=x_train_google.shape[1:3]))
    model_google.add(Dense(32))
    model_google.add(Dense(1))
    model_google.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    print('Train...')
    model_google.fit(x_train_google,y_train_google,validation_data=(x_test_google,y_test_google),callbacks=[monitor,checkpoint_google],verbose=2, epochs=10)  

print('Loading the best model') 
print()
model_google.load_weights('./best_weights_lstm_google.hdf5')


# In[86]:


pred_google= model_google.predict(x_test_google)

score_google= np.sqrt(mean_squared_error(pred_google,y_test_google))
print("Score (RMSE): {}".format(score_google))
score_r2_google= r2_score(pred_google,y_test_google)
print("Score (R2): {}".format(score_r2_google))


# In[87]:


chart_regression(pred_google.flatten(),y_test_google)


# ###  Royal Dutch Shell

# In[141]:


shell_df= pd.read_csv('RDS-B.csv')


# In[142]:


shell_df = shell_df.drop(['Date', 'Adj Close'], axis = 1)
shell_df = shell_df.dropna()
shell_df = shell_df.reset_index(drop=True)


# In[143]:


shell_close_df = shell_df[['Close']]


# In[144]:


# Normalize the input columns
normalize_numeric_minmax(shell_df,"Open")
normalize_numeric_minmax(shell_df,"High") 
normalize_numeric_minmax(shell_df,"Low") 
normalize_numeric_minmax(shell_df,"Volume") 
normalize_numeric_minmax(shell_df,"Close")


# In[146]:



x_shell,y_shell = to_sequences_new(7,shell_df, shell_close_df['Close'])


# In[147]:


#Split for train and test
x_train_shell, x_test_shell, y_train_shell, y_test_shell = train_test_split(x_shell,y_shell, test_size=0.3, random_state=42)


# In[148]:


# set up checkpoint
checkpoint_shell = ModelCheckpoint(filepath="./best_weights_lstm_shell.hdf5", verbose=1, save_best_only=True)


# In[149]:


for i in range(7):
    print(i)
    
    print('Build model...')
    model_shell = Sequential()

    model_shell.add(LSTM(64, recurrent_dropout=0.1, input_shape=x_train_shell.shape[1:3]))
    model_shell.add(Dense(32))
    model_shell.add(Dense(1))
    model_shell.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
    print('Train...')
    model_shell.fit(x_train_shell,y_train_shell,validation_data=(x_test_shell,y_test_shell),callbacks=[monitor,checkpoint_shell],verbose=2, epochs=10)  

print('Loading the best model') 
print()
model_shell.load_weights('./best_weights_lstm_shell.hdf5')


# In[150]:


pred_shell= model_shell.predict(x_test_shell)


# In[151]:


score_shell= np.sqrt(mean_squared_error(pred_shell,y_test_shell))
print("Score (RMSE): {}".format(score_shell))
score_r2_shell= r2_score(pred_shell,y_test_shell)
print("Score (R2): {}".format(score_r2_shell))


# In[152]:


chart_regression(pred_shell.flatten(),y_test_shell)


# ### JP Morgan

# In[153]:


jpMorgan_df= pd.read_csv('JPM.csv')


# In[154]:


jpMorgan_df = jpMorgan_df.drop(['Date', 'Adj Close'], axis = 1)


# In[156]:


jpMorgan_close_df = jpMorgan_df[['Close']]


# In[157]:



normalize_numeric_minmax(jpMorgan_df,"Open")
normalize_numeric_minmax(jpMorgan_df,"High") 
normalize_numeric_minmax(jpMorgan_df,"Low") 
normalize_numeric_minmax(jpMorgan_df,"Volume") 
normalize_numeric_minmax(jpMorgan_df,"Close") 


# In[159]:


x_jp,y_jp = to_sequences_new(7,jpMorgan_df, jpMorgan_close_df['Close'])


# In[160]:


#Split for train and test
x_train_jp, x_test_jp, y_train_jp, y_test_jp = train_test_split(x_jp,y_jp, test_size=0.3, random_state=42)


# In[161]:


# set up checkpointer
checkpoint_jp = ModelCheckpoint(filepath="./best_weights_lstm_jp.hdf5", verbose=1, save_best_only=True)


# In[177]:


for i in range(10):
    print(i)
    
    print('Build model...')
    model_jp = Sequential()
    model_jp.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1, input_shape=x_train_jp.shape[1:3]))
    model_jp.add(Dense(32))
    model_jp.add(Dense(1))
    model_jp.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    print('Train...')
    model_jp.fit(x_train_jp,y_train_jp,validation_data=(x_test_jp,y_test_jp),callbacks=[monitor,checkpoint_jp],verbose=2, epochs=10)  

print('Training finished...Loading the best model') 
print()
model_jp.load_weights('./best_weights_lstm_jp.hdf5')


# In[178]:


pred_jp= model_jp.predict(x_test_jp)


# In[179]:


score_jp= np.sqrt(mean_squared_error(pred_jp,y_test_jp))
print("Score (RMSE): {}".format(score_jp))
score_r2_jp= r2_score(pred_jp,y_test_jp)
print("Score (R2): {}".format(score_r2_jp))


# In[180]:


chart_regression(pred_jp.flatten(),y_test_jp)

