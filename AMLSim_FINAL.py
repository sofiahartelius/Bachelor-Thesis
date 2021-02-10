# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 23:32:21 2021

@author: Sofia Hartelius
Title: AMLSim Final
"""

# In[ ]: # ## 1. PACKAGES ##

# Data processing
import numpy as np  
import pandas as pd # Loading data, line
import random
import os
from tensorflow.keras.utils import to_categorical
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# mxnet
import mxnet.ndarray as nd
from mxnet import autograd, gpu, Context
from mxnet.gluon import Trainer, HybridBlock
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss
from mxnet.gluon.nn import HybridSequential, Activation
from mxnet.initializer import Uniform
from mxnet.ndarray import array
import mxnet.random

# scikit-learn
from sklearn import preprocessing
from sklearn.metrics import ( # Classification performance tools
    classification_report, 
    roc_auc_score, 
    roc_curve 
)
from sklearn.model_selection import KFold

# NetworkX
import networkx as nx

# Time metrics
import time
from datetime import datetime

# Timers
tic_all = time.perf_counter()
tic_csv = time.perf_counter()

processor =  Context("cpu") # cpu()

mxnet.random.seed(2020)

# In[ ]: # ## 2. Load data ##
os.chdir("E:\My Documents\GitHub\Kandidat")

# Specifying dtype
accounts_dtypes = {'ACCOUNT_ID':int, 'CUSTOMER_ID':object, 'INIT_BALANCE':float, 'COUNTRY':object,
                  'ACCOUNT_TYPE':object,'IS_FRAUD':bool,'TX_BEHAVIOR_ID':int}
transactions_dtypes = {'TX_ID':int, 'SENDER_ACCOUNT_ID':int, 'RECEIVER_ACCOUNT_ID':int, 'TX_TYPE':object,
                       'TX_AMOUNT':float, 'TIMESTAMP':int, 'IS_FRAUD':bool, 'ALERT_ID':int}

# 100 vertices - 10K edges
accounts = pd.read_csv(filepath_or_buffer="./100vertices-10Kedges/accounts.csv", sep = ",", dtype=accounts_dtypes) # Nodes
transactions = pd.read_csv(filepath_or_buffer="./100vertices-10Kedges/transactions.csv", sep = ",", dtype=transactions_dtypes) # Vertices
set_splits = 5

# 1K vertices  # Får konstiga värden
# accounts = pd.read_csv(filepath_or_buffer="E:/My Documents/Dataset/AMLSim/1Kvertices-100Kedges/accounts.csv", sep = ",", dtype=accounts_dtypes) # Nodes
# transactions = pd.read_csv(filepath_or_buffer="E:/My Documents/Dataset/AMLSim/1Kvertices-100Kedges/transactions.csv", sep = ",", dtype=transactions_dtypes) # Vertices
# set_splits = 5

#10K vertices - 1M edges
# accounts = pd.read_csv(filepath_or_buffer="E:/My Documents/Dataset/AMLSim/10Kvertices-1Medges/accounts.csv", sep = ",", dtype=accounts_dtypes) # Nodes
# transactions = pd.read_csv(filepath_or_buffer="E:/My Documents/Dataset/AMLSim/10Kvertices-1Medges/transactions.csv", sep = ",", dtype=transactions_dtypes) # Vertices
# set_splits = 5

# 100K vertices - 10M edges
#accounts = pd.read_csv(filepath_or_buffer="E:/My Documents/Dataset/AMLSim/100Kvertices-10Medges/accounts.csv", sep = ",", dtype=accounts_dtypes) # Nodes
#transactions = pd.read_csv(filepath_or_buffer="E:/My Documents/Dataset/AMLSim/100Kvertices-10Medges/transactions.csv", sep = ",", dtype=transactions_dtypes) # Vertices
#set_splits = 10000

# 1M vertices - 100M edges
#accounts = pd.read_csv(filepath_or_buffer="E:/My Documents/Dataset/AMLSim/1Mvertices-100Medges/accounts.csv", sep = ",", dtype=accounts_dtypes) # Nodes
#transactions = pd.read_csv(filepath_or_buffer="E:/My Documents/Dataset/AMLSim/1Mvertices-100Medges/transactions.csv", sep = ",", dtype=transactions_dtypes) # Vertices
#set_splits = 100000

kf = KFold(n_splits = set_splits, shuffle = True, random_state=2020)
kf_iter = kf.split(accounts)
result = next(kf_iter)
print (result)
range_train = result[1]
range_test =  result[0]

with open('range_test.txt', 'w') as filehandle:
    for cell in range_test:
        filehandle.write('%s\n' % cell)

toc_csv = time.perf_counter()
print('\nClock time:', datetime.now().time(), 'CSV read time:', (toc_csv - tic_csv))

# Set Hyper-parameters
n_epochs = 37
n_layer_units = 10 
n_activation_functions = 307
activation_function = 'relu'

# In[ ]: # ## 3. Analyze data ##

print('\nBalance for class IS_FRAUD; True:')
Counter(accounts.IS_FRAUD)

# In[ ]: # ## 4. Preprocess data ###

## 3.1 Process Classes
tic_labels = time.perf_counter()

labels = []

for i,data in enumerate(accounts.IS_FRAUD):
    elements = data
    labels.append(elements)

def encode_label(labels):
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return labels, label_encoder.classes_

labels_encoded, classes = encode_label(labels)
num_classes = len(set(labels))

print('\nClasses: ', set(labels))
print('\nNumber of classes: ', num_classes)

labels = np.asarray(labels)

toc_labels = time.perf_counter()
print('\nClock time:', datetime.now().time(),'\nLabels time:', (toc_labels - tic_labels))

# In[ ]: # ## 5. Building Graph ##
tic_nx_train = time.perf_counter()

G = nx.from_pandas_edgelist(transactions,
                            source = 'SENDER_ACCOUNT_ID',
                            target = 'RECEIVER_ACCOUNT_ID',
                            edge_attr = True,
                            create_using=nx.MultiGraph())
G.add_nodes_from(accounts["ACCOUNT_ID"])                    # Adding nodes
nx.set_node_attributes(G, accounts["IS_FRAUD"], "IS_FRAUD") # Adding class IS_FRAUD
nx.set_node_attributes(G, accounts["INIT_BALANCE"], "INIT_BALANCE") # Adding class INIT_BALANCE
nx.set_node_attributes(G, accounts["TX_BEHAVIOR_ID"], "TX_BEHAVIOR_ID") # Adding class TX_BEHAVIOR_ID
G.name = "AMLSim Graph"

toc_nx_train = time.perf_counter()
print('\nClock time:', datetime.now().time(),'\nBuilding graph time:', (toc_nx_train - tic_nx_train))

N = len(accounts) 

del accounts, transactions, accounts_dtypes

# In[ ]: # ## 6. Extract Adjacency Matrix ##
tic_adjacency = time.perf_counter()

A = nd.array(nx.attr_matrix(G)[0], ctx = processor) #, dtype = np.float64) #, dtype = "int64")

toc_adjacency = time.perf_counter()
print('\nClock time:', datetime.now().time(),'\nExtracting adjacency graph time:', (toc_adjacency - tic_adjacency))

# In[ ]: # ## 7. Extract Feature Matrices ##
tic_feature = time.perf_counter()

X_train = np.array(nx.attr_matrix(G)[1])
X_train = np.expand_dims(X_train,axis=1)

X_test = X_train[range_test]
X_train = X_train[range_train]

toc_feature = time.perf_counter()
print('\nClock time:', datetime.now().time(),'\nExtracting feature graph time:', (toc_feature - tic_feature))

del G
# In[ ]: # ## 8. Building Renormalization trick ##
mxnet.random.seed(2020)
random.seed(2020)

class SpectralRule(HybridBlock):
    def __init__(self, A, in_units, out_units, activation='relu', **kwargs):
        super().__init__(**kwargs)
        I = nd.eye(*A.shape, ctx = processor)
        A_hat = A.copy() + I

        D = nd.sum(A_hat, axis=0)
        D_inv = D**-0.5
        D_inv = nd.diag(D_inv)

        A_hat = D_inv * A_hat * D_inv
        
        self.in_units, self.out_units = in_units, out_units
        
        with self.name_scope():
            self.A_hat = self.params.get_constant('A_hat', A_hat)
            self.W = self.params.get(
                'W', shape=(self.in_units, self.out_units)
            )
            if activation == 'identity':
                self.activation = lambda X: X
            else:
                self.activation = Activation(activation)

    def hybrid_forward(self, F, X, A_hat, W):
        aggregate = F.dot(A_hat, X)
        propagate = self.activation(
            F.dot(aggregate, W))
        return propagate

# In[ ]: # ## 9. Building classifier for binary classification, Logistic Regression ##
    
class LogisticRegressor(HybridBlock):
    def __init__(self, in_units, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.w = self.params.get(
                'w', shape=(1, in_units)
            )

            self.b = self.params.get(
                'b', shape=(1, 1)
            )

    def hybrid_forward(self, F, X, w, b):
        # Change shape of b to comply with MXnet addition API
        b = F.broadcast_axis(b, axis=(0,1), size=(N, 1))
        y = F.dot(X, w, transpose_b=True) + b

        return F.sigmoid(y)

# In[ ]: # ## 10. Define model foundation ##

def build_features(A, X, n_layer_units, n_activation_functions, activation_function_1, activation_function_2 ):
    hidden_layer_specs = [(n_layer_units, activation_function_1), (n_activation_functions, activation_function_2)] # Format: ( 4 units in layer, 2 activation function) 4,12 relu relu
    in_units = in_units=X.shape[1]
  
    features = HybridSequential()
    with features.name_scope():
        for i, (layer_size, activation_func) in enumerate(hidden_layer_specs):
            layer = SpectralRule(
                A, in_units=in_units, out_units=layer_size, 
                activation=activation_func)
            features.add(layer)

            in_units = layer_size
    return features, in_units

def build_model(A, X, n_layer_units, n_activation_functions, activation_function_1, activation_function_2):
    model = HybridSequential()

    with model.name_scope():
        features, out_units = build_features(A, X, n_layer_units, n_activation_functions, activation_function_1, activation_function_2)
        model.add(features)

        classifier = LogisticRegressor(out_units)
        model.add(classifier)

    model.hybridize()
    model.initialize(Uniform(1), ctx = processor)

    return model, features

# In[ ]: # ## 11. Bulding model ##
    
tic_model = time.perf_counter()

X_1 = nd.eye(*A.shape, ctx = processor)
model, features = build_model(A, X_1, n_layer_units = n_layer_units, n_activation_functions = n_activation_functions, activation_function_1 = activation_function, activation_function_2 = activation_function)

toc_model = time.perf_counter()
print('\nClock time:', datetime.now().time(),'\nBuilding model time:', (toc_model - tic_model))
# In[ ]: # ## 12. Define training process ##

def train(model, features, X, X_train, y_train, epochs, opt):
    cross_entropy = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    trainer = Trainer(model.collect_params(), opt) 

    feature_representations = [features(X).asnumpy()]

    for e in range(1, epochs + 1):
        cum_loss = 0
        cum_preds = []

        for i, x in enumerate(X_train.flatten()):
            y = nd.array(y_train, ctx = processor)[i]
            with autograd.record():
                preds = model(X)[x]
                loss = cross_entropy(preds, y)
            loss.backward()
            trainer.step(1) 

            cum_loss += loss.asscalar()
            cum_preds += [preds.asscalar()]

        feature_representations.append(features(X).asnumpy())
            
        if (e % (epochs//10)) == 0:
            print(f"Epoch {e}/{epochs} -- Loss: {cum_loss: .4f}")
            print(cum_preds)
    return feature_representations

def predict(model, X, nodes):
    preds = model(X)[nodes].asnumpy().flatten()
    return np.where(preds >= 0.5, 1, 0)

# In[ ]: # ## 13. Training Model ##
y_train = nd.array(labels_encoded[range_train,1], ctx = processor)
y_test = labels_encoded[range_test,1]

tic_GCN_train = time.perf_counter()
feature_representations = train(model, features, X_1, X_train, y_train, epochs=n_epochs, opt = 'Signum')

toc_GCN_train = time.perf_counter()
print('\nClock time:', datetime.now().time(),'\nEvaluation of GCN model time:', (toc_GCN_train - tic_GCN_train))

# In[ ]: # ## 14. Testing model ##
tic_GCN_predict = time.perf_counter()
y_pred = predict(model, X_1, X_test)

toc_GCN_predict = time.perf_counter()
print('\nClock time:', datetime.now().time(),'\nPrediction of GCN model time:', (toc_GCN_predict - tic_GCN_predict))

# In[ ]: # ## 15. Evaluating model ##
    
print(classification_report(y_test, y_pred))

print('\nROC/AUC score:', roc_auc_score(y_test, y_pred))

toc_all = time.perf_counter()
print('\nClock time:', datetime.now().time(),'\nFull program time:', (toc_all - tic_all))

with open('feature_progress_file.txt', 'w') as filehandle:
    for listitem in feature_representations:
        filehandle.write('%s\n' % listitem)
        
with open('y_pred.txt', 'w') as filehandle:
    for cell in y_pred:
        filehandle.write('%s\n' % cell)
        
with open('y_test.txt', 'w') as filehandle:
    for cell in y_test:
        filehandle.write('%s\n' % cell)

# Thank you!



