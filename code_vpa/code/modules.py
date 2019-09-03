### Modules used for training 

import numpy as np
import os
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import yaml
import h5py


from sklearn.metrics import roc_curve, auc, roc_auc_score
from models import f_define_model
from tensorflow.keras import callbacks




def f_load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def f_get_data(filename):
    '''
    Function to get data from hdf5 files into images, labels and weights.
    '''
    try: 
        hf = h5py.File(filename)

    except Exception as e:
        print(e)
        print("Name of file",filename)
        raise SystemError

    idx=50000
    #idx=None
    images = np.expand_dims(hf['all_events']['hist'][:idx], -1)
    labels = hf['all_events']['y'][:idx]
    weights = hf['all_events']['weight'][:idx]
    weights = np.log(weights+1)

    keys=['images','labels','weights']
    values_dict=dict(zip(keys,[images,labels,weights]))

    return values_dict


def f_train_model(model,inpx,inpy,model_weights,num_epochs=5,batch_size=64):
    '''
    Train model. Returns just history.history
    '''
    cv_fraction=0.33 # Fraction of data for cross validation
    
    history=model.fit(x=inpx, y=inpy,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    callbacks = [callbacks.EarlyStopping(monitor='val_loss', min_delta=0,patience=20, verbose=1),
                                 callbacks.ModelCheckpoint(model_weights, save_best_only=True, monitor='val_loss', mode='min') ],
                    validation_split=cv_fraction,
                    shuffle=True
                )
    
    print("Number of parameters",model.count_params())
    
    return history.history



def f_test_model(model,xdata,ydata):
    '''
    Test model and return array with predictions
    '''
    
#     model.evaluate(xdata,ydata,sample_weights=wts,verbose=1)
    y_pred=model.predict(xdata,verbose=1)
    ### Ensure prediction has the same size as labelled data.
    assert(ydata.shape[0]==y_pred.shape[0]),"Data %s and prediction arrays %s are not of the same size"%(test_y.shape,y_pred.shape)
       
    ##Condition for the case when the prediction is a 2column array 
    ## This complicated condition is needed since the array has shape (n,1) when created, but share (n,) when read from file.
    if (len(y_pred.shape)==2 and y_pred.shape[1]==2) : y_pred=y_pred[:,1]

    return y_pred

def f_plot_learning(history):
    '''Plot learning curves : Accuracy and Validation'''
    fig=plt.figure()
    # Plot training & validation accuracy values
    fig.add_subplot(2,1,1)
    xlim=len(history['acc'])
    
    plt.plot(history['acc'],label='Train',marker='o')
    plt.plot(history['val_acc'],label='Validation',marker='*')
#     plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0,xlim,2))
    
    # Plot loss values
    fig.add_subplot(2,1,2)
    plt.plot(history['loss'],label='Train',marker='o')
    plt.plot(history['val_loss'],label='Validation',marker='*')
#     plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0,xlim,2))

    plt.legend(loc='best')



def f_plot_roc_curve(fpr,tpr):
    '''
    Module for roc plot and printing AUC
    '''
    plt.figure()
    # plt.plot(fpr,tpr)
    plt.scatter(fpr,tpr)
    plt.semilogx(fpr, tpr)
  # Zooms
    plt.xlim([10**-7,1.0])
    plt.ylim([0,1.0])
    # y=x line for comparison
    x=np.linspace(0,1,num=500)
    plt.plot(x,x)
#     plt.xscale('log')
#     plt.xlim(1e-10,1e-5)
    plt.show()

    # AUC 
    auc_val = auc(fpr, tpr)
    print("AUC: ",auc_val)
    

