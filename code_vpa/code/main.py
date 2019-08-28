# The main code for training a CNN on the ATLAS SUSY 2D images
# Created by Venkitesh Ayyar. August 19, 2019

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import time
import argparse
import sys

import subprocess as sp
import pickle
import yaml


## M-L modules
import tensorflow.keras
from tensorflow.keras import layers, models, optimizers, callbacks  # or tensorflow.keras as keras
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tensorflow.keras.models import load_model

## modules from other files
from models import *
from modules import *

    



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and test CNN for ATLAS SUSY data", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    
    add_arg('--config','-c', type=str, default='config.yaml',help='The .yaml file that stores the configuration.')
    add_arg('--train','-tr',  action='store_false' ,dest='train_status' ,help='Has the model been trained?')
    add_arg('--test', '-ts',  action='store_false' ,dest='test_status'  ,help='Has the model been tested?')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--gpu', type=str, choices=['None','maeve','cori'],default='None', help='Whether using gpu, if so, maeve or cori.')    
    add_arg('--model_list', '-mdlst', nargs='+', type=int, dest='mod_lst',help=' Enter the list of model numbers to test ', required=True)

    return parser.parse_args()


if __name__=='__main__':

    args=parse_args()
    print(args)
    ## Note: --train means models needs to be trained. hence train_status=False
    train_status,test_status=args.train_status,args.test_status
    model_lst=args.mod_lst
    ##### Stuff for GPU #####
    if args.gpu!='None': 
        script_loc={'maeve':'/home/vpa/standard_scripts/','cori':'/global/u1/v/vpa/standard_scripts/'}
        ## special imports for setting keras and tensorflow variables.
        sys.path.insert(0,script_loc[args.gpu])
        from keras_tf_parallel_variables import configure_session
        ### Set tensorflow and keras variables
        configure_session(intra_threads=32, inter_threads=2, blocktime=1, affinity='granularity=fine,compact,1,0')
        # Limit GPU usage to just one.
        #export CUDA_DEVICE_ORDER=PCI_BUS_ID
        #export CUDA_VISIBLE_DEVICES=1


    t1=time.time()
    ### Read configuration ###
    config_file=args.config
    config_dict=f_load_config(config_file)
    print(config_dict)

    batch_size=config_dict['training']['batch_size']
    num_epochs=config_dict['training']['n_epochs']

    ### Extract the training and validation data ###
    data_dir=config_dict['data_dir']
    #### Training data
    filename=data_dir+'train.h5'
    train_data_dict=f_get_data(filename)

    #### Test_data
    filename=data_dir+'val.h5'
    test_data_dict=f_get_data(filename)

    train_x,train_y,train_wts=train_data_dict['images'],train_data_dict['labels'],train_data_dict['weights']
    print(train_x.shape,train_y.shape,train_wts.shape)
    t2=time.time()
    print("Time taken to read files",t2-t1)

    model_save_dir=config_dict['output_dir']
    
    for i in model_lst:
        model_name=str(i)
        ### Compile model ###
        fname_model,fname_history='mdl_{0}_weights.h5'.format(model_name),'history_{0}.pickle'.format(model_name)

        ## Define model 
        model=f_define_model(config_dict,name=model_name)

        ### Train model ###
        history=f_train_model(model,train_x,train_y,model_weights=fname_model,num_epochs=num_epochs,batch_size=batch_size)

        ### Save model and history ###
        fname_model,fname_history='model_{0}.h5'.format(model_name),'history_{0}.pickle'.format(model_name)

        model.save(model_save_dir+fname_model)
        with open(model_save_dir+fname_history, 'wb') as f:
            pickle.dump(history, f)
