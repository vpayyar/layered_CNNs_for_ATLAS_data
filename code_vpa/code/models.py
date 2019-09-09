## modify this script to change the model.
### Add models with a new index.

from tensorflow.keras import layers, models, optimizers, callbacks  # or tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

### Import the modules for resnet50
from resnet50 import *
from resnet18 import *

### Defining all the models tried in the study


def f_model_prototype(shape,dropout,**model_dict):
    '''
    General prototype for layered CNNs
    '''
    
    inputs = layers.Input(shape=shape)
    h = inputs
    # Convolutional layers
    conv_sizes=model_dict['conv_size_list'] # Eg. [10,10,10]
    conv_args = dict(kernel_size=model_dict['kernel_size'], activation='relu', padding='same')
    for conv_size in conv_sizes:
        h = layers.Conv2D(conv_size, **conv_args)(h)
        h = layers.MaxPooling2D(pool_size=model_dict['pool_size'])(h)
        ## inner_dropout is None or a float
        if model_dict['inner_dropout']!=None: h = layers.Dropout(rate=model_dict['inner_dropout'])(h)
    h = layers.Flatten()(h)

    # Fully connected  layers
    h = layers.Dense(model_dict['dense_size'], activation='relu')(h)
    h = layers.Dropout(rate=dropout)(h)

    # Ouptut layer
    outputs = layers.Dense(1, activation=model_dict['final_activation'])(h)    
    return outputs,inputs
    
def f_define_model(config_dict,name='1'):
    '''
    Function that defines the model and compiles it.
    '''
    ### Extract info from the config_dict
    shape=config_dict['model']['input_shape']
    learn_rate=config_dict['optimizer']['lr']
    loss_fn=config_dict['training']['loss']
    metrics=config_dict['training']['metrics']
    dropout=config_dict['model']['dropout']
    

    resnet=False ### Variable storing whether the models is resnet or not. This is needed for specifying the loss function.    
    
    # Choose model
    if name=='1':
        model_par_dict={'conv_size_list':[10,10,10],'kernel_size':(3,3),'pool_size':(2,2),
        'inner_dropout':0.1,'dense_size':64,'final_activation':'sigmoid'} 
        
    elif name=='2':
        model_par_dict={'conv_size_list':[10,10,10],'kernel_size':(3,3),'pool_size':(2,2),
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid'} 

    elif name=='3':
        model_par_dict={'conv_size_list':[20,20,20,20],'kernel_size':(3,3),'pool_size':(2,2),
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid'} 
        
    elif name=='4':
        model_par_dict={'conv_size_list':[10,10,10],'kernel_size':(3,3),'pool_size':(4,4),
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid'}
        
    elif name=='5':
        model_par_dict={'conv_size_list':[10,10,10],'kernel_size':(4,4),'pool_size':(4,4),
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid'} 
    
    ### A custom layered cnn is name=0
    elif name=='0': 
        custom_model=True
        
        inputs = layers.Input(shape=shape)
        h = inputs
        # Convolutional layers        
        conv_sizes=[10,10,10]
        conv_args = dict(kernel_size=(4, 4), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv2D(conv_size, **conv_args)(h)
            h = layers.MaxPooling2D(pool_size=(4, 4))(h)
            h = layers.Dropout(rate=dropout)(h)

        h = layers.Flatten()(h)
        # Fully connected  layers
        h = layers.Dense(64, activation='relu')(h)
        h = layers.Dropout(rate=dropout)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)
        
    elif name=='20': # Resnet 50
        inputs = layers.Input(shape=shape)
        model = ResNet50(img_input=inputs)
        #learn_rate=0.00001
        resnet=True

    elif name=='30': # Resnet 50
        model = ResNet18(img_input=inputs)
        #learn_rate=0.00001
        resnet=True
    ## Add more models above
    ############################################
    ####### Compile model ######################
    ############################################

    
    if resnet:
        print("resnet model name",name)
        opt,loss_fn=optimizers.Adam(lr=learn_rate),'sparse_categorical_crossentropy'
    
    else : ## For non resnet models 
        if name!='0':     outputs,inputs=f_model_prototype(shape,dropout,**model_par_dict)
            
        model = models.Model(inputs, outputs)
        opt=optimizers.Adam(lr=learn_rate)
    
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
    #print("model %s"%name)
    #model.summary()

    return model

