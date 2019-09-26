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
    if model_dict['strides']==1: 
        stride_lst=[1]*len(conv_sizes) # Default stride is 1 for each convolution.
    else : 
        stride_lst=model_dict['strides']
    
    conv_args = dict(kernel_size=model_dict['kernel_size'], activation='relu', 'padding='same')
    
    for conv_size,strd in zip(conv_sizes,stride_lst):
        h = layers.Conv2D(conv_size, strides=strd, **conv_args)(h)
        if model_dict['double_conv']:  h = layers.Conv2D(conv_size,strides=strd **conv_args)(h)
        if not model_dict['no_pool']: h = layers.MaxPooling2D(pool_size=model_dict['pool_size'])(h)
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
        model_par_dict={'conv_size_list':[10,10,10],'kernel_size':(3,3),'pool_size':(2,2), 'strides':1, 'no_pool':False,
                'inner_dropout':0.1,'dense_size':64,'final_activation':'sigmoid','double_conv':False} 
        
    elif name=='2':
        model_par_dict={'conv_size_list':[10,10,10],'kernel_size':(3,3),'pool_size':(2,2), 'strides':1, 'no_pool':False,
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid','double_conv':False} 

    elif name=='3':
        model_par_dict={'conv_size_list':[20,20,20,20],'kernel_size':(3,3),'pool_size':(2,2), 'strides':1, 'no_pool':False,
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid','double_conv':False} 
        
    elif name=='4':
        model_par_dict={'conv_size_list':[30,30,30,30,40],'kernel_size':(3,3),'pool_size':(2,2), 'strides':1, 'no_pool':False,
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid','double_conv':False} 

    elif name=='5':
        model_par_dict={'conv_size_list':[30,30,30,30,40],'kernel_size':(3,3),'pool_size':(2,2), 'strides':1, 'no_pool':False,
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid','double_conv':False} 

    elif name=='6':
        model_par_dict={'conv_size_list':[128,128,128,128,128],'kernel_size':(3,3),'pool_size':(2,2), 'strides':1, 'no_pool':False,
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid','double_conv':False} 

    elif name=='7':
        model_par_dict={'conv_size_list':[80,80,80,80,80],'kernel_size':(3,3),'pool_size':(2,2), 'strides':1, 'no_pool':False,
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid','double_conv':False} 

    elif name=='8':
        model_par_dict={'conv_size_list':[256,256,256],'kernel_size':(3,3),'pool_size':(4,4), 'strides':1, 'no_pool':False,
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid','double_conv':False}
      
    elif name=='9':
        model_par_dict={'conv_size_list':[60,60],'kernel_size':(3,3),'pool_size':(4,4), 'strides':1, 'no_pool':False,
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid','double_conv':False}
      
    elif name=='10':
        model_par_dict={'conv_size_list':[128,256,256],'kernel_size':(3,3),'pool_size':(4,4), 'strides':1, 'no_pool':False,
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid','double_conv':False}
      
    elif name=='11':
        model_par_dict={'conv_size_list':[10,10,10],'kernel_size':(3,3),'pool_size':(2,2), 'strides':1, 'no_pool':False,
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid','double_conv':True}
    elif name=='12':
        model_par_dict={'conv_size_list':[64,64,64,64],'kernel_size':(3,3),'pool_size':(2,2), 'strides':1, 'no_pool':False,
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid','double_conv':True}
    elif name=='13':
        model_par_dict={'conv_size_list':[128,128,128,128],'kernel_size':(3,3),'pool_size':(2,2), 'strides':1, 'no_pool':False,
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid','double_conv':True}

    ### Strides instead of pools
    elif name=='14':
        model_par_dict={'conv_size_list':[128,128,128,128],'kernel_size':(3,3),'pool_size':(2,2), 'strides':[1,2,2,1], 'no_pool':True,
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid','double_conv':True}
        
    elif name=='15':
        model_par_dict={'conv_size_list':[128,128,128,128],'kernel_size':(3,3),'pool_size':(2,2), 'strides':[1,2,3,1], 'no_pool':True,
        'inner_dropout':None,'dense_size':64,'final_activation':'sigmoid','double_conv':True}
        
    
     ### A custom layered cnn is name=0
    elif name=='0': 
        custom_model=True
        
        inputs = layers.Input(shape=shape)
        h = inputs
        # Convolutional layers     
        conv_sizes=[128,128,256]
        conv_args = dict(kernel_size=(4, 4), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv2D(conv_size, **conv_args)(h)
            h = layers.Conv2D(conv_size, **conv_args)(h)
            h = layers.Conv2D(conv_size, **conv_args)(h)
            h = layers.Conv2D(conv_size, **conv_args)(h)
            h = layers.MaxPooling2D(pool_size=(2, 2))(h)
            #h = layers.Dropout(rate=dropout)(h)

        h = layers.Flatten()(h)
        # Fully connected  layers
        h = layers.Dense(64, activation='relu')(h)
        h = layers.Dropout(rate=dropout)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)
        
    elif name=='20': # Resnet 50
        inputs = layers.Input(shape=shape)
        model = ResNet50(img_input=inputs)
        learn_rate=0.0001
        resnet=True

    elif name=='21': # Resnet 50
        inputs = layers.Input(shape=shape)
        model = ResNet18(img_input=inputs)
        learn_rate=0.0001
        resnet=True
    ## Add more models above
    ############################################
    ####### Compile model ######################
    ############################################

    elif name=='30': 
        custom_model=True
        
        inputs = layers.Input(shape=shape)
        h = inputs
        # Convolutional layers
        h = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(h)
        h = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(h)
        h = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(h)
        h = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(h)
        h = Flatten()(h)
        h = Dense(512, activation='relu')(h)
        y = Dense(1, activation='sigmoid')(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)
    
    
    if resnet:
        print("resnet model name",name)
        opt,loss_fn=optimizers.Adam(lr=learn_rate),'sparse_categorical_crossentropy'
    
    else : ## For non resnet models 
        if name not in ['0','30']:  ### For non-custom models, use prototype function
            outputs,inputs=f_model_prototype(shape,dropout,**model_par_dict)
            
        model = models.Model(inputs, outputs)
        opt=optimizers.Adam(lr=learn_rate)
    
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
    #print("model %s"%name)
    #model.summary()

    return model

