## modify this script to change the model.
### Add models with a new index.

from tensorflow.keras import layers, models, optimizers, callbacks  # or tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

### Import the modules for resnet50
# from resnet50 import *
# from resnet18 import *

### Defining all the models tried in the study


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
    
    inputs = layers.Input(shape=shape)
    h = inputs
    
    # Choose model
    
    if name=='1':
        # Convolutional layers
        conv_sizes=[10,10,10]
        conv_args = dict(kernel_size=(3, 3), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv2D(conv_size, **conv_args)(h)
            h = layers.MaxPooling2D(pool_size=(2, 2))(h)
            h = layers.Dropout(rate=dropout)(h)
        h = layers.Flatten()(h)

        # Fully connected  layers
        h = layers.Dense(64, activation='relu')(h)
        h = layers.Dropout(rate=dropout)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)
        
    model = models.Model(inputs, outputs)
    #### change loss function for non-resnet models since 'sparse_categorical_crossentropy' throws up an error.
    opt=optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
    #print("model %s"%name)
    #model.summary()

    return model