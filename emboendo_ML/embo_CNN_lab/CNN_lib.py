import numpy as np
from tensorflow.keras import layers, models
from tensorflow import keras

# --------------------------------- UTILITIES ---------------------------------

def bin_forecast(pred,label='real label',up_frc='up forecast',down_frc='down forecast'):

    pred_r=np.round(pred[0][0], decimals=4)


    if pred_r >=0.5: 
        
        print(f'Real: {label} ----> Forecast: {up_frc}')
        print(f'Percentage: {(pred_r)*100} %')

    else:

        print(f'Real: {label} ----> Forecast: {down_frc}')
        print(f'Percentage: {(1-pred_r)*100} %')
    print(' ')

 

# --------------------------------- INPUT GENERATORS ---------------------------------

def cnn_data_bin_gen(input,zero=' ', one= ' '):

    x,y=[],[]

    for kk in input.keys():

        x.append(input[kk]['image'])

        if input[kk]['label']==zero: y.append(0)
        elif input[kk]['label']==one: y.append(1)
        else: print('Binary asigment ERROR')

    X,Y = np.array(x),np.array(y)
    return X,Y

# --------------------------------- MODELS ---------------------------------

MODELS={'A':'image_full_bin','B':'image_conv_bin','C':'image_conv_bin_augmentation'}


def lib_models(mdl:str,im_input_shp=None):

    
    if mdl in MODELS.values():

        if mdl == MODELS['A']:

            model = models.Sequential([
                                      layers.Flatten(input_shape=im_input_shp),
                                      layers.Dense(128, activation='relu'),
                                      layers.Dropout(0.2),
                                      layers.Dense(1, activation='sigmoid'),
                                      ])
            print(f"MODEL LOADED: {MODELS['A']}")

        elif mdl == MODELS['B']:

            model = models.Sequential([
                            layers.Rescaling(1./255, input_shape=im_input_shp),
                            layers.Conv2D(32, (3, 3), activation='relu'),
                            layers.MaxPooling2D((2, 2)),
                            layers.Conv2D(64, (3, 3), activation='relu'),
                            layers.MaxPooling2D((2, 2)),
                            layers.Conv2D(64, (3, 3), activation='relu'),
                            layers.MaxPooling2D((2, 2)),
                            layers.Dropout(0.2),
                            layers.Flatten(),
                            layers.Dense(64, activation='relu'),
                            layers.Dense(1, activation='sigmoid'),
                            ])

            print(f"MODEL LOADED: {MODELS['B']}")

        elif mdl == MODELS['C']:

            data_augmentation = keras.Sequential([
                      layers.RandomFlip("horizontal",input_shape=im_input_shp),
                      layers.RandomRotation(0.1),
                      layers.RandomZoom(0.1),])
                            
            model = models.Sequential([
                            data_augmentation,
                            layers.Rescaling(1./255),
                            layers.Conv2D(32, (3, 3), activation='relu'),
                            layers.MaxPooling2D((2, 2)),
                            layers.Conv2D(64, (3, 3), activation='relu'),
                            layers.MaxPooling2D((2, 2)),
                            layers.Conv2D(64, (3, 3), activation='relu'),
                            layers.MaxPooling2D((2, 2)),
                            layers.Dropout(0.2),
                            layers.Flatten(),
                            layers.Dense(64, activation='relu'),
                            layers.Dense(1, activation='sigmoid'),
                            ])

            print(f"MODEL LOADED: {MODELS['C']}")

        print(' ')

        return model

    else: print(f'ERROR: model "{mdl}" not found')