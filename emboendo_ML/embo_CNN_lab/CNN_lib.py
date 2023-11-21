import numpy as np
import tensorflow as tf

# --------------------------------- UTILITIES ---------------------------------

def bin_forecast(pred,label='real label',up_frc='up forecast',down_frc='down forecast'):

    pred_r=np.round(pred[0][0], decimals=4)


    if pred_r >=0.5: print(f'Real: {label} ----> Forecast: {pred_r*100} % {up_frc}')
    else:print(f'Real: {label} ----> Forecast: {(1-pred_r)*100} % {down_frc}')
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


def lib_models(mdl:str,im_input_shp=None):

    mdl_ls=['image_bin_I']

    if mdl in mdl_ls:

        print(f'MODEL:{mdl}')

        if mdl == 'image_bin_I':

            model = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=im_input_shp),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(1, activation='sigmoid')])

        return model

    else: print(f'ERROR: model "{mdl}" not found')