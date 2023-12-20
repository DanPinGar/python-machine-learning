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

def im_d_bin_gen(input,zero=' ', one= ' '):

    x,y=[],[]

    for kk in input.keys():

        x.append(input[kk]['image'])

        if input[kk]['label']==zero: y.append(0)
        elif input[kk]['label']==one: y.append(1)
        else: print('Binary asigment ERROR')

    X,Y = np.array(x),np.array(y)

    return X,Y

def padd_im(vd_,max_h,max_w):
    vd=[]
    for vid in vd_:

        fil = max_h - vid.shape[0]
        col = max_w - vid.shape[1]
        n = np.pad(vid, ((0, fil), (0, 0)), mode='constant', constant_values=0)
        n = np.pad(n, ((0, 0), (0, col)), mode='constant', constant_values=0)
       
        vd.append(n)
        
    return np.array(vd)

def vid_d_bin_gen(input, zero=' ', one= ' ', pad_type='loop'):

    x,y,dims=[],[],[]

    max_dim_h = max(value['dimHW'][0] for value in input.values())
    max_dim_w = max(value['dimHW'][1] for value in input.values())
    max_frm_n=max([len(R['image']) for R in input.values()])

    for kk,value in input.items():

        vd=padd_im(value['image'].copy(),max_dim_h,max_dim_w)
        vd = vd[:, :, :, np.newaxis]
        x.append(vd)
        dims.append(value['dimHW'])

        if value['label']==zero: y.append(0)
        elif value['label']==one: y.append(1)
        else: print('Binary asigment ERROR')

    for ii, video in enumerate(x):
    
        frames_actual = video.shape[0]
        if frames_actual < max_frm_n:
            
            padding =pad_f(video,max_frm_n,frames_actual,max_dim_h, max_dim_w, type=pad_type)
            x[ii] = np.concatenate([video, padding], axis=0)

    X,Y = np.array(x),np.array(y)
    return X,Y,max_frm_n,max_dim_h,max_dim_w 
    

def pad_f(video,max_frm_n,frames_video,max_dim_h, max_dim_w, type=''):

    if type=='zeros':padding = np.zeros((max_frm_n - frames_video, max_dim_h, max_dim_w, 1),dtype=np.uint8)
    
    elif type=='loop':

        while max_frm_n > frames_actual:

            frames_adding = max_frm_n - frames_actual

            if frames_adding > frames_video:

                if padding is None:padding = video.copy()  
                else:padding = np.concatenate([padding, video], axis=0)

            else:

                if padding is None:padding = video[0:frames_adding].copy()
                else:padding = np.concatenate([padding, video[0:frames_adding]], axis=0)

            frames_actual = frames_video + padding.shape[0]
        
    return padding

# --------------------------------- MODELS ---------------------------------

MODELS={'A':'image_full','B':'image_conv','C':'image_conv_augmentation','D':'video_conv2D','E':'video_conv3D'}


def lib_models(mdl:str,im_input_shp=None):

    
    if mdl in MODELS.values():

        if mdl == MODELS['A']:

            model = models.Sequential([
                                      layers.Flatten(input_shape=im_input_shp),
                                      layers.Dense(128, activation='relu'),
                                      layers.Dropout(0.2),
                                      layers.Dense(128, activation='relu'),
                                      layers.Dropout(0.2),
                                      layers.Dense(128, activation='relu'),
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

        elif mdl == MODELS['D']:

            model = models.Sequential([layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu'), input_shape=im_input_shp),
                             layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
                             layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu')),
                             layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
                             layers.TimeDistributed(layers.Flatten()),
                             layers.LSTM(4),
                             layers.Dense(1, activation='sigmoid'),
                              ])

            print(f"MODEL LOADED: {MODELS['D']}")

        elif mdl == MODELS['E']:

            model = models.Sequential([
                            layers.Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu',input_shape=im_input_shp),
                            layers.MaxPooling3D(pool_size=(2, 2, 2)),
                            layers.Conv3D(filters=32, kernel_size=(1, 3, 3),  activation='relu'),
                            layers.MaxPooling3D(pool_size=(2, 2, 2)),
                            layers.Conv3D(filters=32, kernel_size=(1, 3, 3),  activation='relu'),
                            layers.MaxPooling3D(pool_size=(2, 2, 2)),
                            layers.Conv3D(filters=64, kernel_size=(1, 3, 3),  activation='relu'),
                            layers.Flatten(),
                            layers.Dense(64, activation='relu'),
                            layers.Dense(1,activation='sigmoid')
                            ])

            print(f"MODEL LOADED: {MODELS['E']}")

        print(' ')

        return model

    else: print(f'ERROR: model "{mdl}" not found')