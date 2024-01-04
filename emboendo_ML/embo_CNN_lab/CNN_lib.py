import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
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

def shuffle(x,y,r):

    idx = np.random.permutation(len(x))
    xx,yy=x[idx],y[idx]
    rr=list(np.take(r, idx))

    return xx,yy,rr


# --------------------------------- PLOTS ---------------------------------

def plot_train_eval(history,epochs):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(14,8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')  
    plt.ylabel('Accuracy')  

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range[1:-1], loss[1:-1], label='Training Loss')
    plt.plot(epochs_range[1:-1], val_loss[1:-1], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')  
    plt.ylabel('Loss')  
    plt.show()

def plot_roc_curve(fpr_val,tpr_val,roc_auc_false):

    plt.figure(figsize=(8, 8))

    plt.plot(fpr_val, tpr_val, color='darkorange', lw=2, label=f'AUC = {roc_auc_false:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate(orange)')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Validation Data')
    plt.legend(loc='lower right')
    plt.show()



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


def vid_d_bin_gen(input, zero=' ', one= ' ', pad_type='loop',im_pad_type='center'):

    x,y,dims=[],[],[]

    max_dim_h = max(value['dimHW'][0] for value in input.values())
    max_dim_w = max(value['dimHW'][1] for value in input.values())
    max_frm_n=max([len(R['image']) for R in input.values()])

    for kk,value in input.items():

        vd=padd_im(value['image'].copy().astype(np.float32),max_dim_h,max_dim_w,type=im_pad_type)
        vd=vd/vd.max()
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
    

# --------------------------------- PADDING ---------------------------------

def padd_im(vd_,max_h,max_w,type=''):

    vd=[]

    if type == 'border':

        for vid in vd_:

            fil = max_h - vid.shape[0]
            col = max_w - vid.shape[1]
            n = np.pad(vid, ((0, fil), (0, 0)), mode='constant', constant_values=0)
            n = np.pad(n, ((0, 0), (0, col)), mode='constant', constant_values=0)
        
            vd.append(n)
    
    if type == 'center':

        for vid in vd_:

            fil = max_h - vid.shape[0]
            col = max_w - vid.shape[1]

            n = np.pad(vid, ((fil // 2, (fil + 1) // 2), (col // 2, (col + 1) // 2)), mode='constant', constant_values=0)

            vd.append(n)

    return np.array(vd)


def pad_f(video,max_frm_n,frames_video,max_dim_h, max_dim_w, type=''):

    if type=='zeros':padding = np.zeros((max_frm_n - frames_video, max_dim_h, max_dim_w, 1))#, dtype=float)
    
    elif type=='loop':

        padding = None  
        frames_actual = frames_video

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


# --------------------------------- DATA AUGMENTATION ---------------------------------

def main_aug_f(n,X,Y,R,label=1):

    aug_R=[]
    aug_X=[]

    if label == 1:aug_Y=np.ones(n)
    elif label == 0: aug_Y=np.zeros(n)

    if label == 1:idx_Y = np.where(Y == 1)[0]
    if label == 0:idx_Y = np.where(Y == 0)[0]

    idx_Y = np.random.choice(idx_Y, size=n, replace=False)

    for index in idx_Y:

        video=random_flip(X[index])
        aug_X.append(video)
        aug_R.append('AUG_'+R[index])

    aug_X=np.array(aug_X)

    return aug_X,aug_Y,aug_R


def random_flip(matrix):

    rnd_number= np.random.random()
    video = tf.image.convert_image_dtype(matrix, dtype=tf.float32)

    if rnd_number <0.333:video = tf.image.flip_up_down(video)
    elif rnd_number <0.666:video = tf.image.flip_left_right(video)
    else:
        
        video = tf.image.flip_up_down(video)
        video = tf.image.flip_left_right(video)

    video = tf.clip_by_value(video, 0.0, 1.0)

    return video


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