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

def stats(data,title=' ',rnd=3):

    media = np.mean(data)
    mediana = np.median(data)
    desviacion_estandar = np.std(data)
    varianza = np.var(data)
    minimo = np.min(data)
    maximo = np.max(data)
    percentil_25 = np.percentile(data, 25)
    percentil_75 = np.percentile(data, 75)

    print(' ')
    print(title,':')
    print(f'Valor medio:{round(media,rnd)}')
    print(f'Desviación estandar:{round(desviacion_estandar,rnd)}')
    print(f'Varianza:{round(varianza,rnd)}')
    print(f'Máximo:{round(maximo,rnd)}')
    print(f'Mínimo:{round(minimo,rnd)}')
    print(f'Mediana:{round(mediana,rnd)}')
    print(f'Percentil 25:{round(percentil_25,rnd)}')
    print(f'Percentil 75:{round(percentil_75,rnd)}')




# --------------------------------- PLOTS ---------------------------------

def list_plot(data,title='Plot',ylabel='y'):

    plt.plot(data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()

def simple_plot(x,y,title='Plot',xlabel='x',ylabel='y'):

    plt.plot(x,y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def multiple_plot(x,y11,y12,y21,y22,title= ' '):

    plt.plot(x, y11, linestyle='--', label='Training Accuracy',color='b')
    plt.plot(x, y12, linestyle='--', label='Validation Accuracy',color='r')
    plt.plot(x, y21, linestyle='-', label='Training Loss',color='b')
    plt.plot(x, y22, linestyle='-', label='Validation Loss',color='r')
    plt.legend(loc='lower right')
    plt.title(title)
    plt.xlabel('Epochs')  
    plt.ylabel('Accuracy / Loss')  
   
    plt.show()

def double_plot(x,y11,y12,y21,y22):

    plt.figure(figsize=(14,8))
    plt.subplot(1, 2, 1)
    plt.plot(x, y11, label='Training Accuracy')
    plt.plot(x, y12, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')  
    plt.ylabel('Accuracy')  

    plt.subplot(1, 2, 2)
    plt.plot(x, y21, label='Training Loss')
    plt.plot(x, y22, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')  
    plt.ylabel('Loss')  
    plt.show()


def data_bars_plot(zeros_count,ones_count,zeros_count_val,ones_count_val):

    labels = ['0 Train', '1 Train','0 Validation', '1 Validation']
    plt.bar(labels, [zeros_count, ones_count,zeros_count_val,ones_count_val], color=['green', 'blue','green', 'blue'])

    for i, count in enumerate([zeros_count, ones_count,zeros_count_val,ones_count_val]): plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.title('Data Labels Before Augmentation')
    plt.show()


def plot_train_eval(history,epochs, type='together'):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)
    
    if type == 'together': multiple_plot(epochs_range,acc,val_acc,loss,val_loss,title= 'Training and Validation Accuracy & Loss')
    if type == 'separated': double_plot(epochs_range,acc,val_acc,loss,val_loss)

    

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

def main_aug_f(n,X,Y,R,label=1,typ='Flip'):

    """
    typ: Flip, Rotation, Contrast, Brightness, Noise
    """
    aug_R,aug_X = [],[]

    if label == 1:aug_Y=np.ones(n)
    elif label == 0: aug_Y=np.zeros(n)

    if label == 1:idx_Y = np.where(Y == 1)[0]
    if label == 0:idx_Y = np.where(Y == 0)[0]

    idx = np.random.choice(idx_Y, size=n, replace=False)

    for index in idx:


        if typ=='Flip':

            aug_R.append('FLIP_'+R[index])
            video=random_flip(X[index])
        

        if typ=='Contrast':

            aug_R.append('CNTR_'+R[index])
            video=random_contrast(X[index])

        if typ=='Brightness':

            aug_R.append('BRGHT_'+R[index])
            video=random_brightness(X[index])

        if typ=='Rotation':

            aug_R.append('ROT_'+R[index])
            video=random_rotation(X[index])
        
        aug_X.append(video)

    aug_X=np.array(aug_X)

    return aug_X,aug_Y,aug_R


def random_flip(matrix):

    rnd_number= np.random.random()
    video = tf.image.convert_image_dtype(matrix, dtype=tf.float32)

    if rnd_number <0.5:video = tf.image.flip_up_down(video)
    else: video = tf.image.flip_left_right(video)
    
    video = tf.clip_by_value(video, 0.0, 1.0)

    return video

def random_rotation(matrix):
    
    rnd_number = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
    video = tf.image.convert_image_dtype(matrix, dtype=tf.float32)
    
    video = tf.image.rot90(video, k=rnd_number + 1)  
    video = tf.clip_by_value(video, 0.0, 1.0)

    return video

def random_contrast(matrix):

    video = tf.image.convert_image_dtype(matrix, dtype=tf.float32)
    video = tf.image.random_contrast(video, lower=0.5, upper=10)
    video = tf.clip_by_value(video, 0.0, 1.0)

    return video

def random_brightness(matrix):

    video = tf.image.convert_image_dtype(matrix, dtype=tf.float32)
    video = tf.image.random_brightness(video, max_delta=0.9)
    video = tf.clip_by_value(video, 0.0, 1.0)

    return video

def d_augmentation_logic_encapsulation(X_train_spl,Y_train_spl,recs_train,params):

    zeros_count = np.sum(Y_train_spl == 0)
    ones_count = np.sum(Y_train_spl == 1)
   
    nf1,nf0 = round(params['f1']*ones_count),round(params['f0']*zeros_count)
    nr1,nr0 = round(params['r1']*ones_count),round(params['r0']*zeros_count)
    nc1,nc0 = round(params['c1']*ones_count),round(params['c0']*zeros_count)
    nb1,nb0 = round(params['b1']*ones_count),round(params['b0']*zeros_count)

    Flip_X_1,Flip_Y_1,Flip_recs_1= main_aug_f(nf1,X_train_spl,Y_train_spl,recs_train,label=1,typ='Flip')
    Flip_X_0,Flip_Y_0,Flip_recs_0= main_aug_f(nf0,X_train_spl,Y_train_spl,recs_train,label=0,typ='Flip')

    Rot_X_1,Rot_Y_1,Rot_recs_1= main_aug_f(nr1,X_train_spl,Y_train_spl,recs_train,label=1,typ='Rotation')
    Rot_X_0,Rot_Y_0,Rot_recs_0= main_aug_f(nr0,X_train_spl,Y_train_spl,recs_train,label=0,typ='Rotation')

    Cntr_X_1,Cntr_Y_1,Cntr_recs_1= main_aug_f(nc1,X_train_spl,Y_train_spl,recs_train,label=1,typ='Contrast')
    Cntr_X_0,Cntr_Y_0,Cntr_recs_0= main_aug_f(nc0,X_train_spl,Y_train_spl,recs_train,label=0,typ='Contrast')

    Bgr_X_1,Bgr_Y_1,Bgr_recs_1= main_aug_f(nb1,X_train_spl,Y_train_spl,recs_train,label=1,typ='Brightness')
    Bgr_X_0,Bgr_Y_0,Bgr_recs_0= main_aug_f(nb0,X_train_spl,Y_train_spl,recs_train,label=0,typ='Brightness')

    X_train_spl = np.concatenate((X_train_spl, Flip_X_1,Flip_X_0,Rot_X_1,Rot_X_0,Cntr_X_1,Cntr_X_0,Bgr_X_1,Bgr_X_0), axis=0)
    Y_train_spl = np.concatenate((Y_train_spl, Flip_Y_1,Flip_Y_0,Rot_Y_1,Rot_Y_0,Cntr_Y_1,Cntr_Y_0,Bgr_Y_1,Bgr_Y_0))

    recs_train = recs_train+Flip_recs_1+Flip_recs_0+Rot_recs_1+Rot_recs_0+Cntr_recs_1+Cntr_recs_0+Bgr_recs_1+Bgr_recs_0


    return X_train_spl, Y_train_spl, recs_train


# --------------------------------- MODELS ---------------------------------

MODELS={'A':'image_full','B':'image_conv','C':'image_conv_augmentation','D':'video_conv2D','E':'video_conv3D','F':'video_conv3D_2','F':'conv3D_2D+1'}


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
                    layers.Conv3D(filters=16, kernel_size=(3, 3, 2), activation='relu',input_shape=im_input_shp),
                    layers.MaxPooling3D(pool_size=(4, 4, 4)),
                    layers.Conv3D(filters=32, kernel_size=(3, 3, 2),  activation='relu'),
                    layers.Flatten(),
                    layers.Dropout(0.2),   # si lo subes baja muchisimo
                    layers.Dense(64, activation='relu'),
                    layers.Dense(1,activation='sigmoid')
                    ])

            print(f"MODEL LOADED: {MODELS['E']}")

        elif mdl == MODELS['F']:

            model = models.Sequential([
                layers.Conv3D(filters=16, kernel_size=(3, 3, 2), activation='relu',input_shape=im_input_shp),
                layers.MaxPooling3D(pool_size=(2, 2, 2)),
                layers.Conv3D(filters=32, kernel_size=(3, 3, 2),  activation='relu'),
                layers.Dropout(0.2),
                layers.MaxPooling3D(pool_size=(2, 2, 2)),
                layers.Conv3D(filters=32, kernel_size=(3, 3, 2),  activation='relu'),
                layers.MaxPooling3D(pool_size=(2, 2, 2)),
                layers.Conv3D(filters=16, kernel_size=(3, 3, 1),  activation='relu'),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(1,activation='sigmoid')
                ])

            print(f"MODEL LOADED: {MODELS['F']}")

        elif mdl == MODELS['G']:

            model = models.Sequential([
                layers.Conv3D(filters=16, kernel_size=(3, 3, 1), activation='relu',input_shape=(max_frm_n, HEIGHT, WIDTH, 1)),
                layers.MaxPooling3D(pool_size=(2, 2, 2)),
                layers.Conv3D(filters=32, kernel_size=(3, 3, 1),  activation='relu'),
                layers.MaxPooling3D(pool_size=(2, 2, 2)),
                layers.Conv3D(filters=32, kernel_size=(3, 3, 1),  activation='relu'),
                layers.MaxPooling3D(pool_size=(2, 2, 2)),
                layers.Conv3D(filters=16, kernel_size=(1, 1, 5),  activation='relu'),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dense(1,activation='sigmoid')
                ])
            
            print(f"MODEL LOADED: {MODELS['F']}")

        print(' ')

        return model

    else: print(f'ERROR: model "{mdl}" not found')