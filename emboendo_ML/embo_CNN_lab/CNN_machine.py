import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from sklearn.metrics import roc_curve, auc

import CNN_lib
import CNN_utilities

import os
import numpy as np
import time


def info_print(iter,trains_n,name):

    iter+=1
    print(' ')
    print(f'-------------- {name}: ITERATION {iter}/{trains_n} COMPLETED --------------')
    print(' ')

    return iter


class Gen_Model:

    def __init__(self,aug_params,patients_d_df,name='NO_NAME',path='NO_PATH',input_shape=None,samples=None):

        self.name=name
        self.checkpoint_path=path
        self.input_shape=input_shape
        self.samples=samples
        self.aug_params=aug_params
        self.patients_d_df=patients_d_df

        self.model = self.def_model()
        self.model_compile()

    def def_model(self,conv_str_s = (1, 2, 2) ,conv_str_t = (2, 1, 1) ):
        
        model = models.Sequential([
            layers.Conv3D(filters=32, kernel_size=(1, 3, 3), activation='relu',input_shape=self.input_shape,strides=conv_str_s),
            layers.MaxPooling3D(pool_size=(1, 2, 2)),
            layers.Conv3D(filters=32, kernel_size=(1, 3, 3),  activation='relu',strides=conv_str_s),
            layers.MaxPooling3D(pool_size=(1, 2, 2)),
            layers.Conv3D(filters=64, kernel_size=(1, 3, 3),  activation='relu'),
            layers.MaxPooling3D(pool_size=(2, 1, 1)),
            layers.Conv3D(filters=128, kernel_size=(8, 1, 1),  activation='relu',strides=conv_str_t),
            layers.MaxPooling3D(pool_size=(2, 2, 2)),
            layers.Flatten(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu'),
            layers.Dense(1,activation='sigmoid')])

        return model

    def model_compile(self):
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    def model_perf_param(self,X_eval,Y_eval):

        fpr_val, tpr_val, thresholds_val = roc_curve(Y_eval, self.model.predict(X_eval))
        roc_auc = auc(fpr_val, tpr_val)

        return  roc_auc
         

    def train_model(self, x, y,r,epochs,trains_n,patiens_split):

        iter=0
        histories,roc_aucs=[],[]

        check_points_p = [self.checkpoint_path+'_'+str(n)+'.h5' for n in range(trains_n)]
        checkpoints=[ModelCheckpoint(pp, save_best_only=True, monitor='val_loss',   mode='min', verbose=1) for pp in check_points_p]

        for chk_p in checkpoints:
            
            checkpoint = ModelCheckpoint(self.checkpoint_path, save_best_only=True, monitor='val_loss',   mode='min', verbose=0,load_weights_on_restart=False)  
            X_d,Y_d,recs=CNN_lib.shuffle(x.copy(),y.copy(),r.copy())                                                                                                                                                      # SHUFFLE

            Xx_train_spl, X_eval, Yy_train_spl, Y_eval ,recs_train,recs_eval=CNN_utilities.random_split_by_patients(self.patients_d_df,recs,X_d,Y_d, val_pat_0=patiens_split[0], val_pat_1=patiens_split[1])

            X_train_spl=Xx_train_spl[:self.samples, :, :, :, :]
            Y_train_spl=Yy_train_spl[:self.samples]
            rcs_trn=recs_train[:self.samples]

            X_train_aug, Y_train_aug, r_t = CNN_lib.d_augmentation_logic_encapsulation(X_train_spl,Y_train_spl,rcs_trn,self.aug_params)                                                                   # AUGMENTATION

            X_train,Y_train,recs_train_f=CNN_lib.shuffle(X_train_aug,Y_train_aug,r_t)                                                                                                                       # SHUFFLE
            
            hist=self.model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_eval,Y_eval),callbacks=[chk_p])                                                                                                         # TRAIN
            
            iter=info_print(iter,trains_n,self.name)

            histories.append(hist)
            roc_auc_iter=self.model_perf_param(X_eval,Y_eval)
            roc_aucs.append(roc_auc_iter)

            

        self.histories=histories
        self.roc_aucs=roc_aucs
            