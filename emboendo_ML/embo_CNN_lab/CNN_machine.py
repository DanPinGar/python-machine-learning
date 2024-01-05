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

import numpy as np
import time

class Gen_Model:

    def __init__(self, input_shape,checkpoint_path):

        self.input_shape=input_shape
        self.checkpoint_path=checkpoint_path

        self.check_points()
        self.model = self.def_model()
        self.model_compile()

    def def_model(self):

        model = models.Sequential([
            layers.Conv3D(filters=16, kernel_size=(3, 3, 1), activation='relu',input_shape=self.input_shape),
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
        return model

    def model_compile(self):
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, Y_train,X_eval,Y_eval, epochs):
        
        self.tr_pnt=int(np.shape(Y_train)[0])

        print(f' Number of samples {self.tr_pnt}')

        init_time = time.time()
        self.history=self.model.fit(X_train, Y_train, epochs=epochs,validation_data=(X_eval,Y_eval),callbacks=[self.checkpoint], verbose=0)
        
        end_time = time.time()
        self.trn_time=end_time - init_time

        self.model=load_model(self.checkpoint_path)

        self.model_perf_param(X_eval,Y_eval)
        print(f"Training completed.Elapsed time {self.trn_time} seconds. AUC: {self.roc_auc}")

    
    def model_perf_param(self,X_eval,Y_eval):

        fpr_val, tpr_val, thresholds_val = roc_curve(Y_eval, self.model.predict(X_eval))
        roc_auc = auc(fpr_val, tpr_val)

        self.fpr_val=fpr_val
        self.tpr_val=tpr_val
        self.roc_auc=roc_auc
        
    def check_points(self):

        self.checkpoint = ModelCheckpoint(self.checkpoint_path, save_best_only=True, monitor='val_loss',   mode='min', verbose=0)            


