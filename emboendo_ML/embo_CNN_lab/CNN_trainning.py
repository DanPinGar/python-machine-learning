import gc
gc.collect()

#import fig_lib 
#import CNN_lib
#import CNN_utilities
import numpy as np
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

pkl_train_p='C:\PROJECTS\emboendo\CNN/train_d.pkl'       

with open(pkl_train_p, 'rb') as pikle_file:

    save_trains= pickle.load(pikle_file)

X_train,Y_train =  save_trains[0],save_trains[1]

max_frm_n, HEIGHT, WIDTH=np.shape(X_train)[1],np.shape(X_train)[2],np.shape(X_train)[3]

test_size=0.2
X_train, X_eval, Y_train, Y_eval = train_test_split(X_train, Y_train, test_size=test_size) #,random_state=42)

print('Train data:',len(Y_train),'Evaluation data:',len(Y_eval))

print(' Train data shape:', np.shape(X_train),' Validation data shape:', np.shape(X_eval))


model = models.Sequential([
          layers.Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu',input_shape=(max_frm_n, HEIGHT, WIDTH, 1)),
          layers.MaxPooling3D(pool_size=(2, 2, 2)),
          layers.Conv3D(filters=16, kernel_size=(1, 3, 3),  activation='relu'),
          layers.MaxPooling3D(pool_size=(2, 2, 2)),
          layers.Conv3D(filters=32, kernel_size=(1, 3, 3),  activation='relu'),
          layers.Flatten(),
          layers.Dense(32, activation='relu'),
          layers.Dense(1,activation='sigmoid')
          ])

opt='adam'
lss='binary_crossentropy'
model.compile(optimizer=opt, loss=lss, metrics=['accuracy'])

epch=2
history =model.fit(X_train, Y_train, epochs=epch, validation_data=(X_eval,Y_eval))



print('done')

