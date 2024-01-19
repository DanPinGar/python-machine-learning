import numpy as np
import pandas as pd
import pydicom
import cv2
import os
import matplotlib.pyplot as plt
import re
import pickle

from sklearn.model_selection import train_test_split


PIKLE_SAVED_P='C:\PROJECTS\emboendo\dicom_viewer\_static\labels_d.pkl'
#FILES_PATH = 'C:/PROJECTS/emboendo/Data/AnonymDAnon_Filter_III/DICOM/'
FILES_PATH = 'C:/PROJECTS/emboendo/Data/ALL_RECORDS/DICOM/'

LABELS_FILE_P_1 = 'C:\PROJECTS\emboendo\Data\TEE_Endocarditis_JUL2023.ods'
LABELS_FILE_P_2 = 'C:\PROJECTS\emboendo\Data\EII_HCUV_VAo_VMitral_con_Vegetacion'

FILTER_HEADS=['numero','sexo','edad','ING_embolia_sistemica', 'ING_aneurisma_micotico_A', 'ING_acv_A']



def main_d_df():

    patient_data={}

    for file in os.listdir(FILES_PATH):

        file_p = os.path.join(FILES_PATH, file)
        record = pydicom.dcmread(file_p)
        P_ID=record.PatientID
        P_ID = re.sub(r'[^0-9]', '', P_ID)
        P_ID=int(P_ID)

        if P_ID in patient_data:patient_data[P_ID].append(file)
        else:patient_data[P_ID]=[file]

    df=pd.DataFrame(list(patient_data.items()), columns=['PatientID','Records'])
    df.sort_values(by='PatientID',ascending=True,inplace=True)

    return df

def labels_df(path=PIKLE_SAVED_P):

    with open(path, 'rb') as file:

        df = pickle.load(file)

    return df

def load_rec(record):

    filename = FILES_PATH + record
    return pydicom.dcmread(filename)

def im_data(image_data):
    
    if len(np.shape(image_data))==3: image_data=image_data[ :, :,0]
    normalized_image = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX)
    image_8bit = np.uint8(normalized_image)
    width, height = image_data.shape[1], image_data.shape[0]

    return image_8bit,  width, height

def gen_patients_d_df(json_d):

    patients_recs_d_df=main_d_df()
    patients_labels_d_df=labels_df()
    patients_d_df=pd.merge( patients_labels_d_df,patients_recs_d_df, on='PatientID')

    def find_recs(row):

        rec_true = [record for record in row['Records'] if record in list(json_d.keys())]

        return pd.Series({'recs_crop': rec_true})

    new_col = patients_d_df.apply(find_recs, axis=1)
    patients_d_df = pd.concat([patients_d_df, new_col], axis=1)
    patients_d_df=patients_d_df.drop('Records', axis=1)
    patients_d_df =  patients_d_df[patients_d_df['recs_crop'].apply(lambda x: x != [])]

    return patients_d_df


# -------------------------------- CNN LAB --------------------------------

def save_data(path,data):

    with open(path, 'wb') as pikle_file:
        
        pickle.dump(data, pikle_file)

def simple_check(rec_elm,recs,x_train, y_train):

    try:
        idx=recs.index(rec_elm)
        print(rec_elm,' Label:',y_train[idx])

        plt.imshow(x_train[idx][0], cmap='gray')
        plt.axis('off') 
        plt.show()
    
    except: print('No Luck')

def in_loop_check(rec_elm,recs,x_train, y_train):

    simple_check(rec_elm,recs,x_train, y_train)

    for x,y,r in zip(x_train,y_train,recs):print(r,int(y),np.shape(x),type(x))


def random_split_by_recs(X_d, Y_d,recs, test_size=0.2):

    X_train_spl, X_eval_spl, Y_train_spl, Y_eval_spl = train_test_split(X_d, Y_d, test_size=test_size, shuffle=False) #,random_state=42)

    recs_train =recs[0:len(Y_train_spl)]
    recs_eval =recs[len(Y_train_spl)::]

    return X_train_spl, X_eval_spl, Y_train_spl, Y_eval_spl ,recs_train,recs_eval

def random_split_by_patients(patients_d_df,recs,X_d,Y_d, val_pat_0=5, val_pat_1=3):

    patients_1_ls =patients_d_df.loc[patients_d_df['label']==1,'PatientID'].tolist()
    patients_0_ls =patients_d_df.loc[patients_d_df['label']==0,'PatientID'].tolist()
    patients_1_rnd =np.random.choice(patients_1_ls,val_pat_1,replace=False)
    patients_0_rnd =np.random.choice(patients_0_ls,val_pat_0,replace=False)

    recs_eval_1 = patients_d_df.loc[patients_d_df['PatientID'].isin(patients_1_rnd), 'recs_crop'].tolist()
    recs_eval_1 = [ii for sublist in recs_eval_1 for ii in sublist]

    recs_eval_0 = patients_d_df.loc[patients_d_df['PatientID'].isin(patients_0_rnd), 'recs_crop'].tolist()
    recs_eval_0 = [ii for sublist in recs_eval_0 for ii in sublist]

    recs_eval=recs_eval_0+recs_eval_1
    recs_train = [ii for ii in recs if ii not in recs_eval]
    idx_recs_eval = [recs.index(ii) for ii in recs_eval if ii in recs]
    idx_recs_train = [recs.index(ii) for ii in recs_train if ii in recs]

    X_eval_spl = np.take(X_d, idx_recs_eval, axis=0)
    Y_eval_spl = np.take(Y_d, idx_recs_eval, axis=0)

    X_train_spl = np.take(X_d, idx_recs_train, axis=0)
    Y_train_spl = np.take(Y_d, idx_recs_train, axis=0)

    

    return X_train_spl, X_eval_spl, Y_train_spl, Y_eval_spl ,recs_train,recs_eval