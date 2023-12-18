import numpy as np
import pandas as pd
import pydicom
import cv2
import os
import matplotlib.pyplot as plt
import re
import pickle


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