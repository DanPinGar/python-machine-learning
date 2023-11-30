import numpy as np
import pandas as pd
import pydicom
import cv2
import time
import os
import matplotlib.pyplot as plt
import pyexcel_ods


MAIN_FILE_PATH = 'C:\PROJECTS\emboendo\Data\AnonymDAnon_Filter_II\DICOMDIR'
FILES_PATH = 'C:/PROJECTS/emboendo/Data/AnonymDAnon_Filter_II/DICOM/'
ods_file = 'C:\PROJECTS\emboendo\Data\TEE_Endocarditis_JUL2023.ods'
DS = pydicom.dcmread(MAIN_FILE_PATH)

def main_d_df():

    patient_data = []
    for record in DS.DirectoryRecordSequence:

        if record.DirectoryRecordType == 'PATIENT':

            patient_info = {'PatientID': int(record.PatientID)}
            patient_info['Records']=[]

            for element in record.children[0].children[0].children:

                if os.path.exists(FILES_PATH+element.ReferencedFileID[1]):
                    patient_info['Records'].append(element.ReferencedFileID[1])

            patient_data.append(patient_info)

    df=pd.DataFrame(patient_data)
    df.sort_values(by='PatientID',ascending=True,inplace=True)

    return df

def labels_df():

    _d = pyexcel_ods.get_data(ods_file)

    hoja_trabajo = next(iter(_d))
    data = _d[hoja_trabajo]

    heads=data[0]
    data1=data[1:]

    df = pd.DataFrame(data1, columns=heads)
    filter_heads=['numero','sexo','edad','ING_embolia_sistemica', 'ING_aneurisma_micotico_A', 'ING_acv_A']

    df = df.dropna(subset=['numero'])

    for hh in filter_heads:df[hh] = df[hh].astype(int)

    df=df[filter_heads]
    df['label'] = np.where(df[['ING_embolia_sistemica', 'ING_aneurisma_micotico_A', 'ING_acv_A']].any(axis=1), 1, 0)
    df.columns=['PatientID','Sex','Age','S.E.', 'M.A.', 'A.C.V.','label']

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