from functools import lru_cache
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output

# --------------------------------- UTILIRTIES ---------------------------------


def im_show(imagen,cmap='gray',axis='off'):

    plt.imshow(imagen, cmap=cmap)
    plt.axis(axis) 
    plt.show()

def vid_show(input,t_btw_frm=0.2):

    for ii in range(len(input)):

        imagen = input[ii]

        plt.imshow(imagen, cmap='gray')
        plt.axis('off')
        plt.show()
        time.sleep(t_btw_frm)
        clear_output(wait=True)

#@lru_cache(maxsize=1000)
def acm_param(param=None,pred1=1,pred2=2, error=' PARAM ERROR IN acm_param FUNCTION '):

    if param==None:return np.random.randint(pred1, pred2)
    elif isinstance(param, tuple) and len(param) == 2:return np.random.randint(param[0], param[1])
    elif isinstance(param, (int, float)): return param
    else: print(error)


# ---------------------------------- DATA DICT GENERATORS  ------------------------------

def im_data_gen(height, width,n,typ,color=255,cir_f=False):

    input={typ + str(ii):{} for ii in range(n)}

    for R in input.keys():

        n_rdm=np.random.rand()

        if n_rdm>=0.5:
            
            p1=[np.random.randint(0, width-1),np.random.randint(0, height-1)]
            p2=[np.random.randint(0, width-1),np.random.randint(0, height-1)]
            input[R]['image']=line_im(p1,p2,height=height, width=width,thickness=(1,3),color=color)
            input[R]['label']='line'

        else:
            
            center= [np.random.randint(int(0+width*0.3), int(width - width*0.3)),np.random.randint(int(0+height*0.3), int(height-height*0.3))]
            radio = np.random.randint(12, 17)
            input[R]['image']=circle_im(center,radio,height=height, width=width,thickness=(1,3),color=color,filled=cir_f)
            input[R]['label']='circle'

    return input

def vid_data_gen(height, width,n,n_frm,typ,color=255,cir_f=False):

    input={typ + str(ii):{} for ii in range(n)}

    for R in input.keys():

        vd_ls=[]
        n_frames=np.random.randint(n_frm[0],n_frm[1]+1)
        n_rdm=np.random.rand()

        if n_rdm>=0.5:

            p1=[np.random.randint(0, width-1),np.random.randint(0, height-1)]
            p2=[np.random.randint(0, width-1),np.random.randint(0, height-1)]

            for _ in range(n_frames):

                vd_ls.append(line_im(p1,p2,height=height, width=width,thickness=(1,3),color=color))
                p1[0]+= np.random.randint(-int(width*0.02), int(width*0.02))
                p2[0]+=np.random.randint(-int(width*0.06), int(width*0.06))
                p1[1]+= np.random.randint(-int(height*0.02), int(height*0.02))
                p2[1]+=np.random.randint(-int(height*0.06), int(height*0.06))

            input[R]['label']=1

        else:

            center= [np.random.randint(int(0+width*0.3), int(width - width*0.3)),np.random.randint(int(0+height*0.3), int(height-height*0.3))]
            radio = np.random.randint(12, 17)

            for _ in range(n_frames):
            
                vd_ls.append(circle_im(center,radio,height=height, width=width,thickness=(1,3),color=color,filled=cir_f))
                center[0]+= np.random.randint(-int(width*0.03), int(width*0.03))
                center[1]+= np.random.randint(-int(height*0.03), int(height*0.03))
                radio += np.random.randint(-int(height*0.01), int(height*0.02))

            input[R]['label']=0

        input[R]['image']=np.array(vd_ls)
        input[R]['dimHW']=[height, width]

    return input

# ---------------------------------- FIGURES ------------------------------

def line_im(p1:list,p2:list,height=100, width=100,thickness=None,color=255):
  
    thickness=acm_param(thickness,pred1=1,pred2=3, error=' thickness value ERROR in line function ')
    image = np.zeros((height, width), dtype=np.uint8)
    cv2.line(image, (p1[0], p1[1]), (p2[0], p2[1]), color=color, thickness=thickness)
        
    return image

def circle_im(center:list,radio,height=100, width=100,thickness=None,color=255, filled=False):
    
    if filled:thickness=cv2.FILLED
    thickness=acm_param(thickness,pred1=1,pred2=3, error=' thickness value ERROR in circle function ')
    image = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(image, (center[0],center[1]), radio, color=color, thickness=thickness)
    
    return image