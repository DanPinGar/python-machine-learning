from functools import lru_cache
import numpy as np
import cv2

# --------------------------------- UTILIRTIES ---------------------------------

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
            input[R]['image'],input[R]['label']=line_im(p1,p2,height=height, width=width,thickness=(1,3),color=color)
            
        else:input[R]['image'],input[R]['label']=circle_im(height=height, width=width,thickness=(1,3),color=color,filled=cir_f)

    return input

def vid_data_gen(height, width,n,typ,color=255,cir_f=False):

    input={typ + str(ii):{} for ii in range(n)}

    for R in input.keys():

        n_rdm=np.random.rand()

        if n_rdm>=0.5:
            
            p1=[np.random.randint(0, width-1),np.random.randint(0, height-1)]
            p2=[np.random.randint(0, width-1),np.random.randint(0, height-1)]
            input[R]['image'],input[R]['label']=line_im(p1,p2,height=height, width=width,thickness=(1,3),color=color)

        else:input[R]['image'],input[R]['label']=circle_im(height=height, width=width,thickness=(1,3),color=color,filled=cir_f)

    return input

# ---------------------------------- FIGURES ------------------------------

def line_im(p1:list,p2:list,height=100, width=100,thickness=None,color=255):
  
    thickness=acm_param(thickness,pred1=1,pred2=3, error=' thickness value ERROR in line function ')
    image = np.zeros((height, width), dtype=np.uint8)
    cv2.line(image, (p1[0], p1[1]), (p2[0], p2[1]), color=color, thickness=thickness)
        
    return image, 'line'

def circle_im(height=100, width=100,thickness=None,color=255, filled=False):
    
    if filled:thickness=cv2.FILLED
    thickness=acm_param(thickness,pred1=1,pred2=3, error=' thickness value ERROR in circle function ')
    
    x= np.random.randint(int(0+width*0.3), int(width - width*0.3))
    y=np.random.randint(int(0+height*0.3), int(height-height*0.3))
    radio = np.random.randint(12, 17)
    
    image = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(image, (x,y), radio, color=color, thickness=thickness)
    
    return image, 'circle'