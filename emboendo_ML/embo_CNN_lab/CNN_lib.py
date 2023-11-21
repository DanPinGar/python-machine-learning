import numpy as np


def cnn_data_bin_gen(input,zero=' ', one= ' '):

    x,y=[],[]

    for kk in input.keys():

        x.append(input[kk]['image'])

        y=

    X,Y = np.array(x),np.array(y)
    return X,Y