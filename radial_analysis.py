import numpy as np
import matplotlib.pyplot as plt
import gizmo_analysis as gizmo
import utilities as ut
import scipy
import utilities.io as ut_io
import matplotlib.lines as mlines


def Fe_H_agedependent(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2):
    index = ut.array.get_indices(r[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r_form[:,0], [x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r_form[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    
    
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index5]
    
    if len(Fe_H_cut) == 0:
        return(np.nan)
    
    weight_avg = sum((Fe_H_cut)*part['star']['mass'][index5])/sum(part['star']['mass'][index5])
    
    return(weight_avg)