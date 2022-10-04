import numpy as np
import matplotlib.pyplot as plt
import gizmo_analysis as gizmo
import utilities as ut
import scipy
import utilities.io as ut_io

#sim = ['m12i_res7100/', 'm12f_res7100/', 'm12b_res7100/', 'm11i_res7100/', 'm11h_res7100/', 'm11d_res7100/', 'm12m_res7100/', 'm12r_res7100/', 'm11e_res7100/', ',m12w_res7100/', 'm12c_res7100']

sim = ['m12i_res7100/', 'm12f_res7100/', 'm12b_res7100/']

R90_m12i = np.array([12.7, 11.2, 11.3, 9.9, 8.9, 5.6, 6.1, 4.5, 4.6, 4.3, 6.9, 7, 2.8, 0.4])
R90_m12f = np.array([17.0, 13.8, 15.4, 13.0, 13.1, 12.5, 5.1, 4.0, 4.6, 5.8, 3.1, 6.0, 4.8, 2.2])
R90_m12b = np.array([11.6, 11.7, 10.5, 10.5, 8.2, 9.3, 6.2, 3.6, 2.4, 3.2, 3.6, 6.2, 2.3, 1.7])

R90 = np.vstack([R90_m12i, R90_m12f, R90_m12b])

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



for s, r in zip(sim, R90[0,:]):
    simulation_directory = sim
    part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
    Fe_H = part['star'].prop('metallicity.iron')
    age = part['star'].prop('age')
    
    Fe_H_rad = []
    for a, b in zip(np.arange(0,14), r):
        x = []
        for i in np.arange(0,b,b/10):
                x.append(Fe_H_agedependent(i,i+b/10,-3,3,0,b,-3,3,a,a+1))
        Fe_H_rad.append(x)
    Fe_H_rad = np.array(Fe_H_rad)
