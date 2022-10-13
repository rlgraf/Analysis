import numpy as np
import matplotlib.pyplot as plt
import gizmo_analysis as gizmo
import utilities as ut
import scipy
import utilities.io as ut_io

#sim = ['m12i_res7100/', 'm12f_res7100/', 'm12b_res7100/', 'm11i_res7100/', 'm11h_res7100/', 'm11d_res7100/', 'm12m_res7100/', 'm12r_res7100/', 'm11e_res7100/', ',m12w_res7100/', 'm12c_res7100']

#sim = ['share/Wetzellab/m12i/m12i_res7100_uvb-late/', 'm12f_res7100/', 'm12b_res7100/']

def sim_func():
    sim = ['/Users/Russell/Downloads/m12i_res7100/', '/Users/Russell/Downloads/m12f_res7100/', '/Users/Russell/Downloads/m12b_res7100/']
    return(sim)

def R90_func():
    R90_m12i = np.array([12.7, 11.2, 11.3, 9.9, 8.9, 5.6, 6.1, 4.5, 4.6, 4.3, 6.9, 7, 2.8, 0.4])
    R90_m12f = np.array([17.0, 13.8, 15.4, 13.0, 13.1, 12.5, 5.1, 4.0, 4.6, 5.8, 3.1, 6.0, 4.8, 2.2])
    R90_m12b = np.array([11.6, 11.7, 10.5, 10.5, 8.2, 9.3, 6.2, 3.6, 2.4, 3.2, 3.6, 6.2, 2.3, 1.7])

    R90 = np.vstack([R90_m12i, R90_m12f, R90_m12b])
    return(R90)

# z = 0

def Fe_H_agedependent(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,r,r_form,age,part):
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


def radial_analysis_z_0():
    
    Fe_H_rad_total = []
    slope_total = []
    sim = sim_func()
    R90 = R90_func()
    for s, r90 in zip(sim, R90):
        simulation_directory = s
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        r = part['star'].prop('host.distance.principal.cylindrical')
        r_form = part['star'].prop('form.host.distance.principal.cylindrical')
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
    
        Fe_H_rad = []
        slope = []
        for a, b in zip(np.arange(0,14), r90):
            x = []
            for i in np.arange(0,b,b/10):
                x.append(Fe_H_agedependent(i,i+b/10,-3,3,0,b,-3,3,a,a+1,r,r_form,age,part))
            Fe_H_rad.append(x)
            l = np.arange(0,b,b/10)
            x = np.array(x)
            j, k = np.polyfit(l[np.isfinite(x)],x[np.isfinite(x)],1)
            slope.append(j)
        Fe_H_rad_total.append(Fe_H_rad)
        slope_total.append(slope)
    Fe_H_rad_total = np.array([Fe_H_rad_total])
    slope_total = np.array([slope_total])
    
    ut_io.file_hdf5('/Users/Russell/Downloads/Final Figures/RAD_profile_z_0', Fe_H_rad_total)

# formation

def Fe_H_agedependent_form(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,r_form,r,age,part):
    index = ut.array.get_indices(r_form[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r_form[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r[:,0],[x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index5]
    if len(Fe_H_cut) == 0:
        return(np.nan)
    weight_avg = sum((Fe_H_cut)*part['star']['mass'][index5])/sum(part['star']['mass'][index5])
    return(weight_avg)


def radial_analysis_form():
    
    Fe_H_rad_form_total = []
    slope_form_total = []
    sim = sim_func()
    R90 = R90_func()
    for s, r90 in zip(sim, R90):
        simulation_directory = s
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        r = part['star'].prop('host.distance.principal.cylindrical')
        r_form = part['star'].prop('form.host.distance.principal.cylindrical')
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
    
        Fe_H_rad_form = []
        slope_form = []
        for a_f, b_f in zip(np.arange(0,14), r90):
            x_f = []
            for i_f in np.arange(0,b_f,b_f/10):
                x_f.append(Fe_H_agedependent_form(i_f,i_f+b_f/10,-3,3,0,b_f,-3,3,a_f,a_f+1,r_form,r,age,part))
            Fe_H_rad_form.append(x_f)
            l_f = np.arange(0,b_f,b_f/10)
            x_f = np.array(x_f)
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)], x_f[np.isfinite(x_f)], 1)
            slope_form.append(j_f)
        Fe_H_rad_form_total.append(Fe_H_rad_form)
        slope_form_total.append(slope_form)
    Fe_H_rad_form_total = np.array([Fe_H_rad_form_total])
    slope_form_total = np.array([slope_form_total])
    
    ut_io.file_hdf5('/Users/Russell/Downloads/Final Figures/RAD_profile_form', Fe_H_rad_form_total)