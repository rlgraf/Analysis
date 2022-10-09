import numpy as np
import matplotlib.pyplot as plt
import gizmo_analysis as gizmo
import utilities as ut
import scipy
import utilities.io as ut_io

#sim = ['m12i_res7100/', 'm12f_res7100/', 'm12b_res7100/', 'm11i_res7100/', 'm11h_res7100/', 'm11d_res7100/', 'm12m_res7100/', 'm12r_res7100/', 'm11e_res7100/', ',m12w_res7100/', 'm12c_res7100']

#sim = ['share/Wetzellab/m12i/m12i_res7100/', 'm12f_res7100/', 'm12b_res7100/']
sim = ['m12i_res7100/', 'm12f_res7100/', 'm12b_res7100/']

# z = 0

def Fe_H_agedependent_sd(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,x9,x10):
    index = ut.array.get_indices(r[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r_form[:,0], [x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r_form[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    index6 = ut.array.get_indices(r[:,1]*180/np.pi, [x9,x10], prior_indices = index5)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index6]
    if len(Fe_H_cut) == 0:
        return(np.nan)
    sd = np.std(Fe_H_cut)
    return(sd)

Fe_H_azim_total = []
slope_azim_total = []
for s in sim:
    simulation_directory = s
    part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
    r = part['star'].prop('host.distance.principal.cylindrical')
    r_form = part['star'].prop('form.host.distance.principal.cylindrical')
    Fe_H = part['star'].prop('metallicity.iron')
    age = part['star'].prop('age')
    
    Fe_H_azim = []
    slope_azim = []
    for a in np.arange(0,14):
        Fe_H_azim_pre = []
        for a_pre in np.arange(0,1,0.05):
            std_vs_rad = []
            for i in np.arange(0,15,1):
                std_vs_rad.append(Fe_H_agedependent_sd(i,i+1,0,1,0,15,0,3,a,a+a_pre+0.05,0,360))
            Fe_H_azim_pre.append(std_vs_rad)
        Fe_H_azim_pre = np.array(Fe_H_azim_pre)
        Fe_H_azim_pre_mean = np.nanmean(Fe_H_azim_pre,0)
        Fe_H_azim.append(Fe_H_azim_pre_mean)
        l = np.arange(0,15)
        Fe_H_azim_pre_mean = np.array(Fe_H_azim_pre_mean)
        j, k = np.polyfit(l[np.isfinite(Fe_H_azim_pre_mean)],Fe_H_azim_pre_mean[np.isfinite(Fe_H_azim_pre_mean)],1)
        slope_azim.append(j)
    Fe_H_azim_total.append(Fe_H_azim)
    slope_azim_total.append(slope_azim)
Fe_H_azim_total = np.array(Fe_H_azim_total)
slope_azim_total = np.array(slope_azim_total)
    
# formation

def Fe_H_agedependent_sd_form(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,x9,x10):
    index = ut.array.get_indices(r_form[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r_form[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r[:,0],[x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    index6 = ut.array.get_indices(r[:,1]*180/np.pi, [x9,x10], prior_indices = index5)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index6]
    if len(Fe_H_cut) == 0:
        return(np.nan)
    sd = np.std(Fe_H_cut)
    return(sd)

Fe_H_azim_form_total = []
slope_azim_form_total = []
for s in sim:
    simulation_directory = s
    part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
    r = part['star'].prop('host.distance.principal.cylindrical')
    r_form = part['star'].prop('form.host.distance.principal.cylindrical')
    Fe_H = part['star'].prop('metallicity.iron')
    age = part['star'].prop('age')
    
    Fe_H_azim_form = []
    slope_azim_form = []
    for a_f in np.arange(0,14):
        Fe_H_azim_pre_f = []
        for a_f_pre in np.arange(0,1,0.05):
            std_vs_rad_f = []
            for i_f in np.arange(0,15,1):
                std_vs_rad_f.append(Fe_H_agedependent_sd_form(i_f,i_f+1,0,1,0,15,0,3,a_f,a_f+a_f_pre+0.05,0,360))
            Fe_H_azim_pre_f.append(std_vs_rad_f)
        Fe_H_azim_pre_f = np.array(Fe_H_azim_pre_f)
        Fe_H_azim_pre_mean_f = np.nanmean(Fe_H_azim_pre_f,0)
        Fe_H_azim_form.append(Fe_H_azim_pre_mean_f)
        l_f = np.arange(0,15)
        Fe_H_azim_pre_mean_f = np.array(Fe_H_azim_pre_mean_f)
        j_f, k_f = np.polyfit(l_f[np.isfinite(Fe_H_azim_pre_mean_f)],Fe_H_azim_pre_mean_f[np.isfinite(Fe_H_azim_pre_mean_f)],1)
        slope_azim_form.append(j_f)
    Fe_H_azim_form_total.append(Fe_H_azim_form)
    slope_azim_form_total.append(slope_azim_form)
Fe_H_azim_form_total = np.array(Fe_H_azim_form_total)
slope_azim_form_total = np.array(slope_azim_form_total)  