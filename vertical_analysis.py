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

Fe_H_ver_r_1_2_total = []
slope_ver_r_1_2_total = []
for s in sim:
    simulation_directory = s
    part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
    r = part['star'].prop('host.distance.principal.cylindrical')
    r_form = part['star'].prop('form.host.distance.principal.cylindrical')
    Fe_H = part['star'].prop('metallicity.iron')
    age = part['star'].prop('age')
    
    Fe_H_ver_r_1_2 = []
    slope_ver_r_1_2 = []
    for a in np.arange(0,13):
        x = []
        for i in np.arange(0,1,0.1):
                x.append(Fe_H_agedependent(1,2,i,i+0.1,0,15,0,1,a,a+1))
        Fe_H_ver_r_1_2.append(x)
        l = np.arange(0,1,0.1)
        x = np.array(x)
        j, k = np.polyfit(l[np.isfinite(x)],x[np.isfinite(x)],1)
        slope_ver_r_1_2.append(j)
    Fe_H_ver_r_1_2_total.append(Fe_H_ver_r_1_2)
    slope_ver_r_1_2_total.append(slope_ver_r_1_2)
Fe_H_ver_r_1_2_total = np.array([Fe_H_ver_r_1_2_total])
slope_ver_r_1_2_total = np.array([slope_ver_r_1_2_total])

Fe_H_ver_r_4_5_total = []
slope_ver_r_4_5_total = []
for s in sim:
    simulation_directory = s
    part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
    r = part['star'].prop('host.distance.principal.cylindrical')
    r_form = part['star'].prop('form.host.distance.principal.cylindrical')
    Fe_H = part['star'].prop('metallicity.iron')
    age = part['star'].prop('age')
    
    Fe_H_ver_r_4_5 = []
    slope_ver_r_4_5 = []
    for a in np.arange(0,13):
        x = []
        for i in np.arange(0,1,0.1):
                x.append(Fe_H_agedependent(4,5,i,i+0.1,0,15,0,1,a,a+1))
        Fe_H_ver_r_4_5.append(x)
        l = np.arange(0,1,0.1)
        x = np.array(x)
        j, k = np.polyfit(l[np.isfinite(x)],x[np.isfinite(x)],1)
        slope_ver_r_4_5.append(j)
    Fe_H_ver_r_4_5_total.append(Fe_H_ver_r_4_5)
    slope_ver_r_4_5_total.append(slope_ver_r_4_5)
Fe_H_ver_r_4_5_total = np.array([Fe_H_ver_r_4_5_total])
slope_ver_r_4_5_total = np.array([slope_ver_r_4_5_total])

Fe_H_ver_r_7_8_total = []
slope_ver_r_7_8_total = []
for s in sim:
    simulation_directory = s
    part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
    r = part['star'].prop('host.distance.principal.cylindrical')
    r_form = part['star'].prop('form.host.distance.principal.cylindrical')
    Fe_H = part['star'].prop('metallicity.iron')
    age = part['star'].prop('age')
    
    Fe_H_ver_r_7_8 = []
    slope_ver_r_7_8 = []
    for a in np.arange(0,13):
        x = []
        for i in np.arange(0,1,0.1):
                x.append(Fe_H_agedependent(7,8,i,i+0.1,0,15,0,1,a,a+1))
        Fe_H_ver_r_7_8.append(x)
        l = np.arange(0,1,0.1)
        x = np.array(x)
        j, k = np.polyfit(l[np.isfinite(x)],x[np.isfinite(x)],1)
        slope_ver_r_7_8.append(j)
    Fe_H_ver_r_7_8_total.append(Fe_H_ver_r_7_8)
    slope_ver_r_7_8_total.append(slope_ver_r_7_8)
Fe_H_ver_r_7_8_total = np.array([Fe_H_ver_r_7_8_total])
slope_ver_r_7_8_total = np.array([slope_ver_r_7_8_total])

#ut_io.file_hdf5('Final Figures/VER_profile_r_1_z_0', Fe_H_ver_r_1_2_total)
#ut_io.file_hdf5('Final Figures/VER_profile_r_4_5_z_0', Fe_H_ver_r_4_5_total)
#ut_io.file_hdf5('Final Figures/VER_profile_r_7_8_z_0', Fe_H_ver_r_7_8_total)

# formation

def Fe_H_agedependent_form(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2):
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

Fe_H_ver_r_1_2_form_total = []
slope_ver_r_1_2_form_total = []
for s in sim:
    simulation_directory = s
    part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
    r = part['star'].prop('host.distance.principal.cylindrical')
    r_form = part['star'].prop('form.host.distance.principal.cylindrical')
    Fe_H = part['star'].prop('metallicity.iron')
    age = part['star'].prop('age')
    
    Fe_H_ver_r_1_2_form = []
    slope_ver_r_1_2_form = []
    for a_f in np.arange(0,13):
        x_f = []
        for i_f in np.arange(0,1,0.1):
                x_f.append(Fe_H_agedependent_form(0,15,i_f,i_f+0.1,1,2,0,1,a_f,a_f+1))
        Fe_H_ver_r_1_2_form.append(x_f)
        l_f = np.arange(0,1,0.1)
        x_f = np.array(x_f)
        j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
        slope_ver_r_1_2_form.append(j_f)
    Fe_H_ver_r_1_2_form_total.append(Fe_H_ver_r_1_2_form)
    slope_ver_r_1_2_form_total.append(slope_ver_r_1_2_form)
Fe_H_ver_r_1_2_form_total = np.array([Fe_H_ver_r_1_2_form_total])
slope_ver_r_1_2_form_total = np.array([slope_ver_r_1_2_form_total])

Fe_H_ver_r_4_5_form_total = []
slope_ver_r_4_5_form_total = []
for s in sim:
    simulation_directory = s
    part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
    r = part['star'].prop('host.distance.principal.cylindrical')
    r_form = part['star'].prop('form.host.distance.principal.cylindrical')
    Fe_H = part['star'].prop('metallicity.iron')
    age = part['star'].prop('age')
    
    Fe_H_ver_r_4_5_form = []
    slope_ver_r_4_5_form = []
    for a_f in np.arange(0,13):
        x_f = []
        for i_f in np.arange(0,1,0.1):
                x_f.append(Fe_H_agedependent_form(0,15,i_f,i_f+0.1,4,5,0,1,a_f,a_f+1))
        Fe_H_ver_r_1_2_form.append(x_f)
        l_f = np.arange(0,1,0.1)
        x_f = np.array(x_f)
        print(l_f)
        print(x_f)
        j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
        slope_ver_r_4_5_form.append(j_f)
    Fe_H_ver_r_4_5_form_total.append(Fe_H_ver_r_4_5_form)
    slope_ver_r_4_5_form_total.append(slope_ver_r_4_5_form)
Fe_H_ver_r_4_5_form_total = np.array([Fe_H_ver_r_4_5_form_total])
slope_ver_r_4_5_form_total = np.array([slope_ver_r_4_5_form_total])

Fe_H_ver_r_7_8_form_total = []
slope_ver_r_7_8_form_total = []
for s in sim:
    simulation_directory = s
    part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
    r = part['star'].prop('host.distance.principal.cylindrical')
    r_form = part['star'].prop('form.host.distance.principal.cylindrical')
    Fe_H = part['star'].prop('metallicity.iron')
    age = part['star'].prop('age')
    
    Fe_H_ver_r_7_8_form = []
    slope_ver_r_7_8_form = []
    for a_f in np.arange(0,13):
        x_f = []
        for i_f in np.arange(0,1,0.1):
                x_f.append(Fe_H_agedependent_form(0,15,i_f,i_f+0.1,7,8,0,1,a_f,a_f+1))
        Fe_H_ver_r_1_2_form.append(x_f)
        l_f = np.arange(0,1,0.1)
        x_f = np.array(x_f)
        print(l_f)
        print(x_f)
        j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
        slope_ver_r_7_8_form.append(j_f)
    Fe_H_ver_r_7_8_form_total.append(Fe_H_ver_r_7_8_form)
    slope_ver_r_7_8_form_total.append(slope_ver_r_7_8_form)
Fe_H_ver_r_7_8_form_total = np.array([Fe_H_ver_r_7_8_form_total])
slope_ver_r_7_8_form_total = np.array([slope_ver_r_7_8_form_total])

#ut_io.file_hdf5('Final Figures/VER_profile_r_1_2_form', Fe_H_ver_r_1_2_form_total)
#ut_io.file_hdf5('Final Figures/VER_profile_r_4_5_form', Fe_H_ver_r_4_5_form_total)
#ut_io.file_hdf5('Final Figures/VER_profile_r_7_8_form', Fe_H_ver_r_7_8_form_total)