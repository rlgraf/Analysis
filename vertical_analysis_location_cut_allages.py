#!/usr/bin/env python3
#SBATCH --job-name=vertical_analysis_location_cut_allages
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=64G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=48:00:00
#SBATCH --output=vertical_analysis_location_cut_allages%j.txt
#SBATCH --mail-user=rlgraf@ucdavis.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
import os
import utilities.io as ut_io
# print run-time and CPU information
ScriptPrint = ut_io.SubmissionScriptClass("slurm")
# Analysis code
# Import programs

import numpy as np
import matplotlib.pyplot as plt
import gizmo_analysis as gizmo
import utilities as ut
import scipy
import utilities.io as ut_io
import weightedstats as ws


def sim_func():
    sim = ['/share/wetzellab/m12i/m12i_r7100_uvb-late/', '/share/wetzellab/m12c/m12c_r7100', '/share/wetzellab/m12f/m12f_r7100',  '/share/wetzellab/m12m/m12m_r7100','/share/wetzellab/m12b/m12b_r7100','/share/wetzellab/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/share/wetzellab/m12_elvis/m12_elvis_RomulusRemus_r4000', '/share/wetzellab/m12_elvis/m12_elvis_ThelmaLouise_r4000']
    return(sim)


# z = 0

def Fe_H_agedependent(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,r,r_form,age,part, particle_thresh = 10):
    index = ut.array.get_indices(r[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r_form[:,0], [x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r_form[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index5]
    if len(Fe_H_cut) < particle_thresh:
        return(np.nan)
    weight_avg = ws.weighted_median(Fe_H_cut, part['star']['mass'][index5])
    return(weight_avg)

def vertical_analysis_z_0():
    
    Fe_H_ver_r_1_2_total = []
    slope_ver_r_1_2_total = []
    sim = sim_func()
    for s in sim:
        simulation_directory = s
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        if s in ['/share/wetzellab/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/share/wetzellab/m12_elvis/m12_elvis_RomulusRemus_r4000', '/share/wetzellab/m12_elvis/m12_elvis_ThelmaLouise_r4000']:
            r_array = [part['star'].prop('host1.distance.principal.cylindrical'), part['star'].prop('host2.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host1.distance.principal.cylindrical'), part['star'].prop('form.host2.distance.principal.cylindrical')]
        else:           
            r_array = [part['star'].prop('host.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host.distance.principal.cylindrical')]    
            
        for r, r_form in zip(r_array, r_form_array):    
            Fe_H_ver_r_1_2 = []
            slope_ver_r_1_2 = []
            for i in np.arange(0,1,0.1):
                Fe_H_ver_r_1_2.append(Fe_H_agedependent(1,2,i,i+0.1,0,15,0,1,0,14,r,r_form,age,part))
            l = np.arange(0,1,0.1)
            j, k = np.polyfit(l[np.isfinite(Fe_H_ver_r_1_2)],Fe_H_ver_r_1_2[np.isfinite(Fe_H_ver_r_1_2)],1)
            slope_ver_r_1_2.append(j)
            Fe_H_ver_r_1_2_total.append(Fe_H_ver_r_1_2)
            slope_ver_r_1_2_total.append(slope_ver_r_1_2)
    Fe_H_ver_r_1_2_total = np.array([Fe_H_ver_r_1_2_total])
    slope_ver_r_1_2_total = np.array([slope_ver_r_1_2_total])

    Fe_H_ver_r_4_5_total = []
    slope_ver_r_4_5_total = []
    sim = sim_func()
    for s in sim:
        simulation_directory = s
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
                             
        if s in ['/share/wetzellab/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/share/wetzellab/m12_elvis/m12_elvis_RomulusRemus_r4000', '/share/wetzellab/m12_elvis/m12_elvis_ThelmaLouise_r4000']:
            r_array = [part['star'].prop('host1.distance.principal.cylindrical'), part['star'].prop('host2.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host1.distance.principal.cylindrical'), part['star'].prop('form.host2.distance.principal.cylindrical')]
        else:           
            r_array = [part['star'].prop('host.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host.distance.principal.cylindrical')]
            
        for r, r_form in zip(r_array, r_form_array):
            Fe_H_ver_r_4_5 = []
            slope_ver_r_4_5 = []
            for i in np.arange(0,1,0.1):
                Fe_H_ver_r_4_5.append(Fe_H_agedependent(4,5,i,i+0.1,0,15,0,1,0,14,r,r_form,age,part))
            l = np.arange(0,1,0.1)
            j, k = np.polyfit(l[np.isfinite(Fe_H_ver_r_4_5)],Fe_H_ver_pre_mean[np.isfinite(Fe_H_ver_r_4_5)],1)
            slope_ver_r_4_5.append(j)
            Fe_H_ver_r_4_5_total.append(Fe_H_ver_r_4_5)
            slope_ver_r_4_5_total.append(slope_ver_r_4_5)
    Fe_H_ver_r_4_5_total = np.array([Fe_H_ver_r_4_5_total])
    slope_ver_r_4_5_total = np.array([slope_ver_r_4_5_total])

    Fe_H_ver_r_7_8_total = []
    slope_ver_r_7_8_total = []
    sim = sim_func()
    for s in sim:
        simulation_directory = s
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        if s in ['/share/wetzellab/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/share/wetzellab/m12_elvis/m12_elvis_RomulusRemus_r4000', '/share/wetzellab/m12_elvis/m12_elvis_ThelmaLouise_r4000']:
            r_array = [part['star'].prop('host1.distance.principal.cylindrical'), part['star'].prop('host2.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host1.distance.principal.cylindrical'), part['star'].prop('form.host2.distance.principal.cylindrical')]
        else:           
            r_array = [part['star'].prop('host.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host.distance.principal.cylindrical')]    
            
        for r, r_form in zip(r_array, r_form_array):
            Fe_H_ver_r_7_8 = []
            slope_ver_r_7_8 = []
            for i in np.arange(0,1,0.1):
                Fe_H_ver_r_7_8.append(Fe_H_agedependent(7,8,i,i+0.1,0,15,0,1,0,14,r,r_form,age,part))
            l = np.arange(0,1,0.1)
            j, k = np.polyfit(l[np.isfinite(Fe_H_ver_r_7_8)],Fe_H_ver_r_7_8[np.isfinite(Fe_H_ver_r_7_8)],1)
            slope_ver_r_7_8.append(j)
            Fe_H_ver_r_7_8_total.append(Fe_H_ver_r_7_8)
            slope_ver_r_7_8_total.append(slope_ver_r_7_8)
    Fe_H_ver_r_7_8_total = np.array([Fe_H_ver_r_7_8_total])
    slope_ver_r_7_8_total = np.array([slope_ver_r_7_8_total])

    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_profile_r_1_2_z_0_location_cut_median_allages', Fe_H_ver_r_1_2_total)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_profile_r_4_5_z_0_location_cut_median_allages', Fe_H_ver_r_4_5_total)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_profile_r_7_8_z_0_location_cut_median_allages', Fe_H_ver_r_7_8_total)
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_slope_r_1_2_z_0_location_cut_median_allages', slope_ver_r_1_2_total)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_slope_r_4_5_z_0_location_cut_median_allages', slope_ver_r_4_5_total)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_slope_r_7_8_z_0_location_cut_median_allages', slope_ver_r_7_8_total)

# formation

def Fe_H_agedependent_form(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,r_form,r,age,part, particle_thresh = 4):
    index = ut.array.get_indices(r_form[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r_form[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r[:,0],[x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index5]
    if len(Fe_H_cut) < particle_thresh:
        return(np.nan)
    weight_avg = ws.weighted_median(Fe_H_cut, part['star']['mass'][index5])
    return(weight_avg)

def vertical_analysis_form():
    
    Fe_H_ver_r_1_2_form_total = []
    slope_ver_r_1_2_form_total = []
    sim = sim_func()
    for s in sim:
        simulation_directory = s
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        if s in ['/share/wetzellab/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/share/wetzellab/m12_elvis/m12_elvis_RomulusRemus_r4000', '/share/wetzellab/m12_elvis/m12_elvis_ThelmaLouise_r4000']:
            r_array = [part['star'].prop('host1.distance.principal.cylindrical'), part['star'].prop('host2.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host1.distance.principal.cylindrical'), part['star'].prop('form.host2.distance.principal.cylindrical')]
        else:           
            r_array = [part['star'].prop('host.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host.distance.principal.cylindrical')]   
        
        for r, r_form in zip(r_array, r_form_array):
            Fe_H_ver_r_1_2_form = []
            slope_ver_r_1_2_form = []
            for i_f in np.arange(0,1,0.1):
                Fe_H_ver_r_1_2_form.append(Fe_H_agedependent_form(1,2,i_f,i_f+0.1,0,15,0,1,0,14,r_form,r,age,part))
            l_f = np.arange(0,1,0.1)
            j_f, k_f = np.polyfit(l_f[np.isfinite(Fe_H_ver_r_1_2_form)],Fe_H_ver_r_1_2_form[np.isfinite(Fe_H_ver_r_1_2_form)],1)
            slope_ver_r_1_2_form.append(j_f)
            Fe_H_ver_r_1_2_form_total.append(Fe_H_ver_r_1_2_form)
            slope_ver_r_1_2_form_total.append(slope_ver_r_1_2_form)
    Fe_H_ver_r_1_2_form_total = np.array([Fe_H_ver_r_1_2_form_total])
    slope_ver_r_1_2_form_total = np.array([slope_ver_r_1_2_form_total])

    Fe_H_ver_r_4_5_form_total = []
    slope_ver_r_4_5_form_total = []
    sim = sim_func()
    for s in sim:
        simulation_directory = s
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
    
        if s in ['/share/wetzellab/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/share/wetzellab/m12_elvis/m12_elvis_RomulusRemus_r4000', '/share/wetzellab/m12_elvis/m12_elvis_ThelmaLouise_r4000']:
            r_array = [part['star'].prop('host1.distance.principal.cylindrical'), part['star'].prop('host2.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host1.distance.principal.cylindrical'), part['star'].prop('form.host2.distance.principal.cylindrical')]
        else:           
            r_array = [part['star'].prop('host.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host.distance.principal.cylindrical')]   
        
        for r, r_form in zip(r_array, r_form_array):
            Fe_H_ver_r_4_5_form = []
            slope_ver_r_4_5_form = []
            for i_f in np.arange(0,1,0.1):
                Fe_H_ver_r_4_5_form.append(Fe_H_agedependent_form(4,5,i_f,i_f+0.1,0,15,0,1,0,14,r_form,r,age,part))
            l_f = np.arange(0,1,0.1)
            j_f, k_f = np.polyfit(l_f[np.isfinite(Fe_H_ver_r_4_5_form)],Fe_H_ver_r_4_5_form[np.isfinite(Fe_H_ver_r_4_5_form)],1)
            slope_ver_r_4_5_form.append(j_f)
            Fe_H_ver_r_4_5_form_total.append(Fe_H_ver_r_4_5_form)
            slope_ver_r_4_5_form_total.append(slope_ver_r_4_5_form)
    Fe_H_ver_r_4_5_form_total = np.array([Fe_H_ver_r_4_5_form_total])
    slope_ver_r_4_5_form_total = np.array([slope_ver_r_4_5_form_total])

    Fe_H_ver_r_7_8_form_total = []
    slope_ver_r_7_8_form_total = []
    sim = sim_func()
    for s in sim:
        simulation_directory = s
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
    
        if s in ['/share/wetzellab/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/share/wetzellab/m12_elvis/m12_elvis_RomulusRemus_r4000', '/share/wetzellab/m12_elvis/m12_elvis_ThelmaLouise_r4000']:
            r_array = [part['star'].prop('host1.distance.principal.cylindrical'), part['star'].prop('host2.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host1.distance.principal.cylindrical'), part['star'].prop('form.host2.distance.principal.cylindrical')]
        else:           
            r_array = [part['star'].prop('host.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host.distance.principal.cylindrical')]   
            
        for r, r_form in zip(r_array, r_form_array):
            Fe_H_ver_r_7_8_form = []
            slope_ver_r_7_8_form = []
            for i_f in np.arange(0,1,0.1):
                Fe_H_ver_r_7_8_form.append(Fe_H_agedependent_form(7,8,i_f,i_f+0.1,0,15,0,1,0,14,r_form,r,age,part))
            l_f = np.arange(0,1,0.1)
            j_f, k_f = np.polyfit(l_f[np.isfinite(Fe_H_ver_r_7_8_form)],Fe_H_ver_r_7_8_form[np.isfinite(Fe_H_ver_r_7_8_form)],1)
            slope_ver_r_7_8_form.append(j_f)
            Fe_H_ver_r_7_8_form_total.append(Fe_H_ver_r_7_8_form)
            slope_ver_r_7_8_form_total.append(slope_ver_r_7_8_form)
    Fe_H_ver_r_7_8_form_total = np.array([Fe_H_ver_r_7_8_form_total])
    slope_ver_r_7_8_form_total = np.array([slope_ver_r_7_8_form_total])

    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_profile_r_1_2_form_location_cut_median_allages', Fe_H_ver_r_1_2_form_total)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_profile_r_4_5_form_location_cut_median_allages', Fe_H_ver_r_4_5_form_total)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_profile_r_7_8_form_location_cut_median_allages', Fe_H_ver_r_7_8_form_total)
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_slope_r_1_2_form_location_cut_median_allages', slope_ver_r_1_2_form_total)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_slope_r_4_5_form_location_cut_median_allages', slope_ver_r_4_5_form_total)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_slope_r_7_8_form_location_cut_median_allages', slope_ver_r_7_8_form_total)
    
vertical_analysis_z_0()
vertical_analysis_form()