#!/usr/bin/env python3
#SBATCH --job-name=azimuthal_analysis_7half_8half_obs50_FINAL
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=64G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=48:00:00
#SBATCH --output=azimuthal_analysis_7half_8half_obs50_FINAL_%j.txt
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


import numpy as np
import matplotlib.pyplot as plt
import gizmo_analysis as gizmo
import utilities as ut
import scipy
import utilities.io as ut_io

def sim_func():
    sim = ['/group/awetzelgrp/m12i/m12i_r7100_uvb-late/', '/group/awetzelgrp/m12c/m12c_r7100', '/group/awetzelgrp/m12f/m12f_r7100',  '/group/awetzelgrp/m12m/m12m_r7100','/group/awetzelgrp/m12b/m12b_r7100', '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000', '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000']
    return(sim)

def R90_func():
    R90_m12i = np.array([12.7, 11.2, 11.3, 9.9, 8.9, 5.6, 6.1, 4.5, 4.6, 4.3, 6.9, 7, 2.8, 0.4])
    R90_m12c = np.array([11.8, 11, 10.3, 7.9, 7.4, 6.7, 5.5, 5.7, 7.5, 7, 3.8, 5.9, 4.3, 3.4])
    R90_m12f = np.array([17.0, 13.8, 15.4, 13.0, 13.1, 12.5, 5.1, 4.0, 4.6, 5.8, 3.1, 6.0, 4.8, 2.2])
    R90_m12m = np.array([12.7, 11.9, 11.1, 9.6, 9.8, 9.5, 10.8, 11.6, 10.5, 10, 8.1, 7.7, 3.1, 1.8])
    R90_m12b = np.array([11.6, 11.7, 10.5, 10.5, 8.2, 9.3, 6.2, 3.6, 2.4, 3.2, 3.6, 6.2, 2.3, 1.7])
    R90_Romeo = np.array([16.8, 17.2, 16, 15.7, 13.8, 13.3, 11.1, 9.7, 10.2, 6.1, 6.7, 6.3, 2.8, 2.8])
    R90_Juliet = np.array([16, 16, 13.5, 8.6, 6.5, 4.3, 3.8, 4.1, 3.1, 3.5, 8.6, 5, 4.1, 1.9])
    R90_Romulus = np.array([16.9, 18.8, 17.6, 15.3, 14.6, 9.4, 6.4, 9.4, 5.5, 8.4, 6.1, 4.3, 3.6, 2.7])
    R90_Remus = np.array([16.2, 15.5, 15.4, 14.5, 13.4, 10.2, 8.9, 8.4, 6.8, 4.3, 4, 5.6, 5.7, 1.7])
    R90_Thelma = np.array([15.1, 13, 9.9, 8.2, 7.7, 7.8, 7.6, 9, 8.8, 9.6, 6.5, 5.6, 3.1, 0.6])
    R90_Louise = np.array([17.3, 16.4, 14.5, 12.7, 12.4, 11.2, 9.2, 5.1, 5.1, 7.6, 5.4, 4.7, 5.6, 1.1])
    
    
    R90 = np.vstack([R90_m12i, R90_m12c, R90_m12f,  R90_m12m, R90_m12b, R90_Romeo, R90_Juliet, R90_Romulus, R90_Remus, R90_Thelma, R90_Louise])
    return(R90)

def R90_z_0_func():
    R90_z_0 = np.array([12.7,11.8,17.0,12.7,11.6,16.8,16,16.9,16.2,15.1,17.3])
    return(R90_z_0)


# z = 0

def Fe_H_agedependent_sd(x1,x2,x3,x4,x5,x6,a1,a2,x9,x10,r,r_form,age_obs,part):
    index = ut.array.get_indices(r[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    a_form = part['star'].prop('form.scalefactor')
    scaled_radius = r_form[:,0]/a_form
    index3 = ut.array.get_indices(scaled_radius, [x5,x6], prior_indices = index2)
    #index4 = ut.array.get_indices(abs(r_form[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age_obs, [a1,a2], prior_indices = index3)
    index6 = ut.array.get_indices(r[:,1]*180/np.pi, [x9,x10], prior_indices = index5)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index6]
    if len(Fe_H_cut) == 0:
        return(np.nan)
    sd = np.std(Fe_H_cut)
    return(sd)

def azimuthal_analysis_z_0():
    Fe_H_azim_total = []
    slope_azim_total = []
    sim = sim_func()
    R90 = R90_func()
    R90_z_0 = R90_z_0_func()
    LG_counter = 0
    for q, s in enumerate(sim):
        simulation_directory = s
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        age_obs = age*10**(np.random.normal(0, np.log10(1.50), age.size))
        # for observational uncertainties of 0%, 10%, and 25%, use values in np.log10() of 0.00, 0.10, and 1.25 respectively
        
        if s in ['/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000', '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000']:
            r_array = [part['star'].prop('host1.distance.principal.cylindrical'), part['star'].prop('host2.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host1.distance.principal.cylindrical'), part['star'].prop('form.host2.distance.principal.cylindrical')]
        else:           
            r_array = [part['star'].prop('host.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host.distance.principal.cylindrical')] 
    
    
        for c, (r, r_form) in enumerate(zip(r_array, r_form_array)):
            Fe_H_azim = []
            slope_azim = []
            LG_counter += c
            r90 = R90[q+LG_counter]
            for a,b in zip(np.arange(0,14),r90):
                Fe_H_azim_pre = []
                for a_pre in np.arange(0,1,0.05):
                    Fe_H_azim_pre.append(Fe_H_agedependent_sd(7.5,8.5,0,3,0,30,a+a_pre,a+a_pre+0.05,0,360,r,r_form,age_obs,part))  
                Fe_H_azim_pre = np.array(Fe_H_azim_pre)
                Fe_H_azim_pre_mean = np.nanmean(Fe_H_azim_pre,0)
                Fe_H_azim.append(Fe_H_azim_pre_mean)
            
            Fe_H_azim_total.append(Fe_H_azim)
    Fe_H_azim_total = np.array(Fe_H_azim_total)
    
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/AZIM_profile_z_0_7half_8half_obs50_FINAL', Fe_H_azim_total) 
    
    
azimuthal_analysis_z_0()