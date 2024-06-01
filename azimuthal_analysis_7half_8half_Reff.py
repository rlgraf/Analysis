#!/usr/bin/env python3
#SBATCH --job-name=azimuthal_analysis_7half_8half_Reff
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=64G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=48:00:00
#SBATCH --output=azimuthal_analysis_7half_8half_Reff_%j.txt
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

R90_all_z_0 = np.array([10.1, 9.2, 13.5, 11.9, 9.2, 14.2, 9.5, 14.8, 12.4, 11.7, 12.6])
R90_all_z_0_median = np.median(R90_all_z_0)


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
    for q, (s,r90) in enumerate(zip(sim,R90_all_z_0)):
        simulation_directory = s
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        age_obs = age*10**(np.random.normal(0, np.log10(1), age.size))
        
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
            R = 8/R90_all_z_0_median*r90
            for a in np.arange(0,14):
                Fe_H_azim_pre = []
                for a_pre in np.arange(0,1,0.05):
                    Fe_H_azim_pre.append(Fe_H_agedependent_sd(R-0.5,R+0.5,0,3,0,30,a+a_pre,a+a_pre+0.05,0,360,r,r_form,age_obs,part))  
                Fe_H_azim_pre = np.array(Fe_H_azim_pre)
                Fe_H_azim_pre_mean = np.nanmean(Fe_H_azim_pre,0)
                Fe_H_azim.append(Fe_H_azim_pre_mean)
            
            Fe_H_azim_total.append(Fe_H_azim)
    Fe_H_azim_total = np.array(Fe_H_azim_total)
    
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/AZIM_profile_z_0_7half_8half_Reff', Fe_H_azim_total) 
    
    
azimuthal_analysis_z_0()