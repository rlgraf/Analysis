#!/usr/bin/env python3
#SBATCH --job-name=azimuthal_angmom_analysis_new
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=64G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=48:00:00
#SBATCH --output=azimuthal_angmom_analysis_new_%j.txt
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
    sim = ['/share/wetzellab/m12i/m12i_r7100_uvb-late/', '/share/wetzellab/m12c/m12c_r7100', '/share/wetzellab/m12f/m12f_r7100',  '/share/wetzellab/m12m/m12m_r7100','/share/wetzellab/m12b/m12b_r7100', '/share/wetzellab/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/share/wetzellab/m12_elvis/m12_elvis_RomulusRemus_r4000', '/share/wetzellab/m12_elvis/m12_elvis_ThelmaLouise_r4000']
    return(sim)

# z = 0

def Fe_H_agedependent_sd(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,x9,x10,x11,x12,r,r_form,age,part,angmom_totvalues,particle_thresh = 100):
    index = ut.array.get_indices(angmom_totvalues, [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r_form[:,0], [x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r_form[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    index6 = ut.array.get_indices(r[:,1]*180/np.pi, [x9,x10], prior_indices = index5)
    index7 = ut.array.get_indices(r[:,0], [x11,x12], prior_indices = index6)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index7]
    if len(Fe_H_cut) == 0:
        return(np.nan)
    sd = np.std(Fe_H_cut)
    return(sd)

def angmom_func(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,r,r_form,age,part, particle_thresh = 100):
    index = ut.array.get_indices(r[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r_form[:,0], [x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r_form[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    angmom = part['star'].prop('host.velocity.principal.cylindrical')[:,1]*r[:,0]
    index6 = ut.array.get_indices(angmom, [0,np.inf], prior_indices = index5)
    angmom_cut = angmom[index6]
    if len(angmom_cut) < particle_thresh:
           return(np.nan)
    mean_angmom = np.nanmean(angmom_cut)
    #max_angmom = max_angmom_pre[1]
    return(mean_angmom)

def azimuthal_analysis_z_0():
    Fe_H_azim_total = []
    
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
            angmom_totvalues = part['star'].prop('host.velocity.principal.cylindrical')[:,1]*r[:,0]
            angmom_mean = angmom_func(7,8,0,3,0,15,0,3,0,1,r,r_form,age,part,particle_thresh = 100)
            Fe_H_azim = []
            for a in np.arange(0,1,0.05):
                Fe_H_angmom = Fe_H_agedependent_sd(angmom_mean - 0.07,angmom_mean + 0.07,0,3,0,15,0,3,a,a+0.05,0,360,0,15,r,r_form,age,part,angmom_totvalues,particle_thresh = 100)
                Fe_H_azim.append(Fe_H_angmom)    
            Fe_H_azim_mean = np.nanmean(Fe_H_azim,0)
            Fe_H_azim_total.append(Fe_H_azim_mean)
    Fe_H_azim_total = np.array(Fe_H_azim_total)
            
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/AZIM_profile_z_0_angmom_new', Fe_H_azim_total) 
    
  

    
# formation

def Fe_H_agedependent_sd_form(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,x9,x10,x11,x12,r,r_form,age,part,angmom_totvalues_form,particle_thresh = 100):
    index = ut.array.get_indices(angmom_totvalues_form, [x1,x2])
    index2 = ut.array.get_indices(abs(r_form[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r[:,0], [x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    index6 = ut.array.get_indices(r[:,1]*180/np.pi, [x9,x10], prior_indices = index5)
    index7 = ut.array.get_indices(r[:,0], [x11,x12], prior_indices = index6)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index7]
    if len(Fe_H_cut) == 0:
        return(np.nan)
    sd = np.std(Fe_H_cut)
    return(sd)

def angmom_form_func(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,r,r_form,age,part, particle_thresh = 100):
    index = ut.array.get_indices(r_form[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r[:,0], [x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r_form[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    angmom = part['star'].prop('host.velocity.principal.cylindrical')[:,1]*r[:,0]
    index6 = ut.array.get_indices(angmom, [0,np.inf], prior_indices = index5)
    angmom_cut = angmom[index6]
    if len(angmom_cut) < particle_thresh:
           return(np.nan)
    mean_angmom = np.nanmean(angmom_cut)
    #max_angmom = max_angmom_pre[1]
    return(mean_angmom)

def azimuthal_analysis_form():
    Fe_H_azim_form_total = []
    
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
            angmom_totvalues_form = part['star'].prop('host.velocity.principal.cylindrical')[:,1]*r_form[:,0]
            angmom_mean_form = angmom_form_func(7,8,0,3,0,15,0,3,0,1,r,r_form,age,part,particle_thresh = 100)
            Fe_H_azim_form = []
            for a_f in np.arange(0,1,0.05):
                Fe_H_angmom_form = Fe_H_agedependent_sd_form(angmom_mean_form - 0.07,angmom_mean_form + 0.07,0,3,0,15,0,3,a_f,a_f+0.05,0,360,0,15,r,r_form,age,part,angmom_totvalues_form,particle_thresh = 100)
                Fe_H_azim_form.append(Fe_H_angmom_form)    
            Fe_H_azim_mean_form = np.nanmean(Fe_H_azim_form,0)
            Fe_H_azim_form_total.append(Fe_H_azim_mean_form)
    Fe_H_azim_form_total = np.array(Fe_H_azim_form_total)
            
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/AZIM_profile_form_angmom_new', Fe_H_azim_form_total) 
    
    
azimuthal_analysis_z_0()
azimuthal_analysis_form()