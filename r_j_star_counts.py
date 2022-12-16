#!/usr/bin/env python3
#SBATCH --job-name=r_j_star_counts.py
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=64G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=48:00:00
#SBATCH --output=r_j_star_counts_%j.txt
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

def Fe_H_agedependent_sd(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,r,r_form,age,part):
    index = ut.array.get_indices(r[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r_form[:,0], [x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r_form[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index5]
    if len(Fe_H_cut) == 0:
        return(np.nan)
    sd = np.std(Fe_H_cut)
    return(len(Fe_H_cut))


def angmom_func(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,r,r_form,age,part, particle_thresh = 100):
    
    index = ut.array.get_indices(abs(r[:,2]), [x3,x4])
    index2 = ut.array.get_indices(abs(r_form[:,2]), [x7,x8], prior_indices = index)
    index3 = ut.array.get_indices(age, [a1,a2], prior_indices = index2)
    
    index4 = ut.array.get_indices(r[:,0], [x1,x2], prior_indices = index3)
    index5 = ut.array.get_indices(r_form[:,0], [x5,x6], prior_indices = index4)
    
    angmom = part['star'].prop('host.velocity.principal.cylindrical')[:,1]*r[:,0]
    index6 = ut.array.get_indices(angmom, [0,np.inf], prior_indices = index5)
    angmom_cut = angmom[index6]
    if len(angmom_cut) < particle_thresh:
        return(np.nan)
    mean_angmom = np.mean(angmom_cut)
    index7 = ut.array.get_indices(angmom, [mean_angmom - 0.02*mean_angmom, mean_angmom + 0.02*mean_angmom], prior_indices = index3)
    angmom_range = angmom[index7]
    return(len(angmom_range))

def azimuthal_analysis_z_0():
    range_r = []
    range_j = []
    
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
            angmom_len = angmom_func(7,8,0,3,7,8,0,3,0,1,r,r_form,age,part,particle_thresh = 100)
            range_j.append(angmom_len)
           
            r_len = Fe_H_agedependent_sd(7,8,0,3,7,8,0,3,0,1,r,r_form,age,part)
            range_r.append(r_len)
            
    range_j = np.array(range_j)
    range_r = np.array(range_r)
            
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/AZIM_angmom_z_0_starcount_0.02', range_j)  
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/AZIM_radius_z_0_starcount_0.02', range_r) 
 
           
# formation

def Fe_H_agedependent_form_sd(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,r,r_form,age,part):
    index = ut.array.get_indices(r_form[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r_form[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r[:,0], [x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    
    Fe_H_form = part['star'].prop('metallicity.iron')
    Fe_H_cut_form = Fe_H_form[index5]
    if len(Fe_H_cut_form) == 0:
        return(np.nan)
    sd_f = np.std(Fe_H_cut_form)
    return(len(Fe_H_cut_form))

def angmom_func_form(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,r,r_form,age,part, particle_thresh = 100):
    
    index = ut.array.get_indices(abs(r_form[:,2]), [x3,x4])
    index2 = ut.array.get_indices(abs(r[:,2]), [x7,x8], prior_indices = index)
    index3 = ut.array.get_indices(age, [a1,a2], prior_indices = index2)
    
    index4 = ut.array.get_indices(r_form[:,0], [x1,x2], prior_indices = index3)
    index5 = ut.array.get_indices(r[:,0], [x5,x6], prior_indices = index4)
    
    angmom_form = part['star'].prop('form.host.velocity.principal.cylindrical')[:,1]*r[:,0]
    index6 = ut.array.get_indices(angmom_form, [0,np.inf], prior_indices = index5)
    angmom_cut_form = angmom_form[index6]
    if len(angmom_cut_form) < particle_thresh:
           return(np.nan)
    mean_angmom_form = np.mean(angmom_cut_form)
    index7 = ut.array.get_indices(angmom_form, [mean_angmom_form - 0.02*mean_angmom_form, mean_angmom_form + 0.02*mean_angmom_form], prior_indices = index3)
    angmom_range_f = angmom_form[index7]
    return(len(angmom_range_f))

def azimuthal_analysis_form():
    range_r_f = []
    range_j_f = []
    
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
            angmom_totvalues_form = part['star'].prop('form.host.velocity.principal.cylindrical')[:,1]*r[:,0]
            angmom_len_form = angmom_func_form(7,8,0,3,7,8,0,3,0,1,r,r_form,age,part,particle_thresh = 100)
            range_j_f.append(angmom_len_form)
           
            r_len_form = Fe_H_agedependent_form_sd(7,8,0,3,7,8,0,3,0,1,r,r_form,age,part)
            range_r_f.append(r_len_form)
           
    range_j_f = np.array(range_j_f)
    range_r_f = np.array(range_r_f)
            
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/AZIM_angmom_form_starcount_0.02', range_j_f)  
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/AZIM_radius_form_starcount_0.02', range_r_f)
    
azimuthal_analysis_z_0()
azimuthal_analysis_form()