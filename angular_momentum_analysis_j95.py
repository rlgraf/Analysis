#!/usr/bin/env python3
#SBATCH --job-name=angular_momentum_analysis_j95
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=32G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=03:00:00
#SBATCH --output=angular_momentum_analysis_j95%j.txt
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


#def sim_func():
    #sim = ['/Users/Russell/Downloads/m12i_res7100/', '/Users/Russell/Downloads/m12f_res7100/', '/Users/Russell/Downloads/m12b_res7100/']
    #return(sim)
    
#'share/Wetzellab/m12_elvis/m12_elvis_res7100'

def sim_func():
    sim = ['/share/wetzellab/m12i/m12i_r7100_uvb-late/', '/share/wetzellab/m12c/m12c_r7100', '/share/wetzellab/m12f/m12f_r7100',  '/share/wetzellab/m12m/m12m_r7100','/share/wetzellab/m12b/m12b_r7100', '/share/wetzellab/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/share/wetzellab/m12_elvis/m12_elvis_RomulusRemus_r4000', '/share/wetzellab/m12_elvis/m12_elvis_ThelmaLouise_r4000']
    return(sim)

def R90_func():
    R90 = np.array([12.7, 11.8, 17.0, 12.7, 11.6, 16.8, 16, 16.9, 16.2, 15.1, 17.3])
    return(R90)


# z = 0

def Fe_H_agedependent(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,r,r_form,age,part,angmom_totvalues,particle_thresh = 100):
    index = ut.array.get_indices(angmom_totvalues, [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(angmom_totvalues, [x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r_form[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index5]
    if len(Fe_H_cut) < particle_thresh:
        return(np.nan)
    weight_avg = sum((Fe_H_cut)*part['star']['mass'][index5])/sum(part['star']['mass'][index5])
    return(weight_avg)

def angmom_func(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,r,r_form,age,part,particle_thresh = 100):
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
    max_angmom_pre = np.percentile(angmom_cut, [0,95])
    max_angmom = max_angmom_pre[1]
    return(max_angmom)             

def radial_analysis_z_0():
    
    Fe_H_rad_total = []
    slope_total = []
    sim = sim_func()
    R90 = R90_func()
    LG_counter = 0
    for q, s in enumerate(sim):
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
            
        for j, (r, r_form) in enumerate(zip(r_array,r_form_array)):
            Fe_H_rad = []
            slope = []
            LG_counter += j
            r90 = R90[q+LG_counter]
            x = []
            angmom_totvalues = part['star'].prop('host.velocity.principal.cylindrical')[:,1]*r[:,0]
            angmom_max = angmom_func(0,r90,0,3,0,r90,0,3,0,1,r,r_form,age,part,particle_thresh = 100)
            for i in np.arange(0,angmom_max,angmom_max/10):
                x.append(Fe_H_agedependent(i,i+angmom_max/10,0,3,0,angmom_max,-3,3,0,1,r,r_form,age,part,angmom_totvalues))
            Fe_H_rad.append(x)
            l = np.arange(0,angmom_max,angmom_max/10)
            x = np.array(x)
            if np.isnan(x).all():
                slope.append(np.nan)
            else:
                j, k = np.polyfit(l[np.isfinite(x)],x[np.isfinite(x)],1)
                slope.append(j)
            Fe_H_rad_total.append(Fe_H_rad)
            slope_total.append(slope)
    Fe_H_rad_total = np.array([Fe_H_rad_total])
    slope_total = np.array([slope_total])
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/angular_momentum_j95_profile_z_0', Fe_H_rad_total)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/angular_momentum_j95_slope_z_0', slope_total)

# formation

def Fe_H_agedependent_form(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,r_form,r,age,part, angmom_totvalues_form, particle_thresh = 100):
    index = ut.array.get_indices(angmom_totvalues_form, [x1,x2])
    index2 = ut.array.get_indices(abs(r_form[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(angmom_totvalues_form,[x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index5]
    if len(Fe_H_cut) < particle_thresh:
        return(np.nan)
    weight_avg = sum((Fe_H_cut)*part['star']['mass'][index5])/sum(part['star']['mass'][index5])
    return(weight_avg)
                     
def angmom_form_func(x1,x2,x3,x4,x5,x6,x7,x8,a1,a2,r,r_form,age,part,particle_thresh = 100):
    index = ut.array.get_indices(r[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r_form[:,0], [x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(abs(r_form[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index4)
    angmom = part['star'].prop('form.host.velocity.principal.cylindrical')[:,1]*r_form[:,0]
    index6 = ut.array.get_indices(angmom, [0,np.inf], prior_indices = index5)
    angmom_cut = angmom[index6]
    if len(angmom_cut) < particle_thresh:
        return(np.nan)
    max_angmom_pre = np.percentile(angmom_cut, [0,95])
    max_angmom = max_angmom_pre[1]
    return(max_angmom)             


def radial_analysis_form():
    
    Fe_H_rad_form_total = []
    slope_form_total = []
    sim = sim_func()
    R90 = R90_func()
    LG_counter = 0
    for q, s in enumerate(sim):
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
            
        for j, (r, r_form) in enumerate(zip(r_array,r_form_array)):    
            Fe_H_rad_form = []
            slope_form = []
            LG_counter += j
            r90 = R90[q+LG_counter]
            x_f = []
            angmom_totvalues_form = part['star'].prop('form.host.velocity.principal.cylindrical')[:,1]*r_form[:,0]
            angmom_max_form = angmom_form_func(0,r90,0,3,0,r90,0,3,0,1,r_form,r,age,part,particle_thresh = 100)
            for i_f in np.arange(0,angmom_max_form,angmom_max_form/10):
                x_f.append(Fe_H_agedependent_form(i_f,i_f+angmom_max_form/10,0,3,0,angmom_max_form,-3,3,0,1,r_form,r,age,part,angmom_totvalues_form))
            Fe_H_rad_form.append(x_f)
            l_f = np.arange(0,angmom_max_form,angmom_max_form/10)
            x_f = np.array(x_f)
            if np.isnan(x_f).all():
                slope_form.append(np.nan)
            else:
                j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
                slope_form.append(j_f)
            Fe_H_rad_form_total.append(Fe_H_rad_form)
            slope_form_total.append(slope_form)
    Fe_H_rad_form_total = np.array([Fe_H_rad_form_total])
    slope_form_total = np.array([slope_form_total])
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/angular_momentum_j95_profile_form', Fe_H_rad_form_total)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/angular_momentum_j95_slope_form', slope_form_total)
    
radial_analysis_z_0()
radial_analysis_form()