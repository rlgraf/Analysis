#!/usr/bin/env python3
#SBATCH --job-name=vertical_analysis_vs_R_Z2_FINAL
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=32G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=012:00:00
#SBATCH --output=vertical_analysis_vs_R_Z2_FINAL_%j.txt
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
#pip install weightedstats==0.4.1
#conda install -c conda-forge weightedstats
import numpy as np
import matplotlib.pyplot as plt
import gizmo_analysis as gizmo
import utilities as ut
import scipy
import utilities.io as ut_io
import weightedstats as ws


#def sim_func():
    #sim = ['/Users/Russell/Downloads/m12i_res7100/', '/Users/Russell/Downloads/m12f_res7100/', '/Users/Russell/Downloads/m12b_res7100/']
    #return(sim)
    
#'share/Wetzellab/m12_elvis/m12_elvis_res7100'

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
    
    R90_stack = np.vstack([R90_m12i, R90_m12c, R90_m12f,  R90_m12m, R90_m12b, R90_Romeo, R90_Juliet, R90_Romulus, R90_Remus, R90_Thelma, R90_Louise])
    R90 = np.mean(R90_stack,0)
    return(R90)


# z = 0.

def Fe_H_agedependent(x1,x2,x3,x4,x5,x6,a1,a2,r,r_form,age,part, particle_thresh = 4):
    index = ut.array.get_indices(r[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    a_form = part['star'].prop('form.scalefactor')
    scaled_radius = r_form[:,0]/a_form
    index3 = ut.array.get_indices(scaled_radius, [x5,x6], prior_indices = index2)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index3)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index5]
    if len(Fe_H_cut) < particle_thresh:
        return(np.nan)
    weight_avg = ws.weighted_median(Fe_H_cut, part['star']['mass'][index5])
    return(weight_avg)


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
    
        if s in ['/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000', '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000']:
            r_array = [part['star'].prop('host1.distance.principal.cylindrical'), part['star'].prop('host2.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host1.distance.principal.cylindrical'), part['star'].prop('form.host2.distance.principal.cylindrical')]
        else:           
            r_array = [part['star'].prop('host.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host.distance.principal.cylindrical')]    
            
        for j, (r, r_form) in enumerate(zip(r_array,r_form_array)):
            Fe_H_rad = []
            slope = []
            LG_counter += j
            for R in np.arange(0.5,15.5,1):
                x = []
                for i in np.arange(0,2,0.1):
                # for evaluating profile out to |Z| < 1 kpc, use np.arange(0,1,0.1)
                    x.append(Fe_H_agedependent(R,R+1,i,i+0.1,0,30,0,14,r,r_form,age,part))
                Fe_H_rad.append(x)
                l = np.arange(0,2,0.1)
                # for evaluating slope out to |Z| < 1 kpc, use np.arange(0,1,0.1)
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
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_profile_z_0_vs_R_Z2_FINAL', Fe_H_rad_total)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_slope_z_0_vs_R_Z2_FINAL', slope_total)

# formation

def Fe_H_agedependent_form(x1,x2,x3,x4,x5,x6,a1,a2,r_form,r,age,part, particle_thresh = 4):
    index = ut.array.get_indices(r_form[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r_form[:,2]), [x3,x4], prior_indices = index)
    a_form = part['star'].prop('form.scalefactor')
    scaled_radius = r_form[:,0]/a_form
    index3 = ut.array.get_indices(r[:,0],[x5,x6], prior_indices = index2)
    #index4 = ut.array.get_indices(abs(r[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index3)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index5]
    if len(Fe_H_cut) < particle_thresh:
        return(np.nan)
    weight_avg = ws.weighted_median(Fe_H_cut, part['star']['mass'][index5])
    return(weight_avg)


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
        
        if s in ['/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000', '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000']:
            r_array = [part['star'].prop('host1.distance.principal.cylindrical'), part['star'].prop('host2.distance.principal.cylindrical')]
            r_array_spherical = [part['star'].prop('host1.distance.principal.spherical'), part['star'].prop('host2.distance.principal.spherical')]
            r_form_array = [part['star'].prop('form.host1.distance.principal.cylindrical'), part['star'].prop('form.host2.distance.principal.cylindrical')]
            r_form_array_spherical = [part['star'].prop('form.host1.distance.principal.spherical'), part['star'].prop('form.host2.distance.principal.spherical')]
        else:
            r_array = [part['star'].prop('host.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host.distance.principal.cylindrical')]
            r_array_spherical = [part['star'].prop('host.distance.principal.spherical')]
            r_form_array_spherical = [part['star'].prop('form.host.distance.principal.spherical')]
            
        for j, (r, r_form) in enumerate(zip(r_array,r_form_array)):    
            Fe_H_rad_form = []
            slope_form = []
            LG_counter += j
            for R_f in np.arange(0.5,15.5,1):
                x_f = []
                for i_f in np.arange(0,2,0.1):
                # for evaluating profile out to |Z| < 1 kpc, use np.arange(0,1,0.1)
                    x_f.append(Fe_H_agedependent_form(R_f,R_f+1,i_f,i_f+0.1,0,30,0,14,r_form,r,age,part))
                Fe_H_rad_form.append(x_f)    
                l_f = np.arange(0,2,0.1)
                # for evaluating slope out to |Z| < 1 kpc, use np.arange(0,1,0.1)
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
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_profile_form_vs_R_Z2_FINAL', Fe_H_rad_form_total)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/VER_slope_form_vs_R_Z2_FINAL', slope_form_total)
    
radial_analysis_z_0()
radial_analysis_form()