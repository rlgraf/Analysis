#!/usr/bin/env python3
#SBATCH --job-name=radial_analysis_surface_density_ratio_and_Fe_H
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=32G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=08:00:00
#SBATCH --output=radial_analysis_surface_density_ratio_and_Fe_H_%j.txt
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
import gizmo_analysis as gizmo
import utilities as ut
import scipy
import utilities.io as ut_io
import weightedstats as ws

# formation

def Fe_H_agedependent_form(x1,x2,x3,x4,x5,x6,a1,a2,r_form,r_spherical,age,part, particle_thresh = 16):
    index = ut.array.get_indices(r_form[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r_form[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r_spherical[:,0],[x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(age, [a1,a2], prior_indices = index3)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index4]
    if len(Fe_H_cut) < particle_thresh:
        return(np.nan)
    weight_avg = ws.weighted_median(Fe_H_cut, part['star']['mass'][index4])
    return(weight_avg)

def log_surf_dens_ratio(x1,x2,x3,x4,x5,x6,a1,a2,r_form,r_spherical,age,part):
    index = ut.array.get_indices(r_form[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r_form[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(r_spherical[:,0],[x5,x6], prior_indices = index2)
    index4 = ut.array.get_indices(age, [a1,a2], prior_indices = index3)
    surf_dens_star = np.sum(part['star']['mass'][index4])/(np.pi*(x2**2 - x1**2))
    surf_dens_gas = np.sum(part['gas']['mass'][index4])/(np.pi*(x2**2 - x1**2))
    log_surf_dens_ratio = np.log10(surf_dens_star/surf_dens_gas)
    return(log_surf_dens_ratio)


def radial_analysis_form():
    
    Fe_H_rad_form_all_galaxies = []
    surf_dens_ratio_all_galaxies = []
    
    ### m12i
    
    simulation_directory = '/group/awetzelgrp/m12i/m12i_r7100_uvb-late/'
 
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', red, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.carbon')
        age = part['star'].prop('age')
        
        r = part['star'].prop('host.distance.principal.cylindrical')
        r_form = part['star'].prop('form.host.distance.principal.cylindrical')
        r_spherical = part['star'].prop('host.distance.principal.spherical')
        r_form_spherical = part['star'].prop('form.host.distance.principal.spherical')
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,15,0.5):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+0.5,-3,3,0,30,0,0.1,r_form,r_spherical,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+0.5,-3,3,0,30,0,1,r_form,r_spherical,age,part))
        Fe_H_rad_form_at_snapshot.append(x_f)
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio)                           
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
                                   
    ### m12c
    
    simulation_directory = '/group/awetzelgrp/m12c/m12c_r7100'
 
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', red, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.carbon')
        age = part['star'].prop('age')
        
        r = part['star'].prop('host.distance.principal.cylindrical')
        r_form = part['star'].prop('form.host.distance.principal.cylindrical')
        r_spherical = part['star'].prop('host.distance.principal.spherical')
        r_form_spherical = part['star'].prop('form.host.distance.principal.spherical')
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,15,0.5):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+0.5,-3,3,0,30,0,0.1,r_form,r_spherical,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+0.5,-3,3,0,30,0,1,r_form,r_spherical,age,part))
        Fe_H_rad_form_at_snapshot.append(x_f)
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio)                           
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot)
                                   
    ### m12f
    
    simulation_directory = '/group/awetzelgrp/m12f/m12f_r7100'
 
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', red, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.carbon')
        age = part['star'].prop('age')
        
        r = part['star'].prop('host.distance.principal.cylindrical')
        r_form = part['star'].prop('form.host.distance.principal.cylindrical')
        r_spherical = part['star'].prop('host.distance.principal.spherical')
        r_form_spherical = part['star'].prop('form.host.distance.principal.spherical')
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,15,0.5):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+0.5,-3,3,0,30,0,0.1,r_form,r_spherical,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+0.5,-3,3,0,30,0,1,r_form,r_spherical,age,part))
        Fe_H_rad_form_at_snapshot.append(x_f)
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio)                           
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot)                               
    
    ### m12m
    
    simulation_directory = '/group/awetzelgrp/m12m/m12m_r7100'
 
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', red, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.carbon')
        age = part['star'].prop('age')
        
        r = part['star'].prop('host.distance.principal.cylindrical')
        r_form = part['star'].prop('form.host.distance.principal.cylindrical')
        r_spherical = part['star'].prop('host.distance.principal.spherical')
        r_form_spherical = part['star'].prop('form.host.distance.principal.spherical')
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,15,0.5):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+0.5,-3,3,0,30,0,0.1,r_form,r_spherical,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+0.5,-3,3,0,30,0,1,r_form,r_spherical,age,part))
        Fe_H_rad_form_at_snapshot.append(x_f)
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio)                           
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot)              
    
    ### m12b
    
    simulation_directory = '/group/awetzelgrp/m12b/m12b_r7100'
 
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', red, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.carbon')
        age = part['star'].prop('age')
        
        r = part['star'].prop('host.distance.principal.cylindrical')
        r_form = part['star'].prop('form.host.distance.principal.cylindrical')
        r_spherical = part['star'].prop('host.distance.principal.spherical')
        r_form_spherical = part['star'].prop('form.host.distance.principal.spherical')
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,15,0.5):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+0.5,-3,3,0,30,0,0.1,r_form,r_spherical,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+0.5,-3,3,0,30,0,1,r_form,r_spherical,age,part))
        Fe_H_rad_form_at_snapshot.append(x_f)
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio)                           
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot)   
                                   
    ### Romeo
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'
 
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', red, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.carbon')
        age = part['star'].prop('age')
        
        r = part['star'].prop('host1.distance.principal.cylindrical')
        r_form = part['star'].prop('form.host1.distance.principal.cylindrical')
        r_spherical = part['star'].prop('host1.distance.principal.spherical')
        r_form_spherical = part['star'].prop('form.host1.distance.principal.spherical')
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,15,0.5):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+0.5,-3,3,0,30,0,0.1,r_form,r_spherical,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+0.5,-3,3,0,30,0,1,r_form,r_spherical,age,part))
        Fe_H_rad_form_at_snapshot.append(x_f)
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio)                           
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot)
                                   
    ### Juliet
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'
 
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', red, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.carbon')
        age = part['star'].prop('age')
        
        r = part['star'].prop('host2.distance.principal.cylindrical')
        r_form = part['star'].prop('form.host2.distance.principal.cylindrical')
        r_spherical = part['star'].prop('host2.distance.principal.spherical')
        r_form_spherical = part['star'].prop('form.host2.distance.principal.spherical')
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,15,0.5):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+0.5,-3,3,0,30,0,0.1,r_form,r_spherical,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+0.5,-3,3,0,30,0,1,r_form,r_spherical,age,part))
        Fe_H_rad_form_at_snapshot.append(x_f)
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio)                           
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot)   
                                   
    ### Romulus
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000'
 
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', red, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.carbon')
        age = part['star'].prop('age')
        
        r = part['star'].prop('host1.distance.principal.cylindrical')
        r_form = part['star'].prop('form.host1.distance.principal.cylindrical')
        r_spherical = part['star'].prop('host1.distance.principal.spherical')
        r_form_spherical = part['star'].prop('form.host1.distance.principal.spherical')
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,15,0.5):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+0.5,-3,3,0,30,0,0.1,r_form,r_spherical,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+0.5,-3,3,0,30,0,1,r_form,r_spherical,age,part))
        Fe_H_rad_form_at_snapshot.append(x_f)
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio)                           
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot)
                                   
    ### Remus
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000'
 
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', red, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.carbon')
        age = part['star'].prop('age')
        
        r = part['star'].prop('host2.distance.principal.cylindrical')
        r_form = part['star'].prop('form.host2.distance.principal.cylindrical')
        r_spherical = part['star'].prop('host2.distance.principal.spherical')
        r_form_spherical = part['star'].prop('form.host2.distance.principal.spherical')
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,15,0.5):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+0.5,-3,3,0,30,0,0.1,r_form,r_spherical,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+0.5,-3,3,0,30,0,1,r_form,r_spherical,age,part))
        Fe_H_rad_form_at_snapshot.append(x_f)
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio)                           
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot)  
                                   
    ### Thelma
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'
 
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', red, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.carbon')
        age = part['star'].prop('age')
        
        r = part['star'].prop('host1.distance.principal.cylindrical')
        r_form = part['star'].prop('form.host1.distance.principal.cylindrical')
        r_spherical = part['star'].prop('host1.distance.principal.spherical')
        r_form_spherical = part['star'].prop('form.host1.distance.principal.spherical')
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,15,0.5):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+0.5,-3,3,0,30,0,0.1,r_form,r_spherical,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+0.5,-3,3,0,30,0,1,r_form,r_spherical,age,part))
        Fe_H_rad_form_at_snapshot.append(x_f)
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio)                           
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot)
                                   
    ### Louise
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'
 
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', red, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.carbon')
        age = part['star'].prop('age')
        
        r = part['star'].prop('host2.distance.principal.cylindrical')
        r_form = part['star'].prop('form.host2.distance.principal.cylindrical')
        r_spherical = part['star'].prop('host2.distance.principal.spherical')
        r_form_spherical = part['star'].prop('form.host2.distance.principal.spherical')
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,15,0.5):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+0.5,-3,3,0,30,0,0.1,r_form,r_spherical,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+0.5,-3,3,0,30,0,1,r_form,r_spherical,age,part))
        Fe_H_rad_form_at_snapshot.append(x_f)
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio)                           
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot)                           
                                                                  
    Fe_H_rad_form_all_galaxies = np.array(Fe_H_rad_form_all_galaxies)   
    surf_dens_ratio_all_galaxies = np.array(surf_dens_ratio_all_galaxies)                               
        
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/Fe_H_rad_form_all_galaxies_surfdensratiotrack', Fe_H_rad_form_all_galaxies)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/surf_dens_ratio_all_galaxies_surfdensratiotrack', surf_dens_ratio_all_galaxies)
    
radial_analysis_form()