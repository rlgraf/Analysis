#!/usr/bin/env python3
#SBATCH --job-name=surface_density_profile_star_100Myr_raw_radcentered_CHECKPOINT
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=80G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=16:00:00
#SBATCH --output=surface_density_profile_star_100Myr_raw_radcentered_CHECKPOINT_%j.txt
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



def surf_dens_log_frac(x1,x2,x3,x4,a1,a2,r,age,part,particle_thresh = 4):
    
    
    index = ut.array.get_indices(r[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(age, [a1,a2], prior_indices = index2)
    if len(part['star']['mass'][index3]) < particle_thresh:
           return(np.nan)
           
    surf_dens_lookback = np.sum(part['star']['mass'][index3])/(np.pi*(x2**2 - x1**2))
    return(surf_dens_lookback)



def surf_dens_analysis_gas():
    
    surf_dens_ratio_gas_all_galaxies = []
    
### m12i

    simulation_directory = '/group/awetzelgrp/m12i/m12i_r7100_uvb-late/'                                                                     
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        
        age = part['star'].prop('age')
        r = part['star'].prop('host.distance.pirncipal.cylindrical')
        
        x = []
        for i in np.arange(0.5,15):
            x.append(surf_dens_log_frac(i,i+1,-3,3,0,0.1,r,age,part))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
        
       
    ### m12c

    simulation_directory = '/group/awetzelgrp/m12c/m12c_r7100'       
           
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        
        age = part['star'].prop('age')
        r = part['star'].prop('host.distance.pirncipal.cylindrical')
        
        x = []
        for i in np.arange(0.5,15):
            x.append(surf_dens_log_frac(i,i+1,-3,3,0,0.1,r,age,part))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
        
    
    ### m12f
 
    simulation_directory = '/group/awetzelgrp/m12f/m12f_r7100'  
           
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        
        age = part['star'].prop('age')
        r = part['star'].prop('host.distance.pirncipal.cylindrical')
        
        x = []
        for i in np.arange(0.5,15):
            x.append(surf_dens_log_frac(i,i+1,-3,3,0,0.1,r,age,part))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
        
    
    ### m12m

    simulation_directory = '/group/awetzelgrp/m12m/m12m_r7100' 
           
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        
        age = part['star'].prop('age')
        r = part['star'].prop('host.distance.pirncipal.cylindrical')
        
        x = []
        for i in np.arange(0.5,15):
            x.append(surf_dens_log_frac(i,i+1,-3,3,0,0.1,r,age,part))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
        
           
    ### m12b

    simulation_directory = '/group/awetzelgrp/m12b/m12b_r7100'                                                      
    
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        
        age = part['star'].prop('age')
        r = part['star'].prop('host.distance.pirncipal.cylindrical')
        
        x = []
        for i in np.arange(0.5,15):
            x.append(surf_dens_log_frac(i,i+1,-3,3,0,0.1,r,age,part))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
        
    
    ### Romeo

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'    
           
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        
        age = part['star'].prop('age')
        r = part['star'].prop('host1.distance.pirncipal.cylindrical')
        
        x = []
        for i in np.arange(0.5,15):
            x.append(surf_dens_log_frac(i,i+1,-3,3,0,0.1,r,age,part))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
        
    
    ### Juliet

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'  
           
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        
        age = part['star'].prop('age')
        r = part['star'].prop('host2.distance.pirncipal.cylindrical')
        
        x = []
        for i in np.arange(0.5,15):
            x.append(surf_dens_log_frac(i,i+1,-3,3,0,0.1,r,age,part))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
        
                
    ### Romulus

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000' 
           
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        
        age = part['star'].prop('age')
        r = part['star'].prop('host1.distance.pirncipal.cylindrical')
        
        x = []
        for i in np.arange(0.5,15):
            x.append(surf_dens_log_frac(i,i+1,-3,3,0,0.1,r,age,part))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
        
    
    ### Remus

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000' 
           
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        
        age = part['star'].prop('age')
        r = part['star'].prop('host2.distance.pirncipal.cylindrical')
        
        x = []
        for i in np.arange(0.5,15):
            x.append(surf_dens_log_frac(i,i+1,-3,3,0,0.1,r,age,part))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
        
    
     ### Themla

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'  
           
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        
        age = part['star'].prop('age')
        r = part['star'].prop('host1.distance.pirncipal.cylindrical')
        
        x = []
        for i in np.arange(0.5,15):
            x.append(surf_dens_log_frac(i,i+1,-3,3,0,0.1,r,age,part))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
        
    
    ### Louise
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'  

    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        
        age = part['star'].prop('age')
        r = part['star'].prop('host2.distance.pirncipal.cylindrical')
        
        x = []
        for i in np.arange(0.5,15):
            x.append(surf_dens_log_frac(i,i+1,-3,3,0,0.1,r,age,part))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
        
        
    surf_dens_ratio_gas_all_galaxies = np.array(surf_dens_ratio_gas_all_galaxies)
           
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/surface_density_profile_star_100Myr_raw_radcentered_CHECKPOINT', surf_dens_ratio_gas_all_galaxies)

surf_dens_analysis_gas()