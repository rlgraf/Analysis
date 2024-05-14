#!/usr/bin/env python3
#SBATCH --job-name=surface_density_profile_gas_raw
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=80G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=16:00:00
#SBATCH --output=surface_density_profile_gas_raw_%j.txt
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



def surf_dens_log_frac(x1,x2,x3,x4,x5,x6,x7,x8,r,r_z0,part,part_z0,particle_thresh = 4):
    
    index = ut.array.get_indices(r_z0[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r_z0[:,2]), [x3,x4], prior_indices = index)
    surf_dens_z0 = np.sum(part_z0['gas']['mass'][index2])/(np.pi*(x2**2 - x1**2))
    
    index3 = ut.array.get_indices(r[:,0], [x5,x6])
    index4 = ut.array.get_indices(abs(r[:,2]), [x7,x8], prior_indices = index3)
    surf_dens_lookback = np.sum(part['gas']['mass'][index4])/(np.pi*(x6**2 - x5**2))
    
    frac_surf_lookback_surf_z0 = surf_dens_lookback/surf_dens_z0
    return(surf_dens_lookback)

surf_dens_ratio_gas_all_galaxies = []


def surf_dens_analysis_gas():
    
    surf_dens_ratio_gas_all_galaxies = []
    
### m12i

    simulation_directory = '/group/awetzelgrp/m12i/m12i_r7100_uvb-late/'                                                                     
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part_z0 = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        
        r_z0 = part_z0['gas'].prop('host.distance.principal.cylindrical')
        r = part['gas'].prop('host.distance.principal.cylindrical')
        
        x = []
        for i in np.arange(0,15,0.5):
            x.append(surf_dens_log_frac(i,i+0.5,-3,3,i,i+0.5,-3,3,r,r_z0,part,part_z0))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
    
    ### m12c

    simulation_directory = '/group/awetzelgrp/m12c/m12c_r7100'                                                                
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part_z0 = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        
        r_z0 = part_z0['gas'].prop('host.distance.principal.cylindrical')
        r = part['gas'].prop('host.distance.principal.cylindrical')
        
        x = []
        for i in np.arange(0,15,0.5):
            x.append(surf_dens_log_frac(i,i+0.5,-3,3,i,i+0.5,-3,3,r,r_z0,part,part_z0))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
    
    ### m12f
 
    simulation_directory = '/group/awetzelgrp/m12f/m12f_r7100'                                                              
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part_z0 = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        
        r_z0 = part_z0['gas'].prop('host.distance.principal.cylindrical')
        r = part['gas'].prop('host.distance.principal.cylindrical')
        
        x = []
        for i in np.arange(0,15,0.5):
            x.append(surf_dens_log_frac(i,i+0.5,-3,3,i,i+0.5,-3,3,r,r_z0,part,part_z0))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
    
    ### m12m

    simulation_directory = '/group/awetzelgrp/m12m/m12m_r7100'                                                       
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part_z0 = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        
        r_z0 = part_z0['gas'].prop('host.distance.principal.cylindrical')
        r = part['gas'].prop('host.distance.principal.cylindrical')
        
        x = []
        for i in np.arange(0,15,0.5):
            x.append(surf_dens_log_frac(i,i+0.5,-3,3,i,i+0.5,-3,3,r,r_z0,part,part_z0))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
    
    ### m12b

    simulation_directory = '/group/awetzelgrp/m12b/m12b_r7100'                                                      
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part_z0 = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        
        r_z0 = part_z0['gas'].prop('host.distance.principal.cylindrical')
        r = part['gas'].prop('host.distance.principal.cylindrical')
        
        x = []
        for i in np.arange(0,15,0.5):
            x.append(surf_dens_log_frac(i,i+0.5,-3,3,i,i+0.5,-3,3,r,r_z0,part,part_z0))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
    
    ### Romeo

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'                                         
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part_z0 = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        
        r_z0 = part_z0['gas'].prop('host1.distance.principal.cylindrical')
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        
        x = []
        for i in np.arange(0,15,0.5):
            x.append(surf_dens_log_frac(i,i+0.5,-3,3,i,i+0.5,-3,3,r,r_z0,part,part_z0))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
    
    ### Juliet

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'                                         
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part_z0 = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        
        r_z0 = part_z0['gas'].prop('host2.distance.principal.cylindrical')
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        
        x = []
        for i in np.arange(0,15,0.5):
            x.append(surf_dens_log_frac(i,i+0.5,-3,3,i,i+0.5,-3,3,r,r_z0,part,part_z0))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
                
    ### Romulus

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000'                                      
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part_z0 = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        
        r_z0 = part_z0['gas'].prop('host1.distance.principal.cylindrical')
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        
        x = []
        for i in np.arange(0,15,0.5):
            x.append(surf_dens_log_frac(i,i+0.5,-3,3,i,i+0.5,-3,3,r,r_z0,part,part_z0))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
    
    ### Remus

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000'                             
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part_z0 = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        
        r_z0 = part_z0['gas'].prop('host2.distance.principal.cylindrical')
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        
        x = []
        for i in np.arange(0,15,0.5):
            x.append(surf_dens_log_frac(i,i+0.5,-3,3,i,i+0.5,-3,3,r,r_z0,part,part_z0))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)     
    
     ### Themla

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'                                   
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part_z0 = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        
        r_z0 = part_z0['gas'].prop('host1.distance.principal.cylindrical')
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        
        x = []
        for i in np.arange(0,15,0.5):
            x.append(surf_dens_log_frac(i,i+0.5,-3,3,i,i+0.5,-3,3,r,r_z0,part,part_z0))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)
    
    ### Louise

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'                           
    surf_dens_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for red in part_snapshots:
        part_z0 = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position'], assign_hosts_rotation=True)
        
        r_z0 = part_z0['gas'].prop('host2.distance.principal.cylindrical')
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        
        x = []
        for i in np.arange(0,15,0.5):
            x.append(surf_dens_log_frac(i,i+0.5,-3,3,i,i+0.5,-3,3,r,r_z0,part,part_z0))
        surf_dens_at_snapshot.append(x)   
    surf_dens_ratio_gas_all_galaxies.append(surf_dens_at_snapshot)            
                
    surf_dens_ratio_gas_all_galaxies = np.array(surf_dens_ratio_gas_all_galaxies)
           
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/surface_density_profile_gas_raw_pthresh4', surf_dens_ratio_gas_all_galaxies)

surf_dens_analysis_gas()