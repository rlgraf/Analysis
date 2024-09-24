#!/usr/bin/env python3
#SBATCH --job-name=velocity_dispersion_gas_diffredshifts_massweighted_multisnapavg_m12i_m12c_m12f
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=99G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=40:00:00
#SBATCH --output=velocity_dispersion_gas_diffredshifts_massweighted_multisnapavg_m12i_m12c_m12f_%j.txt
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


def weighted_std(values, weights):
    
    average = np.average(values, weights = weights)
    variance = np.average((values - average)**2, weights = weights)
    return(np.sqrt(variance))

def velocity_dispersion_gas(x1,x2,x3,x4,r,v,part):
    
    index = ut.array.get_indices(r[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    vel_rad = v[:,0]
    #massfrac_iron = part['gas']['massfraction'][:, 10]
    #return(weighted_std(vel_rad[index2], massfrac_iron[index2]*part['gas']['mass'][index2]))
    return(weighted_std(vel_rad[index2], part['gas']['mass'][index2]))
           
def radial_vel_disp_gas():
    
    radial_vel_disp_gas_all_galaxies = []
    
### m12i
    
    simulation_directory = '/group/awetzelgrp/m12i/m12i_r7100_uvb-late/'
    
    vel_disp_at_snapshot = []
    
    snapshot_array = np.array([600, 590, 589, 588, 587, 586, 585, 584, 583, 582])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'snapshot', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    snapshot_array = np.array([470, 469, 468, 467, 466, 465, 464, 463, 462, 461])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'snapshot', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    snapshot_array = np.array([354, 353, 352, 351, 350, 349, 348, 347, 346, 345])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'snapshot', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    snapshot_array = np.array([230, 229, 228, 227, 226, 225, 224, 223, 222, 221])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'snapshot', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    

    
   ### m12c

    simulation_directory = '/group/awetzelgrp/m12c/m12c_r7100'                                                      
    vel_disp_at_snapshot = []
    
    snapshot_array = np.array([600, 590, 589, 588, 587, 586, 585, 584, 583, 582])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'snapshot', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    snapshot_array = np.array([470, 469, 468, 467, 466, 465, 464, 463, 462, 461])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'snapshot', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    snapshot_array = np.array([354, 353, 352, 351, 350, 349, 348, 347, 346, 345])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'snapshot', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    snapshot_array = np.array([230, 229, 228, 227, 226, 225, 224, 223, 222, 221])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'snapshot', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    
    ### m12f

    simulation_directory = '/group/awetzelgrp/m12f/m12f_r7100'                                                    
    vel_disp_at_snapshot = []
    
    snapshot_array = np.array([600, 590, 589, 588, 587, 586, 585, 584, 583, 582])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'snapshot', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    snapshot_array = np.array([470, 469, 468, 467, 466, 465, 464, 463, 462, 461])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'snapshot', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    snapshot_array = np.array([354, 353, 352, 351, 350, 349, 348, 347, 346, 345])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'snapshot', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    snapshot_array = np.array([230, 229, 228, 227, 226, 225, 224, 223, 222, 221])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'snapshot', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    
    radial_vel_disp_gas_all_galaxies = np.array(radial_vel_disp_gas_all_galaxies)
           
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/velocity_dispersion_gas_diffredshifts_massweighted_multisnapavg_m12i_m12c_m12f', radial_vel_disp_gas_all_galaxies)

radial_vel_disp_gas()