#!/usr/bin/env python3
#SBATCH --job-name=radial_mean_velocity_gas_diff_redshifts_metal_multisnapavg
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=99G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=160:00:00
#SBATCH --output=radial_mean_velocity_gas_diff_redshifts_metal_multisnapavg_%j.txt
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


def sim_func():
    sim = ['/group/awetzelgrp/m12i/m12i_r7100_uvb-late/', '/group/awetzelgrp/m12c/m12c_r7100', '/group/awetzelgrp/m12f/m12f_r7100',  '/group/awetzelgrp/m12m/m12m_r7100','/group/awetzelgrp/m12b/m12b_r7100', '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000', '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000']
    return(sim)


def velocity_dispersion_gas(x1,x2,x3,x4,r,v,part):
    
    index = ut.array.get_indices(r[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    vel_rad = v[:,0]
    massfrac_iron = part['gas']['massfraction'][:, 10]
    return(np.average(vel_rad[index2], weights = massfrac_iron[index2]*part['gas']['mass'][index2]))
           
def radial_vel_disp_gas():
    
    radial_vel_disp_gas_all_galaxies = []
    
### m12i
    
    simulation_directory = '/group/awetzelgrp/m12i/m12i_r7100_uvb-late/'
    
    vel_disp_at_snapshot = []
    
    snapshot_array = np.array([600, 590, 589, 588, 587, 586, 585, 584, 583, 582])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
    
    ### m12m

    simulation_directory = '/group/awetzelgrp/m12m/m12m_r7100'                                               
    vel_disp_at_snapshot = []
    
    snapshot_array = np.array([600, 590, 589, 588, 587, 586, 585, 584, 583, 582])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
    
    ### m12b

    simulation_directory = '/group/awetzelgrp/m12b/m12b_r7100'                                         
    vel_disp_at_snapshot = []
    
    snapshot_array = np.array([600, 590, 589, 588, 587, 586, 585, 584, 583, 582])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
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
    
    ### Romeo

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'                                        
    vel_disp_at_snapshot = []
    
    snapshot_array = np.array([600, 590, 589, 588, 587, 586, 585, 584, 583, 582])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        v = part['gas'].prop('host1.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        v = part['gas'].prop('host1.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        v = part['gas'].prop('host1.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        v = part['gas'].prop('host1.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    
    ### Juliet

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'                                        
    vel_disp_at_snapshot = []
    
    snapshot_array = np.array([600, 590, 589, 588, 587, 586, 585, 584, 583, 582])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        v = part['gas'].prop('host2.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        v = part['gas'].prop('host2.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        v = part['gas'].prop('host2.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        v = part['gas'].prop('host2.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    
    ### Romulus

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000'                                     
    vel_disp_at_snapshot = []
    
    snapshot_array = np.array([600, 590, 589, 588, 587, 586, 585, 584, 583, 582])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        v = part['gas'].prop('host1.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        v = part['gas'].prop('host1.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        v = part['gas'].prop('host1.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        v = part['gas'].prop('host1.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    
    ### Remus

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000'                         
    vel_disp_at_snapshot = []
    
    snapshot_array = np.array([600, 590, 589, 588, 587, 586, 585, 584, 583, 582])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        v = part['gas'].prop('host2.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        v = part['gas'].prop('host2.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        v = part['gas'].prop('host2.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        v = part['gas'].prop('host2.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    
    ### Thelma

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'                               
    vel_disp_at_snapshot = []
    
    snapshot_array = np.array([600, 590, 589, 588, 587, 586, 585, 584, 583, 582])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        v = part['gas'].prop('host1.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        v = part['gas'].prop('host1.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        v = part['gas'].prop('host1.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host1.distance.principal.cylindrical')
        v = part['gas'].prop('host1.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    
    ### Louise

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'                    
    vel_disp_at_snapshot = []
    
    snapshot_array = np.array([600, 590, 589, 588, 587, 586, 585, 584, 583, 582])
    vel_disp_array = []
    for red in snapshot_array:
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        v = part['gas'].prop('host2.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        v = part['gas'].prop('host2.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        v = part['gas'].prop('host2.velocity.principal.cylindrical')  
        
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
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host2.distance.principal.cylindrical')
        v = part['gas'].prop('host2.velocity.principal.cylindrical')  
        
        x = []
        for i in np.arange(0,15,15/50):
            x.append(velocity_dispersion_gas(i,i+15/50,-3,3,r,v,part))
        vel_disp_array.append(x)
    vel_disp_mean = np.nanmean(vel_disp_array,0)    
    vel_disp_at_snapshot.append(vel_disp_mean)
    
    del(part)
    
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    
    radial_vel_disp_gas_all_galaxies = np.array(radial_vel_disp_gas_all_galaxies)
           
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/velocity_mean_gas_z_0_diffredshifts_metalmassweighted_multisnapavg', radial_vel_disp_gas_all_galaxies)

radial_vel_disp_gas()