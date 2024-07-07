#!/usr/bin/env python3
#SBATCH --job-name=radial_velocity_dispersion_gas_metalmassweighted_fullprofile_thirdR98_R98_young
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=99G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=40:00:00
#SBATCH --output=radial_velocity_dispersion_gas_metalmassweighted_fullprofile_thirdR98_R98_young_%j.txt
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

def weighted_std(values, weights):
    
    average = np.average(values, weights = weights)
    variance = np.average((values - average)**2, weights = weights)
    return(np.sqrt(variance))

def velocity_dispersion_gas(x1,x2,x3,x4,r,v,part):
    
    index = ut.array.get_indices(r[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    vel_rad = v[:,0]
    massfrac_iron = part['gas']['massfraction'][:, 10]
    #return(weighted_std(vel_rad[index2], massfrac_iron[index2]*part['gas']['mass'][index2]))
    weighted_standard_dev = weighted_std(vel_rad[index2], massfrac_iron[index2]*part['gas']['mass'][index2])
    avg = np.median(weighted_standard_dev)
    return(avg)
           
def radial_vel_disp_gas():
    
    radial_vel_disp_gas_all_galaxies = []
    
### m12i

    simulation_directory = '/group/awetzelgrp/m12i/m12i_r7100_uvb-late/'  
    R98_young_m12i = np.array([17.16, 16.70, 16.84, 16.28, 11.74, 9.95, 10.66, 7.29, 8.32, 7.36, 9.96])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_m12i):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        x = []
        for i in np.arange(r98/3,r98,(r98 - r98/3)/20):
            x.append(velocity_dispersion_gas(i,i+(r98 - r98/3)/20,-3,3,r,v,part))
        vel_disp_at_snapshot.append(x)
            
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
   ### m12c

    simulation_directory = '/group/awetzelgrp/m12c/m12c_r7100'       
    R98_young_m12c = np.array([13.91, 13.99, 12.52, 10.37, 9.50, 7.62, 6.25, 12.11, 10.95, 11.12, 9.95])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_m12c):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        x = []
        for i in np.arange(r98/3,r98,(r98 - r98/3)/20):
            x.append(velocity_dispersion_gas(i,i+(r98 - r98/3)/20,-3,3,r,v,part))
        vel_disp_at_snapshot.append(x)
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### m12f

    simulation_directory = '/group/awetzelgrp/m12f/m12f_r7100'     
    R98_young_m12f = np.array([24.68, 21.76, 20.19, 17.96, 16.84, 15.99, 13.76, 6.16, 6.22, 8.75, 6.61])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_m12f):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        x = []
        for i in np.arange(r98/3,r98,(r98 - r98/3)/20):
            x.append(velocity_dispersion_gas(i,i+(r98 - r98/3)/20,-3,3,r,v,part))
        vel_disp_at_snapshot.append(x)
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### m12m

    simulation_directory = '/group/awetzelgrp/m12m/m12m_r7100'      
    R98_young_m12m = np.array([15.11, 15.29, 13.13, 11.91, 11.57, 11.43, 14.42, 14.63, 13.50, 11.70, 6.98])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_m12m):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        x = []
        for i in np.arange(r98/3,r98,(r98 - r98/3)/20):
            x.append(velocity_dispersion_gas(i,i+(r98 - r98/3)/20,-3,3,r,v,part))
        vel_disp_at_snapshot.append(x)
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### m12b

    simulation_directory = '/group/awetzelgrp/m12b/m12b_r7100'
    R98_young_m12b = np.array([13.76, 14.52, 13.69, 14.25, 12.80, 13.85, 11.65, 6.87, 4.58, 5.43, 5.33])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_m12b):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        x = []
        for i in np.arange(r98/3,r98,(r98 - r98/3)/20):
            x.append(velocity_dispersion_gas(i,i+(r98 - r98/3)/20,-3,3,r,v,part))
        vel_disp_at_snapshot.append(x)
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### Romeo

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'       
    R98_young_Romeo = np.array([27.73, 27.21, 22.97, 23.21, 21.37, 17.69, 14.39, 12.74, 11.04, 10.60, 9.09])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_Romeo):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        x = []
        for i in np.arange(r98/3,r98,(r98 - r98/3)/20):
            x.append(velocity_dispersion_gas(i,i+(r98 - r98/3)/20,-3,3,r,v,part))
        vel_disp_at_snapshot.append(x)
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### Juliet

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'     
    R98_young_Juliet = np.array([22.18, 21.61, 19.41, 16.85, 7.87, 5.88, 5.52, 7.19, 5.44, 3.18, 7.15])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_Juliet):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        x = []
        for i in np.arange(r98/3,r98,(r98 - r98/3)/20):
            x.append(velocity_dispersion_gas(i,i+(r98 - r98/3)/20,-3,3,r,v,part))
        vel_disp_at_snapshot.append(x)
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### Romulus

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000'     
    R98_young_Romulus = np.array([24.25, 23.94, 24.05, 22.02, 18.30, 14.68, 8.52, 12.10, 7.85, 9.11, 6.91])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_Romulus):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        x = []
        for i in np.arange(r98/3,r98,(r98 - r98/3)/20):
            x.append(velocity_dispersion_gas(i,i+(r98 - r98/3)/20,-3,3,r,v,part))
        vel_disp_at_snapshot.append(x)
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### Remus

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000'    
    R98_young_Remus = np.array([26.79, 25.93, 23.42, 20.53, 18.59, 13.74, 12.59, 14.47, 7.46, 7.02, 5.92])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_Remus):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        x = []
        for i in np.arange(r98/3,r98,(r98 - r98/3)/20):
            x.append(velocity_dispersion_gas(i,i+(r98 - r98/3)/20,-3,3,r,v,part))
        vel_disp_at_snapshot.append(x)
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### Thelma

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'  
    R98_young_Thelma = np.array([19.26, 16.42, 15.99, 10.52, 14.61, 13.26, 14.14, 13.64, 12.29, 10.26, 8.98])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_Thelma):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        x = []
        for i in np.arange(r98/3,r98,(r98 - r98/3)/20):
            x.append(velocity_dispersion_gas(i,i+(r98 - r98/3)/20,-3,3,r,v,part))
        vel_disp_at_snapshot.append(x)
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### Louise

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'     
    R98_young_Louise = np.array([25.63, 24.48, 24.34, 19.17, 17.50, 15.62, 11.19, 7.50, 8.10, 11.41, 8.44])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_Louise):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        x = []
        for i in np.arange(r98/3,r98,(r98 - r98/3)/20):
            x.append(velocity_dispersion_gas(i,i+(r98 - r98/3)/20,-3,3,r,v,part))
        vel_disp_at_snapshot.append(x)
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    radial_vel_disp_gas_all_galaxies = np.array(radial_vel_disp_gas_all_galaxies)
           
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/velocity_dispersion_gas_metalmassweighted_fullprofile_thirdR98_R98_young', radial_vel_disp_gas_all_galaxies)

radial_vel_disp_gas()