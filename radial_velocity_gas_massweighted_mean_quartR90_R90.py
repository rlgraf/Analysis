#!/usr/bin/env python3
#SBATCH --job-name=radial_velocity_gas_massweighted_mean_quartR90_R90
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=99G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=40:00:00
#SBATCH --output=radial_velocity_gas_massweighted_mean_quartR90_R90_%j.txt
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
    #massfrac_iron = part['gas']['massfraction'][:, 10]
    return(weighted_std(vel_rad[index2], part['gas']['mass'][index2]))
           
def radial_vel_disp_gas():
    
    radial_vel_disp_gas_all_galaxies = []
    
### m12i

    simulation_directory = '/group/awetzelgrp/m12i/m12i_r7100_uvb-late/'  
    R90_young_m12i = np.array([13.74, 12.61, 12.17, 12.42, 9.32, 6.56, 6.68, 5.31, 4.66, 5.66, 8.70, 7.44, 5.59])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for (red,r90) in zip(part_snapshots,R90_young_m12i):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        vel_disp_at_snapshot.append(velocity_dispersion_gas(r90/4,r90,-3,3,r,v,part))
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
   ### m12c

    simulation_directory = '/group/awetzelgrp/m12c/m12c_r7100'       
    R90_young_m12c = np.array([10.52, 11.25, 10.28, 8.30, 7.70, 6.58, 5.33, 5.31, 9.34, 8.88, 9.59, 5.66, 2.91])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for (red,r90) in zip(part_snapshots,R90_young_m12c):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        vel_disp_at_snapshot.append(velocity_dispersion_gas(r90/4,r90,-3,3,r,v,part))
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### m12f

    simulation_directory = '/group/awetzelgrp/m12f/m12f_r7100'     
    R90_young_m12f = np.array([18.82, 15.19, 17.20, 15.08, 14.13, 12.75, 8.12, 4.78, 4.81, 6.49, 3.37, 6.52, 6.29])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for (red,r90) in zip(part_snapshots,R90_young_m12f):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        vel_disp_at_snapshot.append(velocity_dispersion_gas(r90/4,r90,-3,3,r,v,part))
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### m12m

    simulation_directory = '/group/awetzelgrp/m12m/m12m_r7100'      
    R90_young_m12m = np.array([13.37, 12.26, 10.59, 9.77, 9.40, 9.92, 11.85, 11.25, 11.36, 10.50, 4.63, 7.94, 4.22])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for (red,r90) in zip(part_snapshots,R90_young_m12m):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        vel_disp_at_snapshot.append(velocity_dispersion_gas(r90/4,r90,-3,3,r,v,part))
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### m12b

    simulation_directory = '/group/awetzelgrp/m12b/m12b_r7100'
    R90_young_m12b = np.array([10.69, 11.34, 10.51, 11.01, 8.97, 11.43, 8.87, 4.16, 2.44, 4.19, 3.81, 7.43, 3.87])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for (red,r90) in zip(part_snapshots,R90_young_m12b):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        vel_disp_at_snapshot.append(velocity_dispersion_gas(r90/4,r90,-3,3,r,v,part))
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### Romeo

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'       
    R90_young_Romeo = np.array([23.79, 24.80, 19.42, 18.98, 17.14, 14.88, 11.70, 10.71, 9.65, 6.82, 7.68, 5.16, 4.15])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for (red,r90) in zip(part_snapshots,R90_young_Romeo):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        vel_disp_at_snapshot.append(velocity_dispersion_gas(r90/4,r90,-3,3,r,v,part))
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### Juliet

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'     
    R90_young_Juliet = np.array([19.45, 15.41, 13.70, 9.25, 5.71, 3.90, 4.20, 5.77, 2.89, 2.24, 4.85, 6.26, 5.12])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for (red,r90) in zip(part_snapshots,R90_young_Juliet):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        vel_disp_at_snapshot.append(velocity_dispersion_gas(r90/4,r90,-3,3,r,v,part))
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### Romulus

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000'     
    R90_young_Romulus = np.array([17.00, 20.04, 19.70, 16.90, 14.08, 10.84, 6.82, 9.20, 5.35, 5.36, 4.30, 4.05, 4.49])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for (red,r90) in zip(part_snapshots,R90_young_Romulus):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        vel_disp_at_snapshot.append(velocity_dispersion_gas(r90/4,r90,-3,3,r,v,part))
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### Remus

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000'    
    R90_young_Remus = np.array([21.08, 19.38, 18.47, 17.16, 13.77, 11.37, 10.01, 9.91, 5.47, 5.38, 4.68, 7.42, 5.90])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for (red,r90) in zip(part_snapshots,R90_young_Remus):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        vel_disp_at_snapshot.append(velocity_dispersion_gas(r90/4,r90,-3,3,r,v,part))
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### Thelma

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'  
    R90_young_Thelma = np.array([16.01, 11.71, 11.69, 8.77, 9.76, 8.99, 9.94, 9.89, 8.77, 7.39, 7.95, 6.88, 4.18])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for (red,r90) in zip(part_snapshots,R90_young_Thelma):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        vel_disp_at_snapshot.append(velocity_dispersion_gas(r90/4,r90,-3,3,r,v,part))
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    ### Louise

    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'     
    R90_young_Louise = np.array([22.91, 22.01, 18.36, 14.70, 13.15, 11.24, 9.16, 5.18, 5.47, 9.42, 7.44, 3.70, 3.51])
    
    vel_disp_at_snapshot = []
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038, 2.39130425, 3.60431647])
    for (red,r90) in zip(part_snapshots,R90_young_Louise):
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction'], assign_hosts_rotation=True)
        
        r = part['gas'].prop('host.distance.principal.cylindrical')
        v = part['gas'].prop('host.velocity.principal.cylindrical')       
        
        vel_disp_at_snapshot.append(velocity_dispersion_gas(r90/4,r90,-3,3,r,v,part))
        
    radial_vel_disp_gas_all_galaxies.append(vel_disp_at_snapshot)
    del(part)
    
    radial_vel_disp_gas_all_galaxies = np.array(radial_vel_disp_gas_all_galaxies)
           
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/velocity_gas_massweighted_mean_quartR90_R90', radial_vel_disp_gas_all_galaxies)

radial_vel_disp_gas()