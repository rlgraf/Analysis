#!/usr/bin/env python3
#SBATCH --job-name=radial_velocity_dispersion_gas
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=72G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=08:00:00
#SBATCH --output=radial_velocity_dispersion_gas_%j.txt
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
    vel_rad = abs(v[:,0])
    return(np.nanmean(vel_rad[index2]))
           
def radial_vel_disp_gas():
    
    vel_disp_total = []
    sim = sim_func()
    LG_counter = 0
    for q, s in enumerate(sim):
        simulation_directory = s
        part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, simulation_directory, properties = ['position', 'velocity']assign_hosts_rotation=True)
           
        if s in ['/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000', '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000']:
            r_array = [part['gas'].prop('host1.distance.principal.cylindrical'), part['gas'].prop('host2.distance.principal.cylindrical')]
            v_array = [part['gas'].prop('host1.velocity.principal.cylindrical'), part['gas'].prop('host2.velocity.principal.cylindrical')]
        else:
            r_array = [part['gas'].prop('host.distance.principal.cylindrical')]
            v_array = [part['gas'].prop('host.velocity.principal.cylindrical')]
            
        for j, (r,v) in enumerate(zip(r_array, v_array)):
            LG_counter += j
            x = []
            for i in np.arange(0,15,15/50):
                x.append(velocity_dispersion_gas(0,i+15/50,-3,3,r,v,part))
            vel_disp_total.append(x)
        del(part)
           
    vel_disp_total = np.array(vel_disp_total)
           
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/velocity_dispersion_gas_z_0', vel_disp_total)
    

radial_vel_disp_gas()