#!/usr/bin/env python3
#SBATCH --job-name=half_mass_radius_analysis_percentile_form_100Myr_CHECKPOINT
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=32G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=08:00:00
#SBATCH --output=half_mass_radius_analysis_percentile_form_100Myr_CHECKPOINT_%j.txt
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


#def sim_func():
    #sim = ['/Users/Russell/Downloads/m12i_res7100/', '/Users/Russell/Downloads/m12f_res7100/', '/Users/Russell/Downloads/m12b_res7100/']
    #return(sim)
    
#'share/Wetzellab/m12_elvis/m12_elvis_res7100'

def sim_func():
    sim = ['/group/awetzelgrp/m12i/m12i_r7100_uvb-late/', '/group/awetzelgrp/m12c/m12c_r7100', '/group/awetzelgrp/m12f/m12f_r7100',  '/group/awetzelgrp/m12m/m12m_r7100','/group/awetzelgrp/m12b/m12b_r7100', '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000', '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000']
    return(sim)


def half_mass_radius_form(x1,x2,x3,x4,a1,a2,r_form_cylindrical,r_form_spherical,age,part):
    
    index = ut.array.get_indices(abs(r_form_cylindrical[:,2]), [x1,x2])
    a_form = part['star'].prop('form.scalefactor')
    scaled_radius = r_form_spherical[:,0]/a_form
    index2 = ut.array.get_indices(scaled_radius, [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(age, [a1,a2], prior_indices = index2)
    if len(r_form_cylindrical[:,0][index3]) == 0:
        return(np.nan)
    else:
        mass_cut = ut.math.percentile_weighted(r_form_cylindrical[:,0][index3], 50, weights = part['star']['mass'][index3])
        return(mass_cut)
                                                                                      

def half_mass_radius_analysis_form():
    
    half_mass_radius_galaxy = []
    sim = sim_func()
    LG_counter = 0
    for q, s in enumerate(sim):
        simulation_directory = s
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        age = part['star'].prop('age')
    
        if s in ['/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000', '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000']:
            r_array_spherical = [part['star'].prop('host1.distance.principal.spherical'), part['star'].prop('host2.distance.principal.spherical')]
            r_form_array_spherical = [part['star'].prop('form.host1.distance.principal.spherical'), part['star'].prop('form.host2.distance.principal.spherical')]
            r_array_cylindrical = [part['star'].prop('host1.distance.principal.cylindrical'), part['star'].prop('host2.distance.principal.cylindrical')]
            r_form_array_cylindrical = [part['star'].prop('form.host1.distance.principal.cylindrical'), part['star'].prop('form.host2.distance.principal.cylindrical')]
        else:
            r_array_spherical = [part['star'].prop('host.distance.principal.spherical')]
            r_form_array_spherical = [part['star'].prop('form.host.distance.principal.spherical')]
            r_array_cylindrical = [part['star'].prop('host.distance.principal.cylindrical')]
            r_form_array_cylindrical = [part['star'].prop('form.host.distance.principal.cylindrical')]
            
        for j, (r_spherical, r_form_spherical,r_cylindrical, r_form_cylindrical) in enumerate(zip(r_array_spherical,r_form_array_spherical,r_array_cylindrical,r_form_array_cylindrical)):
            half_mass_radius_at_age = []
            LG_counter += j
            for a in np.arange(0,14):
                half_mass_radius_at_age.append(half_mass_radius_form(-3,3,0,30,a,a+0.1,r_form_cylindrical,r_form_spherical,age,part))
            half_mass_radius_galaxy.append(half_mass_radius_at_age)
    half_mass_radius_galaxy = np.array(half_mass_radius_galaxy)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/50_mass_radius_form_100Myr_CHECKPOINT', half_mass_radius_galaxy)
    
half_mass_radius_analysis_form()