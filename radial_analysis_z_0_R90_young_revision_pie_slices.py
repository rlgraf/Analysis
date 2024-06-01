#!/usr/bin/env python3
#SBATCH --job-name=radial_analysis_z_0_R90_young_revision_pie_slices
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=32G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=020:00:00
#SBATCH --output=radial_analysis_z_0_R90_young_revision_pie_slices_%j.txt
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

#def sim_func():
    #sim = ['/group/awetzelgrp/m12i/m12i_r7100_uvb-late/', '/group/awetzelgrp/m12c/m12c_r7100', '/group/awetzelgrp/m12f/m12f_r7100',  '/group/awetzelgrp/m12m/m12m_r7100','/group/awetzelgrp/m12b/m12b_r7100', '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000', '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000']
    #return(sim)
    
def sim_func():
    sim = ['/group/awetzelgrp/m12c/m12c_r7100']
    return(sim)

R90 = np.array([12.7, 11.8, 13.5, 12.7, 11.6, 16.8, 16, 16.9, 16.2, 15.1, 17.3])

# z = 0.

def Fe_H_agedependent(x1,x2,x3,x4,x5,x6,a1,a2,x7,x8,r,r_form,age,part, particle_thresh = 4):
    index = ut.array.get_indices(r[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r[:,2]), [x3,x4], prior_indices = index)
    a_form = part['star'].prop('form.scalefactor')
    scaled_radius = r_form[:,0]/a_form
    index3 = ut.array.get_indices(scaled_radius, [x5,x6], prior_indices = index2)
    #index4 = ut.array.get_indices(abs(r_form[:,2]), [x7,x8], prior_indices = index3)
    index5 = ut.array.get_indices(age, [a1,a2], prior_indices = index3)
    index6 = ut.array.get_indices(r[:,1]*360/(2*np.pi), [x7,x8], prior_indices = index5)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index6]
    if len(Fe_H_cut) < particle_thresh:
        return(np.nan)
    weight_avg = ws.weighted_median(Fe_H_cut, part['star']['mass'][index6])
    return(weight_avg)


def radial_analysis_z_0():
    
    Fe_H_rad_total = []
    slope_total = []
    sim = sim_func()
    LG_counter = 0
    for q, (s,r90) in enumerate(zip(sim, R90)):
        simulation_directory = s
        part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.carbon')
        age = part['star'].prop('age')
    
        if s in ['/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500', '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000', '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000']:
            r_array = [part['star'].prop('host1.distance.principal.cylindrical'), part['star'].prop('host2.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host1.distance.principal.cylindrical'), part['star'].prop('form.host2.distance.principal.cylindrical')]
        else:
            r_array = [part['star'].prop('host.distance.principal.cylindrical')]
            r_form_array = [part['star'].prop('form.host.distance.principal.cylindrical')]
            
        for j, (r, r_form) in enumerate(zip(r_array,r_form_array)):
            slope = []
            LG_counter += j
            for a in np.arange(0,14):
                pie_slope = []
                for angle in np.arange(0,360,30):
                    x = []
                    for i in np.arange(0,r90,r90/50):
                        x.append(Fe_H_agedependent(i,i+r90/50,-3,3,0,30,a,a+1,angle,angle+30,r,r_form,age,part))
                    l = np.arange(0,r90,r90/50)
                    x = np.array(x)
                    if np.isnan(x).all():
                        pie_slope.append(np.nan)
                    else:
                        j, k = np.polyfit(l[np.isfinite(x)],x[np.isfinite(x)],1)
                        pie_slope.append(j)
                slope.append(pie_slope)
            slope_total.append(slope)  
    slope_total = np.array([slope_total])
   
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/RAD_slope_z_0_R90_young_revision_pie_slices_m12c', slope_total)

radial_analysis_z_0()