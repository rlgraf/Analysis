#!/usr/bin/env python3
#SBATCH --job-name=heatmap_m12i_5_6
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=32G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=03:00:00
#SBATCH --output=heatmap_m12i_5_6%j.txt
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
import matplotlib.pyplot as plt
import gizmo_analysis as gizmo
import utilities as ut
import scipy
import utilities.io as ut_io
import weightedstats as ws


def heatmap():
    sim = '/share/wetzellab/m12i/m12i_r7100_uvb-late/'
    part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, sim, assign_hosts_rotation=True, assign_formation_coordinates = True)
    coordinates = part['star'].prop( 'host.distance.principal' )
    distance_to_center = part['star'].prop( 'host.distance.total' )
    height = part['star'].prop('host.distance.principal.cylindrical')
    age = part['star'].prop('age')
    index1 = ut.array.get_indices(height[:,2], [-3,3])
    index2 = ut.array.get_indices(distance_to_center, [0,15], prior_indices = index1)
    index3 = ut.array.get_indices(age, [5,6], prior_indices = index2)
    
    Fe_H = part['star'].prop('metallicity.iron')

    x_coord = coordinates[:,0][index3]
    y_coord = coordinates[:,1][index3]
    Fe_H_cut = Fe_H[index3]
    Fe_H_weighted = sum((Fe_H_cut)*part['star']['mass'][index3])/sum(part['star']['mass'][index3])

    heatmap_data = np.vstack([x_coord, y_coord, Fe_H_cut])

    ut_io.file_hdf5('/home/rlgraf/Final_Figures/heatmap_data_m12i_5_6_trial3', heatmap_data)

heatmap()