#!/usr/bin/env python3
#SBATCH --job-name=2D_velocity_map
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=32G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=01:00:00
#SBATCH --output=velocity_map%j.txt
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
    sim = '/share/wetzellab/m12b/m12b_r7100'
    part = gizmo.io.Read.read_snapshots(['gas'], 'redshift', 0, sim, assign_hosts_rotation=True, assign_formation_coordinates = True)
    coordinates = part['gas'].prop( 'host.distance.principal' )
    distance_to_center = part['gas'].prop( 'host.distance.total' )
    height = part['gas'].prop('host.distance.principal.cylindrical')
    velocity = part['gas'].prop('host.velocity.principal.cylindrical')
    velocity_azim = velocity[:,1]
    index1 = ut.array.get_indices(height[:,2], [-3,3])
    index2 = ut.array.get_indices(distance_to_center, [0,25], prior_indices = index1)
    index3 = ut.array.gat_indices(velocity_azim, [0,10000], prior_indices = index2)
   
    

    x_coord = coordinates[:,0][index3]
    y_coord = coordinates[:,1][index3]
    velocity_circ = velocity_azim[index3]

    heatmap_data = np.vstack([x_coord, y_coord, velocity_circ])

    ut_io.file_hdf5('/home/rlgraf/Final_Figures/158_2D_velocity_map', heatmap_data)

heatmap()