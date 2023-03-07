#!/usr/bin/env python3
#SBATCH --job-name=heatmap_m12i
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=32G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=03:00:00
#SBATCH --output=heatmap_m12i%j.txt
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
    part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0, simulation_directory, assign_hosts_rotation=True, assign_formation_coordinates = True)
    coordinates = part['star'].prop( 'host.distance' )
    distance_to_center = part['gas'].prop( 'host.distance.total' )
    is_in_galaxy = distance_to_center < 15
    Fe_H = part['star'].prop('metallicity.iron')

    x_coord = coordinates[:,0][is_in_galaxy]
    y_coord = coordinates[:,1][is_in_galaxy]

    heatmap_data = np.vstack([x_coord, y_coord, Fe_H])

    ut_io.file_hdf5('/home/rlgraf/Final_Figures/heatmap_data_m12i', heatmap_data)

heatmap()