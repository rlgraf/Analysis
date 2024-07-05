#!/usr/bin/env python3
#SBATCH --job-name=radial_analysis_surface_density_ratio_and_Fe_H_100Myr_quartR98_R98_young
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=72G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=36:00:00
#SBATCH --output=radial_analysis_surface_density_ratio_and_Fe_H_100Myr_quartR98_R98_young_%j.txt
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

# formation

def Fe_H_agedependent_form(x1,x2,x3,x4,a1,a2,r_star,age,part, particle_thresh = 4):
    #index = ut.array.get_indices(r_spherical_star[:,0],[x5,x6])
    index2 = ut.array.get_indices(r_star[:,0], [x1,x2])
    index3 = ut.array.get_indices(abs(r_star[:,2]), [x3,x4], prior_indices = index2)
    index4 = ut.array.get_indices(age, [a1,a2], prior_indices = index3)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index4]
    if len(Fe_H_cut) < particle_thresh:
        return(np.nan)
    weight_avg = ws.weighted_median(Fe_H_cut, part['star']['mass'][index4])
    return(weight_avg)

def log_surf_dens_ratio(x1,x2,x3,x4,a1,a2,r_star,r_gas,age,part, particle_thresh = 4):
    #index = ut.array.get_indices(r_spherical_star[:,0],[x5,x6])
    index2 = ut.array.get_indices(r_star[:,0], [x1,x2])
    index3 = ut.array.get_indices(abs(r_star[:,2]), [x3,x4], prior_indices = index2)
    index4 = ut.array.get_indices(age, [a1,a2], prior_indices = index3)
    surf_dens_star = np.sum(part['star']['mass'][index4])/(np.pi*(x2**2 - x1**2))
    
    index5 = ut.array.get_indices(r_gas[:,0], [x1,x2])
    index6 = ut.array.get_indices(abs(r_gas[:,2]), [x3,x4], prior_indices = index5)
    surf_dens_gas = np.sum(part['gas']['mass'][index6])/(np.pi*(x2**2 - x1**2))
    
    if len(part['star']['mass'][index4]) < particle_thresh or len(part['gas']['mass'][index6]) < particle_thresh:
        return(np.nan)
    log_surf_dens_ratio = np.log10(surf_dens_star/surf_dens_gas)
    return(log_surf_dens_ratio)


def radial_analysis_form():
    
    Fe_H_rad_form_all_galaxies = []
    surf_dens_ratio_all_galaxies = []
    
    Fe_H_rad_form_all_galaxies_slope = []
    surf_dens_ratio_all_galaxies_slope = []
    
    ### m12i
    
    simulation_directory = '/group/awetzelgrp/m12i/m12i_r7100_uvb-late/'
    R98_young_m12i = np.array([17.16, 16.70, 16.84, 16.28, 11.74, 9.95, 10.66, 7.29, 8.32, 7.36, 9.96])
        
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []

    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_m12i):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host.distance.principal.cylindrical')
        r_gas = part['gas'].prop('host.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(r98/4,r98,(r98 - r98/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r98/4,r98,(r98 - r98/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r98/4,r98,(r98 - r98/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    
    del(part)
    
     ### m12c
    
    simulation_directory = '/group/awetzelgrp/m12c/m12c_r7100'
 
    R98_young_m12c = np.array([13.91, 13.99, 12.52, 10.37, 9.50, 7.62, 6.25, 12.11, 10.95, 11.12, 9.95])
        
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_m12c):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host.distance.principal.cylindrical')
        r_gas = part['gas'].prop('host.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(r98/4,r98,(r98 - r98/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r98/4,r98,(r98 - r98/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r98/4,r98,(r98 - r98/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    
    del(part)
    
    ### m12f
    
    simulation_directory = '/group/awetzelgrp/m12f/m12f_r7100'
 
    R98_young_m12f = np.array([24.68, 21.76, 20.19, 17.96, 16.84, 15.99, 13.76, 6.16, 6.22, 8.75, 6.61])
        
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_m12f):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host.distance.principal.cylindrical')
        r_gas = part['gas'].prop('host.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(r98/4,r98,(r98 - r98/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r98/4,r98,(r98 - r98/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r98/4,r98,(r98 - r98/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    
    del(part)
    
    ### m12m
    
    simulation_directory = '/group/awetzelgrp/m12m/m12m_r7100'
 
    R98_young_m12m = np.array([15.11, 15.29, 13.13, 11.91, 11.57, 11.43, 14.42, 14.63, 13.50, 11.70, 6.98])
        
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_m12m):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host.distance.principal.cylindrical')
        r_gas = part['gas'].prop('host.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(r98/4,r98,(r98 - r98/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r98/4,r98,(r98 - r98/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r98/4,r98,(r98 - r98/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    
    del(part)
    
    ### m12b
    
    simulation_directory = '/group/awetzelgrp/m12b/m12b_r7100'
 
    R98_young_m12b = np.array([13.76, 14.52, 13.69, 14.25, 12.80, 13.85, 11.65, 6.87, 4.58, 5.43, 5.33])
        
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_m12b):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host.distance.principal.cylindrical')
        r_gas = part['gas'].prop('host.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(r98/4,r98,(r98 - r98/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r98/4,r98,(r98 - r98/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r98/4,r98,(r98 - r98/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    
    del(part)
    
     ### Romeo
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'
    
    R98_young_Romeo = np.array([27.73, 27.21, 22.97, 23.21, 21.37, 17.69, 14.39, 12.74, 11.04, 10.60, 9.09])
    
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_Romeo):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction','form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host1.distance.principal.cylindrical') 
        r_gas = part['gas'].prop('host1.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(r98/4,r98,(r98 - r98/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r98/4,r98,(r98 - r98/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r98/4,r98,(r98 - r98/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    
    del(part)
                                   
    ### Juliet
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'
 
    R98_young_Juliet = np.array([22.18, 21.61, 19.41, 16.85, 7.87, 5.88, 5.52, 7.19, 5.44, 3.18, 7.15])
    
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_Juliet):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction','form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host2.distance.principal.cylindrical') 
        r_gas = part['gas'].prop('host2.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(r98/4,r98,(r98 - r98/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r98/4,r98,(r98 - r98/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r98/4,r98,(r98 - r98/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    
    del(part)
    
    ### Romulus
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000'
 
    R98_young_Romulus = np.array([24.25, 23.94, 24.05, 22.02, 18.30, 14.68, 8.52, 12.10, 7.85, 9.11, 6.91])
    
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_Romulus):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction','form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host1.distance.principal.cylindrical') 
        r_gas = part['gas'].prop('host1.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(r98/4,r98,(r98 - r98/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r98/4,r98,(r98 - r98/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r98/4,r98,(r98 - r98/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    
    del(part)
                                   
    ### Remus
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000'
 
    R98_young_Remus = np.array([26.79, 25.93, 23.42, 20.53, 18.59, 13.74, 12.59, 14.47, 7.46, 7.02, 5.92])
    
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_Remus):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction','form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host2.distance.principal.cylindrical') 
        r_gas = part['gas'].prop('host2.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(r98/4,r98,(r98 - r98/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r98/4,r98,(r98 - r98/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r98/4,r98,(r98 - r98/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    
    del(part)
                                   
    ### Thelma
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'
 
    R98_young_Thelma = np.array([19.26, 16.42, 15.99, 10.52, 14.61, 13.26, 14.14, 13.64, 12.29, 10.26, 8.98])
    
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_Thelma):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction','form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host1.distance.principal.cylindrical') 
        r_gas = part['gas'].prop('host1.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(r98/4,r98,(r98 - r98/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r98/4,r98,(r98 - r98/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r98/4,r98,(r98 - r98/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    
    del(part)
                                   
    ### Louise
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'
 
    R98_young_Louise = np.array([25.63, 24.48, 24.34, 19.17, 17.50, 15.62, 11.19, 7.50, 8.10, 11.41, 8.44])
    
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r98) in zip(part_snapshots,R98_young_Louise):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction','form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host2.distance.principal.cylindrical') 
        r_gas = part['gas'].prop('host2.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(r98/4,r98,(r98 - r98/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+(r98 - r98/4)/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r98/4,r98,(r98 - r98/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r98/4,r98,(r98 - r98/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    
    del(part)
    
    Fe_H_rad_form_all_galaxies = np.array(Fe_H_rad_form_all_galaxies)   
    surf_dens_ratio_all_galaxies = np.array(surf_dens_ratio_all_galaxies)  
    
    Fe_H_rad_form_all_galaxies_slope = np.array(Fe_H_rad_form_all_galaxies_slope)   
    surf_dens_ratio_all_galaxies_slope = np.array(surf_dens_ratio_all_galaxies_slope) 
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/Fe_H_rad_form_all_galaxies_surfdensratiotrack_100Myr_quartR98_R98_young', Fe_H_rad_form_all_galaxies)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/surf_dens_ratio_all_galaxies_surfdensratiotrack_100Myr_quartR98_R98_young', surf_dens_ratio_all_galaxies)
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/Fe_H_rad_form_all_galaxies_surfdensratiotrack_100Myr_quartR98_R98_young_slope', Fe_H_rad_form_all_galaxies_slope)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/surf_dens_ratio_all_galaxies_surfdensratiotrack_100Myr_quartR98_R98_young_slope', surf_dens_ratio_all_galaxies_slope)
    
radial_analysis_form()