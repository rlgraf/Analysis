#!/usr/bin/env python3
#SBATCH --job-name=radial_analysis_surface_density_ratio_and_Fe_H_100Myr_R90_young
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=72G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=36:00:00
#SBATCH --output=radial_analysis_surface_density_ratio_and_Fe_H_100Myr_R90_young_%j.txt
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

def Fe_H_agedependent_form(x1,x2,x3,x4,a1,a2,r_star,age,part, particle_thresh = 16):
    #index = ut.array.get_indices(r_spherical_star[:,0],[x5,x6])
    index2 = ut.array.get_indices(r_star[:,0], [x1,x2])
    index3 = ut.array.get_indices(abs(r_star[:,2]), [x3,x4], prior_indices = index2)
    index4 = ut.array.get_indices(age, [a1,a2], prior_indices = index3)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index4]
    if len(Fe_H_cut) < particle_thresh:
        return(np.nan)
    else:
        weight_avg = ws.weighted_median(Fe_H_cut, part['star']['mass'][index4])
        return(weight_avg)

def log_surf_dens_ratio(x1,x2,x3,x4,a1,a2,r_star,r_gas,age,part):
    #index = ut.array.get_indices(r_spherical_star[:,0],[x5,x6])
    index2 = ut.array.get_indices(r_star[:,0], [x1,x2])
    index3 = ut.array.get_indices(abs(r_star[:,2]), [x3,x4], prior_indices = index2)
    index4 = ut.array.get_indices(age, [a1,a2], prior_indices = index3)
    surf_dens_star = np.sum(part['star']['mass'][index4])/(np.pi*(x2**2 - x1**2))
    
    index5 = ut.array.get_indices(r_gas[:,0], [x1,x2])
    index6 = ut.array.get_indices(abs(r_gas[:,2]), [x3,x4], prior_indices = index5)
    surf_dens_gas = np.sum(part['gas']['mass'][index6])/(np.pi*(x2**2 - x1**2))
    
    if surf_dens_star == 0 or surf_dens_gas == 0:
        return(np.nan)
    else:
        log_surf_dens_ratio = np.log10(surf_dens_star/surf_dens_gas)
        return(log_surf_dens_ratio)


def radial_analysis_form():
    
    Fe_H_rad_form_all_galaxies = []
    surf_dens_ratio_all_galaxies = []
    
    Fe_H_rad_form_all_galaxies_slope = []
    surf_dens_ratio_all_galaxies_slope = []
    
    ### m12i
    
    simulation_directory = '/group/awetzelgrp/m12i/m12i_r7100_uvb-late/'
    R90_young_m12i = np.array([13.76, 12.63, 12.18, 12.47, 9.36, 6.62, 6.99, 5.47, 4.73, 5.79, 9.10])
        
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []

    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r90) in zip(part_snapshots,R90_young_m12i):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host.distance.principal.cylindrical')
        r_gas = part['gas'].prop('host.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,r90,r90/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+r90/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+r90/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(0,r90,r90/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(0,r90,r90/20)
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
 
    R90_young_m12c = np.array([10.55, 11.28, 10.31, 8.33, 7.82, 6.65, 5.50, 5.42, 9.37, 8.99, 9.65])
        
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r90) in zip(part_snapshots,R90_young_m12c):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host.distance.principal.cylindrical')
        r_gas = part['gas'].prop('host.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,r90,r90/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+r90/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+r90/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(0,r90,r90/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(0,r90,r90/20)
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
 
    R90_young_m12f = np.array([18.86, 15.20, 17.24, 15.10, 14.21, 12.84, 8.17, 4.92, 5.18, 6.63, 3.65])
        
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r90) in zip(part_snapshots,R90_young_m12f):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host.distance.principal.cylindrical')
        r_gas = part['gas'].prop('host.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,r90,r90/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+r90/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+r90/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(0,r90,r90/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(0,r90,r90/20)
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
 
    R90_young_m12m = np.array([13.38, 12.27, 10.60, 9.79, 9.43, 9.96, 11.89, 11.33, 11.44, 10.52, 4.75])
        
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r90) in zip(part_snapshots,R90_young_m12m):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host.distance.principal.cylindrical')
        r_gas = part['gas'].prop('host.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,r90,r90/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+r90/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+r90/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(0,r90,r90/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(0,r90,r90/20)
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
 
    R90_young_m12b = np.array([10.70, 11.38, 10.54, 11.14, 9.01, 11.47, 8.89, 4.24, 3.06, 4.37, 4.22])
        
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r90) in zip(part_snapshots,R90_young_m12b):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host.distance.principal.cylindrical')
        r_gas = part['gas'].prop('host.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,r90,r90/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+r90/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+r90/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(0,r90,r90/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(0,r90,r90/20)
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
    
    R90_young_Romeo = np.array([23.82, 24.85, 19.44, 19.04, 17.18, 14.92, 11.73, 10.77, 9.74, 6.96, 7.79])
    
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r90) in zip(part_snapshots,R90_young_Romeo):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction','form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host1.distance.principal.cylindrical') 
        r_gas = part['gas'].prop('host1.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,r90,r90/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+r90/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+r90/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(0,r90,r90/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(0,r90,r90/20)
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
 
    R90_young_Juliet = np.array([19.48, 15.48, 13.85, 9.38, 5.98, 4.00, 4.49, 5.87, 3.27, 2.50, 5.12])
    
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r90) in zip(part_snapshots,R90_young_Juliet):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction','form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host2.distance.principal.cylindrical') 
        r_gas = part['gas'].prop('host2.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,r90,r90/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+r90/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+r90/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(0,r90,r90/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(0,r90,r90/20)
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
 
    R90_young_Romulus = np.array([17.03, 20.09, 19.77, 17.01, 14.19, 10.91, 6.90, 9.27, 5.48, 5.51, 4.54])
    
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r90) in zip(part_snapshots,R90_young_Romulus):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction','form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host1.distance.principal.cylindrical') 
        r_gas = part['gas'].prop('host1.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,r90,r90/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+r90/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+r90/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(0,r90,r90/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(0,r90,r90/20)
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
 
    R90_young_Remus = np.array([21.15, 19.39, 18.49, 17.18, 13.78, 11.42, 10.04, 9.99, 5.61, 5.70, 4.80])
    
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r90) in zip(part_snapshots,R90_young_Remus):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction','form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host2.distance.principal.cylindrical') 
        r_gas = part['gas'].prop('host2.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,r90,r90/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+r90/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+r90/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(0,r90,r90/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(0,r90,r90/20)
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
 
    R90_young_Thelma = np.array([16.03, 11.76, 11.75, 8.81, 9.86, 9.12, 10.04, 9.96, 8.80, 7.53, 8.12])
    
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r90) in zip(part_snapshots,R90_young_Thelma):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction','form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host1.distance.principal.cylindrical') 
        r_gas = part['gas'].prop('host1.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,r90,r90/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+r90/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+r90/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(0,r90,r90/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(0,r90,r90/20)
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
 
    R90_young_Louise = np.array([22.95, 22.07, 18.38, 14.74, 13.21, 11.27, 9.21, 5.30, 5.57, 9.61, 7.70])
    
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    
    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r90) in zip(part_snapshots,R90_young_Louise):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'massfraction','form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host2.distance.principal.cylindrical') 
        r_gas = part['gas'].prop('host2.distance.principal.cylindrical')
        
        x_f = []
        surf_dens_ratio = []
        for i_f in np.arange(0,r90,r90/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+r90/20,-3,3,0,0.1,r_star,age,part))
            surf_dens_ratio.append(log_surf_dens_ratio(i_f,i_f+r90/20,-3,3,0,0.1,r_star,r_gas,age,part))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(0,r90,r90/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(0,r90,r90/20)
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
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/Fe_H_rad_form_all_galaxies_surfdensratiotrack_100Myr_R90_young', Fe_H_rad_form_all_galaxies)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/surf_dens_ratio_all_galaxies_surfdensratiotrack_100Myr_R90_young', surf_dens_ratio_all_galaxies)
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/Fe_H_rad_form_all_galaxies_surfdensratiotrack_100Myr_R90_young_slope', Fe_H_rad_form_all_galaxies_slope)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/surf_dens_ratio_all_galaxies_surfdensratiotrack_100Myr_R90_young_slope', surf_dens_ratio_all_galaxies_slope)
    
radial_analysis_form()