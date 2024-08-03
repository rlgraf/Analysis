#!/usr/bin/env python3
#SBATCH --job-name=radial_analysis_surface_density_ratio_and_Fe_H_100Myr_quartR90_R90_young_CHECK2
#SBATCH --partition=high2  # peloton node: 32 cores, 7.8 GB per core, 250 GB total
##SBATCH --partition=high2m  # peloton high-mem node: 32 cores, 15.6 GB per core, 500 GB total
#SBATCH --mem=99G  # need to specify memory if you set the number of tasks (--ntasks) below
##SBATCH --nodes=1  # if you specify this, the number of nodes, do not set memory (--mem) above
##SBATCH --ntasks-per-node=1  # (MPI) tasks per node
#SBATCH --ntasks=1  # (MPI) tasks total
#SBATCH --cpus-per-task=1  # (OpenMP) threads per (MPI) task
#SBATCH --time=72:00:00
#SBATCH --output=radial_analysis_surface_density_ratio_and_Fe_H_100Myr_quartR90_R90_young_CHECK2_%j.txt
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
    index = ut.array.get_indices(r_star[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r_star[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(age, [a1,a2], prior_indices = index2)
    Fe_H = part['star'].prop('metallicity.iron')
    Fe_H_cut = Fe_H[index3]
    if len(Fe_H_cut) < particle_thresh:
        return(np.nan)
    weight_avg = ws.weighted_median(Fe_H_cut, part['star']['mass'][index3])
    return(weight_avg)

def log_surf_dens_ratio(x1,x2,x3,x4,a1,a2,r_star,r_gas,age,part, particle_thresh = 4):
    index = ut.array.get_indices(r_star[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r_star[:,2]), [x3,x4], prior_indices = index)
    index3 = ut.array.get_indices(age, [a1,a2], prior_indices = index2)
    surf_dens_star = np.sum(part['star']['mass'][index3])/(np.pi*(x2**2 - x1**2))
    
    index4 = ut.array.get_indices(r_gas[:,0], [x1,x2])
    index5 = ut.array.get_indices(abs(r_gas[:,2]), [x3,x4], prior_indices = index4)
    surf_dens_gas = np.sum(part['gas']['mass'][index5])/(np.pi*(x2**2 - x1**2))
    
    if len(part['star']['mass'][index3]) < particle_thresh or len(part['gas']['mass'][index5]) < particle_thresh:
        return(np.nan)
    log_surf_dens_ratio = np.log10(surf_dens_star/surf_dens_gas)
    return(log_surf_dens_ratio)

def weighted_std(values, weights):
    
    average = np.average(values, weights = weights)
    variance = np.average((values - average)**2, weights = weights)
    return(np.sqrt(variance))

def velocity_dispersion_gas(x1,x2,x3,x4,r_gas,v_gas,part):
    
    index = ut.array.get_indices(r_gas[:,0], [x1,x2])
    index2 = ut.array.get_indices(abs(r_gas[:,2]), [x3,x4], prior_indices = index)
    vel_ver = v_gas[:,2]
    massfrac_iron = part['gas']['massfraction'][:,]
    weighted_standard_dev = weighted_std(vel_ver[index2], massfrac_iron[index2]*part['gas']['mass'][index2])
    avg = np.median(weighted_standard_dev)
    #return(avg)
    print(np.shape(massfrac_iron))


def radial_analysis_form():
    
    Fe_H_rad_form_all_galaxies = []
    surf_dens_ratio_all_galaxies = []
    surf_dens_ratio_over_v_vel_all_galxies = []
    
    Fe_H_rad_form_all_galaxies_slope = []
    surf_dens_ratio_all_galaxies_slope = []
    surf_dens_ratio_over_v_vel_all_galxies_slope = []
    
    ### m12i
    
    simulation_directory = '/group/awetzelgrp/m12i/m12i_r7100_uvb-late/'
    R90_young_m12i = np.array([13.74, 12.61, 12.17, 12.42, 9.32, 6.56, 6.68, 5.31, 4.66, 5.66, 8.70])
        
    Fe_H_rad_form_at_snapshot = []
    surf_dens_ratio_at_snapshot = []
    surf_dens_ratio_over_v_vel_at_snapshot = []

    Fe_H_rad_form_at_snapshot_slope = []
    surf_dens_ratio_at_snapshot_slope = []
    surf_dens_ratio_over_v_vel_at_snapshot_slope = []
    
    
    
    part_snapshots = np.array([0, 0.07350430, 0.15441179, 0.24850890, 0.35344830, 0.47764710, 0.62273902, 0.79942691, 1.02572345, 1.38636363, 1.73913038])
    for (red,r90) in zip(part_snapshots,R90_young_m12i):
        part = gizmo.io.Read.read_snapshots(['star','gas'], 'redshift', red, simulation_directory, properties = ['mass', 'position', 'velocity', 'massfraction', 'form.scalefactor', 'id'], elements = ['Fe'], assign_hosts_rotation=True, assign_formation_coordinates = True)
        Fe_H = part['star'].prop('metallicity.iron')
        age = part['star'].prop('age')
        
        r_star = part['star'].prop('host.distance.principal.cylindrical')
        r_gas = part['gas'].prop('host.distance.principal.cylindrical')
        v_gas = part['gas'].prop('host.velocity.principal.cylindrical')   
        
        x_f = []
        surf_dens_ratio = []
        surf_dens_ratio_over_v_vel = []
        
        for i_f in np.arange(r90/4,r90,(r90 - r90/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,age,part))
            log_surf_ratio = log_surf_dens_ratio(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,r_gas,age,part)
            surf_dens_ratio.append(log_surf_ratio)
            
            v_vel_computed = velocity_dispersion_gas(i_f,i_f+(r90 - r90/4)/20,-3,3,r_gas,v_gas,part)
            surf_dens_ratio_over_v_vel.append(np.log10((10**log_surf_ratio)/v_vel_computed))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r90/4,r90,(r90 - r90/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
            
        surf_dens_ratio_over_v_vel_at_snapshot.append(surf_dens_ratio_over_v_vel)
        l_f_sv = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio_over_v_vel = np.array(surf_dens_ratio_over_v_vel)
        if np.isnan(surf_dens_ratio_over_v_vel).all():
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(np.nan)
        else:
            j_f_sv, k_f_sv = np.polyfit(l_f_sv[np.isfinite(surf_dens_ratio_over_v_vel)],surf_dens_ratio_over_v_vel[np.isfinite(surf_dens_ratio_over_v_vel)],1)
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(j_f_sv)                                      
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    surf_dens_ratio_over_v_vel_all_galaxies.append(surf_dens_ratio_over_v_vel_at_snapshot) 
                                              
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    surf_dens_ratio_over_v_vel_all_galaxies_slope.append(surf_dens_ratio_over_v_vel_at_snapshot_slope)                                           
    
    del(part)
    
     ### m12c
    
    simulation_directory = '/group/awetzelgrp/m12c/m12c_r7100'
 
    R90_young_m12c = np.array([10.52, 11.25, 10.28, 8.30, 7.70, 6.58, 5.33, 5.31, 9.34, 8.88, 9.59])
        
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
        v_gas = part['gas'].prop('host.velocity.principal.cylindrical')   
        
        x_f = []
        surf_dens_ratio = []
        surf_dens_ratio_over_v_vel = []
        
        for i_f in np.arange(r90/4,r90,(r90 - r90/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,age,part))
            log_surf_ratio = log_surf_dens_ratio(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,r_gas,age,part)
            surf_dens_ratio.append(log_surf_ratio)
            
            v_vel_computed = velocity_dispersion_gas(i_f,i_f+(r90 - r90/4)/20,-3,3,r_gas,v_gas,part)
            surf_dens_ratio_over_v_vel.append(np.log10((10**log_surf_ratio)/v_vel_computed))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r90/4,r90,(r90 - r90/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
            
        surf_dens_ratio_over_v_vel_at_snapshot.append(surf_dens_ratio_over_v_vel)
        l_f_sv = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio_over_v_vel = np.array(surf_dens_ratio_over_v_vel)
        if np.isnan(surf_dens_ratio_over_v_vel).all():
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(np.nan)
        else:
            j_f_sv, k_f_sv = np.polyfit(l_f_sv[np.isfinite(surf_dens_ratio_over_v_vel)],surf_dens_ratio_over_v_vel[np.isfinite(surf_dens_ratio_over_v_vel)],1)
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(j_f_sv)                                      
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    surf_dens_ratio_over_v_vel_all_galaxies.append(surf_dens_ratio_over_v_vel_at_snapshot) 
                                              
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    surf_dens_ratio_over_v_vel_all_galaxies_slope.append(surf_dens_ratio_over_v_vel_at_snapshot_slope)                                           
    
    del(part)
    
    ### m12f
    
    simulation_directory = '/group/awetzelgrp/m12f/m12f_r7100'
 
    R90_young_m12f = np.array([18.82, 15.19, 17.20, 15.08, 14.13, 12.75, 8.12, 4.78, 4.81, 6.49, 3.37])
        
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
        v_gas = part['gas'].prop('host.velocity.principal.cylindrical')   
        
        x_f = []
        surf_dens_ratio = []
        surf_dens_ratio_over_v_vel = []
        
        for i_f in np.arange(r90/4,r90,(r90 - r90/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,age,part))
            log_surf_ratio = log_surf_dens_ratio(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,r_gas,age,part)
            surf_dens_ratio.append(log_surf_ratio)
            
            v_vel_computed = velocity_dispersion_gas(i_f,i_f+(r90 - r90/4)/20,-3,3,r_gas,v_gas,part)
            surf_dens_ratio_over_v_vel.append(np.log10((10**log_surf_ratio)/v_vel_computed))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r90/4,r90,(r90 - r90/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
            
        surf_dens_ratio_over_v_vel_at_snapshot.append(surf_dens_ratio_over_v_vel)
        l_f_sv = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio_over_v_vel = np.array(surf_dens_ratio_over_v_vel)
        if np.isnan(surf_dens_ratio_over_v_vel).all():
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(np.nan)
        else:
            j_f_sv, k_f_sv = np.polyfit(l_f_sv[np.isfinite(surf_dens_ratio_over_v_vel)],surf_dens_ratio_over_v_vel[np.isfinite(surf_dens_ratio_over_v_vel)],1)
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(j_f_sv)                                      
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    surf_dens_ratio_over_v_vel_all_galaxies.append(surf_dens_ratio_over_v_vel_at_snapshot) 
                                              
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    surf_dens_ratio_over_v_vel_all_galaxies_slope.append(surf_dens_ratio_over_v_vel_at_snapshot_slope)                                           
    
    del(part)
    
    ### m12m
    
    simulation_directory = '/group/awetzelgrp/m12m/m12m_r7100'
 
    R90_young_m12m = np.array([13.37, 12.26, 10.59, 9.77, 9.40, 9.92, 11.85, 11.25, 11.36, 10.50, 4.63])
        
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
        v_gas = part['gas'].prop('host.velocity.principal.cylindrical')   
        
        x_f = []
        surf_dens_ratio = []
        surf_dens_ratio_over_v_vel = []
        
        for i_f in np.arange(r90/4,r90,(r90 - r90/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,age,part))
            log_surf_ratio = log_surf_dens_ratio(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,r_gas,age,part)
            surf_dens_ratio.append(log_surf_ratio)
            
            v_vel_computed = velocity_dispersion_gas(i_f,i_f+(r90 - r90/4)/20,-3,3,r_gas,v_gas,part)
            surf_dens_ratio_over_v_vel.append(np.log10((10**log_surf_ratio)/v_vel_computed))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r90/4,r90,(r90 - r90/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
            
        surf_dens_ratio_over_v_vel_at_snapshot.append(surf_dens_ratio_over_v_vel)
        l_f_sv = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio_over_v_vel = np.array(surf_dens_ratio_over_v_vel)
        if np.isnan(surf_dens_ratio_over_v_vel).all():
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(np.nan)
        else:
            j_f_sv, k_f_sv = np.polyfit(l_f_sv[np.isfinite(surf_dens_ratio_over_v_vel)],surf_dens_ratio_over_v_vel[np.isfinite(surf_dens_ratio_over_v_vel)],1)
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(j_f_sv)                                      
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    surf_dens_ratio_over_v_vel_all_galaxies.append(surf_dens_ratio_over_v_vel_at_snapshot) 
                                              
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    surf_dens_ratio_over_v_vel_all_galaxies_slope.append(surf_dens_ratio_over_v_vel_at_snapshot_slope)                                           
    
    del(part)
    
    ### m12b
    
    simulation_directory = '/group/awetzelgrp/m12b/m12b_r7100'
 
    R90_young_m12b = np.array([10.69, 11.34, 10.51, 11.01, 8.97, 11.43, 8.87, 4.16, 2.44, 4.19, 3.81])
        
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
        v_gas = part['gas'].prop('host.velocity.principal.cylindrical')   
        
        x_f = []
        surf_dens_ratio = []
        surf_dens_ratio_over_v_vel = []
        
        for i_f in np.arange(r90/4,r90,(r90 - r90/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,age,part))
            log_surf_ratio = log_surf_dens_ratio(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,r_gas,age,part)
            surf_dens_ratio.append(log_surf_ratio)
            
            v_vel_computed = velocity_dispersion_gas(i_f,i_f+(r90 - r90/4)/20,-3,3,r_gas,v_gas,part)
            surf_dens_ratio_over_v_vel.append(np.log10((10**log_surf_ratio)/v_vel_computed))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r90/4,r90,(r90 - r90/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
            
        surf_dens_ratio_over_v_vel_at_snapshot.append(surf_dens_ratio_over_v_vel)
        l_f_sv = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio_over_v_vel = np.array(surf_dens_ratio_over_v_vel)
        if np.isnan(surf_dens_ratio_over_v_vel).all():
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(np.nan)
        else:
            j_f_sv, k_f_sv = np.polyfit(l_f_sv[np.isfinite(surf_dens_ratio_over_v_vel)],surf_dens_ratio_over_v_vel[np.isfinite(surf_dens_ratio_over_v_vel)],1)
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(j_f_sv)                                      
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    surf_dens_ratio_over_v_vel_all_galaxies.append(surf_dens_ratio_over_v_vel_at_snapshot) 
                                              
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    surf_dens_ratio_over_v_vel_all_galaxies_slope.append(surf_dens_ratio_over_v_vel_at_snapshot_slope)                                           
    
    del(part)
    
     ### Romeo
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'
    
    R90_young_Romeo = np.array([23.79, 24.80, 19.42, 18.98, 17.14, 14.88, 11.70, 10.71, 9.65, 6.82, 7.68])
    
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
        v_gas = part['gas'].prop('host1.velocity.principal.cylindrical')   
        
        x_f = []
        surf_dens_ratio = []
        surf_dens_ratio_over_v_vel = []
        
        for i_f in np.arange(r90/4,r90,(r90 - r90/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,age,part))
            log_surf_ratio = log_surf_dens_ratio(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,r_gas,age,part)
            surf_dens_ratio.append(log_surf_ratio)
            
            v_vel_computed = velocity_dispersion_gas(i_f,i_f+(r90 - r90/4)/20,-3,3,r_gas,v_gas,part)
            surf_dens_ratio_over_v_vel.append(np.log10((10**log_surf_ratio)/v_vel_computed))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r90/4,r90,(r90 - r90/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
            
        surf_dens_ratio_over_v_vel_at_snapshot.append(surf_dens_ratio_over_v_vel)
        l_f_sv = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio_over_v_vel = np.array(surf_dens_ratio_over_v_vel)
        if np.isnan(surf_dens_ratio_over_v_vel).all():
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(np.nan)
        else:
            j_f_sv, k_f_sv = np.polyfit(l_f_sv[np.isfinite(surf_dens_ratio_over_v_vel)],surf_dens_ratio_over_v_vel[np.isfinite(surf_dens_ratio_over_v_vel)],1)
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(j_f_sv)                                      
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    surf_dens_ratio_over_v_vel_all_galaxies.append(surf_dens_ratio_over_v_vel_at_snapshot) 
                                              
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    surf_dens_ratio_over_v_vel_all_galaxies_slope.append(surf_dens_ratio_over_v_vel_at_snapshot_slope)                                           
   
    del(part)
                                   
    ### Juliet
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomeoJuliet_r3500'
 
    R90_young_Juliet = np.array([19.45, 15.41, 13.70, 9.25, 5.71, 3.90, 4.20, 5.77, 2.89, 2.24, 4.85])
    
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
        v_gas = part['gas'].prop('host2.velocity.principal.cylindrical')   
        
        x_f = []
        surf_dens_ratio = []
        surf_dens_ratio_over_v_vel = []
        
        for i_f in np.arange(r90/4,r90,(r90 - r90/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,age,part))
            log_surf_ratio = log_surf_dens_ratio(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,r_gas,age,part)
            surf_dens_ratio.append(log_surf_ratio)
            
            v_vel_computed = velocity_dispersion_gas(i_f,i_f+(r90 - r90/4)/20,-3,3,r_gas,v_gas,part)
            surf_dens_ratio_over_v_vel.append(np.log10((10**log_surf_ratio)/v_vel_computed))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r90/4,r90,(r90 - r90/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
            
        surf_dens_ratio_over_v_vel_at_snapshot.append(surf_dens_ratio_over_v_vel)
        l_f_sv = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio_over_v_vel = np.array(surf_dens_ratio_over_v_vel)
        if np.isnan(surf_dens_ratio_over_v_vel).all():
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(np.nan)
        else:
            j_f_sv, k_f_sv = np.polyfit(l_f_sv[np.isfinite(surf_dens_ratio_over_v_vel)],surf_dens_ratio_over_v_vel[np.isfinite(surf_dens_ratio_over_v_vel)],1)
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(j_f_sv)                                      
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    surf_dens_ratio_over_v_vel_all_galaxies.append(surf_dens_ratio_over_v_vel_at_snapshot) 
                                              
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    surf_dens_ratio_over_v_vel_all_galaxies_slope.append(surf_dens_ratio_over_v_vel_at_snapshot_slope)                                           
    
    del(part)
    
    ### Romulus
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000'
 
    R90_young_Romulus = np.array([17.00, 20.04, 19.70, 16.90, 14.08, 10.84, 6.82, 9.20, 5.35, 5.36, 4.30])
    
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
        v_gas = part['gas'].prop('host1.velocity.principal.cylindrical')   
        
        x_f = []
        surf_dens_ratio = []
        surf_dens_ratio_over_v_vel = []
        
        for i_f in np.arange(r90/4,r90,(r90 - r90/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,age,part))
            log_surf_ratio = log_surf_dens_ratio(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,r_gas,age,part)
            surf_dens_ratio.append(log_surf_ratio)
            
            v_vel_computed = velocity_dispersion_gas(i_f,i_f+(r90 - r90/4)/20,-3,3,r_gas,v_gas,part)
            surf_dens_ratio_over_v_vel.append(np.log10((10**log_surf_ratio)/v_vel_computed))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r90/4,r90,(r90 - r90/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
            
        surf_dens_ratio_over_v_vel_at_snapshot.append(surf_dens_ratio_over_v_vel)
        l_f_sv = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio_over_v_vel = np.array(surf_dens_ratio_over_v_vel)
        if np.isnan(surf_dens_ratio_over_v_vel).all():
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(np.nan)
        else:
            j_f_sv, k_f_sv = np.polyfit(l_f_sv[np.isfinite(surf_dens_ratio_over_v_vel)],surf_dens_ratio_over_v_vel[np.isfinite(surf_dens_ratio_over_v_vel)],1)
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(j_f_sv)                                      
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    surf_dens_ratio_over_v_vel_all_galaxies.append(surf_dens_ratio_over_v_vel_at_snapshot) 
                                              
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    surf_dens_ratio_over_v_vel_all_galaxies_slope.append(surf_dens_ratio_over_v_vel_at_snapshot_slope)                                           
    
    del(part)
                                   
    ### Remus
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_RomulusRemus_r4000'
 
    R90_young_Remus = np.array([21.08, 19.38, 18.47, 17.16, 13.77, 11.37, 10.01, 9.91, 5.47, 5.38, 4.68])
    
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
        v_gas = part['gas'].prop('host2.velocity.principal.cylindrical')   
        
        x_f = []
        surf_dens_ratio = []
        surf_dens_ratio_over_v_vel = []
        
        for i_f in np.arange(r90/4,r90,(r90 - r90/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,age,part))
            log_surf_ratio = log_surf_dens_ratio(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,r_gas,age,part)
            surf_dens_ratio.append(log_surf_ratio)
            
            v_vel_computed = velocity_dispersion_gas(i_f,i_f+(r90 - r90/4)/20,-3,3,r_gas,v_gas,part)
            surf_dens_ratio_over_v_vel.append(np.log10((10**log_surf_ratio)/v_vel_computed))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r90/4,r90,(r90 - r90/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
            
        surf_dens_ratio_over_v_vel_at_snapshot.append(surf_dens_ratio_over_v_vel)
        l_f_sv = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio_over_v_vel = np.array(surf_dens_ratio_over_v_vel)
        if np.isnan(surf_dens_ratio_over_v_vel).all():
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(np.nan)
        else:
            j_f_sv, k_f_sv = np.polyfit(l_f_sv[np.isfinite(surf_dens_ratio_over_v_vel)],surf_dens_ratio_over_v_vel[np.isfinite(surf_dens_ratio_over_v_vel)],1)
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(j_f_sv)                                      
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    surf_dens_ratio_over_v_vel_all_galaxies.append(surf_dens_ratio_over_v_vel_at_snapshot) 
                                              
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    surf_dens_ratio_over_v_vel_all_galaxies_slope.append(surf_dens_ratio_over_v_vel_at_snapshot_slope)                                           
    
    del(part)
                                   
    ### Thelma
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'
 
    R90_young_Thelma = np.array([16.01, 11.71, 11.69, 8.77, 9.76, 8.99, 9.94, 9.89, 8.77, 7.39, 7.95])
    
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
        v_gas = part['gas'].prop('host1.velocity.principal.cylindrical')   
        
        x_f = []
        surf_dens_ratio = []
        surf_dens_ratio_over_v_vel = []
        
        for i_f in np.arange(r90/4,r90,(r90 - r90/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,age,part))
            log_surf_ratio = log_surf_dens_ratio(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,r_gas,age,part)
            surf_dens_ratio.append(log_surf_ratio)
            
            v_vel_computed = velocity_dispersion_gas(i_f,i_f+(r90 - r90/4)/20,-3,3,r_gas,v_gas,part)
            surf_dens_ratio_over_v_vel.append(np.log10((10**log_surf_ratio)/v_vel_computed))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r90/4,r90,(r90 - r90/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
            
        surf_dens_ratio_over_v_vel_at_snapshot.append(surf_dens_ratio_over_v_vel)
        l_f_sv = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio_over_v_vel = np.array(surf_dens_ratio_over_v_vel)
        if np.isnan(surf_dens_ratio_over_v_vel).all():
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(np.nan)
        else:
            j_f_sv, k_f_sv = np.polyfit(l_f_sv[np.isfinite(surf_dens_ratio_over_v_vel)],surf_dens_ratio_over_v_vel[np.isfinite(surf_dens_ratio_over_v_vel)],1)
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(j_f_sv)                                      
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    surf_dens_ratio_over_v_vel_all_galaxies.append(surf_dens_ratio_over_v_vel_at_snapshot) 
                                              
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    surf_dens_ratio_over_v_vel_all_galaxies_slope.append(surf_dens_ratio_over_v_vel_at_snapshot_slope)                                           
    
    del(part)
                                   
    ### Louise
    
    simulation_directory = '/group/awetzelgrp/m12_elvis/m12_elvis_ThelmaLouise_r4000'
 
    R90_young_Louise = np.array([22.91, 22.01, 18.36, 14.70, 13.15, 11.24, 9.16, 5.18, 5.47, 9.42, 7.44])
    
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
        v_gas = part['gas'].prop('host2.velocity.principal.cylindrical')   
        
        x_f = []
        surf_dens_ratio = []
        surf_dens_ratio_over_v_vel = []
        
        for i_f in np.arange(r90/4,r90,(r90 - r90/4)/20):
            x_f.append(Fe_H_agedependent_form(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,age,part))
            log_surf_ratio = log_surf_dens_ratio(i_f,i_f+(r90 - r90/4)/20,-3,3,0,0.1,r_star,r_gas,age,part)
            surf_dens_ratio.append(log_surf_ratio)
            
            v_vel_computed = velocity_dispersion_gas(i_f,i_f+(r90 - r90/4)/20,-3,3,r_gas,v_gas,part)
            surf_dens_ratio_over_v_vel.append(np.log10((10**log_surf_ratio)/v_vel_computed))
            
        Fe_H_rad_form_at_snapshot.append(x_f)
        l_f = np.arange(r90/4,r90,(r90 - r90/4)/20)
        x_f = np.array(x_f)
        if np.isnan(x_f).all():
            Fe_H_rad_form_at_snapshot_slope.append(np.nan)
        else:
            j_f, k_f = np.polyfit(l_f[np.isfinite(x_f)],x_f[np.isfinite(x_f)],1)
            Fe_H_rad_form_at_snapshot_slope.append(j_f)
            
        surf_dens_ratio_at_snapshot.append(surf_dens_ratio) 
        l_f_s = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio = np.array(surf_dens_ratio)
        if np.isnan(surf_dens_ratio).all():
            surf_dens_ratio_at_snapshot_slope.append(np.nan)
        else:
            j_f_s, k_f_s = np.polyfit(l_f_s[np.isfinite(surf_dens_ratio)],surf_dens_ratio[np.isfinite(surf_dens_ratio)],1)
            surf_dens_ratio_at_snapshot_slope.append(j_f_s)
            
        surf_dens_ratio_over_v_vel_at_snapshot.append(surf_dens_ratio_over_v_vel)
        l_f_sv = np.arange(r90/4,r90,(r90 - r90/4)/20)
        surf_dens_ratio_over_v_vel = np.array(surf_dens_ratio_over_v_vel)
        if np.isnan(surf_dens_ratio_over_v_vel).all():
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(np.nan)
        else:
            j_f_sv, k_f_sv = np.polyfit(l_f_sv[np.isfinite(surf_dens_ratio_over_v_vel)],surf_dens_ratio_over_v_vel[np.isfinite(surf_dens_ratio_over_v_vel)],1)
            surf_dens_ratio_over_v_vel_at_snapshot_slope.append(j_f_sv)                                      
        
    Fe_H_rad_form_all_galaxies.append(Fe_H_rad_form_at_snapshot)
    surf_dens_ratio_all_galaxies.append(surf_dens_ratio_at_snapshot) 
    surf_dens_ratio_over_v_vel_all_galaxies.append(surf_dens_ratio_over_v_vel_at_snapshot) 
                                              
    Fe_H_rad_form_all_galaxies_slope.append(Fe_H_rad_form_at_snapshot_slope)
    surf_dens_ratio_all_galaxies_slope.append(surf_dens_ratio_at_snapshot_slope)
    surf_dens_ratio_over_v_vel_all_galaxies_slope.append(surf_dens_ratio_over_v_vel_at_snapshot_slope)                                           
    
    del(part)
    
    Fe_H_rad_form_all_galaxies = np.array(Fe_H_rad_form_all_galaxies)   
    surf_dens_ratio_all_galaxies = np.array(surf_dens_ratio_all_galaxies)
    surf_dens_ratio_over_v_vel_all_galaxies = np.array(surf_dens_ratio_all_galaxies) 
    
    Fe_H_rad_form_all_galaxies_slope = np.array(Fe_H_rad_form_all_galaxies_slope)   
    surf_dens_ratio_all_galaxies_slope = np.array(surf_dens_ratio_all_galaxies_slope) 
    surf_dens_ratio_over_v_vel_all_galaxies_slope = np.array(surf_dens_ratio_all_galaxies_slope)                                      
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/Fe_H_rad_form_all_galaxies_surfdensratiotrack_100Myr_quartR90_R90_young_CHECK2', Fe_H_rad_form_all_galaxies)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/surf_dens_ratio_all_galaxies_surfdensratiotrack_100Myr_quartR90_R90_young_CHECK2', surf_dens_ratio_all_galaxies)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/surf_dens_ratio_over_v_vel_all_galaxies_surfdensratiotrack_100Myr_quartR90_R90_young_CHECK2', surf_dens_ratio_over_v_vel_all_galaxies)
    
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/Fe_H_rad_form_all_galaxies_surfdensratiotrack_100Myr_quartR90_R90_young_slope_CHECK2', Fe_H_rad_form_all_galaxies_slope)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/surf_dens_ratio_all_galaxies_surfdensratiotrack_100Myr_quartR90_R90_young_slope_CHECK2', surf_dens_ratio_all_galaxies_slope)
    ut_io.file_hdf5('/home/rlgraf/Final_Figures/surf_dens_ratio_over_v_vel_all_galaxies_surfdensratiotrack_100Myr_quartR90_R90_young_CHECK2', surf_dens_ratio_over_v_vel_all_galaxies_slope)
    
radial_analysis_form()