import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy import constants as const
import bondi_analytic as bondi
from matplotlib_settings import *
from ylabel_dictionary import *
import pickle
import os
import pdb
from scipy.ndimage import uniform_filter1d

import glob
import h5py
import pyharm

k=const.k_B
G=const.G
c=const.c
m_p=const.m_p
mu = 0.5*m_p #0.62*m_p # ????

default_params = {'xlim': None, 'show_rscale': False, 'show_divisions': True, 'show_rb': False,
        'boxcar_factor': 0, 'tmax_list': None, 'average_factor': 2, 'linestyle_list': None,
        'label_list': None, 'rescale': False, 'show_init': 0, 'trim_zone': True}

def get_zone_num(run_num, nzones=None):
    if nzones > 1 :zone_num = (run_num % (2 * (nzones - 1)))
    else: zone_num = 0
    if zone_num > nzones - 1: zone_num = 2 * (nzones - 1) - zone_num
    return zone_num

def get_mask(n_zones, r_save, overlap = None, base=8):
    # masking
    mask = []
    dx1 = np.log10(r_save[0][1]/r_save[0][0])
    x1_out = np.log10(r_save[0][-1]) + dx1 / 2.
    x1_in = np.log10(r_save[0][0]) - dx1 / 2.
    res = int(round(len(r_save[0]) / (x1_out - x1_in) * np.log10(base**2))) # resolution
    if n_zones > 1:
        if overlap is None: # calculate overlap with zone = 0
            overlap = res // 4
        for zone in range(n_zones):
            n_radii = len(r_save[zone])
            mask_temp = np.full(n_radii, True, dtype=bool)
            if zone > 0:
                mask_temp[-int(overlap):] = False
            if zone < n_zones-1:
                mask_temp[:int(overlap)] = False
            mask += [mask_temp]
        # further check if there is still an overlap, if there is, prioritize larger ann first
        for zone in range(n_zones):
            if zone > 0:
                rin_larger_ann = np.power(10., np.log10(r_save[zone-1][mask[zone-1]][0]) - dx1 / 2.)
                still_overlaps = r_save[zone] > rin_larger_ann
                mask[zone][still_overlaps] = False
    else: # onezone
        mask += [np.full(len(r_save[0]), True, dtype=bool)]
    return mask, res

def readQuantity(dictionary, quantity):

  invert = False
  if quantity == 'beta':
    #It's better to use inverse beta, if we have it.
    # Hyerin (08/09/23) modified such that Pg and Pb are averaged separately.
    try:
      quantity_index = dictionary['quantities'].index('inv_beta')
      invert = True
    except:
      print("inv_beta doesn't exist, so we will stick with beta.")
      quantity_index = dictionary['quantities'].index(quantity)
    profiles = [[list[quantity_index] for list in sublist] for sublist in dictionary['profiles']]
  elif quantity == 'eta':  #TODO this needs to be corrected such that Mdot and Edot averaged over separately first
    quantity_index_numerator = dictionary['quantities'].index('Edot')
    quantity_index_denominator = dictionary['quantities'].index('Mdot')
    profiles = [[1.-np.array(list[quantity_index_numerator])/np.array(list[quantity_index_denominator]) for list in sublist] for sublist in dictionary['profiles']]
  elif quantity == 'etaMdot':
    quantity_index_numerator = dictionary['quantities'].index('Edot')
    quantity_index_denominator = dictionary['quantities'].index('Mdot')
    profiles = [[np.array(list[quantity_index_denominator])-np.array(list[quantity_index_numerator]) for list in sublist] for sublist in dictionary['profiles']]
  #elif quantity == 'Omega':
    #quantity_index_numerator = dictionary['quantities'].index('Omega')
    #quantity_index_denominator = dictionary['quantities'].index('u^t')
    #profiles = [[abs(np.array(list[quantity_index_numerator])/np.array(list[quantity_index_denominator])) for list in sublist] for sublist in dictionary['profiles']]
  elif quantity == 'Pg':
    try:
      quantity_index = dictionary['quantities'].index('Pg')
      profiles = [[list[quantity_index] for list in sublist] for sublist in dictionary['profiles']]
    except:
      gam = 5./3. # TODO: I'm currently assuming that the adiabatic index is always this.
      quantity_index = dictionary['quantities'].index('u')
      profiles = [[np.array(list[quantity_index])*(gam-1.) for list in sublist] for sublist in dictionary['profiles']]
  elif quantity == 'Pb':
    quantity_index = dictionary['quantities'].index('b')
    profiles = [[np.array(list[quantity_index])**2/2. for list in sublist] for sublist in dictionary['profiles']]
  else:
    quantity_index = dictionary['quantities'].index(quantity)
    profiles = [[list[quantity_index] for list in sublist] for sublist in dictionary['profiles']]
  return profiles, invert

def assignTimeBins(D, profiles, ONEZONE=False, num_time_chunk=4, zone_time_average_fraction=0.5, factor=2, tmax=None):
  # Hyerin (06/20/23) assign to different time bins
  # tmax: the maximum time to plot in tB units
  n_profiles = len(profiles)
  radii = D['radii']
  n_zones = D['nzone']
  try: n_zones_eff = D['nzone_eff']
  except : n_zones_eff = n_zones
  gam=5./3.
  try:
    rB = 80.*D["r_sonic"]**2/(27.*gam)
  except:
    rB = 1.8e5 #1e5
  tB = np.power(rB,3./2)

  t_first = np.min(D["times"][0])
  t_last = np.max(D["times"][-1])
  if tmax is not None:
      # number of annuli that has its r_out > rB
      n_ann_out = 0
      for i in range(n_zones_eff):
          if radii[i][-1]>rB: n_ann_out += 1
      if ONEZONE: n_ann_out = 2 # this makes t_atB = t_total for onezone
      t_last_temp = t_atB_to_t_total(tmax*tB, n_ann_out) 
      if t_last > t_last_temp: 
          t_last = t_last_temp
          print('t<{:.3g}'.format(t_last))
      else: print("The run hasn't reached {}tB. Instead using {:.3g}tB".format(tmax, t_total_to_t_atB(t_last,n_ann_out)/tB))
  tDivList = np.array([t_first+(t_last-t_first)/np.power(factor,i+1) for i in range(num_time_chunk)])
  tDivList = tDivList[::-1] # in increasing time order

  #if 'moving_rin' in D['runName'] or 'combine_outer' in D['runName']: # temporary Hyerin (07/31/23)
  #  n_zones_eff -= 2
  usableProfiles = [[[] for _ in range(num_time_chunk)] for _ in range(n_zones_eff)] # (n_zones, num_time_chunk) dimension
  r_save = [None]*n_zones_eff
  num_save = np.zeros((n_zones_eff,num_time_chunk))
  try: base = D["base"]
  except: base = 8

  for i,profile in enumerate(profiles):
    times = D["times"][i]
    run_idx = D["runIndices"][i]
    if n_zones_eff == 1: iteration = 0
    else: iteration = np.maximum(np.ceil(run_idx/(n_zones_eff-1)),1)
    if 1: #iteration % 2 == 1 or run_idx % (n_zones-1)==0: # selecting only outwards direction
        if len(times)<1:
            #pdb.set_trace()
            continue # skip this iteration
        if tmax is not None and times[-1]> t_last and (not ONEZONE):
            continue
        sorting = np.argwhere(tDivList<times[0]) # put one annulus run in the same time bin, even if one zone run corresponds to multiple bins
        #sorting = np.argwhere([np.any(tDiv<times) for tDiv in tDivList]) # put one annulus run in the same time bin, if any of the time in one annulus is in the range of the bin
        bin_num = sorting[-1,0] if len(sorting)>0 else None # take the largest number as the bin
        zone_num = get_zone_num(run_idx, n_zones_eff) #n_zones_eff-1 -int(np.floor(np.log(radii[i][0])/np.log(base))-(base<2))
       
        # taken from above
        if len(times) > 1 and bin_num is not None or ONEZONE: # If there's only 1 output, skip this run
          delt = np.ones(len(times)) #np.gradient(times) #
          if ONEZONE:
            for bin_num in range(num_time_chunk):
              if bin_num == num_time_chunk-1:
                averaging_mask = times > tDivList[bin_num]
              else:
                averaging_mask = (times > tDivList[bin_num]) & (times < tDivList[bin_num+1])
              integrand = np.transpose(np.squeeze(np.array(profile))[averaging_mask])
              usableProfiles[zone_num][bin_num].append(np.sum(integrand*delt[averaging_mask], axis=1) / np.sum(delt[averaging_mask]))
          else:
            averaging_mask = times >= times[-1] - (times[-1]-times[0])*zone_time_average_fraction

            #Always include the last snapshot. Specify this with zone_time_average_fraction == 0.
            if np.sum(averaging_mask) == 0:
              averaging_mask[-1] = True

            #An average, taking care to weight different timesteps appropriately.  There's an annoying squeeze and transpose here.
            #if quantity == "Mdot": # only selecting positive mdots
                #profile = np.array(profile)
                #profile[profile<0] = 0
            integrand = np.transpose(np.squeeze(np.array(profile))[averaging_mask])
            usableProfiles[zone_num][bin_num].append(np.sum(integrand*delt[averaging_mask], axis=1) / np.sum(delt[averaging_mask]))

            num_save[zone_num][bin_num] += 1 # record how many saves per bin

          if r_save[zone_num] is None:
            r_save[zone_num] = radii[i]

  for zone in range(n_zones_eff):
      # catch any cases where there are no data
      try: zone_number_sequence = np.array(D["zones"])
      except:
        zone_number_sequence = np.array([get_zone_num(i, n_zones_eff) for i in range(len(radii))])
      matchingIndices = np.where(zone_number_sequence == zone)[0]
      if len(usableProfiles[zone][0]) == 0 and not ONEZONE:
          fill_i = np.argmin([abs(tDivList[0]-D["times"][i][0]) for i in matchingIndices])
          fill_idx = matchingIndices[fill_i]
          print("Found an empty bin, filling with the closest starting at t = {:.3g} and the # of the annulus is {}".format(D["times"][fill_idx][0], fill_idx)) #D["times"][fill_idx][0], D["times"][fill_idx+1][0], D["times"][fill_idx-1][0], tDivList[0])
          
          times = D["times"][fill_idx]
          delt = np.gradient(times)
          averaging_mask = times >= times[-1] - (times[-1]-times[0])*zone_time_average_fraction
          integrand = np.transpose(np.squeeze(np.array(profiles[fill_idx]))[averaging_mask])
          usableProfiles[zone][0].append(np.sum(integrand*delt[averaging_mask], axis=1) / np.sum(delt[averaging_mask]))
          r_save[zone] = radii[fill_idx]
          num_save[zone][0] += 1

  tDivList = np.append(tDivList,t_last)
  return tDivList, usableProfiles, r_save, num_save

def plotIC(ax,profiles, n_zones_eff, zone_number_sequence, radii, invert, color, quantity=None):
    init_r = np.array([])
    init_plot = np.array([])

    n_radii = len(radii[0])
    for zone in range(n_zones_eff):
        matchingIndices = np.where(zone_number_sequence == zone)[0]
        mask = np.full(len(radii[matchingIndices[0]]), True, dtype=bool)
        if zone > 0:
          mask[-int(n_radii/2):] = False
        init_r = np.concatenate([init_r,radii[matchingIndices[0]][mask]])
        init_plot = np.concatenate([init_plot,profiles[matchingIndices[0]][0][mask]])

    order = np.argsort(init_r)
    if "Omega" in quantity:
        init_plot = (init_plot * np.power(init_r, 3./2)) # normalize by Omega_K
    if invert: ax.plot(np.array(init_r[order]), 1./init_plot[order], color=color, ls=':', lw=1, alpha=1)
    else: ax.plot(np.array(init_r[order]), init_plot[order], color=color, ls=':', lw=1, alpha=1)
    return ax


def plotProfiles(listOfPickles, quantity, output=None, colormap='turbo', color_list=None, linestyle_list=None, figsize=(8,6), flip_sign=False, show_divisions=True, show_rb=False, zone_time_average_fraction=0, 
  xlabel=None, ylabel=None, xlim=None, ylim=None, label_list=None, fig_ax=None, formatting=True, finish=True, rescale=False, rescale_value=1, rescale_Mdot=False, rescale_rho=False, flatten_rho=False, cycles_to_average=1, trim_zone=True, show_init=False, show_gizmo=False, show_bondi=False, show_rscale=False, num_time_chunk=4, boxcar_factor=0, average_factor=2, eta_norm_Bondi=False, lw=2, tmax_list=None):

  if isinstance(listOfPickles, str):
    listOfPickles = [listOfPickles]

  if linestyle_list is None:
    linestyle_list = ['-']*len(listOfPickles)
  times_list = [None]*len(listOfPickles)
  Mdot_list = [None]*len(listOfPickles)
  phib_list = [None]*len(listOfPickles)
  rB_list = [None]*len(listOfPickles)
  if isinstance(rescale, bool):
    rescale = [rescale]*len(listOfPickles)

  #Changes some defaults.
  matplotlib_settings()

  #If you want, provide your own figure and axis.  Good for multipanel plots.
  if fig_ax is None:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
  else:
    fig, ax = fig_ax

  if flatten_rho and quantity=='rho':
      # override
      num_time_chunk=1
      color_list=['k']*len(listOfPickles)
      show_gizmo=False
      show_bondi=False
      show_rscale=False
      show_divisions=False
      boxcar_factor=4
      label_list=[None]*len(listOfPickles)
  

  #Profiles are pre-computed.
  #See ../compute_scripts/computeProfiles.py for how this file is formatted.

  for sim_index in range(len(listOfPickles)):
    with open(listOfPickles[sim_index], 'rb') as openFile:
      D = pickle.load(openFile)
    try:
        r_sonic = D["r_sonic"]
    except:
        r_sonic = np.sqrt(1e5)
    gam = 5./3.
    rB = 80. * r_sonic**2 / (27. * gam)
    
    radii = D['radii']
    #Formula that produces the zone number of a given run.  It would have been better to have this in some other file though.
    n_zones = D['nzone']
    try: n_zones_eff = D['nzone_eff']
    except: n_zones_eff = n_zones
    #print(n_zones)
    #if 'moving_rin' in D['runName'] or 'combine_outer' in D['runName']: # temporary Hyerin (07/31/23)
    #  n_zones_eff -= 2
    try:
        base = D["base"]
    except:
        base = 8
    if n_zones_eff > 1:
      #zone_number_sequence = np.array([np.abs(np.abs(n_zones-1 - (i % (2*n_zones-2)))-(n_zones-1)) for i in range(len(radii))])
      try:
        zone_number_sequence = np.array(D["zones"])
      except:
        #zone_number_sequence = np.array([n_zones_eff-1 - int(np.floor(np.log(radii[i][0])/np.log(base))) for i in range(len(radii))])
        zone_number_sequence = np.array([get_zone_num(i, n_zones_eff) for i in range(len(radii))])
    else:
      zone_number_sequence = np.full(len(radii), 0)
    
    if 'onezone' in listOfPickles[sim_index]: ONEZONE = True
    else: ONEZONE = False
    if type(tmax_list) == list: tmax = tmax_list[sim_index]
    else: tmax = tmax_list
    
    rescalingFactor = 1.0
    if rescale[sim_index]:
        #Find rescaling factor.
        rescalingFactor = 1./rescale_value
    if (rescale_Mdot and 'Mdot' in quantity) or (rescale_rho and quantity == 'rho'):
        # only rescale the Mdot or rho value depending on the density at Bondi radius
        Mdot_analytic = bondi.get_quantity_for_rarr([rB], 'Mdot', rs=r_sonic)[0]
        rho_analytic = bondi.get_quantity_for_rarr([rB*100], 'rho', rs=r_sonic)[0]
        profiles, invert = readQuantity(D, 'rho')
        _, usableProfiles_rho, r_save, _ = assignTimeBins(D, profiles, ONEZONE, 1, zone_time_average_fraction, average_factor, tmax) # TODO: generalize such that the rescaling factor changes over time
        mask_list, res = get_mask(n_zones_eff, r_save, base=base)
        dist_rB = rB
        #zone_rB = 0
        rho_save = min(np.mean(usableProfiles_rho[0][0], axis=0)) # method1. replace with minimum rho
        for zone in range(n_zones_eff):
            mask = mask_list[zone]
            # method1. replace with minimum rho
            if r_save[zone][0] < rB:
                rhomin_temp = min(np.mean(usableProfiles_rho[zone][0], axis=0))
                if rhomin_temp < rho_save: rho_save = rhomin_temp
            
            # method2. replace with rho at rB
            #if r_save[zone][-1] > rB: #r_save[zone][0] < rB and 
            if min(abs(r_save[zone][mask]-rB)) < dist_rB:
                irB = np.argmin(abs(r_save[zone][mask]-rB))
                zone_rB = zone
                dist_rB = abs(r_save[zone][mask][irB]-rB)
        #print(zone_rB)
        rho_save = np.mean(usableProfiles_rho[zone_rB][0], axis=0)[mask_list[zone_rB]][irB] # method2. replace with rho at rB
        if ('Mdot' in quantity): rescalingFactor = rho_analytic/Mdot_analytic/rho_save
        if (quantity == 'rho'): rescalingFactor = 1./rho_save
        print("rB={:.3g}, rho_save={:.3g}, rho_analytic={:.3g}".format(rB, rho_save, rho_analytic))
        rB_list[sim_index] = rB

    if cycles_to_average > 0:
      if color_list is None:
        color_list = [None]*len(listOfPickles)
      if label_list is None:
        label_list = [None]*len(listOfPickles)
      profiles, invert = readQuantity(D, quantity)
      profiles = profiles
      n_profiles = len(profiles)

      r_plot = np.array([])
      values_plot = np.array([])
      #Only plot the very last iteration.  Recommended for comparing different initial conditions.
      last_time = 0
      for zone in range(n_zones_eff):
        matchingIndices = np.where(zone_number_sequence == zone)[0]
        if zone in [0,n_zones_eff-1]:
          #Tricky edge case: there are half as many instances of the zones at the ends.
          cycles_to_average_halved = int(np.ceil(cycles_to_average/2))
        else: # keep original cycles_to_average
          cycles_to_average_halved = cycles_to_average
        indicesToAverage = matchingIndices[-np.min([len(matchingIndices),cycles_to_average_halved]):]
        print(sim_index, zone, len(matchingIndices))

        #Before moving on, we're going to average profiles within each individual run.
        selectedProfiles = [profiles[i] for i in indicesToAverage]
        selectedProfileTimes = [D['times'][i] for i in indicesToAverage]
        usableProfiles = []
            

        for run_index in range(len(selectedProfiles)):
          t = selectedProfileTimes[run_index]
          if t[-1] > last_time:
            last_time = t[-1]
          if len(t) > 1: # If there's only 1 output, skip this run
            delt = np.gradient(t)
            averaging_mask = t >= t[-1] - (t[-1]-t[0])*zone_time_average_fraction

            #Always include the last snapshot. Specify this with zone_time_average_fraction == 0.
            if np.sum(averaging_mask) == 0:
              averaging_mask[-1] = True

            #An average, taking care to weight different timesteps appropriately.  There's an annoying squeeze and transpose here.
            integrand = np.transpose(np.squeeze(np.array(selectedProfiles[run_index]))[averaging_mask])
            usableProfiles.append(np.sum(integrand*delt[averaging_mask], axis=1) / np.sum(delt[averaging_mask]))
          else: # delete the history of the skipped run
            indicesToAverage = np.delete(indicesToAverage, run_index)
            #selectedProfiles = np.delete(selectedProfiles, run_index)
            #selectedProfileTimes = np.delete(selectedProfileTimes, run_index)

        times_list[sim_index] = last_time

        #Now, given an average within each run, average each separate run.
        plottable = np.mean(usableProfiles, axis=0)

        #Flip the quantity upside-down, usually for inv_beta.
        if invert:
          plottable = 1.0 / plottable

        #Next, optionally mask out regions that likely have wall glitches by only taking the central half of the radii
        finalMatchingIndex = indicesToAverage[-1]
        n_radii = len(radii[finalMatchingIndex])
        mask = np.full(n_radii, True, dtype=bool)
        if n_zones_eff > 1:
          if trim_zone:
            if zone > 0:
              mask[-int(n_radii/4):] = False
            if zone < n_zones_eff - 1:
              mask[:int(n_radii/4)] = False

        r_plot = np.concatenate([r_plot,radii[finalMatchingIndex][mask]])
        values_plot = np.concatenate([values_plot,rescalingFactor*plottable[mask]*(-1)**int(flip_sign)])

      r_plot = np.squeeze(np.array(r_plot))
      values_plot = np.squeeze(np.array(values_plot))
      order = np.argsort(r_plot)
      if label_list is None: label = '__nolegend__'
      else: label = label_list[sim_index]
      ax.plot(r_plot[order], values_plot[order], color=color_list[sim_index], ls=linestyle_list[sim_index], lw=2,label=label)
      if show_init and (quantity == "rho" or quantity == "T" or quantity=="beta" or quantity=='u^r'):
        # show initial conditions
        ax = plotIC(ax,profiles, n_zones_eff, zone_number_sequence, radii, invert, 'k')


    else:
      # HYERIN: split into time chunks
      print(listOfPickles[sim_index])
      
      if quantity == 'eta':
          profiles, invert = readQuantity(D, 'Edot')
          tDivList, usableProfiles_num, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor, tmax)
          profiles, _ = readQuantity(D, 'Mdot')
          _, usableProfiles_den, _, _ = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor, tmax)
          #usableProfiles = 1.-usableProfiles_num/usableProfiles_den
      elif quantity == 'etaMdot':
          profiles, invert = readQuantity(D, 'Edot')
          tDivList, usableProfiles_num, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor, tmax)
          profiles, _ = readQuantity(D, 'Mdot')
          _, usableProfiles_den, _, _ = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor, tmax)
      elif quantity == 'u^r':
          profiles, invert = readQuantity(D, 'Mdot')
          tDivList, usableProfiles_num, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor, tmax)
          profiles, _ = readQuantity(D, 'rho')
          _, usableProfiles_den, _, _ = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor, tmax)
      #elif quantity == 'beta':
      #    profiles, invert = readQuantity(D, 'Pg')
      #    tDivList, usableProfiles_num, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor, tmax)
      #    try: profiles, _ = readQuantity(D, 'Pb')
      #    except: continue
      #    _, usableProfiles_den, _, _ = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor, tmax)
      elif quantity == 'phib':
          try: profiles, invert = readQuantity(D, 'Phib')
          except:  continue
          tDivList, usableProfiles_num, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor,tmax)
          profiles, _ = readQuantity(D, 'Mdot')
          _, usableProfiles_den, _, _ = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor,tmax)
      else:
          try: profiles, invert = readQuantity(D, quantity)
          except: continue
          tDivList, usableProfiles, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor, tmax)
      
      if color_list is None: colors=plt.cm.gnuplot(np.linspace(0.9,0.3,num_time_chunk))
      #if color_list is None: colors=plt.cm.plasma(np.linspace(0.9,0.1,num_time_chunk))
      else: colors= [color_list[sim_index]]
      for b in range(num_time_chunk):
        r_plot = np.array([])
        values_plot = np.array([])
        print("{}: t={:.3g}-{:.3g}, # of run summed is {}".format(b,tDivList[b],tDivList[b+1], num_save[:,b]))
        for zone in range(n_zones_eff):
          #Now, given an average within each run, average each separate run.
          if quantity == 'eta':
            plottable_num = np.mean(usableProfiles_num[zone][b], axis=0)
            plottable_den = np.mean(usableProfiles_den[zone][b], axis=0)
            if zone == n_zones_eff-1: # r=10
                i10 = np.argmin(abs(r_save[zone]-10))
                if eta_norm_Bondi: Mdot_save = rescale_value # TODO: change to directly calculating Mdot_B
                else: Mdot_save = plottable_den[i10]
            plottable = plottable_den- plottable_num #+ 4.5e-3
            #plottable = np.mean(usableProfiles[zone][b],axis=0)
            invert = False
          elif quantity == 'etaMdot':
            plottable_num = np.mean(usableProfiles_num[zone][b], axis=0)
            plottable_den = np.mean(usableProfiles_den[zone][b], axis=0)
            plottable = plottable_den - plottable_num
            invert = False
          elif quantity == "u^r":
            plottable_num = np.mean(usableProfiles_num[zone][b], axis=0)
            plottable_den = np.mean(usableProfiles_den[zone][b], axis=0)*(4.*np.pi*r_save[zone]**2) # density averaged
            plottable = -plottable_num/plottable_den
            invert = False
          elif "Omega" in quantity:
            plottable = (np.mean(usableProfiles[zone][b], axis=0) * np.power(r_save[zone],3./2)) # normalize by Omega_K
          elif "phib" in quantity:
            plottable = np.mean(usableProfiles_num[zone][b], axis=0)
            plottable_den = np.mean(usableProfiles_den[zone][b], axis=0)
            if zone == n_zones_eff-1: # r=10
                i10 = np.argmin(abs(r_save[zone]-10))
                Mdot_save = plottable_den[i10]
          #elif quantity == "beta":
          #  plottable_num = np.mean(usableProfiles_num[zone][b], axis=0)
          #  plottable_den = np.mean(usableProfiles_den[zone][b], axis=0)
          #  plottable = plottable_num/plottable_den 
          #  invert = False
          else:
            plottable = np.mean(usableProfiles[zone][b], axis=0)

          #Flip the quantity upside-down, usually for inv_beta.
          if invert:
            plottable = 1.0 / plottable

          #Next, optionally mask out regions that likely have wall glitches by only taking the central half of the radii
          #if 'moving_rin' in D['runName']: # temporary Hyerin (07/31/23)
          #  n_radii = int(len(r_save[n_zones_eff-1])*2/(n_zones+1))
          #  
          #  mask = np.full(len(r_save[zone]), True, dtype=bool)
          #  if trim_zone:
          #      if zone != n_zones_eff-1:
          #          mask[:]=False
          #else:
          #  n_radii = len(r_save[n_zones_eff-1])#len(r_save[zone]) # only take smallest annuli'ls n_radii
          #  mask = np.full(len(r_save[zone]), True, dtype=bool)
          #  if n_zones_eff > 1:
          #    if trim_zone:
          #      if zone > 0:
          #        mask[-int(n_radii/4):] = False
          #      if zone < n_zones_eff - 1:
          #        mask[:int(n_radii/4)] = False
          mask_list, res = get_mask(n_zones_eff, r_save, base=base)
          mask = mask_list[zone]

          r_plot = np.concatenate([r_plot,r_save[zone][mask]])
          values_plot = np.concatenate([values_plot,rescalingFactor*plottable[mask]*(-1)**int(flip_sign)])

        r_plot = np.squeeze(np.array(r_plot))
        values_plot = np.squeeze(np.array(values_plot))
        if quantity == "eta":
            values_plot /= Mdot_save # divide by  Mdot at r=10
        if quantity == "phib":
            values_plot /= np.sqrt(Mdot_save)
        order = np.argsort(r_plot)
        if label_list is None: label = 't={:.5g} - {:.5g}'.format(tDivList[b],tDivList[b+1]) #r'$2^{{{}}} - 2^{{{}}}$'.format(-num_time_chunk+b,-num_time_chunk+b+1)#+r' [$t_{\rm run}$]' #
        else: label = label_list[sim_index]
        if flatten_rho and quantity=='rho': values_plot *= np.power(r_plot, 1.1) # just to test Xu+23 rho ~ r^{-0.8}
        if boxcar_factor == 0:
            boxcar_avged = values_plot[order]
        else:
            boxcar_avged = uniform_filter1d(values_plot[order], size=res//boxcar_factor) # (07/12/23) boxcar averaging
        ax.plot(r_plot[order], boxcar_avged, color=colors[b], ls=linestyle_list[sim_index], lw=lw, label=label)
        if (quantity=='eta' or (num_time_chunk == 1) or quantity=='Omega') and linestyle_list[sim_index]!='': ax.plot(r_plot[order], -boxcar_avged, color=colors[b], ls=':', lw=lw+1)
        if rescale_Mdot and quantity == 'Mdot':
            iEH = np.argmin(abs(r_plot[order]-2))
            Mdot_list[sim_index] = boxcar_avged[iEH]
            print("Mdot/MdotB = {:.3g}".format(Mdot_list[sim_index]))
        if quantity == 'T':
            Tinf = boxcar_avged[-1]
            print("RB = {:.5g} from Tinf = {:.5g}".format(1./(gam * Tinf),Tinf))
            rB_list[sim_index] = 1./(gam*Tinf) # replace if this is available
        if quantity == 'phib':
            iEH = np.argmin(abs(r_plot[order]-2))
            phib_list[sim_index] = boxcar_avged[iEH]
            print("phib at EH = {:.5g}".format(phib_list[sim_index]))

        if show_init and (quantity == "rho" or quantity == "T" or quantity=="beta" or quantity=='u^r' or 'Omega' in quantity):
            # show initial conditions
            #pdb.set_trace()
            ax = plotIC(ax,profiles, n_zones_eff, zone_number_sequence, radii, invert, color='k', quantity=quantity)

  #Formatting
  if formatting:
    if xlabel is None:
      xlabel = 'Radius [$r_g$]'
      ax.set_xlabel(xlabel)
    if ylabel is None:
      ylabel = variableToLabel(quantity)
      if any(rescale) or (rescale_Mdot and 'Mdot' in quantity) or (rescale_value != 1):
        ylabel = ylabel.replace('arb. units', r'$\dot{M}_B$')
      if eta_norm_Bondi and quantity=='eta':
        ylabel = r'$\overline{\dot{M}-\dot{E}}/\dot{M}_B$'
      if flatten_rho and quantity=='rho':
        ylabel = r'$\langle \rho \rangle r^{1.1}$ [arb. units]'
      if flip_sign: ylabel = '-' + ylabel
      ax.set_ylabel(ylabel)  
      ax.set_xscale('log')
      ax.set_yscale('log')
      ax.set_xlim(xlim)
      ax.set_ylim(ylim)
    if show_divisions:
      divisions = []
      #for zone in range(n_zones):
      #  divisions.append(radii[zone][-1])
      #  if (zone == n_zones-1) | (zone == n_zones-2):
      #    divisions.append(radii[zone][0])
      divisions = [base**i for i in range(n_zones+2)]
      for div in divisions:
        ax.plot([div]*2, ax.get_ylim(), alpha=0.2, color='grey', lw=1)


  if show_rb:
      ax.axvline(rB, color='grey', lw=1, alpha=1, ls='--')
  if show_bondi:
    # Bondi analytic overplotting
    xlim = ax.get_xlim()
    r_bondi = np.logspace(np.log10(max(2,xlim[0])), np.log10(xlim[1]), 50)
    analytic_sol = bondi.get_quantity_for_rarr(r_bondi, quantity, rs=r_sonic)
    if quantity == "eta": 
        print(bondi.get_quantity_for_rarr(r_bondi, 'eta', rs=r_sonic)[0], bondi.get_quantity_for_rarr(r_bondi, 'RB', rs=r_sonic)[0])
    if quantity == "Mdot" and rescale_value != 1:
        analytic_sol *= rescalingFactor
    #    analytic_sol = np.ones(len(r_bondi))*(-1.)*27.*gam/(80.*(gam-1)*r_sonic**2) # look at my rel. Bondi note where gam=5/3
    #    print(analytic_sol[0])

    if analytic_sol is not None:# and not rescale:
      if quantity == "rho": label = "Bondi analytic"
      else: label = "__nolegend__"
      if rescale_Mdot and quantity == 'Mdot':#any(rescale):
          analytic_sol /= analytic_sol #*= rescalingFactor
      ax.plot(r_bondi, analytic_sol, color='slategrey',label=label, lw=10, ls='-', zorder=-100,alpha=0.5)
      ax.plot(r_bondi, -analytic_sol, color='slategrey', lw=10, ls=':', zorder=-100,alpha=0.5)
    #rb = 1e5
    #rho0 = bondi.get_quantity_for_rarr([1e8], quantity, rs=r_sonic)
    #if 'rho' in quantity:
    #    ax.plot(r_bondi,rho0 * (r_bondi + rb) / r_bondi,'k:')
  if show_gizmo:
    dump_temp = pyharm.load_dump(sorted(glob.glob(sorted(glob.glob(D["runName"]+"/*[0-9][0-9][0-9][0-9][0-9]/"))[0]+"*phdf"))[0])
    fname_gz = dump_temp["datfn"].replace("../","/n/holylfs05/LABS/bhi/Users/hyerincho/grmhd//data/")
    if "txt" in fname_gz:
        dat_gz=np.loadtxt(fname_gz)#"/n/holylfs05/LABS/bhi/Users/hyerincho/grmhd//data/gizmo/031623_100Myr/dat.txt")
        r_gizmo=dat_gz[:,0]
        rho_gizmo = dat_gz[:,1]
        T_gizmo=dat_gz[:,2]
        vr_gizmo=dat_gz[:,3]
    else:
        #fname = "/n/holylfs05/LABS/bhi/Users/hyerincho/grmhd/data/gizmo/021924_bp10_64_all/new_magbp10_only_exact_snap_deposit64_010.hdf5" 
        f = h5py.File(fname_gz,"r")
        coord = f["/PartType0_dimless/Coordinates"][:]
        rho = f["/PartType0_dimless/Density"][:]#*1e10*u.Msun/u.kpc**3
        theta = f["/PartType0_dimless/Temperature"][:]#*u.K
        v_c = f["/PartType0_dimless/Velocities"][:]#*u.km/u.s
        B = f["/PartType0_dimless/MagneticFields"][:]
        beta = 2 * rho * theta / (B ** 2).sum(axis=1)
        # only keeping valid coordinates
        zero_idx = (coord[:,1] == 0) | (coord[:,2] == 0) # if th or phi = 0, remove that coordinates
        coord_valid = coord[~zero_idx,:]
        rho_valid = rho[~zero_idx]
        theta_valid = theta[~zero_idx]
        v_c_valid = v_c[~zero_idx,:]
        beta_valid = beta[~zero_idx]
        r_valid = coord_valid[:,0]
        r_valid_set = sorted(set(r_valid))
        th_valid_set = sorted(set(coord_valid[:,1]))
        phi_valid_set = sorted(set(coord_valid[:,2]))
        n_radii = len(r_valid_set)
        n_angles = len(th_valid_set)*len(phi_valid_set)
        for r_temp in r_valid_set:
            if (len(np.where(r_valid == r_temp)[0]) != n_angles):
                print("ERROR")
        idx = np.argsort(r_valid)
        r_gizmo = r_valid[idx].reshape(-1,n_angles)[:,0]
        rho_gizmo = rho_valid[idx].reshape(-1,n_angles).mean(axis=1)
        T_gizmo = theta_valid[idx].reshape(-1,n_angles).mean(axis=1)
        vr_gizmo = v_c_valid[idx, 0].reshape(-1,n_angles).mean(axis=1)
        beta_gizmo = 1./((1./beta_valid[idx].reshape(-1,n_angles)).mean(axis=1))
    to_plot=None
    min_rgizmo = r_gizmo[0]
    if quantity=='rho':
      to_plot = rho_gizmo
    elif quantity=='u':
      to_plot = T_gizmo*rho_gizmo*3/2
    elif quantity=='Pg':
      to_plot = T_gizmo*rho_gizmo
    elif quantity=='T':
      to_plot = T_gizmo
    elif quantity=='beta':
      to_plot = beta_gizmo
    else:
      r_gizmo=[] ;to_plot = None #[]
    if to_plot is not None:
        if quantity == "rho": label = "GIZMO"
        else: label = "__nolegend__"
        ax.plot(r_gizmo,to_plot,'b-',lw=5,label=label, zorder=-100,alpha=0.3)

    # extend inwards with expected Bondi solution
    if len(r_gizmo)>1:
        r_bondi = np.logspace(np.log10(2), np.log10(min_rgizmo), 50)
    else:
        xlim = ax.get_xlim()
        r_bondi = np.logspace(np.log10(max(2,xlim[0])), np.log10(xlim[1]), 50)
    label = "__nolegend__"
    # extending GIZMO solution to Bondi solution
    #if quantity != "Mdot":
    #    analytic_sol = bondi.get_quantity_for_rarr(r_bondi, quantity, rs=r_sonic)
    #    if quantity == "rho" or quantity == "T":
    #        analytic_sol *= to_plot[0]/analytic_sol[-1]
    #        if quantity == "rho": label = "Bondi extension"
    #else:
    #    rho = bondi.get_quantity_for_rarr([min_rgizmo],'rho',rs=r_sonic)
    #    Mdot = bondi.get_quantity_for_rarr([min_rgizmo], 'Mdot', rs=r_sonic)
    #    analytic_sol = [Mdot/rho[0] * rho_gizmo[0]]*len(r_bondi)
    #if analytic_sol is not None: ax.plot(r_bondi,analytic_sol,'b-.',label=label,lw=5, zorder=-100,alpha=0.2)

  if show_rscale!=False:
    # show density scalings
    rarr= np.logspace(np.log10(2),np.log10(r_sonic),20) #*1000
    if "rho" in quantity and (show_rscale==True or "rho" in show_rscale):
        if r_sonic > 100: 
            #factor1 = 7e-8*1e5/r_sonic**2#7e-7
            factor1 = 1e-8*1e5/r_sonic**2#7e-7
            factor2 = factor1 * 5000
        else: 
            factor1 = 1e5/r_sonic**2*5e-9 #8e-9#
            factor2 = factor1* 10
        ax.plot(rarr,np.power(rarr/1e3,-1)*factor1,'g-',alpha=0.5,lw=2)#,label=r'$r^{-1}$')
        ax.text(rarr[len(rarr)//3], np.power(rarr[len(rarr)//3]/1e3,-1)*factor1/np.power(r_sonic,2./3), r'$r^{-1}$')
        ax.plot(rarr,np.power(rarr/1e3,-3./2.)*factor2,'b-',alpha=0.5,lw=2)#,label=r'$r^{-3/2}$')
        ax.text(rarr[len(rarr)//2], np.power(rarr[len(rarr)//2]/1e3,-3./2.)*factor2*2, r'$r^{-3/2}$', clip_on=True)
        #ax.plot(rarr,np.power(rarr/1e3,-1.5)*factor,'g:',alpha=0.3,lw=10,label=r'$r^{-1.5}$')
        #rb=1e5
        #rho0=np.power(3e-6,1.5)
        #ax.plot(rarr,rho0 * (rarr + rb) / rarr,'k-')
        #ax.plot(rarr,np.power(rarr/1e3,-1/2)*factor,'g:',alpha=0.3,lw=10,label=r'$r^{-1/2}$')
    if "T" in quantity and (show_rscale==True or "T" in show_rscale):
        if r_sonic > 100: factor=5e-2
        else: factor=1e-1
        ax.plot(rarr,np.power(rarr,-1)*factor,'b-',alpha=0.5,lw=2)#,label=r'$r^{-1}$')
        ax.text(rarr[len(rarr)//4], np.power(rarr[len(rarr)//4],-1)*factor/np.power(r_sonic,3./5), r'$r^{-1}$')
    if quantity == "beta" and (show_rscale==True or "beta" in show_rscale):
        # TODO: should I show this? idk
        factor=1e7/r_sonic**2#
        ax.plot(rarr,np.power(rarr/1e3,3/2.)*factor,'g-',alpha=0.5,lw=2) #,label=r'$r^{3/2}$')
        ax.text(rarr[-1], np.power(rarr[-1]/1e3,3/2)*factor/10, r'$r^{3/2}$')
    if quantity == "phib" and (show_rscale==True or "phib" in show_rscale):
        factor=1/2#
        ax.plot(rarr,np.power(rarr,1)*factor,'g-',alpha=0.5,lw=2)#,label=r'$r^{1}$')
        ax.text(rarr[len(rarr)//2], np.power(rarr[len(rarr)//2],1)*factor/3, r'$r^{1}$')
    if quantity == "u^r" and (show_rscale==True or "u^r" in show_rscale):
        factor=1/10
        ax.plot(rarr,np.power(rarr,-1/2)*factor,'g-',alpha=0.5,lw=2)#,label=r'$r^{1}$')
        ax.text(rarr[len(rarr)//2], np.power(rarr[len(rarr)//2],-1/2)*factor/10, r'$r^{-1/2}$')
  # legends
  if formatting:
    for sim_index in range(len(listOfPickles)):
      if cycles_to_average > 0: ax.plot([], [], color=color_list[sim_index], lw=2, label=label_list[sim_index], ls=linestyle_list[sim_index]) # +', t={:.2g}'.format(times_list[sim_index])
      ax.legend(loc='best', frameon=False)
      fig.tight_layout()

  #Either show or save.
  if finish:
    if output is None:
      fig.show()
    else:
      fig.savefig(output)
      plt.close(fig)
 
  #if rescale_Mdot and quantity=='Mdot':
  #    return [Mdot_list, rB_list]
  #if quantity == 'T':
  #    return rB_list
  #if quantity == 'phib':
  #    return phib_list
  #else: return None
  if fig_ax is not None:
        return (fig, ax)

if __name__ == '__main__':
