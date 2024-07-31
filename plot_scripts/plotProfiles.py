import os
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle

def get_mask(dictionary):
    # basic parsing
    base = dictionary["base"]
    n_zones_eff = dictionary["nzones_eff"]
    active_range = dictionary['active_range']
    radii = dictionary['radii']
    n_radii = len(radii)
    
    # figure out the base resolution
    dx1 = np.log10(radii[1]/radii[0])
    x1_out = np.log10(radii[-1]) + dx1 / 2.
    x1_in = np.log10(radii[0]) - dx1 / 2.
    res = int(round(len(radii) / (x1_out - x1_in) * np.log10(base**2))) # resolution
    overlap = res // 4
    
    mask = []
    for zone in range(n_zones_eff):
        mask_temp = np.full(n_radii, True, dtype=bool)
        mask_temp[radii < active_range[zone][0]] = False
        mask_temp[radii > active_range[zone][1]] = False

        # mask the overlap region
        active = np.argwhere(mask_temp)[:,0]
        mask_temp[:active[0]+overlap] = False
        mask_temp[active[-1]+1-overlap:] = False
        mask += [mask_temp]

    return mask

def readQuantity(dictionary, quantity):
    invert = False
    if quantity == 'beta':
        if 'inv_beta' in dictionary['quantities']:
            quantity = 'inv_beta'
            invert = True
        else:
            print("inv_beta doesn't exist, so we will stick with beta.")
        quantity_index = dictionary['quantities'].index(quantity)
        profiles = [list[quantity_index] for list in dictionary['profiles']]
    elif quantity == 'Pg':
        if 'Pg' in dictionary['quantities']:
            quantity_index = dictionary['quantities'].index('Pg')
            profiles = [list[quantity_index] for list in dictionary['profiles']]
        else:
            try:
                gam = dictionary['gam']
            except:
                gam = 5./3.
            quantity_index = dictionary['quantities'].index('u')
            profiles = [np.array(list[quantity_index]) * (gam-1.) for list in dictionary['profiles']]
    elif quantity == 'Pb':
        quantity_index = dictionary['quantities'].index('b')
        profiles = [np.array(list[quantity_index])**2/2. for list in dictionary['profiles']]
    else:
        # just reading the pre-calculated quantities
        quantity_index = dictionary['quantities'].index(quantity)
        profiles = [list[quantity_index] for list in dictionary['profiles']]
    return profiles, invert


def timeAvgPerBin(dictionary, tDivList, binNumList, quantity, perzone_avg_frac=0.5):
    """
    Calculate time averages for each time bin. This only works when there is no need for combining more than one averaged quantities.
    """
    # basic parsing
    zones = dictionary["zones"]
    n_zones_eff = dictionary["nzones_eff"]
    ncycle_per_zone = dictionary["ncycle_per_zone"]
    n0_zone = dictionary["n0_zone"]
    t0_zone = dictionary["t0_zone"]
    times = dictionary["times"]
    cycles = dictionary["cycles"]

    # derived
    switch_on_ncycle = (ncycle_per_zone > 0)

    profiles, invert = readQuantity(dictionary, quantity)
    num_time_chunk = len(tDivList) - 1
    
    # list initialization
    sortedProfiles = [[[] for _ in range(num_time_chunk)] for _ in range(n_zones_eff)] # (n_zones_eff, num_time_chunk) dimension
    avgedProfiles = [[[] for _ in range(num_time_chunk)] for _ in range(n_zones_eff)] # (n_zones_eff, num_time_chunk) dimension

    # switch criteria
    if switch_on_ncycle:
        switch_list = cycles
        switch_pt = set(n0_zone)
    else:
        switch_list = times
        switch_pt = set(t0_zone)
    switch_pt = sorted(switch_pt) + [switch_list[-1]]

    # TODO: (07/29/24) do I need dt weight?
    for i, profile in enumerate(profiles):
        zone_num = zones[i]
        bin_num = binNumList[i]
        if bin_num is not None:
            switch_num = np.argwhere((switch_list[i] >= switch_pt[:-1]) & (switch_list[i] <= switch_pt[1:]))
            if len(switch_num) > 1: print("ERROR: can't identify when this output is switched!")
            else: switch_num = switch_num[0,0]
            #if switch_list[i] >= (switch_pt[switch_num] + switch_pt[switch_num + 1]) * perzone_avg_frac: # realized that this only works for perzone_avg_frac=0.5
            if switch_pt[switch_num + 1] - switch_list[i] <= (switch_pt[switch_num + 1] - switch_pt[switch_num]) * perzone_avg_frac:
                # only when it is last (perzone_avg_frac), stage for averaging
                sortedProfiles[zone_num][bin_num].append(profile)
    
    for b in range(num_time_chunk):
        for zone in range(n_zones_eff):
            if len(sortedProfiles[zone][b]) == 0:
                # empty
                continue
            else:
                avgedProfiles[zone][b] = np.mean(sortedProfiles[zone][b], axis=0)

    return avgedProfiles, invert

    #tDivList, usableProfiles, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor, tmax)

def calcFinalTimeAvg(dictionary, tDivList, binNumList, quantity, perzone_avg_frac=0.5, mask_list=None):
    """
    Put together the final time averages. If needed, do extra operations.
    """
    radii = dictionary['radii']
    n_zones_eff = dictionary["nzones_eff"]
    num_time_chunk = len(tDivList) - 1
    
    # list initialization
    avgedProfiles = [[[] for _ in range(num_time_chunk)] for _ in range(n_zones_eff)] # (n_zones_eff, num_time_chunk) dimension
    
    if quantity == "eta":
        avgedProfiles_Mdot, invert = timeAvgPerBin(dictionary, tDivList, binNumList, 'Mdot', perzone_avg_frac=perzone_avg_frac)
        avgedProfiles_Edot, invert = timeAvgPerBin(dictionary, tDivList, binNumList, 'Edot', perzone_avg_frac=perzone_avg_frac)
        i10 = np.argmin(abs(radii - 10))
        for b in range(num_time_chunk):
            Mdot10 = avgedProfiles_Mdot[n_zones_eff - 1][b][i10] # Mdot at r = 10
            for zone in range(n_zones_eff):
                if len(avgedProfiles_Edot[zone][b]) > 0:
                    avgedProfiles[zone][b] = (avgedProfiles_Mdot[zone][b] - avgedProfiles_Edot[zone][b]) / Mdot10
    else:
        avgedProfiles, invert = timeAvgPerBin(dictionary, tDivList, binNumList, quantity, perzone_avg_frac=perzone_avg_frac)
    
    # list initialization
    rList = [[] for _ in range(num_time_chunk)]
    valuesList = [[] for _ in range(num_time_chunk)]

    # combine zones for each time bin
    for b in range(num_time_chunk):
        r_combined = np.array([])
        values_combined = np.array([])
        for zone in range(n_zones_eff):
            if mask_list is None: mask = np.full(len(radii), True, dtype=bool)
            else: mask = mask_list[zone]
            profile = avgedProfiles[zone][b]
            if len(profile) > 0:
                r_combined = np.concatenate([r_combined, radii[mask]])
                values_combined = np.concatenate([values_combined, profile[mask]])# * (-1) ** int(flip_sign)])
        rList[b] = r_combined
        valuesList[b] = values_combined

    # TODO: mask, rescale, zone_time_average_fraction
    #if invert:
    #    # Flip the quantity upside-down, usually for inv_beta.
    #    plottable = 1.0 / plottable
    return rList, valuesList

def setTimeBins(dictionary, num_time_chunk=4, time_bin_factor=2):
    n_zones_eff = dictionary["nzones_eff"]
    times = dictionary["times"]
    t_first = times[0]
    t_last = times[-1]

    # list initialization
    binNumList = [None for _ in range(len(times))] #np.full(len(times), np.nan)
    
    # TODO: add tmax option

    tDivList = np.array([t_first+(t_last-t_first)/np.power(time_bin_factor,i+1) for i in range(num_time_chunk)])
    tDivList = tDivList[::-1] # in increasing time order
    tDivList = np.append(tDivList,t_last)
    
    for i, time in enumerate(times):
        bin_num = np.argwhere((time >= tDivList[:-1]) & (time <= tDivList[1:]))
        if len(bin_num) > 1:
            print("ERROR: one profile sorted into more than 1 time bins")
        elif len(bin_num) < 1:
            continue
        else:
            binNumList[i] = int(bin_num[0, 0])

    return tDivList, binNumList

def plotProfileQuantity(radii, profile, figsize=(8,6), mask_list=None):
    plt.subplots(1,1,figsize=figsize)

    #n_zones_eff = len(profile)
    num_time_chunk = len(profile)
    for b in range(num_time_chunk):
        if len(radii[b]) > 0:
            plt.loglog(radii[b], profile[b])
    #    for zone in range(n_zones_eff):
    #        if len(profile[zone][b]) == 0:
    #            # empty
    #            continue
    #        else:
    #            if mask_list is None: mask = np.full(len(radii), True, dtype=bool)
    #            else: mask = mask_list[zone]
    #            print(zone, b)
    #            plt.loglog(radii[mask], profile[zone][b][mask])

    plt.savefig("./temp.png", bbox_inches='tight')


def plotProfiles(pkl_name, quantity_list, plot_dir=None, fig_ax=None, color_list=None, linestyle_list=None, figsize=(8, 6), flip_sign=False, show_divisions=True, show_rb=False, perzone_avg_frac=0.5, num_time_chunk=4):
    # Changes some defaults.
    #matplotlib_settings()
    
    # If you want, provide your own figure and axis.  Good for multipanel plots.
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, ax = fig_ax
    
    with open(pkl_name, 'rb') as openFile:
        D = pickle.load(openFile)
        
        tDivList, binNumList = setTimeBins(D, num_time_chunk)
        mask_list = get_mask(D)

        radii, profiles = calcFinalTimeAvg(D, tDivList, binNumList, 'eta', perzone_avg_frac=perzone_avg_frac, mask_list=mask_list)

        plotProfileQuantity(radii, profiles, figsize=figsize) #, mask_list=mask_list)

        #for quantity in quantity_list:
            #profiles = readQuantity(D, quantity)
            #output = plot_dir + "/profile_" + quantity + ".pdf"


if __name__ == "__main__":
    pkl_name = "../data_products/061724_fastvc/combineout_restructured_profiles_all.pkl"

    plot_dir = "../plots/test"  # common directory
    os.makedirs(plot_dir, exist_ok=True)
    plot_dir = "/".join(pkl_name.split("/")[:-1])  # run specific directory
    os.makedirs(plot_dir, exist_ok=True)

    quantityList = ['Ldot', 'rho', 'eta', 'Mdot', 'b', 'K', 'beta', 'Edot', 'u', 'T', 'abs_u^r', 'abs_u^phi', 'abs_u^th', 'u^r', 'u^phi',  'u^th', "abs_Omega","Omega"] 
    #'Etot', 
    plotProfiles(pkl_name, quantityList, plot_dir=plot_dir, perzone_avg_frac=0.5)  
    # , zone_time_average_fraction=avg_frac, cycles_to_average=cta, color_list=colors, linestyle_list=linestyles, label_list=listOfLabels, rescale=False, rescale_Mdot=True, flatten_rho=flatten_rho, \
