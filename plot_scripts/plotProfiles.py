import os
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle

from matplotlib_settings import *
from ylabel_dictionary import *


def get_mask(dictionary):
    # basic parsing
    base = dictionary["base"]
    n_zones_eff = dictionary["nzones_eff"]
    active_range = dictionary["active_range"]
    radii = dictionary["radii"]
    n_radii = len(radii)

    # figure out the base resolution
    dx1 = np.log10(radii[1] / radii[0])
    x1_out = np.log10(radii[-1]) + dx1 / 2.0
    x1_in = np.log10(radii[0]) - dx1 / 2.0
    res = int(round(len(radii) / (x1_out - x1_in) * np.log10(base**2)))  # resolution
    overlap = res // 4

    mask = []
    for zone in range(n_zones_eff):
        mask_temp = np.full(n_radii, True, dtype=bool)
        mask_temp[radii < active_range[zone][0]] = False
        mask_temp[radii > active_range[zone][1]] = False

        # mask the overlap region
        if n_zones_eff > 1:
            active = np.argwhere(mask_temp)[:, 0]
            if zone > 0:
                mask_temp[: active[0] + overlap] = False
            if zone < n_zones_eff - 1:
                mask_temp[active[-1] + 1 - overlap :] = False
        mask += [mask_temp]

        # further check if there is still an overlap, if there is, prioritize smaller ann first
        if zone > 0:
            zone_temp = zone - 1
            while len(radii[mask[zone_temp]]) > 0 and zone_temp >= 0:
                rout_smaller_ann = np.power(10.0, np.log10(radii[mask[zone - 1]][-1]) + dx1 / 2.0)
                zone_temp -= 1
            still_overlaps = radii <= rout_smaller_ann
            mask[zone][still_overlaps] = False

    return mask


def readQuantity(dictionary, quantity):
    invert = False
    if quantity == "beta":
        if "inv_beta" in dictionary["quantities"]:
            quantity = "inv_beta"
            invert = True
        else:
            print("inv_beta doesn't exist, so we will stick with beta.")
        quantity_index = dictionary["quantities"].index(quantity)
        profiles = [list[quantity_index] for list in dictionary["profiles"]]
    elif quantity == "Pg":
        if "Pg" in dictionary["quantities"]:
            quantity_index = dictionary["quantities"].index("Pg")
            profiles = [list[quantity_index] for list in dictionary["profiles"]]
        else:
            try:
                gam = dictionary["gam"]
            except:
                gam = 5.0 / 3.0
            quantity_index = dictionary["quantities"].index("u")
            profiles = [np.array(list[quantity_index]) * (gam - 1.0) for list in dictionary["profiles"]]
    elif quantity == "Pb":
        quantity_index = dictionary["quantities"].index("b")
        profiles = [np.array(list[quantity_index]) ** 2 / 2.0 for list in dictionary["profiles"]]
    else:
        # just reading the pre-calculated quantities
        quantity_index = dictionary["quantities"].index(quantity)
        profiles = [list[quantity_index] for list in dictionary["profiles"]]
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
    switch_on_ncycle = ncycle_per_zone > 0

    profiles, invert = readQuantity(dictionary, quantity)
    num_time_chunk = len(tDivList) - 1

    # list initialization
    sortedProfiles = [[[] for _ in range(num_time_chunk)] for _ in range(n_zones_eff)]  # (n_zones_eff, num_time_chunk) dimension
    avgedProfiles = [[[] for _ in range(num_time_chunk)] for _ in range(n_zones_eff)]  # (n_zones_eff, num_time_chunk) dimension

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
            switch_num = np.argwhere((switch_list[i] >= switch_pt[:-1]) & (switch_list[i] < switch_pt[1:]))
            if len(switch_num) > 1:
                print("ERROR: can't identify when this output is switched!")
            else:
                switch_num = switch_num[0, 0]
            # if switch_list[i] >= (switch_pt[switch_num] + switch_pt[switch_num + 1]) * perzone_avg_frac: # realized that this only works for perzone_avg_frac=0.5
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

    # tDivList, usableProfiles, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor, tmax)


def calcFinalTimeAvg(dictionary, tDivList, binNumList, quantity, perzone_avg_frac=0.5, mask_list=None):
    """
    Put together the final time averages. If needed, do extra operations.
    """
    radii = dictionary["radii"]
    n_zones_eff = dictionary["nzones_eff"]
    num_time_chunk = len(tDivList) - 1

    # list initialization
    avgedProfiles = [[[] for _ in range(num_time_chunk)] for _ in range(n_zones_eff)]  # (n_zones_eff, num_time_chunk) dimension

    if quantity == "eta" or quantity == "eta_EM" or quantity == "eta_Fl":
        avgedProfiles_Mdot, _ = timeAvgPerBin(dictionary, tDivList, binNumList, "Mdot", perzone_avg_frac=perzone_avg_frac)
        i10 = np.argmin(abs(radii - 10))

    if quantity == "eta":
        avgedProfiles_Edot, invert = timeAvgPerBin(dictionary, tDivList, binNumList, "Edot", perzone_avg_frac=perzone_avg_frac)
        for b in range(num_time_chunk):
            if len(avgedProfiles_Mdot[0][b]) > 0:
                Mdot10 = avgedProfiles_Mdot[0][b][i10]  # Mdot at r = 10
            else:
                continue
            for zone in range(n_zones_eff):
                if len(avgedProfiles_Edot[zone][b]) > 0:
                    avgedProfiles[zone][b] = (avgedProfiles_Mdot[zone][b] - avgedProfiles_Edot[zone][b]) / Mdot10
    elif quantity == "eta_Fl":
        avgedProfiles_EdotFl, invert = timeAvgPerBin(dictionary, tDivList, binNumList, "Edot_Fl", perzone_avg_frac=perzone_avg_frac)
        for b in range(num_time_chunk):
            if len(avgedProfiles_Mdot[0][b]) > 0:
                Mdot10 = avgedProfiles_Mdot[0][b][i10]  # Mdot at r = 10
            else:
                continue
            for zone in range(n_zones_eff):
                if len(avgedProfiles_EdotFl[zone][b]) > 0:
                    avgedProfiles[zone][b] = (avgedProfiles_Mdot[zone][b] - avgedProfiles_EdotFl[zone][b]) / Mdot10
    elif quantity == "eta_EM":
        avgedProfiles_EdotEM, invert = timeAvgPerBin(dictionary, tDivList, binNumList, "Edot_EM", perzone_avg_frac=perzone_avg_frac)
        for b in range(num_time_chunk):
            if len(avgedProfiles_Mdot[0][b]) > 0:
                Mdot10 = avgedProfiles_Mdot[0][b][i10]  # Mdot at r = 10
            else:
                continue
            for zone in range(n_zones_eff):
                if len(avgedProfiles_EdotEM[zone][b]) > 0:
                    avgedProfiles[zone][b] = -avgedProfiles_EdotEM[zone][b] / Mdot10
    # TODO: need separate procedure for u^r?
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
            if mask_list is None:
                mask = np.full(len(radii), True, dtype=bool)
            else:
                mask = mask_list[zone]
            profile = avgedProfiles[zone][b]
            if len(profile) > 0:
                r_combined = np.concatenate([r_combined, radii[mask]])
                values_combined = np.concatenate([values_combined, profile[mask]])  # * (-1) ** int(flip_sign)])
        rList[b] = r_combined
        valuesList[b] = values_combined

    # TODO: rescale
    if invert:
        # Flip the quantity upside-down, usually for inv_beta.
        valuesList = [1.0 / valuesList[b] if (len(valuesList[b]) > 0) else valuesList[b] for b in range(num_time_chunk)]
    return rList, valuesList


def setTimeBins(dictionary, num_time_chunk=4, time_bin_factor=2):
    n_zones_eff = dictionary["nzones_eff"]
    times = dictionary["times"]
    t_first = times[0]
    t_last = times[-1]

    # list initialization
    binNumList = [None for _ in range(len(times))]  # np.full(len(times), np.nan)

    # TODO: add tmax option

    tDivList = np.array([t_first + (t_last - t_first) / np.power(time_bin_factor, i + 1) for i in range(num_time_chunk)])
    tDivList = tDivList[::-1]  # in increasing time order
    tDivList = np.append(tDivList, t_last)

    for i, time in enumerate(times):
        bin_num = np.argwhere((time >= tDivList[:-1]) & (time < tDivList[1:]))
        if len(bin_num) > 1:
            print("ERROR: one profile sorted into more than 1 time bins")
        elif len(bin_num) < 1:
            continue
        else:
            binNumList[i] = int(bin_num[0, 0])

    return tDivList, binNumList


def plotProfileQuantity(ax, radii, profile, tDivList, colors=None, label=None):
    # n_zones_eff = len(profile)
    num_time_chunk = len(profile)
    if colors is None:
        colors = plt.cm.gnuplot(np.linspace(0.9, 0.3, num_time_chunk))
    for b in range(num_time_chunk):
        if label is None:
            label_use = "t={:.5g} - {:.5g}".format(tDivList[b], tDivList[b + 1])
        else:
            label_use = label
        if len(radii[b]) > 0:
            ax.plot(radii[b], profile[b], color=colors[b], lw=2, label=label_use)
    #    for zone in range(n_zones_eff):
    #        if len(profile[zone][b]) == 0:
    #            # empty
    #            continue
    #        else:
    #            if mask_list is None: mask = np.full(len(radii), True, dtype=bool)
    #            else: mask = mask_list[zone]
    #            print(zone, b)
    #            plt.loglog(radii[mask], profile[zone][b][mask])
    ax.legend()


def plotProfiles(
    pkl_name,
    quantity_list,
    plot_dir="../plots/test",
    fig_ax=None,
    color_list=None,
    label=None,
    linestyle_list=None,
    formatting=True,
    figsize=(8, 6),
    flip_sign=False,
    show_divisions=True,
    show_rb=False,
    perzone_avg_frac=0.5,
    num_time_chunk=4,
):
    # Changes some defaults.
    matplotlib_settings()

    # If you want, provide your own figure and axis.  Good for multipanel plots.
    if fig_ax is not None:
        fig, axes = fig_ax
        ax1d = axes.reshape(-1)

    plotrc = {}
    with open(pkl_name, "rb") as openFile:
        D = pickle.load(openFile)

    tDivList, binNumList = setTimeBins(D, num_time_chunk)
    mask_list = get_mask(D)

    for i, quantity in enumerate(quantity_list):
        if fig_ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            ax = ax1d[i]  # here we assume that the number of axes passed = number of quantities

        radii, profiles = calcFinalTimeAvg(D, tDivList, binNumList, quantity, perzone_avg_frac=perzone_avg_frac, mask_list=mask_list)
        if i == 0:
            for b in range(len(tDivList) - 1):
                print("{}: t={:.3g}-{:.3g}".format(b, tDivList[b], tDivList[b + 1]))

        plotProfileQuantity(ax, radii, profiles, tDivList, colors=color_list, label=label)

        # Formatting
        if formatting:
            ax.set_xlabel("Radius [$r_g$]")
            ylabel = variableToLabel(quantity)
            # if eta_norm_Bondi and quantity=='eta':
            #    ylabel = r'$\overline{\dot{M}-\dot{E}}/\dot{M}_B$'
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.set_yscale("log")
            # ax.set_xlim(xlim); ax.set_ylim(ylim)

        if fig_ax is None:
            output = plot_dir + "/profile_" + quantity + ".png"  # pdf"
            plt.savefig(output, bbox_inches="tight")
            plt.close()
            print("saved to " + output)

    if fig_ax is not None:
        return (fig, axes)


if __name__ == "__main__":
    # pkl_name = "../data_products/051224_bondi_kerr/00000_profiles_all.pkl"
    # pkl_name = "../data_products/061724_fastvc/combineout_restructured_profiles_all.pkl"
    # pkl_name = "../data_products/061724_fastvc/combineout_ncycle200_profiles_all.pkl"
    # pkl_name = "../data_products/061724_fastvc/combineout_nocap_profiles_all.pkl"
    pkl_name = "../data_products/061724_fastvc/combineout_ismr/dirichlet_and_no_recon_floor_profiles_all.pkl"  # _nolongtin #a0.5_ #
    pkl_name = "../data_products/061724_fastvc/combineout_ismr_a0.5_bfluxc/test_new_dump_cadence_profiles_all.pkl"  # #moverin_ _nolongtin #
    # pkl_name = "../data_products/061724_fastvc/combineout_ismr_a0.5_ncycle50_profiles_all.pkl"
    # pkl_name = "../data_products/080724_fastvc_consistentB/mad_stock_profiles_all.pkl"
    # pkl_name = "../data_products/081424_a0.5_bfluxc_moverin_profiles_all.pkl" #_nocap
    # pkl_name = "../data_products/081524_a0.5_bflux0_moverin_profiles_all.pkl" #tchar_
    # pkl_name = "../data_products/081524_a0.5_ncycle50_profiles_all.pkl"
    pkl_name = "../data_products/081624_a0.5_bfluxc_moverin_longtin4_profiles_all.pkl"

    plot_dir = "../plots/test"  # common directory
    os.makedirs(plot_dir, exist_ok=True)
    # plot_dir = "/".join(pkl_name.split("/")[:-1])  # run specific directory
    # os.makedirs(plot_dir, exist_ok=True)

    quantityList = [
        "Mdot",
        "beta",
        "eta",
        "rho",
        "eta_Fl",
        "eta_EM",
    ]  # ["Ldot", "rho", "eta", "Mdot", "b", "K", "beta", "Edot", "u", "T", "abs_u^r", "abs_u^phi", "abs_u^th", "u^r", "u^phi", "u^th", "abs_Omega", "Omega"]
    #'Etot',
    print(pkl_name)
    plotProfiles(pkl_name, quantityList, plot_dir=plot_dir, perzone_avg_frac=0.5, num_time_chunk=6)
    # , zone_time_average_fraction=avg_frac, cycles_to_average=cta, color_list=colors, linestyle_list=linestyles, label_list=listOfLabels, rescale=False, rescale_Mdot=True, flatten_rho=flatten_rho, \
