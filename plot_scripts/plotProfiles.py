import os
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle

from matplotlib_settings import *
from ylabel_dictionary import *
from plot_utils import *
import bondi_analytic as bondi

def get_mask(dictionary):
    # basic parsing
    base = dictionary["base"]
    n_zones_eff = dictionary["nzones_eff"]
    active_range = dictionary["active_range"]
    radii = dictionary["radii"]
    ncycle_per_zone = dictionary["ncycle_per_zone"]
    n_radii = len(radii)

    # figure out the base resolution
    dx1 = np.log10(radii[1] / radii[0])
    x1_out = np.log10(radii[-1]) + dx1 / 2.0
    x1_in = np.log10(radii[0]) - dx1 / 2.0
    res = int(round(len(radii) / (x1_out - x1_in) * np.log10(base**2)))  # resolution
    overlap = res // 4
    low_cadence = (ncycle_per_zone > 0) and (ncycle_per_zone < 100)

    mask = []
    for zone in range(n_zones_eff):
        mask_temp = np.full(n_radii, True, dtype=bool)
        if active_range[zone] is not None:
            mask_temp[radii < active_range[zone][0]] = False
            mask_temp[radii > active_range[zone][1]] = False

        # mask the overlap region
        active = np.argwhere(mask_temp)[:, 0]
        if n_zones_eff > 1 and len(active) > 0:
            if zone > 0:
                mask_temp[: active[0] + overlap] = False
            if zone < n_zones_eff - 1 and not low_cadence:
                mask_temp[active[-1] + 1 - overlap :] = False
        mask += [mask_temp]

        # further check if there is still an overlap, if there is, prioritize smaller ann first
        if 0: #zone > 0:
            zone_temp = zone - 1
            #if len(radii[mask[zone_temp]]) > 0 and zone_temp >= 0:
            while len(radii[mask[zone_temp]]) > 0 and zone_temp >= 0:
                rout_smaller_ann = np.power(10.0, np.log10(radii[mask[zone_temp]][-1]) + dx1 / 2.0)
                zone_temp -= 1
            still_overlaps = radii <= rout_smaller_ann
            mask[zone][still_overlaps] = False
        
    if 1:
        # or, prioritize larger ann first
        for zone in range(n_zones_eff - 1):
            rin_larger_ann = np.power(10.0, np.log10(radii[mask[zone + 1]][0]) - dx1 / 2.0)
            still_overlaps = radii >= rin_larger_ann
            mask[zone][still_overlaps] = False

    return mask




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


def calcFinalTimeAvg(dictionary, tDivList, binNumList, quantity, perzone_avg_frac=0.5, mask_list=None, rescale=False):
    """
    Put together the final time averages. If needed, do extra operations.
    """
    radii = dictionary["radii"]
    n_zones_eff = dictionary["nzones_eff"]
    num_time_chunk = len(tDivList) - 1
    save_rho = rescale and ('Mdot' in quantity)

    # list initialization
    avgedProfiles = [[[] for _ in range(num_time_chunk)] for _ in range(n_zones_eff)]  # (n_zones_eff, num_time_chunk) dimension

    if "eta" in quantity or quantity == "u^r" or quantity == "phib":
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
    elif quantity == "u^r":
        avgedProfiles_rho, invert = timeAvgPerBin(dictionary, tDivList, binNumList, "rho", perzone_avg_frac=perzone_avg_frac)
        for b in range(num_time_chunk):
            for zone in range(n_zones_eff):
                if len(avgedProfiles_Mdot[zone][b]) > 0:
                    avgedProfiles[zone][b] = avgedProfiles_Mdot[zone][b] / (avgedProfiles_rho[zone][b] * (4.0 * np.pi * radii**2))
    elif quantity == 'phib':
        avgedProfiles_Phib, invert = timeAvgPerBin(dictionary, tDivList, binNumList, "Phib", perzone_avg_frac=perzone_avg_frac)
        for b in range(num_time_chunk):
            if len(avgedProfiles_Mdot[0][b]) > 0:
                Mdot10 = avgedProfiles_Mdot[0][b][i10]  # Mdot at r = 10
            else:
                continue
            for zone in range(n_zones_eff):
                if len(avgedProfiles_Phib[zone][b]) > 0:
                    avgedProfiles[zone][b] = avgedProfiles_Phib[zone][b] / np.sqrt(Mdot10)
    else:
        avgedProfiles, invert = timeAvgPerBin(dictionary, tDivList, binNumList, quantity, perzone_avg_frac=perzone_avg_frac)

    if save_rho:
        avgedProfiles_rho, _ = timeAvgPerBin(dictionary, tDivList, binNumList, 'rho', perzone_avg_frac=perzone_avg_frac)

    # list initialization
    rList = [[] for _ in range(num_time_chunk)]
    valuesList = [[] for _ in range(num_time_chunk)]
    if save_rho: valuesListRho = [[] for _ in range(num_time_chunk)]

    # combine zones for each time bin
    for b in range(num_time_chunk):
        r_combined = np.array([])
        values_combined = np.array([])
        if save_rho: values_combined_rho = np.array([])
        for zone in range(n_zones_eff):
            if mask_list is None:
                mask = np.full(len(radii), True, dtype=bool)
            else:
                mask = mask_list[zone]
            profile = avgedProfiles[zone][b]
            if save_rho: profile_rho = avgedProfiles_rho[zone][b]
            if len(profile) > 0:
                r_combined = np.concatenate([r_combined, radii[mask]])
                values_combined = np.concatenate([values_combined, profile[mask]])  # * (-1) ** int(flip_sign)])
                if save_rho: values_combined_rho = np.concatenate([values_combined_rho, profile_rho[mask]])  # * (-1) ** int(flip_sign)])
        rList[b] = r_combined
        valuesList[b] = values_combined
        if save_rho: valuesListRho[b] = values_combined_rho

    # any final operations
    ## normalization
    if "Omega" in quantity:
        valuesList = [valuesList[b] * np.power(radii, 3.0 / 2) for b in range(num_time_chunk)]  # normalize by Omega_K
    ## TODO: rescale for Mdot
    if save_rho:
        # rescale Mdot depending on the density at Bondi radius
        r_sonic = dictionary['dump']['rs']
        rB = bondi.get_quantity_for_rarr([1], 'RB', rs=r_sonic)[0]
        Mdot_analytic = bondi.get_quantity_for_rarr([rB], 'Mdot', rs=r_sonic)[0]
        rho_analytic = bondi.get_quantity_for_rarr([100 * rB], 'rho', rs=r_sonic)[0]
        for b in range(num_time_chunk):
            rho_save = min(valuesListRho[b][radii < 5 * rB])
            print(rho_save, rho_analytic / (Mdot_analytic * rho_save))
            valuesList[b] *= rho_analytic / (Mdot_analytic * rho_save)

    ## invert
    if invert:
        # Flip the quantity upside-down, usually for inv_beta.
        valuesList = [1.0 / valuesList[b] if (len(valuesList[b]) > 0) else valuesList[b] for b in range(num_time_chunk)]
    return rList, valuesList


def setTimeBins(dictionary, num_time_chunk=4, time_bin_factor=2, tmax=None):
    n_zones_eff = dictionary["nzones_eff"]
    times = dictionary["times"]
    t_first = times[0]
    t_last = times[-1]

    # list initialization
    binNumList = [None for _ in range(len(times))]  # np.full(len(times), np.nan)

    if tmax is not None:
        if t_last > tmax:
            t_last = tmax
        else: print("The run hasn't reached {}tg. Instead using {:.3g}tg".format(tmax, t_last))

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


def plotProfileQuantity(ax, radii, profile, tDivList, colors=None, label=None, linestyle="-"):
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
            ax.plot(radii[b], profile[b], color=colors[b], lw=2, label=label_use, ls=linestyle)
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

def plotIC(ax, dictionary, quantity):
    radii = dictionary["radii"]
    profile, invert = readQuantity(dictionary, quantity)
    init_profile = profile[0]
    if invert: init_profile = 1. / init_profile
    valid_radii = radii > 2
    ax.plot(radii[valid_radii], init_profile[valid_radii], 'k:', lw=1)

def plotProfiles(
    pkl_name,
    quantity_list,
    plot_dir="../plots/test",
    fig_ax=None,
    color_list=None,
    label=None,
    linestyle=None,
    formatting=True,
    figsize=(8, 6),
    flip_sign=False,
    show_divisions=True,
    show_rb=False,
    show_init=False,
    perzone_avg_frac=0.5,
    num_time_chunk=4,
    time_bin_factor=2,
    tmax=None,
    rescale=False,
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

    tDivList, binNumList = setTimeBins(D, num_time_chunk, time_bin_factor=time_bin_factor, tmax=tmax)
    mask_list = get_mask(D)

    for i, quantity in enumerate(quantity_list):
        if fig_ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            ax = ax1d[i]  # here we assume that the number of axes passed = number of quantities

        radii, profiles = calcFinalTimeAvg(D, tDivList, binNumList, quantity, perzone_avg_frac=perzone_avg_frac, mask_list=mask_list, rescale=rescale)
        if i == 0:
            for b in range(len(tDivList) - 1):
                print("{}: t={:.3g}-{:.3g}".format(b, tDivList[b], tDivList[b + 1]))

        plotProfileQuantity(ax, radii, profiles, tDivList, colors=color_list, label=label, linestyle=linestyle)

        if show_init and (quantity == "rho" or quantity == "T" or quantity=="beta" or quantity=='u^r'):
            plotIC(ax, D, quantity)

        # Formatting
        if formatting:
            ax.set_xlabel("Radius [$r_g$]")
            ylabel = variableToLabel(quantity)
            # if eta_norm_Bondi and quantity=='eta':
            #    ylabel = r'$\overline{\dot{M}-\dot{E}}/\dot{M}_B$'
            if rescale and 'Mdot' in quantity:
                ylabel = ylabel.replace('arb. units', r'$\dot{M}_B$')
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if "eta" in quantity and quantity != "beta":
                ax.set_ylim([1e-3, 4])
            try:
                rEH = D["dump"]["r_eh"]
            except:
                a = 0  # for now
                rEH = 1.0 + np.sqrt(1.0 - a**2)
            xlim = (rEH, ax.get_xlim()[-1])
            ax.set_xlim(xlim)

        if fig_ax is None:
            output = plot_dir + "/profile_" + quantity + ".png"  # pdf"
            plt.savefig(output, bbox_inches="tight")
            plt.close()
            print("saved to " + output)

    if fig_ax is not None:
        return (fig, axes)


if __name__ == "__main__":
    #pkl_name="../data_products/112524_fastervc_a0.5_b2n14/test_combine_in_profiles_all.pkl"
    #pkl_name="../data_products/112524_fastervc_a0.5_b8n4_profiles_all.pkl"
    #pkl_name = "../data_products/100724_a0.5_n8/high_cadence_profiles_all.pkl"  #
    #pkl_name = "../data_products/111524_a0.5_torus_profiles_all.pkl"
    pkl_name = "../data_products/111524_a0.5_oz_128_profiles_all.pkl"
    #pkl_name = "../data_products/112624_fastervc_a0.5_b2n26/reconnect_profiles_all.pkl"
    #pkl_name = "../data_products/112924_a0.5_n8_reconnect_profiles_all.pkl"
    #pkl_name = "../data_products/120824_a0.5_n4_betaflr_profiles_all.pkl"
    #pkl_name = "../data_products/121024_a0.5_n8_reconnect_rot_profiles_all.pkl"
    #pkl_name = "../data_products/121024_a0.9_n4_reconnect_profiles_all.pkl"
    #pkl_name = "../data_products/122624_a0.5_n8_reconnectfix_profiles_all.pkl"
    pkl_name = "../data_products/123024_a0.5_oz_reconnect_profiles_all.pkl"
    pkl_name = "../data_products/123124_a0.5_oz_rdepgmax_profiles_all.pkl"
    pkl_name = "../data_products/010225_a0.5_oz_rdepgmax5_profiles_all.pkl"
    pkl_name = "../data_products/010525_a0.5_n8_ncycle200_profiles_all.pkl"
    pkl_name = "../data_products/010625_a0.5_n8_ncycle200_capped_profiles_all.pkl"
    pkl_name = "../data_products/010625_a0.5_oz_128_profiles_all.pkl"
    #pkl_name = "../data_products/010825_a0.5_n8_tchar_profiles_all.pkl"
    pkl_name = "../data_products/010825_a0.5_n8_ncycle400_capped_profiles_all.pkl"
    #pkl_name = "../data_products/010925_a0.5_b2n26_reconnect_profiles_all.pkl"
    pkl_name = "../data_products/010625_a0.5_oz_128_profiles_all.pkl"
    #pkl_name = "../data_products/010925_a0.9_oz_128_profiles_all.pkl"
    #pkl_name = "../data_products/011225_a0.5_b2n26_ncycle10_reconnect_profiles_all.pkl"
    #pkl_name = "../data_products/011225_a0.5_n8_ncycle50_profiles_all.pkl"
    #pkl_name = "../data_products/011225_a0.5_b2n26_ncycle10/debug_kastaun_profiles_all.pkl"
    #pkl_name = "../data_products/011425_a0.5_b2n26_ncycle10_betafloor_profiles_all.pkl"
    #pkl_name = "../data_products/011525_a0.5_b2n26_ncycle2_betafloor_profiles_all.pkl"
    #pkl_name = "../data_products/012125_a0.5_noncapped_profiles_all.pkl"
    #pkl_name = "../data_products/012325_a0.5_ncycle100_profiles_all.pkl"
    #pkl_name = "../data_products/012725_a0.5_extg_tempmax_profiles_all.pkl"
    #pkl_name = "../data_products/012825_a0.5_extg_tempmax_cap1e6_profiles_all.pkl"
    #pkl_name = "../data_products/012925_a0.5_extg_kastaun_cap2e6_profiles_all.pkl"
    #pkl_name = "../data_products/020325_a0.5_96_profiles_all.pkl"
    #pkl_name = "../data_products/020325_a0.5_beta1000_profiles_all.pkl"
    #pkl_name = "../data_products/020625_a0.5_tmax_beyond_1e6_profiles_all.pkl"
    #pkl_name = "../data_products/021025_a0.5_tmax_normal_profiles_all.pkl"
    #pkl_name = "../data_products/021125_a0.0_b2n26_profiles_all.pkl"
    #pkl_name = "../data_products/021125_a0.5_rcool3e5_profiles_all.pkl"
    #pkl_name = "../data_products/021225_a0.5_lin_profiles_all.pkl"
    #pkl_name = "../data_products/021225_a0.5_rB1e6_profiles_all.pkl"
    #pkl_name = "../data_products/021225_a0.5_fofc_profiles_all.pkl"
    pkl_name = "../data_products/021325_a0.5_n8_profiles_all.pkl"
    #pkl_name = "../data_products/021325_a0.0_011424_profiles_all.pkl"
    #pkl_name = "../data_products/021425_a0.5_rB1e6_reconnect_profiles_all.pkl"
    #pkl_name = "../data_products/021825_a0.5_rB1e6_reconnect_nocool_profiles_all.pkl"
    #pkl_name = "../data_products/021925_a0.5_rB1e6/withbetaflr_profiles_all.pkl"
    #pkl_name = "../data_products/021925_a0.0_b2n26_profiles_all.pkl"
    #pkl_name = "../data_products/022025_a0.5_n8_nc40_profiles_all.pkl"
    pkl_name = "../data_products/022425_a0.0_n8_ncycle8000_profiles_all.pkl"
    #pkl_name = "../data_products/022425_a0.0_b2_tchar_profiles_all.pkl"
    #pkl_name = "../data_products/022525_a0.5_b8n4_profiles_all.pkl"
    #pkl_name = "../data_products/022525_a0.5_b2n14_profiles_all.pkl"
    #pkl_name = "../data_products/022625_a0.5_b2n14_tchar_profiles_all.pkl"
    pkl_name = "../data_products/022625_a0.0_n8_normal_profiles_all.pkl"
    pkl_name = "../data_products/022625_a0.5_safe_profiles_all.pkl"
    #pkl_name = "../data_products/022625_a0.9_n4_profiles_all.pkl"
    pkl_name = "../data_products/022725_a0.0_safe_tchar_profiles_all.pkl"
    #pkl_name = "../data_products/022825_a0.5_n8_rdepgmax_profiles_all.pkl"
    #pkl_name = "../data_products/030225_a0.0_b2_tchar_normal1dw_profiles_all.pkl"
    #pkl_name = "../data_products/030325_a0.5_rdepgmax_nodelrhocap_profiles_all.pkl"
    #pkl_name = "../data_products/030325_a0.0_safe_tchar_profiles_all.pkl"
    pkl_name = "../data_products/030325_a0.9_oz_128_profiles_all.pkl"
    #pkl_name = "../data_products/delta/110624_a0.0_oz_128_profiles_all.pkl"
    pkl_name = "../data_products/030425_a0.5_rdepgmax5_profiles_all.pkl"
    #pkl_name = "../data_products/030425_a0.5_safe_longtin20_profiles_all.pkl"
    pkl_name = "../data_products/030425_a0.5_b8n4_safe_longtin10_profiles_all.pkl"
    #pkl_name = "../data_products/030425_a0.9_b8n4_safe_profiles_all.pkl"
    pkl_name = "../data_products/030425_a0.0_b2n14_safe_profiles_all.pkl"

    plot_dir = "../plots/test"  # common directory
    os.makedirs(plot_dir, exist_ok=True)
    # plot_dir = "/".join(pkl_name.split("/")[:-1])  # run specific directory
    # os.makedirs(plot_dir, exist_ok=True)

    quantityList = [
        "Mdot",
        "beta",
        "eta",
        "rho",
        "T",
        "eta_Fl",
        "eta_EM",
        "u^r",
        "abs_u^r",
        "Omega",
        "abs_Omega",
    ]  # ["Ldot", "rho", "eta", "Mdot", "b", "K", "beta", "Edot", "u", "T", "abs_u^r", "abs_u^phi", "abs_u^th", "u^r", "u^phi", "u^th", "abs_Omega", "Omega"]
    print(pkl_name)
    plotProfiles(pkl_name, quantityList, plot_dir=plot_dir, perzone_avg_frac=1, num_time_chunk=3, time_bin_factor=2, rescale=False, show_init=True)
