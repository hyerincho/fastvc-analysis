import os
import numpy as np
import pdb
import pickle

from matplotlib_settings import *
from ylabel_dictionary import *
from plot_utils import *
import bondi_analytic as bondi


def get_mask(dictionary, prioritize_inner=False):  # True): #
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
        if prioritize_inner and zone > 0:
            zone_temp = zone - 1
            # if len(radii[mask[zone_temp]]) > 0 and zone_temp >= 0:
            while len(radii[mask[zone_temp]]) > 0 and zone_temp >= 0:
                rout_smaller_ann = np.power(10.0, np.log10(radii[mask[zone_temp]][-1]) + dx1 / 2.0)
                zone_temp -= 1
            still_overlaps = radii <= rout_smaller_ann
            mask[zone][still_overlaps] = False

    if not prioritize_inner:
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
    deltBin = [[[] for _ in range(num_time_chunk)] for _ in range(n_zones_eff)]  # (n_zones_eff, num_time_chunk) dimension

    # switch criteria
    if switch_on_ncycle:
        switch_list = cycles
        switch_pt = set(n0_zone)
    else:
        switch_list = times
        switch_pt = set(t0_zone)
    delt = np.gradient(times)
    switch_pt = sorted(switch_pt) + [switch_list[-1]]

    # TODO: (07/29/24) do I need dt weight?
    for i, profile in enumerate(profiles):
        zone_num = zones[i]
        bin_num = binNumList[i]
        if bin_num is not None:
            switch_num = np.argwhere((switch_list[i] > switch_pt[:-1]) & (switch_list[i] <= switch_pt[1:]))
            if len(switch_num) > 1:
                print("ERROR: can't identify when this output is switched!")
            else:
                switch_num = switch_num[0, 0]
            #if zone_num == 0: pdb.set_trace()
            if switch_pt[switch_num + 1] - switch_list[i] <= (switch_pt[switch_num + 1] - switch_pt[switch_num]) * perzone_avg_frac:
                # only when it is last (perzone_avg_frac), stage for averaging
                sortedProfiles[zone_num][bin_num].append(profile)
                #deltBin[zone_num][bin_num].append(delt[i])
                delt = times[i] - times[i-1]
                if zone_num == 0 and delt > 100: delt = 100 # temporary TODO
                deltBin[zone_num][bin_num].append(delt)

    #if quantity == "eta":
    #    i5 = np.argmin(abs(dictionary["radii"] - 5))
    #    np.save("./plotprofiledat.npy",np.array(sortedProfiles[0][num_time_chunk-1])[:,i5])

    for b in range(num_time_chunk):
        for zone in range(n_zones_eff):
            if len(sortedProfiles[zone][b]) == 0:
                # empty
                continue
            else:
                #deltBin[zone][b] = np.gradient(np.array(deltBin[zone][b]))
                avgedProfiles[zone][b] = np.mean(sortedProfiles[zone][b], axis=0)
                #avgedProfiles[zone][b] = (np.sum(np.array([deltBin[zone][b]]).T * sortedProfiles[zone][b], axis=0) / np.sum(deltBin[zone][b])) # trying out delt weighting

    return avgedProfiles, invert


def calcFinalTimeAvg(dictionary, tDivList, binNumList, quantity, perzone_avg_frac=0.5, mask_list=None, rescale=False, flatten_rho=False, use_avged_Mdot=False, flip_sign=False):
    """
    Put together the final time averages. If needed, do extra operations.
    """
    radii = dictionary["radii"]
    n_zones_eff = dictionary["nzones_eff"]
    num_time_chunk = len(tDivList) - 1
    save_rho = rescale and ("Mdot" in quantity)

    # list initialization
    avgedProfiles = [[[] for _ in range(num_time_chunk)] for _ in range(n_zones_eff)]  # (n_zones_eff, num_time_chunk) dimension

    if "eta" in quantity or quantity == "u^r" or quantity == "phib" or "etaB" in quantity:
        avgedProfiles_Mdot, _ = timeAvgPerBin(dictionary, tDivList, binNumList, "Mdot", perzone_avg_frac=perzone_avg_frac)
        i10 = np.argmin(abs(radii - 10))

    if use_avged_Mdot and quantity == "eta": #
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
    elif quantity == "etaMdot":
        avgedProfiles_Edot, invert = timeAvgPerBin(dictionary, tDivList, binNumList, "Edot", perzone_avg_frac=perzone_avg_frac)
        for b in range(num_time_chunk):
            for zone in range(n_zones_eff):
                if len(avgedProfiles_Edot[zone][b]) > 0:
                    avgedProfiles[zone][b] = avgedProfiles_Mdot[zone][b] - avgedProfiles_Edot[zone][b]
    elif quantity == "u^r":
        avgedProfiles_rho, invert = timeAvgPerBin(dictionary, tDivList, binNumList, "rho", perzone_avg_frac=perzone_avg_frac)
        for b in range(num_time_chunk):
            for zone in range(n_zones_eff):
                if len(avgedProfiles_Mdot[zone][b]) > 0:
                    avgedProfiles[zone][b] = avgedProfiles_Mdot[zone][b] / (avgedProfiles_rho[zone][b] * (4.0 * np.pi * radii**2))
    elif use_avged_Mdot and quantity == "phib":
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
        avgedProfiles_rho, _ = timeAvgPerBin(dictionary, tDivList, binNumList, "rho", perzone_avg_frac=perzone_avg_frac)

    # list initialization
    rList = [[] for _ in range(num_time_chunk)]
    valuesList = [[] for _ in range(num_time_chunk)]
    if save_rho:
        valuesListRho = [[] for _ in range(num_time_chunk)]

    # combine zones for each time bin
    for b in range(num_time_chunk):
        r_combined = np.array([])
        values_combined = np.array([])
        if save_rho:
            values_combined_rho = np.array([])
        for zone in range(n_zones_eff):
            if mask_list is None:
                mask = np.full(len(radii), True, dtype=bool)
            else:
                mask = mask_list[zone]
            profile = avgedProfiles[zone][b]
            if save_rho:
                profile_rho = avgedProfiles_rho[zone][b]
            if len(profile) > 0:
                r_combined = np.concatenate([r_combined, radii[mask]])
                values_combined = np.concatenate([values_combined, profile[mask] * (-1) ** int(flip_sign)])
                if save_rho:
                    values_combined_rho = np.concatenate([values_combined_rho, profile_rho[mask]]) #* (-1) ** int(flip_sign)
        if flatten_rho and quantity == "rho":
            values_combined *= np.power(r_combined, 1.1)  # just to test Xu+23 rho ~ r^{-0.8}
        rList[b] = r_combined
        valuesList[b] = values_combined
        if save_rho:
            valuesListRho[b] = values_combined_rho

    # any final operations
    ## normalization
    if "Omega" in quantity:
        try:
            valuesList = [valuesList[b] * np.power(radii, 3.0 / 2) for b in range(num_time_chunk)]  # normalize by Omega_K
        except:
            valuesList = [valuesList[b] * np.power(radii[: len(valuesList[b])], 3.0 / 2) for b in range(num_time_chunk)]
    ## TODO: rescale for Mdot
    if save_rho:
        # rescale Mdot depending on the density at Bondi radius
        r_sonic = dictionary["dump"]["rs"]
        mdot = dictionary["dump"]["mdot"]
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot)[0]
        Mdot_analytic = bondi.get_quantity_for_rarr([rB], "Mdot", rs=r_sonic, mdot=mdot)[0]
        rho_analytic = bondi.get_quantity_for_rarr([100 * rB], "rho", rs=r_sonic, mdot=mdot)[0]
        for b in range(num_time_chunk):
            # rho_save = min(valuesListRho[b])
            #rho_save = min(valuesListRho[b][radii < 5 * rB])
            irB = np.argmin(abs(rList[b] - rB))
            rho_save = valuesListRho[b][irB] # the method used in Cho+24
            print("rho_save={:.5g}, factor={:.5g}".format(rho_save, rho_analytic / (Mdot_analytic * rho_save)))
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
        r_sonic = dictionary["dump"]["rs"]
        mdot = dictionary["dump"]["mdot"]
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot)[0]
        tB = np.power(rB, 3./2)
        if t_last > tmax * tB:
            t_last = tmax * tB
        else:
            print("The run hasn't reached {}tB, or {}tg. Instead using {:.3g}tg".format(tmax, tmax * tB, t_last))
    
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


def plotProfileQuantity(ax, radii, profile, tDivList, colors=None, alpha=1., label=None, linestyle="-", legend=True, print_radius=None, plot_rmax=None):
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
            plot_r = radii[b]
            plot_p = profile[b]
            if plot_rmax is not None:
                keep_r = (plot_r < plot_rmax)
                plot_r = plot_r[keep_r]
                plot_p = plot_p[keep_r]
            ax.plot(plot_r, plot_p, color=colors[b], lw=2, label=label_use, ls=linestyle, alpha=alpha)
            if print_radius is not None:
                i_r = np.argmin(abs(radii[b] - print_radius))
                print("at r={:.5g}, quantity={:.5g}".format(radii[b][i_r], profile[b][i_r]))

    #    for zone in range(n_zones_eff):
    #        if len(profile[zone][b]) == 0:
    #            # empty
    #            continue
    #        else:
    #            if mask_list is None: mask = np.full(len(radii), True, dtype=bool)
    #            else: mask = mask_list[zone]
    #            print(zone, b)
    #            plt.loglog(radii[mask], profile[zone][b][mask])
    if legend:
        ax.legend()


def plotIC(ax, dictionary, quantity):
    radii = dictionary["radii"]
    profile, invert = readQuantity(dictionary, quantity)
    init_profile = profile[0]
    if invert:
        init_profile = 1.0 / init_profile
    valid_radii = radii > 2
    ax.plot(radii[valid_radii], init_profile[valid_radii], "k:", lw=1)


def plotProfiles(
    pkl_name,
    quantity_list,
    plot_dir="../plots/test",
    fig_ax=None,
    color_list=None,
    alpha=1.,
    label=None,
    linestyle=None,
    formatting=True,
    figsize=(8, 6),
    flip_sign=False,
    show_divisions=True,
    show_rb=False,
    show_init=False,
    show_rscale=False,
    perzone_avg_frac=0.5,
    num_time_chunk=4,
    time_bin_factor=2,
    tmax=None,
    rescale=False,
    flatten_rho=False,
    legend_all=True,
    verbose=False,
    prioritize_inner=False,
    use_avged_Mdot=False,
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
    mask_list = get_mask(D, prioritize_inner=prioritize_inner)

    # get important radii
    try:
        rEH = D["dump"]["r_eh"]
    except:
        a = 0  # for now
        rEH = 1.0 + np.sqrt(1.0 - a**2)
    r_sonic = D["dump"]["rs"]
    mdot = D["dump"]["mdot"]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot)[0]

    for i, quantity in enumerate(quantity_list):
        if fig_ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            ax = ax1d[i]  # here we assume that the number of axes passed = number of quantities

        pzaf = perzone_avg_frac
        if D["dump"]["driver/type"] == "kharma": pzaf=1 # if onezone, don't cutoff

        radii, profiles = calcFinalTimeAvg(D, tDivList, binNumList, quantity, perzone_avg_frac=pzaf, mask_list=mask_list, rescale=rescale, flatten_rho=flatten_rho, use_avged_Mdot=use_avged_Mdot, flip_sign=flip_sign)
        if i == 0:
            for b in range(len(tDivList) - 1):
                print("{}: t={:.5g}-{:.5g}".format(b, tDivList[b], tDivList[b + 1]))
    
        print_radius = None
        if verbose:
            if quantity == "Mdot" or quantity == "phib": print_radius = rEH
            elif quantity == "eta": print_radius = rB / 3.
        plotProfileQuantity(ax, radii, profiles, tDivList, colors=color_list, alpha=alpha, label=label, linestyle=linestyle, legend=(legend_all or (i == 0)), print_radius=print_radius)

        if show_init and ((quantity == "rho" and not flatten_rho) or quantity == "T" or quantity == "beta" or quantity == "u^r"):
            plotIC(ax, D, quantity)
        if show_rscale or (show_rb is not False):
            if show_rscale and quantity == 'phib':
                rarr = np.logspace(2, np.log10(rB), 20)
                factor = 0.5
                ax.plot(rarr, np.power(rarr, 1) * factor, "g-", alpha=0.5, lw=2)
                ax.text(rarr[len(rarr) // 2], np.power(rarr[len(rarr) // 2], 1) * factor / 3, r"$r^{1}$")
            if show_rb == True or show_rb == 'grey':
                if show_rb == 'grey': color ='grey'
                elif num_time_chunk==1: color=color_list[0]
                else: color='grey'
                ax.axvline(rB, color=color, lw=1, alpha=1, ls='--')

        # Formatting
        if formatting:
            ax.set_xlabel("Radius [$r_g$]")
            ylabel = variableToLabel(quantity)
            # if eta_norm_Bondi and quantity=='eta':
            #    ylabel = r'$\overline{\dot{M}-\dot{E}}/\dot{M}_B$'
            if rescale and "Mdot" in quantity:
                ylabel = ylabel.replace("arb. units", r"$\dot{M}_B$")
            if flatten_rho and quantity == "rho":
                ylabel = r"$\langle \rho \rangle r^{1.1}$ [arb. units]"
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if "eta" in quantity and quantity != "beta" and quantity != "etaMdot":
                ax.set_ylim([4e-3, 4])
            elif quantity == "beta":
                ax.set_ylim([1e-3, 10])
            elif quantity == "phib":
                ax.set_ylim([10, 1e6])
            elif quantity == "Mdot":
                ax.set_ylim([3e-4, 5])

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
    pkl_name = "../data_products/041625_n4_a0.9_toriilike_jks2_smth2_reconnect_profiles_all.pkl"
    pkl_name = "../data_products/041825_a0.9_rB2e5_jks2_smth2_profiles_all.pkl"
    pkl_name = "../data_products/042225_n4_a0.9_bondi_jks2_nocap_profiles_all.pkl"
    pkl_name = "../data_products/042325_a0.9_rB2e5_bondi_profiles_all.pkl"
    pkl_name = "../data_products/043025_a0.9_rB2e3_bondi_eks_profiles_all.pkl"
    pkl_name = "../data_products/051225_oz_a0.9_toriilike_newflr_profiles_all.pkl"
    #pkl_name = "../data_products/delta/051325_a0.9_rB2e5_bondi_eks_profiles_all.pkl"
    pkl_name = "../data_products/052825_a0.9_rB2e5_bondi_eks_largerout_profiles_all.pkl"
    #pkl_name = "../data_products/052925_a0.0_rB2e5_eks_largerout_profiles_all.pkl"
    #pkl_name = "../data_products/080425_a0.9_rB2e5_mixedinverter_profiles_all.pkl"
    #pkl_name = "../data_products/080425_a0.9_rB2e5_fafout_profiles_all.pkl"
    #pkl_name = "../data_products/080625_a0.9_rB2e6_profiles_all.pkl"
    pkl_name = "../data_products/080625_a0.9_rB2e5_96_profiles_all.pkl"
    pkl_name = "../data_products/091125_a0.9_rB2e5_rdepgmax5_profiles_all.pkl"
    #pkl_name = "../data_products/091525_a0.9_rB2e5_normal-recovery_profiles_all.pkl"
    #pkl_name = "../data_products/092525_a0.9_rB2e4_fafout_profiles_all.pkl"
    pkl_name = "../data_products/092525_a0.9_rB2e3_mom_cons_profiles_all.pkl"
    #pkl_name = "../data_products/092525_a0.9_rB2e5_mom_cons_test_profiles_all.pkl"
    #pkl_name = "../data_products/delta/092525_a0.9_rB2e4_mom_cons_profiles_all.pkl"
    pkl_name = "../data_products/092525_a0.9_rB2e6_momcons_profiles_all.pkl"
    #pkl_name = "../data_products/delta/092425_a0.7_rB2e5_momcons_profiles_all.pkl"
    #pkl_name = "../data_products/delta/102825_a0.5_rB2e4_momcons_profiles_all.pkl"
    #pkl_name = "../data_products/102925_a0.97_rB2e3_profiles_all.pkl"
    pkl_name = "../data_products/101225_n4_a0.9_bondi_nocap_momcons_profiles_all.pkl"
    pkl_name = "../data_products/102125_a0.9_rB2e5_mom_cons_96_profiles_all.pkl"
    #pkl_name = "../data_products/103025_a0.9_rB2e5_mom_cons_g43_profiles_all.pkl"
    #pkl_name = "../data_products/110725_a0.7_rB2e6_momcons_profiles_all.pkl"
    #pkl_name = "../data_products/111725_a0.9_rB2e5_mom_cons_rdepgmax_profiles_all.pkl"
    #pkl_name = "../data_products/111725_n4_a0.1_bondi_nocap_momcons_profiles_all.pkl"
    #pkl_name = "../data_products/120325_a0.9_rB2e5_mom_cons_rdepgmax_uconst_profiles_all.pkl"
    #pkl_name = "../data_products/120525_a0.9_rB2e5_mom_cons_rdepgmax2_uconst_profiles_all.pkl"
    #pkl_name = "../data_products/121025_a0.9_rB2e5_mom_cons_rdepgmax_full_profiles_all.pkl"
    #pkl_name = "../data_products/121325_a0.9_rB2e5_mom_cons_rdepgmax2_profiles_all.pkl"
    #pkl_name = "../data_products/121325_n4_a0.1_bondi_nocap_momcons_profiles_all.pkl"
    #pkl_name = "../data_products/121925_a0.9_rB2e5_mom_cons_rdepgmax2_vshallow_profiles_all.pkl"
    #pkl_name = "../data_products/122125_a0.9_rB2e5_mom_cons_rdepgmax3_uconst_profiles_all.pkl"
    #pkl_name = "../data_products/011226_a0.9_rB2e3_mom_cons_beta100_profiles_all.pkl"
    pkl_name = "../data_products/012026_a0_rB2e4_momcons_profiles_all.pkl"
    #pkl_name = "../data_products/012026_a0_rB2e5_mom_cons_test_profiles_all.pkl"
    #i = 21
    #pkl_name = "../data_products/"+gdirtags[i]+"_profiles_all.pkl"

    plot_dir = "../plots/test"  # common directory
    os.makedirs(plot_dir, exist_ok=True)
    # plot_dir = "/".join(pkl_name.split("/")[:-1])  # run specific directory
    # os.makedirs(plot_dir, exist_ok=True)

    quantityList = [
        #"rho",
        #"Mdot",
        #"beta",
        "eta",
        #"T",
        #"eta_Fl",
        #"eta_EM",
        #"u^r",
        #"abs_u^r",
        #"Omega",
        #"abs_Omega",
        #"phib",
    ]
    print(pkl_name)
    plotProfiles(pkl_name, quantityList, plot_dir=plot_dir, perzone_avg_frac=0.05, num_time_chunk=4, time_bin_factor=2., rescale=True, show_init=True, show_rscale=False, flatten_rho=False, prioritize_inner=False, tmax=700) #, verbose=True) #, use_avged_Mdot=True) #) #, label='__nolegend__',color_list=['g']) #
