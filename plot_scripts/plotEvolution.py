import os
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
from functools import partial

from matplotlib_settings import *
from ylabel_dictionary import *
from plotProfiles import readQuantity
from plot_utils import *


def tg2tcap(t, tcap):
    return t / tcap


def tcap2tg(t, tcap):
    return t * tcap

def plotEvolution(pkl, ax_passed=None, quantity="eta", average_factor=2, xaxis_t=False, use_Mdot_mean=False, scale_tB=False, rescaleMdot=False, color='k', alpha=1., label="__nolegend__", tmax=None, show_avg=False, show_negative=True, perzone_avg_frac=1., only_selectively_show=False, take_mean=False, radius=None):
    matplotlib_settings()
    print(pkl)
    plt.rcParams.update({"font.size": 25})
    if ax_passed is None:
        fig, ax = plt.subplots(1, 1, figsize=(24, 5))  # 8
    else:
        ax = ax_passed

    with open(pkl, "rb") as openFile:
        D = pickle.load(openFile)
    
    dump = D["dump"]
    quantity_arr, times = processTimeSeries(D, quantity, use_Mdot_mean=use_Mdot_mean, rescale=rescaleMdot, tmax=tmax, average_factor=average_factor, radius=radius)
    innermost = np.array(D["zones"]) == 0  # <= 1 #
    innermost = innermost[:len(times)]

    if xaxis_t:
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=dump["bondi/rs"], mdot=dump["bondi/mdot"])[0]
        if scale_tB:
            times /= np.power(rB, 3./2.)
        else:
            try:
                base = dump["multizone/base"]
                offset = int(np.ceil(np.log(8.0) / np.log(base))) - 1
                nzones_eff = dump["Params"]["Multizone/nzones_eff"]
                rout = np.power(base, nzones_eff + offset - 1)  # dump["multizone/combine_out_radius"]
                tcap = rout / np.sqrt(1.0 / rout + 1.0 / rB)
                print("r_cap {:.5g} and t_cap {:.5g}".format(rout, tcap))
                secax = ax.secondary_xaxis("top", functions=(partial(tg2tcap, tcap=tcap), partial(tcap2tg, tcap=tcap)))
                secax.set_xlabel(r"$t/r_{\rm cap}^{3/2}$")
            except:
                print("dump file not found, not adding second x axis")
        
        xaxis = times
        if only_selectively_show:
            quantity_arr[~innermost] = None
            quantity_arr[np.gradient(xaxis) > 1] = None # also flag out any artifact of the restarting
            quantity_arr[quantity_arr < 0] = None # flag out negative quantity
            switch_on_ncycle = D["ncycle_per_zone"] > 0
            if switch_on_ncycle:
                switch_list = D["cycles"][:len(times)]
                switch_pt = set(D["n0_zone"])
                switch_pt = np.array(sorted(switch_pt) + [switch_list[-1]])
                switch_num = np.array([0 if c==0 else np.argwhere((c > switch_pt[:-1]) & (c <= switch_pt[1:]))[0,0] for c in switch_list])
                ignore = (switch_pt[switch_num + 1] - switch_list > (switch_pt[switch_num + 1] - switch_pt[switch_num]) * perzone_avg_frac)
                quantity_arr[ignore] = None
                if take_mean: 
                    for s in np.array(sorted(set(switch_num))):
                        same_zone = (switch_num == s) & (np.isfinite(quantity_arr))
                        if len(np.where(same_zone)[0]) > 1:
                            subzone_mean = np.mean(quantity_arr[same_zone])
                            quantity_arr[same_zone] = subzone_mean
            else:
                print("NOT SUPPORTED YET")
            mask = np.isfinite(quantity_arr)
            if "Omega" in quantity:
                ax.plot(xaxis[mask], quantity_arr[mask], color=color, alpha=alpha, label=label, marker='.', markersize=10)
                ax.axhline(0, color='k', ls=':')
            else: ax.semilogy(xaxis[mask], quantity_arr[mask], color=color, alpha=alpha, label=label, marker='.', markersize=10) #, ls='None')
        else: 
            ax.semilogy(xaxis, quantity_arr, color=color, alpha=alpha, label=label)
            if show_negative: ax.semilogy(xaxis, -quantity_arr, color=color, alpha=alpha, ls=":")
    else:
        xaxis = np.arange(len(quantity_arr))
        if "Omega" in quantity:
            ax.plot(xaxis[innermost], quantity_arr[innermost], "k.")
            ax.axhline(0, color='k', ls=':')
        else:
            ax.semilogy(xaxis[innermost], quantity_arr[innermost], "k.")
        ax.set_xlabel("output #", labelpad=10)
    ylabel = variableToLabel(quantity)
    if rescaleMdot and quantity == "Mdot": ylabel = ylabel.replace('arb. units', r'$\dot{M}_B$')
    ax.set_ylabel(ylabel)

    if quantity == "eta" or quantity == "eta_EM":
        # ax.set_ylim([1e-4,0.4])
        ax.axhline(1, color="k", linestyle=":")
        ax.set_ylim([1e-3, 4])

    # show and print mean
    if (quantity == "phib" or quantity == "eta") and not only_selectively_show:
        if not xaxis_t:
            quantity_arr = quantity_arr[innermost]
        mean = np.mean(quantity_arr[int(float(len(quantity_arr)) / average_factor) :])
        print("mean of " + quantity + " is {:.5g}".format(mean))
        ax.axhline(mean, color="k", lw=3, alpha=0.2)  # horizontal line
        # ax.set_yscale('linear')
        # ax.set_ylim([0, 3 * mean])

    if tmax is not None and scale_tB and show_avg and only_selectively_show:
        i_keep = np.argwhere((times < tmax) & (times > tmax / average_factor))
        #if "052825_a0.9_rB2e5_bondi_eks_largerout" in pkl and quantity == "eta": 
        #    np.save("./plotevolutiondat.npy", quantity_arr[i_keep])
        mean = np.nanmean(quantity_arr[i_keep])
        print("mean of " + quantity + " is {:.5g}".format(mean))
        ax.plot([tmax / average_factor, tmax], [mean, mean], color=color, alpha=0.5, lw=10)

    if ax_passed is None:
        if xaxis_t:
            xlabel = r"$t$"
            if scale_tB: xlabel += r" [$t_B$]"
            ax.set_xlabel(xlabel, labelpad=10)
        ax.set_xlim([xaxis[0], xaxis[-1]])
        fig.tight_layout()
        output = "../plots/plot_evolution_" + quantity + ".png"
        plt.savefig(output, bbox_inches="tight")
        print("saved to " + output)
        plt.close(fig)
    else:
        return ax

def plotOmegaEvolution(pkl, ax_passed=None, xaxis_t=False):
    matplotlib_settings()
    print(pkl)
    plt.rcParams.update({"font.size": 25})
    if ax_passed is None:
        fig, ax = plt.subplots(1, 1, figsize=(24, 5))  # 8
    else:
        ax = ax_passed

    radii = [5, 10] #, 50]
    colors = ['k', 'b', 'g']
    with open(pkl, "rb") as openFile:
        D = pickle.load(openFile)
        times = D["times"]

        for i, radius in enumerate(radii):
            quantity_arr, _ = readTimeSeries(D, "Omega", radius)
            quantity_arr *= np.power(radius, 3.0 / 2)

            if xaxis_t:
                xaxis = times
                ax.plot(xaxis, quantity_arr, color=colors[i])
                ax.plot(xaxis, -quantity_arr, color=colors[i], ls=":")
            else:
                innermost = np.array(D["zones"]) == 0  # <= 1 #
                xaxis = np.arange(len(quantity_arr))
                ax.plot(xaxis[innermost], quantity_arr[innermost], colors[i])
    
    # each dumps midplane slice
    fnames = sorted(glob.glob('../data/' + pkl_name.split('/')[-1].replace('_profiles_all.pkl','') + '/*out0.*.phdf'))
    for fname in fnames[-100:]: # -1000
        dump = pyharm.load_dump(fname, ghost_zones=False)
        file_num = dump["n_dump"]
        #file_num = int(fname.split('/')[-1].split('.')[-2])
        phi_av = phi_average(dump, 'Omega')
        for i, radius in enumerate(radii):
            i_r = np.argmin(abs(dump["r1d"] - radius))
            Omega = phi_av[i_r, dump["nx2"]//2]
            Omega *= np.power(dump["r1d"][i_r], 3./2)
            #print(file_num, Omega)
            ax.plot(file_num, Omega, color=colors[i], marker='.', alpha=0.2) #, markersize=50)


    if xaxis_t:
        ax.set_xlabel("t", labelpad=10)
        try:
            base = dump["multizone/base"]
            offset = int(np.ceil(np.log(8.0) / np.log(base))) - 1
            nzones_eff = dump["Params"]["Multizone/nzones_eff"]
            rout = np.power(base, nzones_eff + offset - 1)  # dump["multizone/combine_out_radius"]
            rB = bondi.get_quantity_for_rarr([1], "RB", rs=dump["bondi/rs"], mdot=dump["bondi/mdot"])[0]
            tcap = rout / np.sqrt(1.0 / rout + 1.0 / rB)
            print("r_cap {:.5g} and t_cap {:.5g}".format(rout, tcap))
            secax = ax.secondary_xaxis("top", functions=(partial(tg2tcap, tcap=tcap), partial(tcap2tg, tcap=tcap)))
            secax.set_xlabel(r"$t/r_{\rm cap}^{3/2}$")
        except:
            print("dump file not found, not adding second x axis")
    else:
        ax.set_xlabel("output #", labelpad=10)
    ax.set_xlim([xaxis[0], xaxis[-1]])
    ylabel = variableToLabel("Omega")
    ax.set_ylabel(ylabel)

    if ax_passed is None:
        fig.tight_layout()
        output = "../plots/plot_omega_evolution.png"
        plt.savefig(output, bbox_inches="tight")
        print("saved to " + output)
        plt.close(fig)
    else:
        return ax

def plotEvolutionMultipanel(pkl, quantities=["Mdot", "eta", "phib"], average_factor=1.2, xaxis_t=True):
    plt.rcParams.update({"font.size": 25})
    plt.rcParams['axes.xmargin'] = 0  
    figsize = (24, 5 * len(quantities))
    fig, ax = plt.subplots(len(quantities), 1, figsize=figsize, sharex=True)

    for i, quantity in enumerate(quantities):
        plotEvolution(pkl, ax_passed=ax[i], quantity=quantity, average_factor=average_factor, xaxis_t=xaxis_t)

    fig.suptitle(pkl.replace("../data_products/", "").replace("_profiles_all.pkl", ""))
    fig.tight_layout()
    output = "../plots/plot_evolution_multipanel.png"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)

def plotCorrelation(pkl, q1="eta", q2="phib", last_factor=2.):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(3, 1, figsize=(16, 12))
    
    with open(pkl, "rb") as openFile:
        D = pickle.load(openFile)
        times = D["times"]

    q1_inv = False
    q2_inv = False
    if "inv" in q1:
        q1_inv = True
        q1 = q1.replace("inv_", "")
    if "inv" in q2:
        q2_inv = True
        q2 = q2.replace("inv_", "")
    q1arr = extractQuantity(D, q1, average_factor=last_factor, return_mean=False, use_Mdot_mean=False) #True)
    q2arr = extractQuantity(D, q2, average_factor=last_factor, return_mean=False, use_Mdot_mean=True) # to prevent nan
    if q1_inv: q1arr = 1. / q1arr
    if q2_inv: q2arr = 1. / q2arr
    p_t = corr(q1arr, q2arr)

    t = np.linspace(0,len(p_t),len(p_t))
    t -= np.mean(t)
    ax[0].plot(t, np.fft.fftshift(abs(p_t)))
    ax[0].axvline(0, color='k', ls=":")
    xaxis = np.arange(len(q1arr))
    ax[1].semilogy(xaxis, q1arr)
    ax[2].semilogy(xaxis, q2arr)

    # plot settings
    ax[0].set_xlabel(r'$\Delta$ output')
    q1_addlabel = ""
    q2_addlabel = ""
    if q1_inv: q1_addlabel = r"$^{-1}$"
    if q2_inv: q2_addlabel = r"$^{-1}$"
    ax[0].set_ylabel('corr(' + variableToLabel(q1) + q1_addlabel + ', ' + variableToLabel(q2) + q2_addlabel + ')')
    ax[1].set_ylabel(variableToLabel(q1) + q1_addlabel)
    ax[2].set_ylabel(variableToLabel(q2) + q2_addlabel)

    # save figure
    fig.suptitle(pkl.replace("../data_products/", "").replace("_profiles_all.pkl", ""))
    fig.tight_layout()
    output = "../plots/plot_corr_" + q1 + "_" + q2 + ".png"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)
    

def plotOmegaFieldEvolution(dirtag, ax_passed=None, xaxis_t=False):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    if ax_passed is None:
        fig, ax = plt.subplots(1, 1, figsize=(24, 5))  # 8
    else:
        ax = ax_passed

    fnames = sorted(glob.glob('../data/' + dirtag + '/*out0.*.phdf'))
    dump = pyharm.load_dump(fnames[0], ghost_zones=False)
    rEH = dump["r_eh"]
    i_rEH = np.argmin(abs(dump["r1d"] - rEH))
    for fname in fnames[-10:]:
        dump = pyharm.load_dump(fname, ghost_zones=False)
        file_num = dump["n_dump"]
        num = phi_average(dump, 'F_0_1')
        den = phi_average(dump, 'F_1_3')
        omegaF = ((num / den) * rEH / dump["a"])[i_rEH, 0]
        print(file_num, omegaF)
        #ax.semilogy(file_num, -num[i_rEH,1], color='k', marker='.') #, markersize=50)
        ax.semilogy(file_num, omegaF, color='b', marker='.') #, markersize=50)


    if xaxis_t:
        ax.set_xlabel("t", labelpad=10)
        try:
            base = dump["multizone/base"]
            offset = int(np.ceil(np.log(8.0) / np.log(base))) - 1
            nzones_eff = dump["Params"]["Multizone/nzones_eff"]
            rout = np.power(base, nzones_eff + offset - 1)  # dump["multizone/combine_out_radius"]
            rB = bondi.get_quantity_for_rarr([1], "RB", rs=dump["bondi/rs"], mdot=dump["bondi/mdot"])[0]
            tcap = rout / np.sqrt(1.0 / rout + 1.0 / rB)
            print("r_cap {:.5g} and t_cap {:.5g}".format(rout, tcap))
            secax = ax.secondary_xaxis("top", functions=(partial(tg2tcap, tcap=tcap), partial(tcap2tg, tcap=tcap)))
            secax.set_xlabel(r"$t/r_{\rm cap}^{3/2}$")
        except:
            print("dump file not found, not adding second x axis")
    else:
        ax.set_xlabel("output #", labelpad=10)
    ylabel = variableToLabel("omegaF")
    ax.set_ylabel(ylabel)

    if ax_passed is None:
        fig.tight_layout()
        output = "../plots/plot_omega_field_evolution.png"
        plt.savefig(output, bbox_inches="tight")
        print("saved to " + output)
        plt.close(fig)
    else:
        return ax

def plotHistogram(pkl, ax_passed=None, quantity="eta", average_factor=2, use_Mdot_mean=False, rescaleMdot=False, tmax=None, perzone_avg_frac=0.05, radius=None, color='k', normalize=True, only_active_zone=True):
    matplotlib_settings()
    print(pkl)
    plt.rcParams.update({"font.size": 25})
    if ax_passed is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        ax = ax_passed

    with open(pkl, "rb") as openFile:
        D = pickle.load(openFile)
    
    dump = D["dump"]
    r_sonic = D["dump"]["rs"]
    mdot = D["dump"]["mdot"]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot)[0]
    rEH = D["dump"]["r_eh"]
    if radius is None:
        if quantity == "Mdot":
            radius = rEH
        elif quantity == "phib":
            radius = rEH
        elif quantity == "eta":
            radius = rB / 3.
    quantity_arr, times = processTimeSeries(D, quantity, use_Mdot_mean=use_Mdot_mean, rescale=rescaleMdot, tmax=tmax, average_factor=average_factor, radius=radius)
    times /= np.power(rB, 3./2.)
    zones = np.array(D["zones"][:len(times)])
    
    switch_on_ncycle = D["ncycle_per_zone"] > 0
    if switch_on_ncycle:
        switch_list = D["cycles"][:len(times)]
        switch_pt = set(D["n0_zone"])
        switch_pt = np.array(sorted(switch_pt) + [switch_list[-1]])
        switch_num = np.array([0 if c==0 else np.argwhere((c > switch_pt[:-1]) & (c <= switch_pt[1:]))[0,0] for c in switch_list])
        keep = (switch_pt[switch_num + 1] - switch_list <= (switch_pt[switch_num + 1] - switch_pt[switch_num]) * perzone_avg_frac)
        quantity_arr = quantity_arr[keep]# = None
        times = times[keep]
        zones = zones[keep]
    else:
        print("NOT SUPPORTED YET")
    if only_active_zone:
        active_range = D["active_range"]
        zone_num = np.where(np.array(active_range)[:,0] * np.sqrt(8.) > radius)[0][0]-1 # TODO: take base instead of 8
        if zone_num < 0: zone_num = 0 # for the innermost zone
        i_keep = (zones == zone_num)
        quantity_arr = quantity_arr[i_keep]
        times = times[i_keep]
    i_keep = np.argwhere((times < tmax) & (times > tmax / average_factor))
    quantity_arr = quantity_arr[i_keep]
    times = times[i_keep]


    log = False
    rng = (np.nanmin(quantity_arr),np.nanmax(quantity_arr))
    if quantity == "Mdot" or quantity == "eta":
        log = True
        quantity_arr = np.log10(quantity_arr)
        if quantity == "eta": rng = (-3,1)
    elif quantity == "phib":
        rng = (10, 60)
        
    counts, bins = np.histogram(quantity_arr, range=rng, bins=10)
    print("N = {}".format(np.sum(counts)))
    if normalize:
        counts = counts / np.sum(counts)
        ax.set_ylim([0,0.8])
    ax.stairs(counts, bins, color=color)

    label = variableToLabel(quantity)
    if rescaleMdot and quantity == "Mdot": label = label.replace('arb. units', r'$\dot{M}_B$')
    if log:
        label = r"$\log_{\rm 10}($" + label + r"$)$"
    ax.set_xlabel(label)

    # calc mean
    if log:
        quantity_arr = np.power(10, quantity_arr)
    mean = np.nanmean(quantity_arr)
    print("mean of " + quantity + " is {:.5g}".format(mean))
    if 1:
        if log:
            mean = np.log10(mean)
        ax.plot(mean, 0.01, color=color, marker='x', ms=10)

    if ax_passed is None:
        fig.tight_layout()
        output = "../plots/plot_histogram_" + quantity + ".png"
        plt.savefig(output, bbox_inches="tight")
        print("saved to " + output)
        plt.close(fig)
    else:
        return ax

def compareHistogram(a=0.9, quantity="eta"):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    if a == 0.9:
        dirtags = [
            "101225_n4_a0.9_bondi_nocap_momcons",
            "092525_a0.9_rB2e3_mom_cons",
            "delta/092525_a0.9_rB2e4_mom_cons",
            "092525_a0.9_rB2e5_mom_cons_test",
            "092525_a0.9_rB2e6_momcons",
                ]
    elif a == 0.7:
        dirtags = [
            "111725_n4_a0.7_bondi_nocap_momcons",
            "delta/102825_a0.7_rB2e3_momcons",
            "delta/102825_a0.7_rB2e4_momcons",
            "delta/092425_a0.7_rB2e5_momcons",
            #"110725_a0.7_rB2e6_momcons",
                ]
    elif a == 0.5:
        dirtags = [
            "111725_n4_a0.5_bondi_nocap_momcons",
            "delta/102825_a0.5_rB2e3_momcons",
            "delta/102825_a0.5_rB2e4_momcons",
            "delta/092425_a0.5_rB2e5_momcons",
            "110725_a0.5_rB2e6_momcons",
                ]
    elif a == 0.3:
        dirtags = [
            "111725_n4_a0.3_bondi_nocap_momcons",
            "delta/102825_a0.3_rB2e3_momcons",
            "delta/102825_a0.3_rB2e4_momcons",
            "delta/092425_a0.3_rB2e5_momcons",
            "110725_a0.3_rB2e6_momcons",
                ]
    elif a == 0.1:
        dirtags = [
            "111725_n4_a0.1_bondi_nocap_momcons",
            "delta/102825_a0.1_rB2e3_momcons",
            "delta/102825_a0.1_rB2e4_momcons",
            "delta/092425_a0.1_rB2e5_momcons",
            "110725_a0.1_rB2e6_momcons",
                ]
    tmaxs = [400] + [700] * (len(dirtags) - 1)
    colors = plt.cm.plasma(np.linspace(0., 1., len(dirtags)))
    for i,dirtag in enumerate(dirtags):
        pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"
        ax = plotHistogram(pkl_name, ax_passed=ax, quantity=quantity, tmax=tmaxs[i], color=colors[i]) #, perzone_avg_frac=0.5)

    ax.set_title(r"$a_*=$" + str(a))

    # save
    output = "../plots/compare_histogram_" + quantity + "_a" + str(a) + ".png"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)

if __name__ == "__main__":
    dirtag = "100724_a0.5_n4"
    # dirtag = "100724_a0.5_n8"
    # dirtag = "100724_a0.5_oz"
    # dirtag = "2023/122723_n4_onezone_wks0.04/00000"
    dirtag = "030325_a0.9_oz_128"
    dirtag = "delta/030525_a0.9_oz"
    dirtag = "delta/032025_a0.9_oz_clearangle"
    # dirtag = "030425_a0.5_safe_longtin20"
    #dirtag = "030725_a0.9_safe_longtin20"
    # dirtag = "030925_a0.7_safe_longtin20"
    #dirtag = "031125_a0.9_cap_correctly"
    dirtag="032125_n4a0.9_toriilike"
    dirtag="041625_n4_a0.9_toriilike_jks2_smth2_reconnect"
    dirtag = "042325_a0.9_rB2e5_bondi"
    dirtag="043025_a0.9_rB2e3_bondi_eks"
    #dirtag="050625_a0.9_rB2e5_oz_test"
    #dirtag="051225_a0.5_rB2e5_toriilike_beta1"
    #dirtag="051225_n4_a0.9_torrilike_nocap_newflr"
    #dirtag="051225_n4_a-0.9_toriilike_eks"
    #dirtag="051225_oz_a0.9_toriilike_newflr"
    dirtag="051225_oz_a0.9_bondi_newflr"
    #dirtag="051225_n4_a0.9_bondi_newflr"
    #dirtag="delta/051325_a0.9_rB2e5_bondi_eks"
    #dirtag="051325_a0.0_rB2e5_eks"
    #dirtag="052125_torus_noehbuffer_noismr_a0.5"
    #dirtag="052725_torus_noehbuffer_noismr_a0.5_diffflr"
    #dirtag="052825_a0.9_rB2e5_bondi_eks_largerout"
    #dirtag="052825_n4_a-0.9_torilike_nocap_newflr"
    #dirtag="080425_a0.9_rB2e5_fafout" #mixedinverter" #avgneighbor" #sigma10" #
    #dirtag="080625_a0.9_rB2e3_mixedinverter"
    #dirtag="080625_a0.9_rB2e6"
    #dirtag="080625_a0.9_rB2e5_96"
    #dirtag="082725_a0.9_rB2e3_fafout"
    #dirtag="091525_a0.9_rB2e5_normal-recovery"
    dirtag="092525_a0.9_rB2e5_mom_cons_test" #_all" #
    #dirtag="092525_a0.9_rB2e6_momcons"
    #dirtag="delta/102825_a0.7_rB2e3_momcons"
    #dirtag="102125_a0.9_rB2e5_mom_cons_96"
    #dirtag="102925_a0.97_rB2e3"
    #dirtag="111025_a0.9_rB2e3_Btor"
    dirtag="111725_a0.9_rB2e3_Btor_T+"
    #dirtag="111725_a0.9_rB2e5_mom_cons_rdepgmax"
    #dirtag="120325_a0.9_rB2e5_mom_cons_rdepgmax_uconst"
    pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"

    average_factor = 1.25 #1.5  # 2 #
    quantities = ["Mdot", 'eta', 'Omega10', 'phib'] #["Mdot", "eta", "phib"]  # 
    #plotEvolutionMultipanel(pkl_name, quantities=quantities, average_factor=average_factor, xaxis_t=True) #False)  # 
    #for q2 in ['Mdot']: #'inv_abs_u^th', 'Omega2', 'Omega5', 'Omega10', 'Omega50', 'phib']:
    #    plotCorrelation(pkl_name, q1='eta', q2=q2) #, last_factor=1.2)
    #plotOmegaEvolution(pkl_name)
    #plotOmegaFieldEvolution(dirtag)
    #plotHistogram(pkl_name, tmax=700)
    for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
        compareHistogram(a, "phib")
