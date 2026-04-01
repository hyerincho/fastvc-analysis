import os
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
from scipy.stats import norm

from matplotlib_settings import *
from ylabel_dictionary import *
from plot_utils import *


def tg2tcap(t, tcap):
    return t / tcap


def tcap2tg(t, tcap):
    return t * tcap

def tg2tB(t, tB):
    return t / tB

def tB2tg(t, tB):
    return t * tB

def lognormal(x,mu=0,sigma=1):
    return 1. / (np.log(10.) * sigma * x * np.sqrt(2. * np.pi)) * np.exp(- np.power(np.log10(x) - mu, 2.) / (2. * sigma ** 2)) 

dirtags = [
    "121325_n4_a0.1_bondi_nocap_momcons",
    "121325_n4_a0.3_bondi_nocap_momcons",
    "121325_n4_a0.5_bondi_nocap_momcons",
    "121325_n4_a0.7_bondi_nocap_momcons",
    "121325_n4_a0.9_bondi_nocap_momcons",
    "delta/102825_a0.1_rB2e3_momcons",
    "delta/102825_a0.3_rB2e3_momcons",
    "delta/102825_a0.5_rB2e3_momcons",
    "delta/102825_a0.7_rB2e3_momcons",
    "092525_a0.9_rB2e3_mom_cons",
    "102925_a0.97_rB2e3",
    "delta/102825_a0.1_rB2e4_momcons",
    "delta/102825_a0.3_rB2e4_momcons",
    "delta/102825_a0.5_rB2e4_momcons",
    "delta/102825_a0.7_rB2e4_momcons",
    "delta/092525_a0.9_rB2e4_mom_cons",
    "delta/092425_a0.1_rB2e5_momcons",
    "delta/092425_a0.3_rB2e5_momcons",
    "delta/092425_a0.5_rB2e5_momcons",
    "delta/092425_a0.7_rB2e5_momcons",
    "092525_a0.9_rB2e5_mom_cons_test",
    "110725_a0.1_rB2e6_momcons",
    "110725_a0.3_rB2e6_momcons",
    "110725_a0.5_rB2e6_momcons",
    "110725_a0.7_rB2e6_momcons",
    "092525_a0.9_rB2e6_momcons",
    ]

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
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=dump["bondi/rs"], mdot=dump["bondi/mdot"], gam=dump["gam"])[0]
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
            rB = bondi.get_quantity_for_rarr([1], "RB", rs=dump["bondi/rs"], mdot=dump["bondi/mdot"], gam=dump["gam"])[0]
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
    #p_t = corr(q1arr, q2arr)
    p_t = np.correlate(q1arr-np.mean(q1arr),q2arr-np.mean(q2arr),'same')

    t = np.linspace(0,len(p_t),len(p_t))
    t -= np.mean(t)
    #ax[0].plot(t, np.fft.fftshift(abs(p_t)))
    ax[0].plot(t, ((p_t)))
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
            rB = bondi.get_quantity_for_rarr([1], "RB", rs=dump["bondi/rs"], mdot=dump["bondi/mdot"], gam=dump["gam"])[0]
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

def plotHistogram(pkl, ax_passed=None, quantity="eta", average_factor=2, use_Mdot_mean=False, rescaleMdot=True, tmax=None, perzone_avg_frac=0.05, radius=None, color='k', normalize=True, only_active_zone=True, label='__nolegend__'):
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
    gam = D["dump"]["gam"]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
    rEH = D["dump"]["r_eh"]
    if radius is None:
        if quantity == "Mdot":
            radius = 5 #rEH #
        elif quantity == "phib":
            radius = rEH
        elif quantity == "eta":
            radius = rB / 3.
        elif quantity == "etaB":
            radius = rB / 3.
        elif quantity == "s":
            radius = 5.
    if D["times"][-1] < tmax * np.power(rB,3./2.):
        tmax = D["times"][-1] / np.power(rB, 3./2.)
        print("new tmax is {:.5g} tB".format(tmax))

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
        try: zone_num = np.where(np.array(active_range)[:,0] * np.sqrt(8.) > radius)[0][0]-1 # TODO: take base instead of 8
        except: zone_num = 0
        if zone_num < 0: zone_num = 0 # for the innermost zone
        i_keep = (zones == zone_num)
        quantity_arr = quantity_arr[i_keep]
        times = times[i_keep]
    i_keep = np.argwhere((times < tmax) & (times > tmax / average_factor))
    quantity_arr = quantity_arr[i_keep][:,0]
    times = times[i_keep][:,0]


    log = False
    quantity_arr = quantity_arr[~np.isnan(quantity_arr)]
    rng = (np.min(quantity_arr),np.max(quantity_arr))
    if quantity == "Mdot" or quantity == "eta" or quantity == "etaB" or quantity == "s":
        log = True
        if quantity == "eta": rng = (-7.5,1.5)#(-3,1)
        elif quantity == "Mdot": rng = (-11,0) #(-5,0)
        elif quantity == "etaB": rng = (-6,-1)
        elif quantity == "s": 
            rng = (-3,2)
            quantity_arr *= -1
    elif quantity == "phib":
        log = True
        rng = (2.3, 4.6) #(1,2) #(10, 60)
    if log:
        quantity_arr = np.log(quantity_arr)
    x_mean = np.nanmean((quantity_arr)) #np.log10(mean**2 / np.sqrt(mean**2+sigmaX**2)) #
    x_std = np.nanstd((quantity_arr)) #np.log10(1. + sigmaX ** 2 / mean **2) #
    print(quantity + " x_mean {:.3g} x_std {:.3g}".format(x_mean, x_std))
        
    counts, bins = np.histogram(quantity_arr, range=rng, bins=20)
    Ntot = np.sum(counts)
    print("N = {}".format(Ntot))

    # test goodness of fit
    if 0:
        expected = (norm.cdf(bins[1:], x_mean, x_std) - norm.cdf(bins[:-1], x_mean, x_std)) * Ntot
        chi2presum = (np.power(counts - expected, 2.) / expected)
        onlysum = (counts > 0)
        chi2 = np.sum(chi2presum[onlysum])
        Gpresum = counts * np.log(counts / expected)
        G = np.sum(Gpresum[onlysum]) * 2.
        df = np.sum(onlysum) - 3 # degrees of freedom
        print("G = {:.5g} df = {}".format(G, df))
    if normalize:
        counts = counts / np.sum(counts * np.diff(bins))
        #ax.set_ylim([0,0.8])
    ax.stairs(counts, bins, color=color, label=label)
    #ax.plot((bins[1:] + bins[:-1])/2., counts, color=color, marker='.', ls='none')
    #ax.hist(quantity_arr, range=rng, bins=10, density=True, color=color, alpha=0.1)

    label = variableToLabel(quantity).replace("\overline","")
    if rescaleMdot and quantity == "Mdot": label = label.replace(' [arb. units]', r'$/\dot{M}_B$')
    if log:
        label = r"$\ln($" + label + r"$)$"
    ax.set_xlabel(label)

    # calc mean
    if log: quantity_arr = np.exp(quantity_arr)
    mean = np.nanmean(quantity_arr)
    print("mean of " + quantity + " is {:.5g} (estimate {:.5g})".format(mean, np.exp(x_mean + x_std**2/2)))
    if log:
        mean_est = (x_mean + x_std ** 2/ 2.)
        ax.axvline(mean_est, color=color, alpha=0.5)

    # log normal dist
    if log:
        x_axis = np.logspace(np.log10(np.exp(rng[0])), np.log10(np.exp(rng[1])), 100)

        #print(np.power(10,x_mean + x_std**2/2.)) #, 0.03, color=color, marker='.', ms=10)
        #print(x_mean, np.log10(mean**2 / np.sqrt(mean**2+sigmaX**2)))
        #print(x_std, np.log10(1. + sigmaX ** 2 / mean **2))
        #ax.plot(np.log10(x_axis), lognormal(x_axis, x_mean, x_std), color=color)
        ax.plot(np.log(x_axis), norm.pdf(np.log(x_axis), x_mean, x_std), color=color, lw=5, alpha=0.3)
    else:
        x_axis = np.linspace(rng[0], rng[1], 100)
        ax.plot(np.log(x_axis), norm.pdf(x_axis, x_mean, x_std), color=color, lw=5, alpha=0.3)

    if ax_passed is None:
        fig.tight_layout()
        output = "../plots/plot_histogram_" + quantity + ".png"
        plt.savefig(output, bbox_inches="tight")
        print("saved to " + output)
        plt.close(fig)
    else:
        return ax

def compareHistogram(a=0.9, quantity="eta", rescaleMdot=True, radius=None):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    if a == 0.9:
        dirtags = [
            #"051225_oz_a0.9_bondi_newflr",
            #"051225_n4_a0.9_bondi_newflr",
            "101225_n4_a0.9_bondi_nocap_momcons",
            "121325_n4_a0.9_bondi_nocap_momcons",
            "092525_a0.9_rB2e3_mom_cons",
            "delta/092525_a0.9_rB2e4_mom_cons",
            "092525_a0.9_rB2e5_mom_cons_test",
            "092525_a0.9_rB2e6_momcons",
                ]
    elif a == 0.7:
        dirtags = [
            #"111725_n4_a0.7_bondi_nocap_momcons",
            "121325_n4_a0.7_bondi_nocap_momcons",
            "delta/102825_a0.7_rB2e3_momcons",
            "delta/102825_a0.7_rB2e4_momcons",
            "delta/092425_a0.7_rB2e5_momcons",
            "110725_a0.7_rB2e6_momcons",
                ]
    elif a == 0.5:
        dirtags = [
            #"111725_n4_a0.5_bondi_nocap_momcons",
            "121325_n4_a0.5_bondi_nocap_momcons",
            "delta/102825_a0.5_rB2e3_momcons",
            "delta/102825_a0.5_rB2e4_momcons",
            "delta/092425_a0.5_rB2e5_momcons",
            "110725_a0.5_rB2e6_momcons",
                ]
    elif a == 0.3:
        dirtags = [
            #"111725_n4_a0.3_bondi_nocap_momcons",
            "121325_n4_a0.3_bondi_nocap_momcons",
            "delta/102825_a0.3_rB2e3_momcons",
            "delta/102825_a0.3_rB2e4_momcons",
            "delta/092425_a0.3_rB2e5_momcons",
            "110725_a0.3_rB2e6_momcons",
                ]
    elif a == 0.1:
        dirtags = [
            #"111725_n4_a0.1_bondi_nocap_momcons",
            "121325_n4_a0.1_bondi_nocap_momcons",
            "delta/102825_a0.1_rB2e3_momcons",
            "delta/102825_a0.1_rB2e4_momcons",
            "delta/092425_a0.1_rB2e5_momcons",
            "110725_a0.1_rB2e6_momcons",
                ]
    tmaxs = [700] * (len(dirtags))
    colors = plt.cm.plasma(np.linspace(0., 1., len(dirtags)))
    for i,dirtag in enumerate(dirtags):
        pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"
        ax = plotHistogram(pkl_name, ax_passed=ax, quantity=quantity, tmax=tmaxs[i], color=colors[i], rescaleMdot=rescaleMdot, radius=radius) #, perzone_avg_frac=0.5)

    ax.set_title(r"$a_*=$" + str(a))

    # save
    output = "../plots/compare_histogram_" + quantity + "_a" + str(a) + ".png"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)

def compareHistogramAll():
    matplotlib_settings()
    plt.rcParams.update({"font.size": 30})
    fig, ax = plt.subplots(2, 5, figsize=(7*5, 6*2), sharex='row',sharey='row')
    plt.subplots_adjust(wspace=0., hspace=0.3)
    ax1d = ax.reshape(-1)

    dirtags = gdirtags
    tmaxs = [700] * (len(dirtags))
    colors = list(plt.cm.coolwarm(np.linspace(0., 1., 5)))
    labels = [r'$R_B=400$',r'$2000$',r'$2\cdot 10^4$',r'$2\cdot 10^5$',r'$2\cdot 10^6$']
    for i,dirtag in enumerate(dirtags):
        pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"
        with open(pkl_name, "rb") as openFile:
            D = pickle.load(openFile)
        nzeff = D["dump"]["Params"]["Multizone/nzones_eff"]
        a = D["dump"]["a"]
        if a > 0:
            iax = np.where(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.97])==a)[0][0]
            for j, quantity in enumerate(["Mdot","eta"]):
                if nzeff==4 or nzeff==8: plotHistogram(pkl_name, ax_passed=ax[j,iax], quantity=quantity, tmax=tmaxs[i], color=colors[nzeff-4], label=labels[nzeff-4])
    
    panel_label = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
    titles = [r'$a_*=0.1$',r'$a_*=0.3$',r'$a_*=0.5$',r'$a_*=0.7$',r'$a_*=0.9$'] #r'$a_*=0$',
    for i in range(5):
        ax1d[i].set_title(titles[i], fontdict={'fontsize': 30})
    #for i in range(len(ax1d)):
    #    ax1d[i].text(0.01, 0.90, panel_label[i], transform=ax1d[i].transAxes)
    ax1d[0].legend(fontsize=25, bbox_to_anchor=(0.02, 0.95), loc='upper left')
    ax1d[0].set_ylim([0,2.])
    ax[1,0].set_ylim([0,1.2])
    ax[0,0].set_ylabel('PDF')
    ax[1,0].set_ylabel('PDF')

    # save
    output = "../plots/compare_histogram_all.pdf"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)

def calcStats(pkl, quantity="eta", average_factor=2, use_Mdot_mean=False, rescaleMdot=True, tmax=None, perzone_avg_frac=0.05, radius=None, only_active_zone=True, verbose=True):
    if verbose: print(pkl)

    with open(pkl, "rb") as openFile:
        D = pickle.load(openFile)
    
    dump = D["dump"]
    r_sonic = dump["rs"]
    mdot = dump["mdot"]
    gam = dump["gam"]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
    rEH = dump["r_eh"]
    a = dump["a"]
    if radius is None:
        if quantity == "Mdot":
            radius = 5 #rEH
        elif quantity == "phib":
            radius = rEH
        elif quantity == "eta":
            radius = rB / 3.
        elif quantity == "etaB":
            radius = rB / 3.
    if D["times"][-1] < tmax * np.power(rB,3./2.):
        tmax = D["times"][-1] / np.power(rB, 3./2.)
        print("new tmax is {:.5g} tB".format(tmax))

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
    quantity_arr = quantity_arr[i_keep][:,0]
    times = times[i_keep][:,0]

    x_mean = np.nanmean(np.log(quantity_arr)) #np.log10(mean**2 / np.sqrt(mean**2+sigmaX**2)) #
    x_std = np.nanstd(np.log(quantity_arr)) #np.log10(1. + sigmaX ** 2 / mean **2) #
    return x_mean, x_std, rB, a, quantity_arr

def compareAllStats(quantity, ax_passed=None, rescaleMdot=True, radius=None):
    import seaborn as sns
    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    if ax_passed is None:
        fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    else:
        ax = ax_passed
    dirtags = gdirtags
    tmaxs = [700] * (len(dirtags))
    tmaxs[4] = 500 # exception
    #colors = np.repeat(plt.cm.plasma(np.linspace(0., 1., 5)), np.array([5, 6, 5, 5, 5]), axis=0)
    colors = list(plt.cm.gnuplot(np.linspace(0.9, 0., 5))) + ['gray', 'lightgreen']
    rBarr = []
    violinarr = []
    mean_save = []
    std_save = []
    a_save = []
    for i,dirtag in enumerate(dirtags):
        pkl = "../data_products/" + dirtag + "_profiles_all.pkl"
        x_mean, x_std, rB, a, quantity_arr = calcStats(pkl, quantity, tmax=tmaxs[i], rescaleMdot=rescaleMdot, radius=radius)
        color = colors[np.where(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0])==a)[0][0]]
        ax[0].plot(rB, x_mean, color=color, marker='.', ms=20)
        rBarr += [rB]
        mean_save += [x_mean]
        std_save += [(x_std)]
        a_save += [a]
        print(quantity + " mean {:.3g} std {:.3g}".format(x_mean, x_std))
        if 0: #a == 0.9 and rB>1e5:
            violinarr += [np.power(10, quantity_arr)]
        #ax[1].plot(rB, np.power(10,x_mean + x_std**2/2.), color=color, marker='.', ms=10)
        ax[1].plot(rB, (x_std), color=color, marker='.', ms=20)
    
    rBlist = np.logspace(2, 7, 10)
    a_save = np.array(a_save)
    rBarr = np.array(rBarr)
    mean_save = np.array(mean_save)
    std_save = np.array(std_save)
    for a in [0,0.1, 0.3, 0.5, 0.7, 0.9]:
        ifit = (a_save == a)
        color = colors[np.where(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.97,0])==a)[0][0]]
        popt, pcov = curve_fit(lin_func, np.log10(rBarr[ifit]), (mean_save[ifit]))
        ax[0].semilogx(rBlist, lin_func(np.log10(rBlist), *popt), color=color, alpha=0.5)
        print("mu fit a={:.3g} b={:.3g}".format(popt[0], popt[1]))
        popt, pcov = curve_fit(lin_func, np.log10(rBarr[ifit]), (std_save[ifit]))
        ax[1].semilogx(rBlist, lin_func(np.log10(rBlist), *popt), color=color, alpha=0.5)
        print("sigma fit a={:.3g} b={:.3g}".format(popt[0], popt[1]))

    if 0:
        ax[0].violinplot(dataset=violinarr, positions=rBarr) #, log_scale=True)
        ax[0].set_yscale('log')

    # plot settings
    for j in range(2):
        ax[j].set_xscale('log')
    if quantity == "eta": ylabel = "eta"
    if quantity == "etaB": ylabel = "dot{E}"
    if quantity == "Mdot": ylabel = "dot{M}"
    ax[0].set_title(rf'$\overline{{\ln\{ylabel}}}$')
    ax[1].set_title(rf'$<\ln\{ylabel}>$')
    ax[0].set_ylim([-9,-1])
    ax[1].set_ylim([0,1.8])
    ax[0].set_xlim([1e2,1e7])

    if ax_passed is None:
        fig.tight_layout()
        output = "../plots/compare_all_stats_" + quantity + ".png"
        plt.savefig(output, bbox_inches="tight")
        print("saved to " + output)
        plt.close(fig)
    else:
        return ax

def plotEvolutionSpin(pkl, ax_passed=None, tmax=700):
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
    a_init = dump["a"]
    rEH = dump["r_eh"]
    Edot_arr, times = processTimeSeries(D, "Edot", tmax=tmax, radius=rEH*3)
    Ldot_arr, _ = processTimeSeries(D, "Ldot", tmax=tmax, radius=rEH*3)
    Mdot_arr, _ = processTimeSeries(D, "Mdot", tmax=tmax, radius=rEH*3)
    innermost = np.array(D["zones"]) == 0
    innermost = innermost[:len(times)]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=dump["bondi/rs"], mdot=dump["bondi/mdot"], gam=dump["gam"])[0]
    tB = np.power(rB, 3./2.)

    Edot_arr[~innermost] = 0
    Ldot_arr[~innermost] = 0

    fedd = 2e-5 # EHT 2019 Paper V
    m_p = const.m_p
    sigmaT = const.sigma_T
    M = 6.5e9*u.Msun
    tg = G * M / c**3
    prefactor = fedd * 4. * (np.pi * G * m_p / (0.1 * sigmaT * c)).to('1/s')

    a_arr = []
    dadt_arr = [0]
    s_arr = [0]
    for i in range(len(times)):
        if i == 0: a_arr += [a_init]
        else: 
            dadt = prefactor * (Ldot_arr[i-1] / Mdot_arr[i-1] - 2. * a_arr[i-1] * Edot_arr[i-1] / Mdot_arr[i-1])
            dadt_arr += [dadt.value]
            s_arr += [(dadt / prefactor).to('')]
            a_arr += [a_arr[i-1] + (dadt * (times[i]-times[i-1]) * tg).to('').value]

    if 1:
        ax.plot(times, a_arr)
        ax.axhline(0, color='k', ls=':')
        ax.axhline(a_init, color='b', ls=':')
        da = 0.01
        ax.set_ylim([a_init-da,a_init+da])
        #ax.plot(times, s_arr)
        #ax.set_yscale('symlog')
    else:
        #t_spin = np.array(a_arr) / np.array(dadt_arr) / tB
        t_spin = (a_init / (np.array(dadt_arr)/u.s) / (tB * tg)).to('')
        ax.plot(times, abs(t_spin), ls='none', marker='x')
        #ax.set_ylim([-1e3,1e3])
        ax.set_yscale('log')
    
    if ax_passed is None:
        secax = ax.secondary_xaxis("top", functions=(partial(tg2tB, tB=tB), partial(tB2tg, tB=tB)))
        secax.set_xlabel(r"$t$ [$t_B$]")
        ax.set_xlabel(r"$t$ [$t_g$]", labelpad=10)
        fig.tight_layout()
        output = "../plots/plot_evolution_spin.png"
        plt.savefig(output, bbox_inches="tight")
        print("saved to " + output)
        plt.close(fig)
    else:
        return ax

def testCorrelation(pkl, ax_passed=None, quantity="eta", average_factor=2, use_Mdot_mean=False, rescaleMdot=True, tmax=700., perzone_avg_frac=0.05, radius=None, color='k', only_active_zone=True):
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
    gam = D["dump"]["gam"]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
    tB = np.power(rB, 3./2.)
    rEH = D["dump"]["r_eh"]
    if radius is None:
        if quantity == "Mdot":
            radius = rEH
        elif quantity == "phib":
            radius = rEH
        elif quantity == "eta":
            radius = rB / 3.
        elif quantity == "etaB":
            radius = rB / 3.
    if tmax is not None:
        if D["times"][-1] < tmax * tB:
            tmax = D["times"][-1] / tB
            print("new tmax is {:.5g} tB".format(tmax))

    quantity_arr, times = processTimeSeries(D, quantity, use_Mdot_mean=use_Mdot_mean, rescale=rescaleMdot, tmax=tmax, average_factor=average_factor, radius=radius)
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
    i_keep = np.argwhere((times < tmax * tB) & (times > tmax * tB / average_factor))
    quantity_arr = quantity_arr[i_keep][:,0]
    times = times[i_keep][:,0]

    x_mean = np.nanmean(quantity_arr) #np.log10(mean**2 / np.sqrt(mean**2+sigmaX**2)) #
    x_std = np.nanstd(quantity_arr) #np.log10(1. + sigmaX ** 2 / mean **2) #
    
    Ntot = len(quantity_arr)
    print("N = {}".format(Ntot))

    #quantity_arr = np.concatenate((quantity_arr,np.zeros(len(quantity_arr)-1)))
    #corrarr = corr(quantity_arr, quantity_arr)
    #corrarr = abs(np.fft.fftshift(corrarr))
    corrarr = np.correlate(quantity_arr-x_mean,quantity_arr-x_mean,'same')
    n = np.linspace(0,len(corrarr)-1,len(corrarr))
    n -= n[len(n)//2]
    i_search = abs(n)<4
    n_search = n[i_search]
    corr_search = corrarr[i_search]
    n_corr = abs(n_search[np.argmin(np.abs(corr_search - np.max(corr_search)/2.))]) + 1
    #print("N_eff = {}".format(int(Ntot /n_corr)))
    print(n_corr)
    baseline = (np.append(corrarr[:len(n)//4],corrarr[-len(n)//4:]))
    #corrarr -= baseline
    corrarr /= np.std(corrarr)
    ax.plot(n, corrarr, color=color)
    ax.axvline(n_corr-1, color=color)
    xrng = 20 #Ntot/2. # min(100, len(corrarr)//2)
    ax.set_xlim([-xrng,xrng])
        
        
    ax.set_xlabel(r"$\Delta n_{Vcycle}$", labelpad=10)
    
    if ax_passed is None:
        fig.tight_layout()
        output = "../plots/test_" + quantity + "_correlation.png"
        plt.savefig(output, bbox_inches="tight")
        print("saved to " + output)
        plt.close(fig)
    else:
        return ax

def testCorrelationAll(quantity="eta"):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    fig, ax = plt.subplots(1, 5, figsize=(8*5, 6)) #, sharey=True)
    fig.subplots_adjust(hspace=0.)
    #colors = list(plt.cm.coolwarm(np.linspace(0., 1., 5)))
    colors = list(plt.cm.gnuplot(np.linspace(0.9, 0., 5))) + ['gray']
    for i,dirtag in enumerate(dirtags):
        pkl = "../data_products/" + dirtag + "_profiles_all.pkl"
        with open(pkl, "rb") as openFile:
            D = pickle.load(openFile)
        a = D["dump"]["a"]
        nzeff = D["dump"]["Params"]["Multizone/nzones_eff"]
        iax = nzeff-4
        icolor = np.where(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.97])==a)[0][0]
        testCorrelation(pkl, ax_passed=ax[iax], quantity=quantity, color=colors[icolor])
    
    output = "../plots/test_" + quantity + "_correlation_all.png"
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
    dirtag="041825_a0.9_rB2e5_jks2_smth2"
    dirtag="092525_a0.9_rB2e3_mom_cons"
    dirtag="092525_a0.9_rB2e5_mom_cons_test" #_all" #
    #dirtag="092525_a0.9_rB2e6_momcons"
    dirtag="102125_a0.9_rB2e5_mom_cons_96"
    #dirtag="102925_a0.97_rB2e3"
    #dirtag="111025_a0.9_rB2e3_Btor"
    #dirtag="111725_a0.9_rB2e3_Btor_T+"
    #dirtag="111725_a0.9_rB2e5_mom_cons_rdepgmax"
    #dirtag="120325_a0.9_rB2e5_mom_cons_rdepgmax_uconst"
    #dirtag="110725_a0.7_rB2e6_momcons"
    #dirtag="011226_a0.9_rB2e3_mom_cons_beta100"
    #dirtag="012226_a0.9_rB2e5_rot-"
    dirtag="012726_a0.9_rB2e5_beta100"
    pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"

    average_factor = 2. #1.5  # 1.25 #
    quantities = ["Mdot", 'eta', 'Omega10', 'phib'] #["norm_Ldot", "Omega10"] #["Mdot", "eta", "phib"]  # 
    plotEvolutionMultipanel(pkl_name, quantities=quantities, average_factor=average_factor, xaxis_t=True) #False)  # 
    #for q2 in ['Mdot']: #'inv_abs_u^th', 'Omega2', 'Omega5', 'Omega10', 'Omega50', 'phib']:
        #plotCorrelation(pkl_name, q1='eta', q2=q2) #, last_factor=1.2)
    #plotOmegaEvolution(pkl_name)
    #plotOmegaFieldEvolution(dirtag)
    #plotHistogram(pkl_name, quantity="eta", tmax=800,average_factor=2)
    for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
        #compareHistogram(a, "Mdot")
        #compareHistogram(a, "s")
        #compareHistogram(a, "eta")
        compareHistogram(a, "phib")
    #compareHistogramAll()
    #compareAllStats("Mdot")#, radius=5) #, rescaleMdot=False)
    #plotEvolutionSpin(pkl_name)
    #testCorrelationAll(quantity="etaB")
