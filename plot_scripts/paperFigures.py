from astropy import units as u
from astropy import constants as const
from plot_utils import *
from plotEvolution import plotEvolution, plotHistogram
from plotCompare import compareRuns
from plotProfiles import setTimeBins, get_mask, calcFinalTimeAvg, plotProfileQuantity

def powerlaw_func(x, a, b):
    return b * np.power(x, a)


k=const.k_B
c=const.c
m_p=const.m_p
mu = 0.62*const.m_p # ????
def rB2T(rB):
    gam = 5./3.
    return ((1. / (gam * np.array(rB))) * mu * c**2 / k).to('K').value

def T2rB(T):
    gam = 5./3.
    theta = (k * np.array(T) * u.K / (mu * c**2)).to('')
    return 1. / (gam * theta)

def rB2M(rB):
    # using Wang & Abel 2008
    gam = 5./3.
    Tvir = ((1. / (gam * np.array(rB))) * mu * c**2 / k).to('K')
    return np.power(Tvir / (4.8e-3 * u.K), 3./2.).to('')

def M2rB(M):
    gam = 5./3.
    Tvir = 4.8e-3 * np.power(np.array(M), 2./3.) * u.K
    theta = (k * Tvir / (mu * c**2)).to('')
    return 1. / (gam * theta)

def compare_prescription_slice():
    from matplotlib import patches
    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    sz = 64
    kwargs = {'native': False, 'log_r':False, 'vmin': -1, 'vmax':1, 'window': (-sz, sz, -sz, sz), 'shading':'flat', 'label':''}
    
    fnums = [11, 8011]
    dirtags = ["043025_n4_a0.9_bondi_bflux0", "041625_n4_a0.9_toriilike_jks2_smth2_reconnect", ]
    labels = ["bflux0", "bflux-const"]
    for i, dirtag in enumerate(dirtags):
        fig, ax = plt.subplots(1, 2, figsize=(15, 7.5), sharey=True)
        plt.subplots_adjust(wspace=0.)
        for j, fnum in enumerate(fnums):
            fname = "../data/" + dirtag + "/highcadence/bondi.out0.{:05d}.phdf".format(fnum)
            dump = pyharm.load_dump(fname)
            if j == 1: kwargs["ylabel"] = False
            else: kwargs["ylabel"] = True
            pyharm.plots.plot_xz(ax[j], dump, 'symlog_FE_EM_A', cbar=False, **kwargs)
        
        fig.suptitle(labels[i])
        ax[0].set_title("end of zone-0", fontsize=27) #-60, 55, 
        ax[1].set_title("end of zone-1", fontsize=27)
        transFigure = fig.transFigure.inverted()
        coord1 = transFigure.transform(ax[0].transData.transform([55,0]))
        coord2 = transFigure.transform(ax[1].transData.transform([-55,0]))
        arrow = patches.FancyArrowPatch(
                    coord1, coord2,
                    shrinkA=0,
                    shrinkB=0,
                    arrowstyle="-|>",
                    transform=fig.transFigure,
                    mutation_scale=20,
                    color="black")
        fig.patches.append(arrow)
        fig.text(0.5, 0.52, r'$8000~\Delta t$', ha='center')
        output = "../plots/compare_prescription_slice_" + labels[i] + ".png"
        plt.savefig(output, bbox_inches="tight")
        print("saved to " + output)


def compare_evolution_rB(a=0.9,show_tl=False):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    quantities = ["Mdot", "eta", "phib"] #, "Omega"]
    figsize = (24, 5 * len(quantities))
    fig, ax = plt.subplots(len(quantities), 1, figsize=figsize, sharex=True)
    if len(quantities) == 1: ax = [ax]
    plt.subplots_adjust(wspace=0., hspace=0)
    avf = 1.25
    pzaf = 0.5
    
    if a == 0.1:
        dirtags = [
            "072125_a0.1_rB2e4",
            "072125_a0.1_rB2e5",
            ]
        tmaxs = [700, 700] 
        labels = ["2e4", "2e5"]
    elif a == 0.3:
        dirtags = [
            "072125_a0.3_rB2e4",
            "072125_a0.3_rB2e5",
            ]
        tmaxs = [700, 700] 
        labels = ["2e4", "2e5"]
    elif a == 0.5:
        dirtags = [
            "062025_a0.5_rB2e3",
            "062025_a0.5_rB2e4",
            "062025_a0.5_rB2e5",
            ]
        tmaxs = [700, 700, 1000] 
        labels = ["2e3", "2e4", "2e5"]
    elif a == 0.7:
        dirtags = [
            "072125_a0.7_rB2e4",
            "072125_a0.7_rB2e5",
            ]
        tmaxs = [700, 700] 
        labels = ["2e4", "2e5"]
    elif a == 0.9:
        dirtags = [
            #"051225_n4_a0.9_bondi_nocap_newflr",
            "101225_n4_a0.9_bondi_nocap_momcons",
            #"043025_a0.9_rB2e3_bondi_eks",
            "092525_a0.9_rB2e3_mom_cons",
            #"delta/051325_a0.9_rB2e5_bondi_eks",
            "delta/092525_a0.9_rB2e4_mom_cons",
            #"052825_a0.9_rB2e5_bondi_eks_largerout",
            "092525_a0.9_rB2e5_mom_cons_test",
            #"080625_a0.9_rB2e5_96",
            ]
        tmaxs = [400, 700, 700, 700, 450,100] 
        avf = 2.
        pzaf = 0.05
        labels = ["4e2", "2e3", "2e4", "2e5", "2e5hi", "2e5T+", "2e5_a0"]


    if show_tl:
        colors = plt.cm.gnuplot(np.linspace(0.9, 0., len(dirtags)-1))
        colors = list(colors) + ['gray']
        alphas = [1, 1, 1, 1, 0.3]
    else:
        colors = plt.cm.gnuplot(np.linspace(0.9, 0., len(dirtags)))
        alphas = [1] * len(dirtags)
    for i, dirtag in enumerate(dirtags):
        pkl = "../data_products/" + dirtag + "_profiles_all.pkl"
        with open(pkl, "rb") as openFile:
            D = pickle.load(openFile)
        r_sonic = D["dump"]["rs"]
        mdot = D["dump"]["mdot"]
        gam = D["dump"]["gam"]
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
        for j, quantity in enumerate(quantities):
            #if quantity == "eta": radius = rB
            #else: radius = None
            plotEvolution(pkl, ax_passed=ax[j], quantity=quantity, average_factor=avf, xaxis_t=True, scale_tB=True, color=colors[i], alpha=alphas[i], rescaleMdot=True, tmax=tmaxs[i], show_avg=True, show_negative=False, label=labels[i], perzone_avg_frac=pzaf, only_selectively_show=True, take_mean=True, radius=None) #, use_Mdot_mean=True)

    if show_tl:
        dirtag = "041825_a0.9_rB2e5_jks2_smth2"
        pkl = "../data_products/" + dirtag + "_profiles_all.pkl"
        with open(pkl, "rb") as openFile:
            D = pickle.load(openFile)
        r_sonic = D["dump"]["rs"]
        mdot = D["dump"]["mdot"]
        gam = D["dump"]["gam"]
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
        for j, quantity in enumerate(['eta', 'phib']):
            plotEvolution(pkl, ax_passed=ax[j+1], quantity=quantity, average_factor=avf, xaxis_t=True, scale_tB=True, color='b', alpha=0.2, rescaleMdot=True, tmax=100, show_avg=True, show_negative=False, label='2e5T+', perzone_avg_frac=pzaf, only_selectively_show=True, take_mean=True, radius=None)

    #ax[0].set_yscale('linear')
    ax[1].legend(ncol=len(dirtags) + show_tl, bbox_to_anchor=(0.22, 0.2), numpoints=1)#, fontsize=20)
    ax[0].set_ylim([1e-4,1]) #([0, 0.15])
    ax[0].set_xlim([0, np.max(tmaxs)]) #([0, 0.15])
    ax[0].set_ylabel(r"$\dot{M}$ [$\dot{M}_B$]")
    ax[1].set_ylabel(r"$\eta$")
    ax[2].set_ylabel(r"$\phi_b$")
    ax[2].set_ylim([6,150])
    #ax[2].set_yscale('linear')
    xlabel = r"$t$ [$t_B$]"
    ax[-1].set_xlabel(xlabel)
    output = "../plots/compare_evolution_rB.png"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)

def compare_quantity_rB():
    from plotEvolution import calcStats

    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    quantities = ["Mdot", "etaMdot", "eta"] #, "phib"]
    figsize = (9, 4 * (len(quantities)))
    fig, ax = plt.subplots(len(quantities), 1, figsize=figsize, sharex=True) #, sharey=True)
    if len(quantities) == 1: ax = [ax]
    plt.subplots_adjust(wspace=0., hspace=0)

    cho2025fig=False
    if cho2025fig:
        dirtags = [
            "060525_n4_a0_bondi_nocap_newflr",
            "051225_n4_a0.9_bondi_nocap_newflr",
            "043025_a0.9_rB2e3_bondi_eks",
            "delta/051325_a0.9_rB2e5_bondi_eks",
            #"043025_a0.9_rB2e5_bondi_eks",
            "052825_a0.9_rB2e5_bondi_eks_largerout",
            "052925_a0.0_rB2e5_eks_largerout",
            #"051325_a0.0_rB2e5_eks",
            ]
        tmaxs = [400, 400, 700, 700, 700, 700] #400]
        colors = list(plt.cm.gnuplot(np.linspace(0.9, 0., len(dirtags)-2))) # + ['k']
        colors = [colors[0]] + colors
        colors = colors + [colors[-1]]
        time_bin_factor = 1.25
        perzone_avg_frac = 0.5
    else:
        dirtags = gdirtags
        #ncorr_Mdot = np.array([3,2,2,3,3,2,2,2,2,2,2,1,2,2,2,2,3,3,4,4,3,4,4,4,4,4]) # TODO recieve this value
        #ncorr_eta = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,2,4,4,3,3])
        #ncorr_etaB = np.array([3,2,2,2,2,3,2,2,2,2,2,2,2,2,2,2,2,4,4,3,3,4,4,4,4,4])
        tmaxs = [700] * (len(dirtags))
        tmaxs[4] = 500 # exception
        #colors = ['r'] + ['k'] * 4 + list(plt.cm.gnuplot(np.linspace(0.9, 0.2, len(dirtags)-4))) # + ['k']
        colors = list(plt.cm.gnuplot(np.linspace(0.9, 0., 5))) + ['gray', 'lightgreen']
        time_bin_factor = 2.
        perzone_avg_frac = 0.05
        labels = [r"$0.1$", r"$0.3$", r"$0.5$", r"$0.7$", r"$0.9$", r"$0.97$", r"$0$"]
    
    mdot_save = []
    eta_save = []
    etaB_save = []
    rB_save = []
    a_save = []
    for i, dirtag in enumerate(dirtags):
        pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"
        print(pkl_name)
        with open(pkl_name, "rb") as openFile:
            D = pickle.load(openFile)
        tDivList, binNumList = setTimeBins(D, 1, time_bin_factor=time_bin_factor, tmax=tmaxs[i])
        mask_list = get_mask(D, prioritize_inner=False) #True)
        r_sonic = D["dump"]["rs"]
        mdot = D["dump"]["mdot"]
        gam = D["dump"]["gam"]
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
        rEH = D["dump"]["r_eh"]
        nzeff = D["dump"]["Params"]["Multizone/nzones_eff"]
        a = D["dump"]["a"]
        rB_save += [rB]#if nzeff > 5: 
        a_save += [a]  #if nzeff > 5: 
        for j, quantity in enumerate(quantities):
            if 1: #quantity != "etaMdot":
                radii, profiles = calcFinalTimeAvg(D, tDivList, binNumList, quantity, perzone_avg_frac=perzone_avg_frac, mask_list=mask_list, rescale=True, use_avged_Mdot=cho2025fig)
                if quantity == "Mdot":
                    r_read = 5 #rEH #
                elif quantity == "phib":
                    r_read = rEH
                elif quantity == "eta":
                    r_read = rB / 3.
                elif quantity == "etaMdot":
                    r_read = rB / 3.
                else:
                    print("WARNING not supported")
                i_r = np.argmin(abs(radii[0] - r_read))
                q_out = profiles[0][i_r]
                print("at r={:.5g}, {}={:.5g}".format(radii[0][i_r], quantity, profiles[0][i_r]))
                if cho2025fig:
                    color=colors[i]
                else:
                    a_ind = np.where(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0])==a)[0][0]
                    color = colors[a_ind]
                mfc = color
                if cho2025fig and (i == 0 or i == len(dirtags) - 1): mfc = 'none'
                if quantity == "Mdot": 
                    #if nzeff > 5: 
                    mdot_save += [q_out]
                    #ncorr_c = ncorr_Mdot[i]
                elif quantity == "eta": 
                    #if nzeff > 5: 
                    eta_save += [q_out]
                    #ncorr_c = ncorr_eta[i]
                elif quantity == "etaMdot": 
                    #if nzeff > 5: 
                    etaB_save += [q_out]
                    #ncorr_c = ncorr_etaB[i]
                if cho2025fig:
                    ax[j].plot(rB, q_out, color=color, marker='.', ms=20, markerfacecolor=mfc)
                else:
                    if quantity == "etaMdot": quantity="etaB"
                    x_mean, x_std, _, _, quantity_arr = calcStats(pkl_name, quantity, average_factor=time_bin_factor, use_Mdot_mean=cho2025fig, rescaleMdot=True, tmax=tmaxs[i], perzone_avg_frac=perzone_avg_frac, radius=r_read, verbose=False)
                    mean_est = np.exp(x_mean + x_std ** 2 / 2.)
                    ncorr = get_lcorr(quantity_arr)
                    #if ncorr != ncorr_c:
                    #    print("WARNING! DIFFERENT NCORR {} {}".format(ncorr,ncorr_c))
                    #if quantity == "etaB": 
                    #    ncorr = ncorr_etaB[i]
                    #    etaB_save += [mean_est]
                    #else:
                    if 1:
                        if abs((np.mean((quantity_arr)) - q_out) / q_out) > 1e-1:
                            pdb.set_trace()
                        label = labels[a_ind]
                        labels[a_ind] = '__nolegend__'
                        ax[j].plot(rB, q_out, color=color, marker='.', ms=10, markerfacecolor=mfc, label=label, ls='')
                    #print("err {:.5g}, sigma {:.5g}, N {}, ncorr {}".format(x_std / np.sqrt(len(quantity_arr)/ncorr), x_std, len(quantity_arr), ncorr))
                    #ax[j].errorbar(rB, np.log10(q_out), yerr=x_std / np.sqrt(len(quantity_arr)/ncorr), color=color, marker='.', ms=10, markerfacecolor=mfc, capsize=6)
                    Neff = len(quantity_arr) / ncorr
                    cox_err = np.sqrt(x_std ** 2 / Neff + x_std ** 4 / (2. * (Neff - 1.))) # 1.96 for 95%, 1 for 68%
                    print("mean_est {:.5g} x_std {:.5g} Neff {} err {:.5g}".format(x_mean + x_std**2/2., x_std, Neff, cox_err))
                    low = mean_est - np.exp(x_mean + x_std ** 2 / 2. - cox_err)
                    high = np.exp(x_mean + x_std ** 2 / 2. + cox_err) - mean_est
                    ax[j].errorbar(rB, mean_est, yerr=[[low], [high]], color=color, capsize=6)
                    if quantity == "Mdot" or quantity == "etaMdot":
                        if quantity == "Mdot": q_est = (-3*a+4.7) * np.power(rB,-0.5)
                        elif quantity == "etaMdot": q_est = (0.98*a+0.13) * np.power(rB,-0.6)
                        if (q_out / q_est) > 2:
                            print("READ!!! a={} R_B={:.2g}: diff is {:.5g}".format(a, rB, q_out/q_est))
                        elif (q_out / q_est) < 1/2:
                            print("READ!!! a={} R_B={:.2g}: diff is {:.5g}".format(a, rB, q_est/q_out))

    if cho2025fig:
        # Cho+24 scaling
        rBlist = np.logspace(2, 6, 10)
        ax[0].plot(rBlist, np.power(rBlist/6, -0.5), 'b')
    else:
        # fit for each BH spin
        rBlist = np.logspace(2, 7, 10)
        rB_save = np.array(rB_save)
        a_save = np.array(a_save)
        mdot_save = np.array(mdot_save)
        eta_save = np.array(eta_save)
        etaB_save = np.array(etaB_save)
        for a in [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97]:
            ifit = (a_save == a)
            color = colors[np.where(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.97,0])==a)[0][0]]
            if a < 0.95:
                popt, pcov = curve_fit(lin_func, np.log10(rB_save[ifit]), np.log10(mdot_save[ifit]))
                print("Mdot/MdotB slope = {:.3g} +- {:.3g} norm r={:.3g}".format(popt[0], np.sqrt(np.diag(pcov)[0]),np.power(1./np.power(10.,popt[1]), 1/popt[0])))
            #popt, pcov = curve_fit(lambda x, b: lin_func(x, -0.5, b), np.log10(rB_save[ifit]), np.log10(mdot_save[ifit]))
            #print("Mdot/MdotB slope = {:.3g} +- {:.3g} norm {:.3g}".format(popt[0], np.sqrt(np.diag(pcov)[0]),np.power(2e5,popt[0])*np.power(10.,popt[1])))
            #print("Mdot/MdotB slope = {:.3g} +- {:.3g} norm {:.3g}".format(popt[0], np.sqrt(np.diag(pcov)[0]),np.power(10.,popt[1])))
            #print("Mdot/MdotB norm {:.3g}+={:.3g}".format(popt[0], np.sqrt(np.diag(pcov)[0])))
            #if not cho2025fig: ax[0].semilogx(rBlist, lin_func(np.log10(rBlist), *popt), color=color, alpha=0.5)
            #if not cho2025fig: ax[0].loglog(rBlist, np.power(10,lin_func(np.log10(rBlist), *popt)), color=color, alpha=0.5)
            if not cho2025fig: ax[0].loglog(rBlist, np.power(10,lin_func(np.log10(rBlist), a=-0.5, b=np.log10(-3*a+4.7))), color=color, alpha=0.5)
            #ifit2 = ifit & (rB_save > 1e3)
            #ax[2].loglog(rB_save[ifit], etaB_save[ifit]/mdot_save[ifit], color=color, marker='.', ms=10, ls='none')
            if a< 0.95:
                popt, pcov = curve_fit(lin_func, np.log10(rB_save[ifit]), np.log10(eta_save[ifit]))
                print("eta slope = {:.3g} +- {:.3g} norm {:.3g}".format(popt[0], np.sqrt(np.diag(pcov)[0]),np.power(2e5,popt[0])*np.power(10.,popt[1])))
            #popt, pcov = curve_fit(lin_func, np.log10(rB_save[ifit]), np.log10((etaB_save/mdot_save)[ifit]))
            #print("eta slope = {:.3g} +- {:.3g} norm r={:.3g}".format(popt[0], np.sqrt(np.diag(pcov)[0]),np.power(1./np.power(10.,popt[1]), 1/popt[0])))
            #if not cho2025fig: ax[1].semilogx(rBlist, lin_func(np.log10(rBlist), *popt), color=color, alpha=0.5)
            #if not cho2025fig: ax[2].loglog(rBlist, np.power(10,lin_func(np.log10(rBlist), *popt)), color=color, alpha=0.3)
            #if not cho2025fig: ax[1].loglog(rBlist, np.power(10,lin_func(np.log10(rBlist), a=[-0.1,0][a>0.5], b=np.log10(np.power(2e5,[0.1,0][a>0.5])*(0.5*a**2-0.16*a+0.03)))), color=color, alpha=0.5)
            if 1:
                Efb = etaB_save[ifit] #mdot_save[ifit] * eta_save[ifit] #
                ax[1].loglog(rB_save[ifit], Efb, color=color, marker='.', ms=10, ls='none')
                #popt, pcov = curve_fit(lin_func, np.log10(rB_save[ifit]), np.log10(Efb))
                if a < 0.95:
                    popt, pcov = curve_fit(lambda x, b: lin_func(x, -0.6, b), np.log10(rB_save[ifit]), np.log10(Efb))
                    print("Efb norm {:.3g}+-{:.3g}".format(popt[0], np.sqrt(np.diag(pcov)[0])))
                #print("Efb slope = {:.3g} +- {:.3g} norm {:.3g}".format(popt[0], np.sqrt(np.diag(pcov)[0]),np.power(10.,popt[1])))
                #ax[1].loglog(rBlist, np.power(10,lin_func(np.log10(rBlist), *popt)), color=color, alpha=0.5)
                #ax[1].loglog(rBlist, np.power(10,lin_func(np.log10(rBlist), a=-0.7, b=popt[0])), color=color, alpha=0.5)
                #ax[1].loglog(rBlist, np.power(10,lin_func(np.log10(rBlist), a=-0.7, b=np.log10(3.3*a+0.6))), color=color, alpha=0.5)
                #ax[2].loglog(rBlist, np.power(10,lin_func(np.log10(rBlist), a=-0.55, b=np.log10(0.6*a**2+0.5*a+0.06))), color=color, alpha=0.5)
                ax[1].loglog(rBlist, np.power(10,lin_func(np.log10(rBlist), a=-0.6, b=np.log10(0.98*a+0.13))), color=color, alpha=0.5)
    
    # Cho+24 a=0 Mdot
    if 0:
        rB24 = [178, 455, 1.6e3, 1.14e4, 1.78e5, 1e6, 1.02e7]
        mdot24 = [0.175, 0.106, 0.0775, 0.0308, 0.00756, 0.00267, 0.000596]
        ax[0].plot(rB24, mdot24, color='g', marker='.', ms=20, markerfacecolor='none')

    # plot settings
    ax[0].set_xscale('log')
    if cho2025fig:
        ax[0].set_ylim([1e-3, 0.4])
        ax[1].text(70, 0.3, r'$a_*=0.9$', color='grey', fontsize=15)
        ax[1].text(70, 0.03, r'$a_*=0.0$', color='grey', fontsize=15)
        ax[1].set_ylim([1e-2, 4])
        ax[1].axhspan(0.1, 1., color='g', alpha=0.1)
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
    else:
        legend = ax[0].legend(fontsize=15,ncol=2, title=r'$a_*$')
        plt.setp(legend.get_title(),fontsize=20)
        ax[0].set_xlim([2e2, 4e6])
        ax[0].set_ylim([5e-4, 0.4])
        #ax[0].set_ylim([-3.5, -0.5])
        ax[2].set_ylim([5e-3, 1])
        #ax[1].set_ylim([-2.5, 0])
        ax[1].set_ylim([1.5e-5, 1e-1])
        ax[2].set_yscale('log')
    for ax_temp in ax:
        ax_temp.set_xlabel(r'$R_B$ [$r_g$]')
    ax[0].axhline(1, color='k', ls=":")
    #cmap_nums = [0.4] * 5  + [0.5] * 10 + [0.4]
    #ax[1].imshow([[z] * len(cmap_nums) for z in cmap_nums],  cmap = 'Greens', extent=[60, 2e6, 0.14, 1.2], interpolation = 'bicubic', aspect='auto', alpha=0.4)
    #ax[2].set_ylim([10, 100])
    if cho2025fig:
        ax[0].set_ylabel(r"$\overline{\dot{M}}(r_H)$ [$\dot{M}_B$]")
        ax[1].set_ylabel(r'$\overline{\eta}(R_B/3)$')
    else:
        ax[0].set_ylabel(r"${\overline{\dot{M}}}$ [$\dot{M}_B$]")
        ax[2].set_ylabel(r'${\overline{\eta}}$')
        ax[1].set_ylabel(r'$\overline{\dot{E}_{\rm fb}}$ [$\dot{M}_B c^2$]')
    # 2nd axis
    if 1:
        secax = ax[0].secondary_xaxis('top', functions=(rB2T, T2rB))
        secax.set_xlabel(r'$T_{\infty}$ [K]') #,fontsize=fontsize, labelpad=7)
    else:
        # rB to T to halo mass
        secax = ax[0].secondary_xaxis('top', functions=(rB2M, M2rB))
        secax.set_xlabel(r'$M_{\rm halo}$ [$M_\odot$]')

    # save
    output = "../plots/compare_quantity_rB.pdf"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)

def show_snapshot(fnum):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from pyharm.plots.overlays import overlay_field
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    
    #fig = plt.figure(figsize=(25,13))
    fig = plt.figure(figsize=(36,12))
    spec_snp = GridSpec(1, 1)[0]
    plt.subplots_adjust(hspace=0.01)
    
    #gs = GridSpecFromSubplotSpec(2, 4, subplot_spec = spec_snp, hspace=0.02, wspace=0.02)
    n_zones = 6 #dump["Params"]["Multizone/nzones_eff"]
    gs = GridSpecFromSubplotSpec(2, n_zones, subplot_spec = spec_snp, hspace=0.02, wspace=0.02)
    ax = []
    for cell in gs: ax += [plt.subplot(cell)]
    ax = np.array(ax).reshape(2,-1)

    plotrc={}
    plotrc.update({'xlabel': False, 'ylabel': False,'xticks': [], 'yticks': [],'cbar': False, 'frame': False, 'no_title': True, 'shading': 'flat'})
    inwards = -1 # 1

    # read file
    dirtag = "052825_a0.9_rB2e5_bondi_eks_largerout" #"043025_a0.9_rB2e5_bondi_eks"
    fn = glob.glob("../data/" + dirtag + "/*{:05d}*.phdf".format(fnum))[0]
    dump = pyharm.load_dump(fn,ghost_zones=False)
    print(dump["n_step"])
    r_sonic = dump["rs"]
    mdot = dump["mdot"]
    gam = dump["gam"]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
    iwv = dump["Params"]["Multizone/i_within_vcycle"]
    nzeff = dump["Params"]["Multizone/nzones_eff"]
    i_zone = abs(iwv - (nzeff - 1))
    print("fnum {} t={:.5g} tg / {:.5g} tB izone{}".format(fnum, dump["t"], dump["t"] / np.power(rB, 3./2), i_zone))
    patch_sz = np.zeros(n_zones)

    axes = ax #np.concatenate((ax[0],ax[1,::-1]))
    for i in range(np.shape(axes)[1]): #, ax1d in enumerate(axes):
        if i < n_zones: 
            sz = 8**(i+1.8)
            window = (-sz, sz, -sz, sz)
            #im1 = pyharm.plots.plot_xz(ax1d, dump, "log_beta", window=window, cmap='plasma', vmin=1e-1, vmax=1e3, **plotrc)
            #im1 = pyharm.plots.plot_xz(ax1d, dump, "log_K", window=window, cmap='jet', vmin=1e-1, vmax=1e6, **plotrc)
            #im1 = pyharm.plots.plot_xz(axes[0,i], dump, "symlog_FE_norho_A", window=window, vmin=-1e-3, vmax=1e-3, **plotrc)
            #im1 = pyharm.plots.plot_xz(axes[0,i], dump, "log_Gamma", window=window, vmin=1, vmax=5, cmap='gist_heat', **plotrc)
            im1 = pyharm.plots.plot_xz(axes[0,i], dump, "log_Theta", window=window, vmin=8e-6, vmax=1, cmap='gist_heat', **plotrc)
            im2 = pyharm.plots.plot_xz(axes[1,i], dump, dump["rho"], window=window, vmin=5e-12, vmax=1e-4, log=True, cmap='turbo', **plotrc)# turbo nipy_spectral half_cut=True, 
            scale = np.power(10,np.floor(np.log10(sz)))
            c= 'white'
            scalebar = AnchoredSizeBar(axes[0,i].transData, scale, r'$10^{:d}\, r_g$'.format(int(np.log10(scale))), 'lower left', pad=0.5, color=c, frameon=False, size_vertical=sz/8**2)
            axes[0,i].add_artist(scalebar)
            axes[0,i].title.set_visible(False)
            
            if i==0:
                axes[0,i].text(-sz*0.9, sz*0.7, r'$T$', color='w', fontsize=40)
                axes[1,i].text(-sz*0.9, sz*0.7, r'$\rho$', color='w', fontsize=40)

                # colorbar for each rows
                for j, im in enumerate([im1, im2]):
                    cb=fig.colorbar(im, cax=axes[j,i].inset_axes((-0.1, 0.15, 0.05, 0.7)))
                    cb.ax.tick_params(labelleft=True, labelright=False)
            
            # show B fields
            at_i = np.argmin(abs(dump["r1d"] - sz * 1.5))
            #overlay_field(axes[1,i], dump, half_cut=True,nlines=15, reverse=True, i_slice=slice(0, at_i), color='k')#, sum=False)

            # show RB
            for j in range(2): 
                if i >= 4:
                    circle1 = plt.Circle((0, 0), rB, ec='grey', fill=False, ls="--", lw=3)
                    axes[j,i].add_artist(circle1)
                    if i==4: axes[j,i].text(-sz*0.9, sz*0.3, r'$R_B$', color='grey', fontsize=30)
                if i == 5:
                    circle1 = plt.Circle((0, 0), 5*rB, ec='grey', fill=False, ls=":", lw=3)
                    axes[j,i].add_artist(circle1)
                    axes[j,i].text(-sz*0.95, sz*0.3, r'$5\,R_B$', color='grey', fontsize=30)
        
            patch_sz[i] = sz
    
    for i in range(np.shape(axes)[1]): #, ax1d in enumerate(axes):
        for j in range(2):
            if i+(inwards>0) < n_zones and i+inwards >= 0: # and i < n_zones:
                rect = patches.Rectangle((-patch_sz[i+inwards],-patch_sz[i+inwards]), 2*patch_sz[i+inwards], 2*patch_sz[i+inwards], linewidth=3, edgecolor='w', facecolor='none')
                axes[j, i].add_patch(rect)
                print(patch_sz[i+inwards])
                con1 = patches.ConnectionPatch(xyA=(inwards*patch_sz[i+inwards], -patch_sz[i+inwards]), xyB=(-inwards*patch_sz[i+inwards],-patch_sz[i+inwards]), coordsA="data", coordsB="data", axesA=axes[j,i], axesB=axes[j,i+inwards], color='w',ls=':', lw=2)
                con2 = patches.ConnectionPatch(xyA=(inwards*patch_sz[i+inwards], patch_sz[i+inwards]), xyB=(-inwards*patch_sz[i+inwards],patch_sz[i+inwards]), coordsA="data", coordsB="data", axesA=axes[j,i], axesB=axes[j,i+inwards], color='w',ls=':', lw=2)

                fig.add_artist(con1)
                fig.add_artist(con2)
            
        # panel numbers
        axes[0,i].text(0.70, 0.93, 'zone-'+str([0,7][inwards>0]-i*(inwards)),transform=axes[0,i].transAxes, fontsize=25, color='w')#, bbox=dict(facecolor='w', edgecolor='k', pad=5.0))
    
    output = "../plots/snapshot.png"
    plt.savefig(output,bbox_inches='tight')
    print("saved to " + output)
    plt.close()

def show_tavged(also_show_rprofile=False):
    from scipy.optimize import curve_fit
    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    
    dirtag = "043025_a0.9_rB2e5_bondi_eks" #"043025_a0.9_rB2e3_bondi_eks" #"delta/051325_a0.9_rB2e5_bondi_eks" #
    tmax = 600 #700 #
    average_factor = 1.25
    perzone_avg_frac = 0.5 # TODO
    quantity = "Theta" #"rho" #"u^r" #"K" #"sigma" #"FE_EM" #_norho" #"T^1_0"

    files = sorted(glob.glob("../data/" + dirtag + "/*out*.phdf"))[::-1]

    # basic info
    dump0 = pyharm.load_dump(files[0], ghost_zones=False)
    r_sonic = dump0["rs"]
    mdot = dump0["mdot"]
    gam = dump0["gam"]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
    tB = np.power(rB, 3./2)
    n_zones = dump0["Params"]["Multizone/nzones_eff"]
    switch_on_ncycle = dump0["Params"]["Multizone/ncycle_per_zone"] > 0
    if switch_on_ncycle:
        pkl = "../data_products/" + dirtag + "_profiles_all.pkl"
        with open(pkl, "rb") as openFile:
            D = pickle.load(openFile)
        switch_pt = set(D["n0_zone"])
        switch_pt = np.array(sorted(switch_pt))
    
    fig, ax = plt.subplots(1 + also_show_rprofile, n_zones, figsize=(8 * n_zones, 8 * (1 + also_show_rprofile))) #, sharey=True)
    if not also_show_rprofile:
        ax = [ax]
    plt.subplots_adjust(wspace=0.02)
    
    # list initialization
    quantity_arr = [0] * n_zones
    num_sum = [0] * n_zones
    rho_arr = [0] * n_zones
    first_time = np.inf
    last_time = np.inf
    if quantity == "rho" or "FE" in quantity: density_weight = False
    else: density_weight = True

    start_fnum=380 #450 #180 #
    for f in files[start_fnum:start_fnum+100]:
        dump = pyharm.load_dump(f, ghost_zones=False)
        iwv = dump["Params"]["Multizone/i_within_vcycle"]
        zone = abs(iwv - (n_zones - 1))
        if dump["t"] > tmax * tB:
            if dump["t"] < last_time: last_time = dump["t"]
            continue
        elif dump["t"] < tmax * tB / average_factor:
            break
        else:
            switch_num = np.argwhere(switch_pt - dump["n_step"] < 0)[-1,0]
            if (switch_pt[switch_num + 1] - dump["n_step"] <= (switch_pt[switch_num + 1] - switch_pt[switch_num]) * perzone_avg_frac):
                if dump["t"] < first_time: first_time = dump["t"]
                if density_weight:
                    quantity_arr[zone] += dump[quantity] * dump["rho"]
                    rho_arr[zone] += dump["rho"]
                else:
                    quantity_arr[zone] += dump[quantity]
                num_sum[zone] += 1
    
    print(num_sum, last_time / tB, first_time / tB)
    for zone in range(n_zones): 
        if num_sum[zone] > 0:
            if density_weight: quantity_arr[zone] /= rho_arr[zone]
            else: quantity_arr[zone] /= num_sum[zone]
            sz = 8**(zone+1.8)
            #vmax = 1e-3; vmin = -vmax
            if quantity == "rho": 
                vmax = 1e-3; vmin=1e-10
            elif quantity == "Theta":
                vmax = 1e0; vmin=1e-6
            #vmax = 1e4; vmin=1e-1
            #vmax = 1e1; vmin=1e-6
            #vmax = 1e-1; vmin = -vmax
            window = (-sz, sz, -sz, sz)
            #pyharm.plots.plot_xz(ax[zone], dump, quantity_arr[zone] * dump["gdet"], native=False, log_r=False, symlog=True, vmin=vmin, vmax=vmax, cbar=0, shading="flat", window=window, average=1) #, xlabel=native, ylabel=native)
            pyharm.plots.plot_xz(ax[0,zone], dump, quantity_arr[zone], native=False, log_r=False, log=True, vmin=vmin, vmax=vmax, cbar=0, shading="flat", window=window, average=1, ylabel=False, yticks=[])

            if rB < sz * 2 and rB > sz / 50:
                circle1 = plt.Circle((0, 0), rB, ec='w', fill=False, ls="--", lw=5)
                ax[0, zone].add_artist(circle1)
            
            if also_show_rprofile:
                irB = np.argmin(abs(dump["r1d"] - rB))
                mean = quantity_arr[zone].mean(axis=-1)
                #mean = (mean + np.flip(mean, axis=1)) / 2.
                thnorth = np.argwhere(dump["th1d"] < np.pi / 6)[:,0]
                thsouth = np.argwhere(dump["th1d"] > np.pi * 5 / 6)[:,0]
                thdisk = np.argwhere((dump["th1d"] < np.pi * 7 / 12) & (dump["th1d"] > np.pi * 5 / 12))[:,0]
                #colors = plt.cm.gnuplot(np.linspace(0.9, 0., (np.shape(mean)[1])//2))
                #for ith in range(len(colors)):
                #    ax[1, zone].loglog(dump["r1d"], mean[:,ith], color=colors[ith])
                #ax[1, zone].loglog(dump["r1d"], mean[:,thnorth].mean(axis=-1), color='r')
                #ax[1, zone].loglog(dump["r1d"], mean[:,thsouth].mean(axis=-1), color='b')
                #ax[1, zone].loglog(dump["r1d"], mean[:,thdisk].mean(axis=-1), color='g')

                # fit powerlaw
                inertial_range = np.argwhere((dump["r1d"] > 10) & (dump["r1d"] < rB / 10))[:,0]
                colors = ['r', 'b', 'k']
                regions = ['N', 'S', 'M']
                for ithselect, thselect in enumerate([thnorth, thsouth, thdisk]):
                    selectmean = mean[:,thselect].mean(axis=-1)
                    ax[1, zone].loglog(dump["r1d"], selectmean, color=colors[ithselect])
                    popt, pcov = curve_fit(lin_func, np.log10(dump["r1d"][inertial_range]), np.log10((mean[:, thselect].mean(axis=-1))[inertial_range]))
                    print("{}: slope = {:.3g} +- {:.3g}".format(regions[ithselect], popt[0], np.sqrt(np.diag(pcov)[0])))
                    ax[1, zone].loglog(dump["r1d"][inertial_range], np.power(10, lin_func(np.log10(dump["r1d"][inertial_range]), *popt)), color=colors[ithselect], lw=10, alpha=0.2)
                #ax[1, zone].loglog(dump["r1d"][:irB], np.power(dump["r1d"][:irB], -1) / 1e2, color='k')
                #ax[1, zone].loglog(dump["r1d"][:irB], np.power(dump["r1d"][:irB], -1.3) / 1e4, color='m')

    output = "../plots/tavged_slice.png"
    plt.savefig(output,bbox_inches='tight')
    print("saved to " + output)
    plt.close()
    
def compare_jet_disk_profile():
    from scipy.optimize import curve_fit
    from plotProfiles import get_mask
    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    
    dirtag = "052825_a0.9_rB2e5_bondi_eks_largerout" #"043025_a0.9_rB2e5_bondi_eks" #"delta/051325_a0.9_rB2e5_bondi_eks" #"043025_a0.9_rB2e3_bondi_eks" #
    tmax = 700 #600 #
    average_factor = 1.25
    perzone_avg_frac = 0.5
    quantity = "rho" #"Theta" #

    files = sorted(glob.glob("../data/" + dirtag + "/*out*.phdf"))[::-1]

    # basic info
    dump0 = pyharm.load_dump(files[0], ghost_zones=False)
    r_sonic = dump0["rs"]
    mdot = dump0["mdot"]
    rEH = dump0["r_eh"]
    rout = dump0["r_out"]
    gam = dump0["gam"]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
    tB = np.power(rB, 3./2)
    n_zones = dump0["Params"]["Multizone/nzones_eff"]
    switch_on_ncycle = dump0["Params"]["Multizone/ncycle_per_zone"] > 0
    pkl = "../data_products/" + dirtag + "_profiles_all.pkl"
    with open(pkl, "rb") as openFile:
        D = pickle.load(openFile)
    if switch_on_ncycle:
        switch_pt = set(D["n0_zone"])
        switch_pt = np.array(sorted(switch_pt))
    else: print("NOT SUPPORTED")
    mask_list = get_mask(D, False)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # list initialization
    quantity_arr = [0] * n_zones
    num_sum = [0] * n_zones
    first_time = np.inf
    last_time = np.inf

    start_fnum=100 #380 #(rB2e5) #180 #450 (rB2e3) #
    for f in files[start_fnum:]: #start_fnum+100]:
        dump = pyharm.load_dump(f, ghost_zones=False)
        iwv = dump["Params"]["Multizone/i_within_vcycle"]
        zone = abs(iwv - (n_zones - 1))
        if dump["t"] > tmax * tB:
            if dump["t"] < last_time: last_time = dump["t"]
            continue
        elif dump["t"] < tmax * tB / average_factor:
            break
        else:
            switch_num = np.argwhere(switch_pt - dump["n_step"] < 0)[-1,0]
            if (switch_pt[switch_num + 1] - dump["n_step"] <= (switch_pt[switch_num + 1] - switch_pt[switch_num]) * perzone_avg_frac):
                if dump["t"] < first_time: first_time = dump["t"]
                quantity_arr[zone] += dump[quantity]
                num_sum[zone] += 1
    
    print(num_sum, last_time / tB, first_time / tB)
    
    # combine all zones data
    values_combined = np.zeros(np.shape(quantity_arr[0]))
    for zone in range(n_zones): 
        if num_sum[zone] > 0:
            quantity_arr[zone] /= num_sum[zone]
            mask = mask_list[zone]
            quantity_arr[zone][~mask] = 0.
            values_combined += quantity_arr[zone]
            
    irB = np.argmin(abs(dump["r1d"] - rB))
    mean = values_combined.mean(axis=-1)
    
    # th range
    thnorth = np.argwhere(dump["th1d"] < np.pi / 6)[:,0]
    thsouth = np.argwhere(dump["th1d"] > np.pi * 5 / 6)[:,0]
    thdisk = np.argwhere((dump["th1d"] < np.pi * 7 / 12) & (dump["th1d"] > np.pi * 5 / 12))[:,0]

    inertial_range = np.argwhere((dump["r1d"] > 10) & (dump["r1d"] < rB / 10))[:,0]
    colors = ['r', 'b', 'k']
    regions = ['N', 'S', 'M']
    for ithselect, thselect in enumerate([thnorth, thsouth, thdisk]):
        selectmean = mean[:,thselect].mean(axis=-1)
        ax.loglog(dump["r1d"], selectmean, color=colors[ithselect], label=regions[ithselect])
        
        # fit powerlaw
        popt, pcov = curve_fit(lin_func, np.log10(dump["r1d"][inertial_range]), np.log10((mean[:, thselect].mean(axis=-1))[inertial_range]))
        print("{}: slope = {:.3g} +- {:.3g}".format(regions[ithselect], popt[0], np.sqrt(np.diag(pcov)[0])))
        ax.loglog(dump["r1d"][inertial_range], np.power(10, lin_func(np.log10(dump["r1d"][inertial_range]), *popt)), color=colors[ithselect], lw=10, alpha=0.1)
        ax.text(1e2, 7e-6, r"$\propto r^{-1.1}$", color='k')
        ax.text(5e1, 4e-8, r"$\propto r^{-1.3}$", color='m')

    # show rB
    ax.axvline(rB, color='gray', lw=1, alpha=1, ls='--')

    # plot settings
    ax.set_xlabel(r"Radius [$r_g$]")
    ylabel = variableToLabel(quantity)
    ax.set_ylabel(ylabel)
    ax.set_xlim([rEH, rout])
    ax.legend()

    output = "../plots/compare_jet_disk_"+quantity+"_profile.png"
    plt.savefig(output,bbox_inches='tight')
    print("saved to " + output)
    plt.close()

def snapshot_Gamma(fnum):
    matplotlib_settings()
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    
    fig, ax = plt.subplots(1,2,figsize=(13,6))
    
    plotrc={}
    sz = 8**(5.8) #5e5
    window = (-sz, sz, -sz, sz)
    plotrc.update({'xlabel': False, 'ylabel': False,'xticks': [], 'yticks': [],'cbar': True, 'frame': False, 'no_title': True, 'shading': 'flat', 'window':window, 'log_r': False}) #, 'label':''})

    # read file
    dirtag = "052825_a0.9_rB2e5_bondi_eks_largerout" #"051225_oz_a0.9_toriilike_newflr" #"043025_a0.9_rB2e5_bondi_eks"
    fn = glob.glob("../data/" + dirtag + "/*{:05d}*.phdf".format(fnum))[0]
    dump = pyharm.load_dump(fn,ghost_zones=False)

    ax[1].loglog(dump["r1d"], np.mean(dump["Gamma"][:,-5:,:],axis=(1,2)))
    
    pyharm.plots.plot_xz(ax[0], dump, "log_Theta", vmin=8e-6, vmax=1, cmap='gist_heat', **plotrc)
    #plotrc['cbar_label']=r'$\gamma$'
    #pyharm.plots.plot_xz(ax[1], dump, "log_Gamma", vmin=1, vmax=4, cmap='Reds', **plotrc)

    # scale
    scale = np.power(10,np.floor(np.log10(sz)))
    scalebar = AnchoredSizeBar(ax[0].transData, scale, r'$10^{:d}\, r_g$'.format(int(np.log10(scale))), 'lower left', pad=0.5, color='w', frameon=False, size_vertical=sz/8**2)
    ax[0].add_artist(scalebar)
    
    # save
    output = "../plots/snapshot_Gamma.png"
    plt.savefig(output,bbox_inches='tight')
    print("saved to " + output)
    plt.close()

def compare_resolution():
    dirtagList = [
            "052825_a0.9_rB2e5_bondi_eks_largerout",
            "080625_a0.9_rB2e5_96"
    ]
    quantityList = ["Mdot", "rho", "beta", "eta"]  # , "eta_Fl", "eta_EM"]
    colorList = ["tab:red", "tab:blue", "tab:orange", "tab:green", "black"]  # colors for each runs
    labelList = ["fiducial", "high_res"]
    plot_dir = "../plots/resolution_comparison"
    
    a = 0.9
    if a == None:
        rEH = 2  # just use a=0 rEH
    else:
        rEH = calc_rEH(a)
    xlim = (rEH, 1e8)

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim, use_avged_Mdot=True, time_bin_factor=1.25, tmax=700)

def compare_feedback_a():
    from scipy.optimize import curve_fit

    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    quantities = ["phib", "eta"]
    figsize = (8 * len(quantities), 6)
    fig, ax = plt.subplots(1, len(quantities), figsize=figsize, sharex=True)
    if len(quantities) == 1: ax = [ax]

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
    tmaxs = [700] * (len(dirtags))
    colors = np.repeat(plt.cm.plasma(np.linspace(0., 1., 5)), np.array([5, 6, 5, 5, 5]), axis=0)
    time_bin_factor = 2.
    perzone_avg_frac = 0.05
    
    mdot_save = []
    rB_save = []
    for i, dirtag in enumerate(dirtags):
        pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"
        print(pkl_name)
        with open(pkl_name, "rb") as openFile:
            D = pickle.load(openFile)
        tDivList, binNumList = setTimeBins(D, 1, time_bin_factor=time_bin_factor, tmax=tmaxs[i])
        mask_list = get_mask(D, prioritize_inner=False) #True)
        r_sonic = D["dump"]["rs"]
        mdot = D["dump"]["mdot"]
        gam = D["dump"]["gam"]
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
        rEH = D["dump"]["r_eh"]
        if D["dump"]["a"] == 0.9: rB_save += [rB]
        for j, quantity in enumerate(quantities):
            radii, profiles = calcFinalTimeAvg(D, tDivList, binNumList, quantity, perzone_avg_frac=perzone_avg_frac, mask_list=mask_list, rescale=True, use_avged_Mdot=False)
            if quantity == "Mdot":
                r_read = rEH
            elif quantity == "phib":
                r_read = rEH
            elif quantity == "eta":
                r_read = rB / 3.
            else:
                print("WARNING not supported")
            i_r = np.argmin(abs(radii[0] - r_read))
            q_out = profiles[0][i_r]
            print("at r={:.5g}, {}={:.5g}".format(radii[0][i_r], quantity, profiles[0][i_r]))
            color = colors[i]
            mfc = color
            if D["dump"]["a"] == 0.9 and quantity == "Mdot": mdot_save += [q_out]
            ax[j].plot(D["dump"]["a"], q_out, color=color, marker='x', markerfacecolor=mfc, ms=10)
            #if j == 0: ax[1].plot(D["dump"]["a"], eta_BZ6(D["dump"]["a"], q_out, 0.09), color='b', marker='x')

    a_arr = np.linspace(0,1,30)
    rEH = calc_rEH(a_arr)
    Omega = a_arr / (2 * rEH)
    ax[1].plot(a_arr, 5*Omega**2, 'k:')
    ax[0].plot(a_arr, 5*rEH**2, 'k:')

    
    # plot settings
    #ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].set_ylim([5, 50])
    ax[1].set_ylim([1e-2, 1])
    ax[0].set_title(r'$\phi_b$ ($r_H$)')
    ax[1].set_title(r'$\eta$ ($R_B/3$)')
    for ax_temp in ax:
        ax_temp.set_xlabel(r'$a_*$')

    # save
    output = "../plots/compare_feedback_a.png"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)

def compare_profiles_all(time_bin_factor=2, tmax=700):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})

    fig, ax = plt.subplots(2, 3, figsize=(27, 2*5), sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0., hspace=0)
    dirtagList = [
        "012026_n4_a0_bondi_nocap_momcons",
        "012026_a0_rB2e3_momcons",
        "012026_a0_rB2e4_momcons",
        "012026_a0_rB2e5_mom_cons_test",
        "012026_a0_rB2e6_momcons",
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
    colors = list(plt.cm.coolwarm(np.linspace(0., 1., 5)))
    #linestyles = [':',(0, (1, 10)),'-.','--','-']
    ax1d = ax.reshape(-1)

    for i, dirtag in enumerate(dirtagList):
        pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"
        print(pkl_name)
        with open(pkl_name, "rb") as openFile:
            D = pickle.load(openFile)
        tDivList, binNumList = setTimeBins(D, 1, time_bin_factor=time_bin_factor, tmax=tmax)
        mask_list = get_mask(D, prioritize_inner=False) #True)
        dump = D["dump"]
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=dump["rs"], mdot=dump["mdot"], gam=dump["gam"])[0]
        nzeff = dump["Params"]["Multizone/nzones_eff"]
        iax = np.where(np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97])==dump["a"])[0][0]
        radii, profiles = calcFinalTimeAvg(D, tDivList, binNumList, "eta", perzone_avg_frac=0.05, mask_list=mask_list, rescale=True)
        color = colors[nzeff-4] #iax]
        if iax == 0: label = [r'$400$',r'$2\cdot 10^3$',r'$2\cdot 10^4$',r'$2\cdot 10^5$',r'$2\cdot 10^6$'][nzeff-4]
        else: label='__nolegend__'
        plotProfileQuantity(ax1d[iax], radii, profiles, tDivList, colors=[color], label=label, linestyle='-', legend=(iax==0), plot_rmax=400*rB)
        if nzeff == 8:
            #_, profiles = calcFinalTimeAvg(D, tDivList, binNumList, "eta_EM", perzone_avg_frac=0.05, mask_list=mask_list, rescale=True)
            #plotProfileQuantity(ax1d[iax], radii, profiles, tDivList, colors=[color], label='__nolegend__', linestyle='-.', legend=False, plot_rmax=1e3)
            try:
                pkl_name = "../data_products/" + dirtag + "_profiles_Eonly.pkl"
                with open(pkl_name, "rb") as openFile:
                    D2 = pickle.load(openFile)
                tDivList, binNumList = setTimeBins(D2, 1, time_bin_factor=time_bin_factor, tmax=tmax)
                mask_list = get_mask(D2, prioritize_inner=False) #True)
                _, profiles = calcFinalTimeAvg(D2, tDivList, binNumList, "eta_KE", perzone_avg_frac=0.05, mask_list=mask_list, rescale=True)
                plotProfileQuantity(ax1d[iax], radii, profiles, tDivList, colors=[color], label=['__nolegend__','kinetic'][iax==1], linestyle='--', legend=False, plot_rmax=400*rB, plot_rmin=1e3)
                _, profiles = calcFinalTimeAvg(D2, tDivList, binNumList, "eta_TE", perzone_avg_frac=0.05, mask_list=mask_list, rescale=True)
                plotProfileQuantity(ax1d[iax], radii, profiles, tDivList, colors=[color], label=['__nolegend__','thermal'][iax==1], linestyle=':', legend=False, plot_rmax=400*rB, plot_rmin=1e3)
                ax1d[iax].axvline(rB, color=color, alpha=0.5)
            except:
                print('error')
    
    # plot settings
    panel_label = [r'(a) $a_*=0$', r'(b) $a_*=0.1$',r'(c) $a_*=0.3$',r'(d) $a_*=0.5$',r'(e) $a_*=0.7$',r'(f) $a_*=0.9$']
    for i in range(np.shape(ax)[0]):
        ax[i,0].set_ylabel(r'$\overline{\eta}$')
    for j in range(np.shape(ax)[1]):
        ax[-1,j].set_xlabel(r'Radius [$r_g$]')
        secax = ax[0,j].secondary_xaxis('top', functions=(rg2pc, pc2rg))
        secax.set_xlabel(r'Radius [pc]')
    for i,ax_temp in enumerate(ax1d):
        ax_temp.text(0.01, 0.90, panel_label[i], transform=ax1d[i].transAxes) #, fontsize=20)
    ax1d[0].set_xscale('log')
    ax1d[0].set_yscale('log')
    #ax[0].legend(ncol=5, bbox_to_anchor=(0.,1.01), loc='lower left', columnspacing=0.8, fontsize=20)
    legend=ax1d[0].legend(fontsize=20, loc='upper right', title=r'$R_B/r_g$', ncol=2)
    ax1d[1].legend(fontsize=20, loc='upper center')
    plt.setp(legend.get_title(),fontsize=20)
    ax1d[0].set_xlim([2, 9e7])
    ax1d[0].set_ylim([2e-3, 1])
        

    # save
    output = "../plots/compare_profiles_all.png"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)

def compare_ic():
    from plotEvolution import calcStats

    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    quantities = ["Mdot", "eta"] #Mdot"]
    figsize = (9, 4 * (len(quantities)))
    fig, ax = plt.subplots(len(quantities), 1, figsize=figsize, sharex=True) #, sharey=True)
    if len(quantities) == 1: ax = [ax]
    plt.subplots_adjust(wspace=0., hspace=0)

    dirtags = [
            "092525_a0.9_rB2e3_mom_cons",
            "011226_a0.9_rB2e3_mom_cons_rot+",
            "011226_a0.9_rB2e3_mom_cons_rot-",
            "011226_a0.9_rB2e3_mom_cons_beta100",
            "092525_a0.9_rB2e5_mom_cons_test",
            "012226_a0.9_rB2e5_rot+",
            "012226_a0.9_rB2e5_rot-",
            "012726_a0.9_rB2e5_beta100",
            #"102125_a0.9_rB2e5_mom_cons_96",
            ]
    tmaxs = [700] * (len(dirtags)-1) + [1200] #-3) + [400, 400, 400]
    colors = ['k', 'b', 'r','g', 'k','b','r','g','c']
    time_bin_factor = 2.
    perzone_avg_frac = 0.05
    
    for i, dirtag in enumerate(dirtags):
        pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"
        print(pkl_name)
        with open(pkl_name, "rb") as openFile:
            D = pickle.load(openFile)
        tDivList, binNumList = setTimeBins(D, 1, time_bin_factor=time_bin_factor, tmax=tmaxs[i])
        mask_list = get_mask(D, prioritize_inner=False) #True)
        r_sonic = D["dump"]["rs"]
        mdot = D["dump"]["mdot"]
        gam = D["dump"]["gam"]
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
        rEH = D["dump"]["r_eh"]
        for j, quantity in enumerate(quantities):
            if 1: #quantity != "etaMdot":
                radii, profiles = calcFinalTimeAvg(D, tDivList, binNumList, quantity, perzone_avg_frac=perzone_avg_frac, mask_list=mask_list, rescale=True, use_avged_Mdot=False)
                if quantity == "Mdot":
                    r_read = 5 #rEH
                elif quantity == "phib":
                    r_read = rEH
                elif quantity == "eta":
                    r_read = rB / 3.
                elif quantity == "etaMdot":
                    r_read = rB / 3.
                else:
                    print("WARNING not supported")
                i_r = np.argmin(abs(radii[0] - r_read))
                q_out = profiles[0][i_r]
                print("at r={:.5g}, {}={:.5g}".format(radii[0][i_r], quantity, profiles[0][i_r]))
                color = colors[i]
                
                if quantity == "etaMdot": quantity="etaB"
                x_mean, x_std, _, _, quantity_arr = calcStats(pkl_name, quantity, average_factor=time_bin_factor, use_Mdot_mean=False, rescaleMdot=True, tmax=tmaxs[i], perzone_avg_frac=perzone_avg_frac, radius=r_read, verbose=False)
                mean_est = np.exp(x_mean + x_std ** 2 / 2.)
                ncorr = get_lcorr(quantity_arr)
                if 1:
                    if abs((np.mean((quantity_arr)) - q_out) / q_out) > 1e-1:
                        pdb.set_trace()
                    ax[j].plot(rB, q_out, color=color, marker='.', ms=10)
                Neff = len(quantity_arr) / ncorr
                cox_err = np.sqrt(x_std ** 2 / Neff + x_std ** 4 / (2. * (Neff - 1.))) # 1.96 for 95%, 1 for 68%
                print("mean_est {:.5g} x_std {:.5g} Neff {:.1g} err {:.5g}".format(x_mean + x_std**2/2., x_std, Neff, cox_err))
                nzeff = D["dump"]["Params"]["Multizone/nzones_eff"]
                low = mean_est - np.exp(x_mean + x_std ** 2 / 2. - cox_err)
                high = np.exp(x_mean + x_std ** 2 / 2. + cox_err) - mean_est
                ax[j].errorbar(rB, mean_est, yerr=[[low], [high]], color=color, capsize=6, alpha=0.5)

    # fitted scaling
    rBlist = np.logspace(2, 7, 10)
    ax[0].plot(rBlist, (-3*0.9+4.7)*np.power(rBlist, -0.5), 'k', alpha=0.5)
    #ax[1].plot(rBlist, (0.98*0.9+0.13)*np.power(rBlist, -0.6), 'k', alpha=0.5)
    
    # plot settings
    ax[0].set_xscale('log')
    ax[0].set_xlim([2e2, 1e6])
    ax[0].set_ylim([5e-4, 0.4])
    #ax[1].set_ylim([1.5e-4, 1e-1])
    ax[1].set_ylim([5e-3, 1])
    for i in range(np.shape(ax)[0]):
        ax[i].set_yscale('log')
        ax[i].set_xlabel(r'$R_B$ [$r_g$]')
    ax[0].set_ylabel(r"${\overline{\dot{M}}}$ [$\dot{M}_B$]")
    #ax[1].set_ylabel(r'${\overline{\eta}}$')
    ax[1].set_ylabel(r'${\overline{\dot{E}}}$ [$\dot{M}_B c^2$]')
    # 2nd axis
    if 1:
        secax = ax[0].secondary_xaxis('top', functions=(rB2T, T2rB))
        secax.set_xlabel(r'$T_{\infty}$ [K]') #,fontsize=fontsize, labelpad=7)
    else:
        # rB to T to halo mass
        secax = ax[0].secondary_xaxis('top', functions=(rB2M, M2rB))
        secax.set_xlabel(r'$M_{\rm halo}$ [$M_\odot$]')

    # save
    output = "../plots/compare_ic.png"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)

def compare_ic_histogram():
    from plotEvolution import calcStats

    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    quantities = ["Mdot", "eta"]
    figsize = (9, 4 * (len(quantities)))
    fig, ax = plt.subplots(len(quantities), 1, figsize=figsize)
    if len(quantities) == 1: ax = [ax]

    dirtags = [
            "092525_a0.9_rB2e3_mom_cons",
            "011226_a0.9_rB2e3_mom_cons_rot+",
            "011226_a0.9_rB2e3_mom_cons_rot-",
            "011226_a0.9_rB2e3_mom_cons_beta100",
            #"092525_a0.9_rB2e5_mom_cons_test",
            #"012226_a0.9_rB2e5_rot+",
            #"012226_a0.9_rB2e5_rot-",
            #"012726_a0.9_rB2e5_beta100",
            #"102125_a0.9_rB2e5_mom_cons_96",
            ]
    tmaxs = [700] * (len(dirtags))
    colors = ['k', 'b', 'r','g', 'k','b','r','g','c']
    time_bin_factor = 2.
    perzone_avg_frac = 0.05
    
    for i, dirtag in enumerate(dirtags):
        pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"
        print(pkl_name)
        for j, quantity in enumerate(quantities):
            color = colors[i]
            ax[j] = plotHistogram(pkl_name, ax_passed=ax[j], quantity=quantity, tmax=tmaxs[i], color=color)

    # save
    output = "../plots/compare_ic_hist.png"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)

def compare_snapshots():
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from pyharm.plots.overlays import overlay_field
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    
    fig = plt.figure(figsize=(30,12))
    spec_snp = GridSpec(1, 1)[0]
    plt.subplots_adjust(hspace=0.01)
    
    gs = GridSpecFromSubplotSpec(1, 5, subplot_spec = spec_snp, hspace=0.02, wspace=0.02)
    ax = []
    for cell in gs: ax += [plt.subplot(cell)]
    ax = np.array(ax).reshape(1,-1)

    plotrc={}
    #plotrc.update({'xlabel': False, 'ylabel': False,'xticks': [], 'yticks': [],'cbar': False, 'frame': False, 'no_title': True, 'shading': 'flat'})
    plotrc.update({'xlabel': False, 'xticks': [], 'yticks': [],'cbar': False, 'shading': 'flat'})
    inwards = -1 # 1

    # read file
    fnums = [
             1162, #1100,
            600,   #800, 
             1906, #1987,
             5551, #1944,
             9047] #7429,
    ifnum = 0
    titles = [r'$R_B/r_g\approx 400$',r'$\approx 2000$',r'$\approx 2\times 10^4$',r'$\approx 2\times 10^5$',r'$\approx 2\times 10^6$']
    
    for dirtag in gdirtags:
        # basic info
        fn = glob.glob("../data/" + dirtag + "/*00000*.phdf")[0]
        dump = pyharm.load_dump(fn,ghost_zones=False)
        a = dump["a"]
        if a == 0.9: #a == 0.5 or 
            fnum = fnums[ifnum]
            ifnum+=1
            fn = glob.glob("../data/" + dirtag + "/*{:05d}*.phdf".format(fnum))[0]
            dump = pyharm.load_dump(fn,ghost_zones=False)
            print(dump["n_step"])
            nzeff = dump["Params"]["Multizone/nzones_eff"]
            axrow = 0#int((a - 0.5)/0.4)
            axcol = nzeff - 4
            r_sonic = dump["rs"]
            mdot = dump["mdot"]
            gam = dump["gam"]
            rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
            sz = 10 * rB # 10
            window = (-sz, sz, -sz, sz)
            if axrow == 0:
                plotrc['label'] = titles[axcol]
                plotrc['no_title'] = False
            else: plotrc['no_title'] = True
            if axcol == 0: plotrc['ylabel'] = True
            else: plotrc['ylabel'] = False
            #im1 = pyharm.plots.plot_xz(ax[axrow,axcol], dump, "log_Theta", window=window, vmin=1e-6, vmax=1, cmap='gist_heat', **plotrc)
            im1 = pyharm.plots.plot_xz(ax[axrow,axcol], dump, "log_T", window=window, vmin=2e6, vmax=5e11, cmap='gist_heat', **plotrc)
            #im1 = pyharm.plots.plot_xz(ax[axrow,axcol], dump, "symlog_FE_norho", window=window, vmin=1e-6, vmax=1, **plotrc) # TODO
            
            iwv = dump["Params"]["Multizone/i_within_vcycle"]
            i_zone = abs(iwv - (nzeff - 1))
            
            print("fnum {} t={:.5g} tg / {:.5g} tB izone{}".format(fnum, dump["t"], dump["t"] / np.power(rB, 3./2), i_zone))
        
            scale = np.power(10,np.floor(np.log10(sz)))
            c= 'white'
            if axrow == 0: 
                scalebar = AnchoredSizeBar(ax[0,axcol].transData, scale, r'$10^{:d}\, r_g$'.format(int(np.log10(scale))), 'lower left', pad=0.5, sep=5, color=c, frameon=False, size_vertical=sz/8**2)
                ax[0,axcol].add_artist(scalebar)
                scale = np.power(10,np.floor(np.log10(rg2pc(sz))))
                scale_rg = pc2rg(scale)
                scalebar = AnchoredSizeBar(ax[0,axcol].transData, scale_rg, r'$10^{:d}\,$pc'.format(int(np.log10(scale))), 'lower right', pad=0.5, sep=5, color=c, frameon=False, size_vertical=sz/8**2)
                ax[0,axcol].add_artist(scalebar)
        
            #if nzeff==4:
            #    ax[axrow,0].text(-sz*0.9, sz*0.7, r'$a_*=$'+str(a), color='w', fontsize=40)
            
            # show RB
            circle1 = plt.Circle((0, 0), rB, ec='grey', fill=False, ls="--", lw=3)
            ax[axrow,axcol].add_artist(circle1)
            if axcol==4: ax[axrow,axcol].text(rB*1.5, -rB*0.5, r'$R_B$', color='grey', fontsize=30)

            if axrow==1:ax[axrow,axcol].title.set_visible(False)

            if 0:
                # show B fields
                at_i = np.argmin(abs(dump["r1d"] - sz * 1.5))
                overlay_field(ax[axrow,axcol], dump, half_cut=True, nlines=30, reverse=True, i_slice=slice(0, at_i), color='w')#, sum=False)

    #ax[0,0].set_ylabel(r'$a_*=0.5$')
    ax[0,0].set_ylabel(r'$a_*=0.9$')
    

    if 1:
        # TODO
        # colorbar for each rows
        cb=fig.colorbar(im1, cax=ax[0,-1].inset_axes((1.01, 0.0, 0.1, 1.0)))
        cb.ax.tick_params(labelleft=False, labelright=True)
        cb.ax.set_ylabel(r'$T$ [K]')
    
    output = "../plots/compare_snapshot.pdf"
    plt.savefig(output,bbox_inches='tight')
    print("saved to " + output)
    plt.close()

def compare_spinup(xaxis_rB=False):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(1, 1, figsize=(7,5))

    dirtags = gdirtags + [
            "011226_a0.9_rB2e3_mom_cons_rot+",
            "011226_a0.9_rB2e3_mom_cons_rot-",
            #"011226_a0.9_rB2e3_mom_cons_beta100",
            "012226_a0.9_rB2e5_rot+",
            "012226_a0.9_rB2e5_rot-",
            "012726_a0.9_rB2e5_beta100",
            #"051225_oz_a0.9_toriilike_newflr",
            "051225_n4_a0.9_toriilike_newflr",
            "051225_n4_a-0.9_toriilike_eks",
            #"012026_a0_rB2e5_mom_cons_test",
            #"051225_n4_a0.9_torrilike_nocap_newflr",
            #"052825_n4_a-0.9_torilike_nocap_newflr",
            #"102125_a0.9_rB2e5_mom_cons_96",
            ]
    labels = [r'$R_B/r_g\approx 400$',r'$2000$', r'$2\times 10^4$', r'$2\times 10^5$',r'$2\times 10^6$']
    tmaxs = [700] * (len(dirtags)-5) + [400,400,400,50, 50]
    if xaxis_rB:
        colors = list(plt.cm.gnuplot(np.linspace(0.9, 0., 5))) + ['gray']
    else: colors = list(plt.cm.binary(np.linspace(0.2, 1., 5)))
    #else: colors = list(plt.cm.coolwarm(np.linspace(0., 1., 5)))
    time_bin_factor = 2.
    perzone_avg_frac = 0.05
    
    a_save = []
    s_save = []
    for i, dirtag in enumerate(dirtags):
        pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"
        print(pkl_name)
        with open(pkl_name, "rb") as openFile:
            D = pickle.load(openFile)
        tDivList, binNumList = setTimeBins(D, 1, time_bin_factor=time_bin_factor, tmax=tmaxs[i])
        mask_list = get_mask(D, prioritize_inner=False) #True)
        r_sonic = D["dump"]["rs"]
        mdot = D["dump"]["mdot"]
        gam = D["dump"]["gam"]
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
        rEH = D["dump"]["r_eh"]
        a = D["dump"]["a"]
        try:
            oz = False
            nzeff = D["dump"]["Params"]["Multizone/nzones_eff"]
        except: 
            oz = True
            nzeff = 4
        
        # TODO: from here
        radii, profiles_j = calcFinalTimeAvg(D, tDivList, binNumList, "norm_Ldot", perzone_avg_frac=perzone_avg_frac, mask_list=mask_list, rescale=True, use_avged_Mdot=False)
        _, profiles_e = calcFinalTimeAvg(D, tDivList, binNumList, "norm_Edot", perzone_avg_frac=perzone_avg_frac, mask_list=mask_list, rescale=True, use_avged_Mdot=False)
        _, profiles_eta = calcFinalTimeAvg(D, tDivList, binNumList, "eta_EM", perzone_avg_frac=perzone_avg_frac, mask_list=mask_list, rescale=True, use_avged_Mdot=False)
        r_read = 5
        i_r = np.argmin(abs(radii[0] - r_read))
        s = profiles_j[0][i_r] - 2. * profiles_e[0][i_r] * a
        #color = colors[np.where(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.97])==a)[0][0]]
        color = colors[max(nzeff-4,0)]
        ms = 10
        marker ='.'
        if oz: 
            marker = '+'
        elif "96" in dirtag:
            color = 'g'
        else:
            if D["dump"]["uphi"]>0: 
                color='b'
                #marker='+'
            elif D["dump"]["uphi"]<0: 
                color='r'
                #marker='x'
            if nzeff < 4:
                marker = '*'
                #if D["dump"]["uphi"]>0: color='b'
                #elif D["dump"]["uphi"]<0: color='r'
            print(s)
        if xaxis_rB: ax.plot(rB, s, color=color, marker=marker, ms=10)
        else:
            label = labels[nzeff-4]
            labels[nzeff-4] = '__nolegend__'
            if nzeff == 8: 
                a_save += [a]
                s_save += [s]
            ax.plot(a, s, color=color, marker=marker, ms=ms, label=label, ls='', mfc='none')
            if 0:
                rEH = calc_rEH(a)
                OmegaH = a / (2 * rEH)
                k=0.35 # 0.5
                s_EM = -profiles_eta[0][i_r] * (1. / (k * OmegaH) - 2. * a) # eq 20 in Lowell+23
                ax.plot(a, s_EM, color=color, alpha=0.5, marker='+')

    # plot settings
    ax.set_ylabel(r'spinup $\overline{s}$')
    if xaxis_rB:
        # x-axis RB
        ax.set_xscale('log')
        ax.set_xlabel(r'$R_B$ [$r_g$]')
        # 2nd axis
        secax = ax.secondary_xaxis('top', functions=(rB2T, T2rB))
        secax.set_xlabel(r'$T_{\infty}$ [K]') #,fontsize=fontsize, labelpad=7)
    else:
        # x-axis a
        ax.set_xlim([-0.1,1])
        ax.set_xticks([-0.1,0.1,0.3,0.5,0.7,0.9])
        a_arr = np.linspace(-0.1,1,100)
        popt, pcov = curve_fit(lin_func, a_save, s_save)
        print(popt)
        #ax.plot(a_arr, lin_func(a_arr, *popt), color='k', alpha=0.5)
        ax.plot(a_arr, lin_func(a_arr, a=-3.7, b=0), color='k', alpha=0.5)
        a_arr = np.linspace(0,1,100)
        ax.plot(a_arr, (0.45-12.53*a_arr-7.8*a_arr**2+9.44*a_arr**3+5.71*a_arr**4-4.03*a_arr**5),'b:', label='RN22 prograde')
        ax.plot(a_arr, -(0.45+12.53*a_arr-7.8*a_arr**2-9.44*a_arr**3+5.71*a_arr**4+4.03*a_arr**5), 'r:', label='RN22 retrograde')
        ax.set_xlabel(r'$a_*$')
        ax.legend(fontsize=12)

    # save
    output = "../plots/compare_spinup.pdf"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)

if __name__ == "__main__":
    #compare_prescription_slice()
    #compare_evolution_rB() #show_tl=True) #0.7)
    compare_quantity_rB()
    #compare_profiles_all()
    #compare_feedback_a()
    #show_snapshot(4783) #5224) #3494) #4000) # old runs 3830) #3997) # 3598) #3493) #3220) #
    #snapshot_Gamma(4783) #2097) #4763)
    #compare_jet_disk_profile()
    #show_tavged(also_show_rprofile=True)
    #compare_resolution()
    #compare_ic()
    #compare_ic_histogram()
    #compare_snapshots()
    #compare_spinup()

