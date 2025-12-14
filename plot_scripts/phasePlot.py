from plot_utils import *
from matplotlib_settings import *
from ylabel_dictionary import *


def compare_eta_phib(dirtags, labels=None, colors=None, average_factor=1.2, use_Mdot_mean=True, tmax=None):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    if colors is None:
        colors = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(dirtags)))  # colors for each runs
    if labels is None:
        labels = dirtags

    for i, dirtag in enumerate(dirtags):
        pkl = "../data_products/" + dirtag + "_profiles_all.pkl"
        with open(pkl, "rb") as openFile:
            D = pickle.load(openFile)

        #eta_mean = extractQuantity(D, "eta", average_factor)
        #phib_mean = extractQuantity(D, "phib", average_factor)
        eta = extractQuantity(D, "eta", tmax, average_factor, False, use_Mdot_mean)
        phib = extractQuantity(D, "phib", tmax, average_factor, False, use_Mdot_mean)
        a = get_spin(D)

        #ax.plot(phib_mean, eta_mean, marker=".", color=colors[i], label=labels[i])
        #ax.errorbar(np.mean(phib), np.mean(eta), xerr=np.std(phib), yerr=np.std(eta), marker=".", color=colors[i], label=labels[i])
        ax.scatter(phib, eta, marker=".", color=colors[i], label=labels[i])

    ax.set_xlim([10, 100])
    ax.set_ylim([1e-2, 5])
    ax.set_xscale('log')
    ax.set_yscale('log')
    xlim = ax.get_xlim()
    phib_xaxis = np.linspace(xlim[0], xlim[1], 10)
    ax.plot(phib_xaxis, eta_BZ6(a, phib_xaxis, 0.05))
    ax.legend()  # fontsize=7)
    ax.set_xlabel(r"$\phi_b$")
    ax.set_ylabel(r"$\eta$")
    fig.tight_layout()
    output = "../plots/compare_eta_phib.png"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)

def scatter_q1_q2(dirtag, ax_passed=None, q1='phib', q2='eta', tmax=None, average_factor=1.2, use_Mdot_mean=True, color_eta=False, color_time=False, logx=False, logy=False, show_cbar=True, use_thinfo_q1=False, only_use_positive_Omega=False, show_hist=False):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 15})
    if ax_passed is None:
        if show_hist:
            fig, axs = plt.subplot_mosaic([['histx', '.'],
                ['scatter', 'histy']],
                figsize=(6, 6)) #, layout='constrained') #, width_ratios=(4, 1), height_ratios=(1, 4)
            ax = axs['scatter']
        else: fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        ax = ax_passed

    pkl = "../data_products/" + dirtag + "_profiles_all.pkl"
    with open(pkl, "rb") as openFile:
        D = pickle.load(openFile)
        a = get_spin(D)
        rEH = calc_rEH(a)

    if use_thinfo_q1:
        # use theta-dependent info instead of already shell averaged data
        if q1 == "Omega": r1 = 10 #or "u^th" in q1 or "u^phi" in q1
        else: r1 = 5
        if q2 == "phib": r2 = rEH
        else: r2 = 5
        num_files = -1 #500 #
        fnames = sorted(glob.glob('../data/' + dirtag + '/*out0.*.phdf'))
        which = "signedavg" #"phiweight" #
        q1_arr = extract_rth_info(fnames, q1, [r1], num_files, which=which)[0]
        if which == "phiweight": q1_arr = np.abs(q1_arr)
        q2_arr = extract_shellsum(fnames, q2, [r2], num_files)[0]
        if only_use_positive_Omega: 
            ind = (extract_rth_info(fnames, "Omega", [10], num_files, which="avg")[0]) > 0
            q1_arr = q1_arr[ind]
            q2_arr = q2_arr[ind]
    else:
        q1_arr = extractQuantity(D, q1, tmax, average_factor, False, use_Mdot_mean, verbose=True)
        q2_arr = extractQuantity(D, q2, tmax, average_factor, False, use_Mdot_mean)
        if only_use_positive_Omega: 
            ind = (extractQuantity(D, "Omega10", tmax, average_factor, False, use_Mdot_mean)) > 0
            q1_arr = q1_arr[ind]
            q2_arr = q2_arr[ind]

    a = get_spin(D)

    if (color_eta or color_time) and not use_thinfo_q1:
        if color_eta:
            color = extractQuantity(D, 'eta', tmax, average_factor, False, use_Mdot_mean)
            norm = matplotlib.colors.LogNorm(vmin=1e-2, vmax=10)
        if color_time:
            color = extractQuantity(D, 'time', tmax, average_factor, False, use_Mdot_mean)
            norm = None
        if only_use_positive_Omega: color = color[ind]
        sc = ax.scatter(q1_arr, q2_arr, marker=".", c=color, norm=norm)
        if ax_passed is None or show_cbar:
            cbar = plt.colorbar(sc)
            cbar.set_label(r'$\eta$', rotation=0)
    else: 
        sc = ax.scatter(q1_arr, q2_arr, marker=".", c='k')
    if show_hist:
        nbins = 10
        if logx: xbins = np.logspace(np.log10(np.min(q1_arr)), np.log10(np.max(q1_arr)), nbins)
        else: xbins = np.linspace(np.min(q1_arr), np.max(q1_arr), nbins)
        if logy: 
            ybins = np.logspace(np.log10(np.min(q2_arr)), np.log10(np.max(q2_arr)), nbins)
            axs['histy'].set_yscale('log')
        else: ybins = np.linspace(np.min(q2_arr), np.max(q2_arr), nbins)
        axs['histx'].hist(q1_arr, bins=xbins)
        axs['histy'].hist(q2_arr, bins=ybins, orientation='horizontal')
        #axs['histy'].set_xscale('log')
        #axs['histx'].set_yscale('log')
        axs['histy'].sharey(ax)
        axs['histx'].sharex(ax)


    if q1 == "phib":
        ax.set_xlim([10, 100])
    elif "Omega" in q1:
        ax.set_xlim([-1, 1])
        #ax.set_xlim([0, 0.5])
        #ax.set_xscale('log')
    elif q1 == "abs_u^th":
        ax.set_xlim([2.5e-3, 1.5e-2])
    if q2 == "eta":
        ax.set_ylim([1e-2, 10])
    elif q2 == "phib":
        ax.set_ylim([10, 100])
    if logx: ax.set_xscale('log')
    else: ax.axvline(0, color='k', ls=':')
    if logy: ax.set_yscale('log')
    xlim = ax.get_xlim()
    if q1 == "phib" and q2 == "eta":
        phib_xaxis = np.linspace(xlim[0], xlim[1], 10)
        ax.plot(phib_xaxis, eta_BZ6(a, phib_xaxis, 0.05))
    ax.set_xlabel(variableToLabel(q1))
    if ax_passed is None:
        ax.set_ylabel(variableToLabel(q2))
        fig.suptitle(dirtag)
        fig.tight_layout()
        output = "../plots/scatter_" + q1 + "_" + q2 + ".png"
        plt.savefig(output, bbox_inches="tight")
        print("saved to " + output)
        plt.close(fig)
    else: return sc

def scatter_Omega_q2(dirtag, q2='eta', tmax=None, average_factor=1.2, use_Mdot_mean=True, color_eta=False, color_time=False, use_midplane_Omega=False, show_hist=False, for_paper=False):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 15})
    if use_midplane_Omega: radii = [None]
    else: radii = [5] #[5, 10, 50]
    if show_hist:
        # just focus on 1 radius r=5
        ax = None
        radii = [5]
    else:
        fig, ax = plt.subplots(1, len(radii), figsize=(6 * len(radii), 6), sharey=True, sharex=True)
    if len(radii) == 1:
        ax = [ax]
    plt.subplots_adjust(wspace=0., hspace=0)
    q1 = "abs_u^th" #"Omega"
    for i, r in enumerate(radii):
        if not use_midplane_Omega: q1 = "Omega" + str(r)
        sc = scatter_q1_q2(dirtag, ax[i], q1 , q2, tmax, average_factor, use_Mdot_mean, color_eta, color_time, False, True, show_cbar=False, use_thinfo_q1=use_midplane_Omega, show_hist=show_hist)

    if not show_hist and (color_eta or color_time) and not use_midplane_Omega: 
        cax = fig.add_axes([.91,.124,.02,.754])
        cbar = fig.colorbar(sc, cax=cax)
        if color_eta: cbar_label=r"$\eta$"
        else: cbar_label = "$t/t_B$"
        cbar.set_label(cbar_label, rotation=0)
    if not show_hist: 
        if q2 == "eta": ax[0].set_ylabel(r'$\eta$')
        else: ax[0].set_ylabel(variableToLabel(q2))
        if not for_paper: fig.suptitle(dirtag)
    #fig.tight_layout()
    output = "../plots/scatter_omega_" + q2 + ".png"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)

def scatter_q_complex(dirtag, ax_passed=None, q='u^th', color_eta=False, show_cbar=True):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 15})
    if ax_passed is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        ax = ax_passed
    # TODO: color for eta

    pkl = "../data_products/" + dirtag + "_profiles_all.pkl"
    with open(pkl, "rb") as openFile:
        D = pickle.load(openFile)
        a = get_spin(D)
        rEH = calc_rEH(a)

    r = 5 #10 #
    num_files = -1 #
    fnames = sorted(glob.glob('../data/' + dirtag + '/*out0.*.phdf'))
    q_arr = extract_rth_info(fnames, q, [r], num_files, which="phiweight")[0]

    a = get_spin(D)

    color = extract_shellsum(fnames, 'eta', [r], num_files)[0]
    sc = ax.scatter(np.real(q_arr), np.imag(q_arr), marker=".", c=color, norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=10))
    ax.axvline(0, color='k', ls=':')
    ax.axhline(0, color='k', ls=':')
    ax.set_xlabel(r'Re[$\langle u^{\theta} e^{i\phi}\rangle$]')
    ax.set_ylabel(r'Im[$\langle u^{\theta} e^{i\phi}\rangle$]')
    if ax_passed is None or show_cbar:
        cbar = plt.colorbar(sc)
        cbar.set_label(r'$\eta$', rotation=0)

    if ax_passed is None:
        fig.suptitle(dirtag)
        fig.tight_layout()
        output = "../plots/scatter_" + q + "_complex.png"
        plt.savefig(output, bbox_inches="tight")
        print("saved to " + output)
        plt.close(fig)
    else: return sc


def compare_kappa(dirtags):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(dirtags)))  # colors for each runs

    for i, dirtag in enumerate(dirtags):
        pkl = "../data_products/" + dirtag + "_profiles_all.pkl"
        with open(pkl, "rb") as openFile:
            D = pickle.load(openFile)

        eta_mean = extractQuantity(D, "eta")
        phib_mean = extractQuantity(D, "phib")
        a = get_spin(D)

        kappa = eta_mean / eta_BZ6(a, phib_mean, 1)

        ax.plot(phib_mean, kappa, marker=".", color=colors[i], label=dirtag)

    ax.legend(fontsize=7)
    ax.axhline(0.044)
    ax.axhline(0.053)
    ax.set_xlabel(r"$\phi_b$")
    ax.set_ylabel(r"$\kappa$")
    fig.tight_layout()
    output = "../plots/compare_kappa.png"
    plt.savefig(output, bbox_inches="tight")
    print("saved to " + output)
    plt.close(fig)


if __name__ == "__main__":
    dirtags = ["032125_torus_tegan", "032425_torus", "032525_torus_safe_floors_normal", "032625_torus_safe_floors2", "032725_torus_safe_floors3", "032825_torus_safe_floors4", "032825_torus_safe_floors_nofofc_sigmamax", "040125_torus_reconnect_sigma", "040225_torus_reconnect_sigma_Tmax", "040325_torus_noehbuffer"]
    dirtags = ["031125_a0.9_cap_correctly", "032125_n4a0.9_toriilike", "042125_n4_a0.9_bondi_jks2", "041625_n4_a0.9_toriilike_jks2_smth2_reconnect"]
    dirtags = [
        #"delta/030525_a0.9_oz",
        #"042125_a0.9_oz_jks",
        #"031125_a0.9_cap_correctly",
        #"042125_n4_a0.9_bondi_jks2",
        #"delta/032025_a0.9_oz_clearangle",
        #"032125_n4a0.9_toriilike",
        #"041625_n4_a0.9_toriilike_jks2_smth2_reconnect",
        #"042225_n4_a0.9_retrograde",
        #"040325_n4_a0.9_torrilike_nocap",
        #"042325_a0.9_rB2e3_bondi",
        #"042325_a0.9_rB2e5_bondi",
        #"041825_a0.9_rB2e5_jks2_smth2",
        #"043025_n4_a0.9_bondi_rot+",
        #"043025_n4_a0.9_bondi_rot-",
        #"042225_n4_a0.9_bondi_jks2_clearangle",
        #"042825_a0.9_rB2e4_bondi_rot",
        # "042325_n4_a0.9_tl_uphi0",
        # "042225_n4_a0.9_toriilike_jks2_nocap",
        #"042225_n4_a0.9_bondi_jks2_nocap"
        #"043025_a0.9_rB2e3_bondi_eks"
        #"030725_a0.9_safe_longtin20"
        #"051225_oz_a0.9_toriilike_newflr"
        "092525_a0.9_rB2e5_mom_cons_test"
    ]
    labels = ['bondi'] #["oz", "oz_jks", "mz", "mz_jks", "oz_tl", "mz_tl", "mz_tl_jks", "mz_tl_-.9", "mz_tl_nocap", "2e3", "2e5", "+", "-"]  # "mz_tl_0","mz_jks_ca", 
    colors = ["black", "tab:blue", "c", "g", "r", "tab:orange", "m", "y", "pink", "peru", "orange", "r", 'm']  # colors for each runs #"gray", 
    compare_eta_phib(dirtags, labels, colors, 2., use_Mdot_mean=False)  # 1.1)
    dirtag="051225_oz_a0.9_bondi_newflr" #"052825_a0.9_rB2e5_bondi_eks_largerout" #"051225_n4_a0.9_bondi_newflr" #"051225_n4_a0.9_bondi_nocap_newflr" #"052825_n4_a-0.9_torilike_nocap_newflr" #"043025_a0.9_rB2e5_bondi_eks" #"051225_n4_a-0.9_toriilike_eks" #"041825_a0.9_rB2e5_jks2_smth2" #"042325_a0.9_rB2e5_bondi" #"042225_n4_a0.9_toriilike_jks2_nocap" #"042225_n4_a0.9_retrograde" #"042125_n4_a0.9_bondi_jks2" #"delta/030525_a0.9_oz" #"041625_n4_a0.9_toriilike_jks2_smth2_reconnect" #
    #scatter_q1_q2(dirtag, q1='abs_u^th', q2='phib', average_factor=2, use_Mdot_mean=False, color_eta=True, logx=True, logy=True, only_use_positive_Omega=True)
    #scatter_q1_q2(dirtag, q1='abs_u^phi', q2='phib', average_factor=2, use_Mdot_mean=False, color_eta=True, logx=True, logy=True)
    #scatter_q1_q2(dirtag, q1='u^th', q2='eta', average_factor=2, use_Mdot_mean=False, color_eta=True, logx=False, logy=True, use_thinfo_q1=True, only_use_positive_Omega=True)
    #scatter_q_complex(dirtag, q='u^th', color_eta=True)
    #scatter_q1_q2(dirtag, q1='u^phi', q2='phib', average_factor=2, use_Mdot_mean=False, color_eta=True, logx=False, logy=True)
    #scatter_Omega_q2(dirtag, "eta", 50, 1e5, use_Mdot_mean=False, color_eta=False, color_time=True, use_midplane_Omega=False, for_paper=True) #, show_hist=True)
    #scatter_Omega_q2(dirtag, "phib", 2, use_Mdot_mean=False, color_eta=True)
    # compare_kappa(dirtags)

    # print(eta_BZ6(0.9375, 56.37, 0.044))
    # print(eta_BZ6(0.9375, 44.8, 0.054))
    # print(eta_BZ6(0.9, 49.695, 0.044))
