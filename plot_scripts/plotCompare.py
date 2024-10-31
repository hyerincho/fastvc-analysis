import glob

from plotProfiles import *


def compareRuns(dirtags, quantities, colors, labels=None, linestyles=None, plot_dir=None, row=None, figsize=None, xlim=None, passed_fig_ax=None):
    matplotlib_settings()

    if len(quantities) <= 3:
        row = 1
    elif row is None:
        row = 2
    col = len(quantities) // row
    if figsize is None:
        figsize = (8 * col, 6 * row)
    if labels is None:
        labels = [None] * len(dirtags)
    if linestyles is None:
        linestyles = [None] * len(dirtags)

    if passed_fig_ax is None:
        fig_ax = plt.subplots(row, col, figsize=figsize, sharex=True)
    else:
        fig_ax = passed_fig_ax

    for i, dirtag in enumerate(dirtags):
        print(dirtag)
        pkl_name = glob.glob("../data_products/" + dirtag + "_profiles_all*.pkl")
        if len(pkl_name) > 1:
            pdb.set_trace()
            print("ERROR: found more than 1 pickle file! Take a look at which one you'd like to use.")
        else:
            pkl_name = pkl_name[0]
        fig_ax = plotProfiles(pkl_name, quantities, plot_dir=plot_dir, perzone_avg_frac=0.5, num_time_chunk=1, fig_ax=fig_ax, color_list=[colors[i]], label=labels[i], linestyle=linestyles[i])

    if xlim is not None:
        ax1d = fig_ax[1].reshape(-1)
        ax1d[0].set_xlim(xlim)

    if passed_fig_ax is not None:
        return fig_ax
    else:
        # save plot
        os.makedirs(plot_dir, exist_ok=True)
        output = plot_dir + "/" + plot_dir.split("/")[-1] + "_comparison.png"
        plt.savefig(output, bbox_inches="tight")
        plt.close()
        print("saved to " + output)


def compareFvcVsOld():
    dirtagList = [
        "061724_fastvc/combineout_restructured",
        "061724_fastvc/combineout_ncycle200_capped_ncycle",
        "061724_fastvc/combineout_nocap",
        "061724_fastvc/save/combineout_ncycle200",
        "061724_fastvc/combineout_ismr",
    ]
    quantityList = ["Mdot", "rho", "beta", "eta"]  # , "eta_Fl", "eta_EM"]
    colorList = ["tab:red", "tab:blue", "tab:orange", "tab:green", "black"]  # colors for each runs
    labelList = ["old", "fvc", "old_nocap", "fvc_nocap", "old_ismr"]
    plot_dir = "../plots/080924_fvc_vs_old"

    xlim = (2, 3e4)

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim)


def compareN4Beta():
    dirtagList = ["100724_a0.5_n4", "101524_a0.5_beta10_rot", "100724_a0.5_beta100_rot"]
    quantityList = ["Mdot", "rho", "beta", "eta", "eta_Fl", "eta_EM", "u^r", "Omega"]
    colorList = ["tab:red", "tab:blue", "tab:orange", "tab:green", "black"]  # colors for each runs
    labelList = ["beta1", "10", "100"]
    plot_dir = "../plots/102124_n4_beta"

    xlim = (2, 3e4)

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim)


def compareSpin(a=0.5):
    linestyleList = None
    if a == None:
        # compare between different spins
        dirtagList = [  # TODO: include oz from old multizone
            "100724_a0.0_n4",
            "100724_a0.1_n4",
            "100724_a0.3_n4",
            "100724_a0.5_n4",
            "100724_a0.7_n4",
            "100724_a0.9_n4",
            "2023/122723_n4_onezone_wks0.04/00000",
            "100724_a0.5_oz",
        ]
        colorList = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(dirtagList) - 2))  # colors for each runs
        colorList = np.append(colorList, [colorList[0], colorList[3]], axis=0)  # for onezones
        labelList = ["0", "0.1", "0.3", "0.5", "0.7", "0.9", "0.0_oz", "0.5_oz"]
        linestyleList = ["solid"] * (len(dirtagList) - 2) + ["dashed"] * 2
        plot_dir = "../plots/101624_different_spin"
    if a == 0.5:
        dirtagList = [
            # "081524_a0.5_oz", # replace to 0909
            # "090924_a0.5_oz",
            "100724_a0.5_oz",
            # "091124_a0.5_production",
            "100724_a0.5_n4",
            "092524_a0.5_bflux0",
            # "092524_a0.5_nomoverin",
            # "092524_a0.5_rot0.5",
            # "100224_a0.5_production"
            # "092324_a0.5_rdepgamx_test", #"092624_a0.5_rdepgmax",
            # "092624_a0.5_sigmaceildown",
            # "092224_a0.5_production_gmax2",
            # "081424_a0.5_bfluxc_moverin",
            # "081524_a0.5_bflux0_moverin",
            # "081624_a0.5_bfluxc_moverin_longtin4"
            # "082024_a0.5_consistentB_onlyfofc"
            # "081924_a0.5_consistentB",
            # "081524_a0.5_ncycle50"
        ]  # , "061724_fastvc/combineout_ismr_a0.5_moverin", "061724_fastvc/combineout_ismr_a0.5_ncycle50"] #, "061724_fastvc/combineout_ismr_a0.5", "061724_fastvc/combineout_ismr_a0.5_bfluxc"] #, "061724_fastvc/combineout_ismr_a0.5_nolongtin"] "051224_bondi_kerr/00000",
        colorList = ["black", "tab:blue", "tab:green", "r", "tab:orange", "m"]  # colors for each runs
        labelList = ["oz", "mz", "mz_bflux0", "mz_slide", "mz_rot", "mz_rdepgmax", "mz_sigmadown", "mz_bfluxc", "mz_bfluxc_cons"]  # , "mz_bfluxc_50"]  # _final', 'mz', 'mz_bfluxc',  'mz_nolongtin']
        plot_dir = "../plots/081224_spin_" + str(a)

    quantityList = ["Mdot", "rho", "beta", "eta", "eta_Fl", "eta_EM", "u^r", "Omega"]
    if a == None:
        rEH = 2  # just use a=0 rEH
    else:
        rEH = 1.0 + np.sqrt(1.0 - a**2)  # TODO: calculation as a fxn of a
    xlim = (rEH, 3e4)

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim, linestyles=linestyleList)


def _main():
    # compareFvcVsOld()
    # compareN4Beta()
    compareSpin(None)


if __name__ == "__main__":
    _main()
