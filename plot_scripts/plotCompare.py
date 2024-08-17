import glob

from plotProfiles import *


def compareRuns(dirtags, quantities, colors, labels=None, plot_dir=None, row=None, figsize=None, xlim=None):
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

    fig_ax = plt.subplots(row, col, figsize=figsize, sharex=True)

    for i, dirtag in enumerate(dirtags):
        print(dirtag)
        pkl_name = glob.glob("../data_products/" + dirtag + "_profiles_all*.pkl")
        if len(pkl_name) > 1:
            pdb.set_trace()
            print("ERROR: found more than 1 pickle file! Take a look at which one you'd like to use.")
        else:
            pkl_name = pkl_name[0]
        fig_ax = plotProfiles(pkl_name, quantities, plot_dir=plot_dir, perzone_avg_frac=0.5, num_time_chunk=1, fig_ax=fig_ax, color_list=[colors[i]], label=labels[i])

    if xlim is not None:
        ax1d = fig_ax[1].reshape(-1)
        ax1d[0].set_xlim(xlim)

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


def compareSpin(a=0.5):
    if a == 0.5:
        dirtagList = [
            "081524_a0.5_oz",
            "081424_a0.5_bfluxc_moverin",
            "081524_a0.5_bflux0_moverin",
        ]  # , "061724_fastvc/combineout_ismr_a0.5_moverin", "061724_fastvc/combineout_ismr_a0.5_ncycle50"] #, "061724_fastvc/combineout_ismr_a0.5", "061724_fastvc/combineout_ismr_a0.5_bfluxc"] #, "061724_fastvc/combineout_ismr_a0.5_nolongtin"] "051224_bondi_kerr/00000",
        quantityList = ["Mdot", "rho", "beta", "eta", "eta_Fl", "eta_EM"]
        colorList = ["black", "tab:blue", "tab:green", "r", "tab:orange", "m"]  # colors for each runs
        labelList = ["oz", "mz_bfluxc", "mz_bflux0", "mz_bflux0_50"]  # _final', 'mz', 'mz_bfluxc', 'mz_moverin', 'mz_nolongtin']
        plot_dir = "../plots/081224_spin_" + str(a)

    rEH = 1.0 + np.sqrt(1.0 - a**2)  # TODO: calculation as a fxn of a
    xlim = (rEH, 3e4)

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim)


def _main():
    # compareFvcVsOld()
    compareSpin()


if __name__ == "__main__":
    _main()
