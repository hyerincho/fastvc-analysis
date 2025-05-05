import glob
import pdb

from plotProfiles import *
import oldplotProfiles as old_script
from plot_utils import *
from ylabel_dictionary import *


def compareRuns(dirtags, quantities, colors, labels=None, linestyles=None, plot_dir=None, row=None, figsize=None, xlim=None, passed_fig_ax=None, tmax=None, rescale=False, time_bin_factor=2):
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

    if not isinstance(tmax, list):
        tmax = [tmax] * len(dirtags)

    if passed_fig_ax is None:
        fig_ax = plt.subplots(row, col, figsize=figsize, sharex=True)
    else:
        fig_ax = passed_fig_ax

    for i, dirtag in enumerate(dirtags):
        use_old = False
        print(dirtag)
        pkl_name = glob.glob("../data_products/" + dirtag + "_profiles_all*.pkl")
        if len(pkl_name) > 1:
            pkl_name = pkl_name[0]
            print("ERROR: found more than 1 pickle file! Using this one: " + pkl_name)
        else:
            pkl_name = pkl_name[0]
        if "kharma_multizone_analysis" in pkl_name:
            use_old = True
        if use_old:
            fig, axes = fig_ax
            # ax1d = axes.reshape(-1)
            # for j, quantity in enumerate(quantities):
            fig_ax = old_script.plotProfiles(pkl_name, quantities, zone_time_average_fraction=0.5, fig_ax=fig_ax, num_time_chunk=1, color=colors[i], label=labels[i], rescale_Mdot=rescale, average_factor=time_bin_factor)
            # fig_ax = (fig, axes)
        else:
            fig_ax = plotProfiles(pkl_name, quantities, plot_dir=plot_dir, perzone_avg_frac=1, num_time_chunk=1, fig_ax=fig_ax, color_list=[colors[i]], label=labels[i], linestyle=linestyles[i], tmax=tmax[i], rescale=rescale, time_bin_factor=time_bin_factor)

    if xlim is not None:
        if len(np.shape(fig_ax[1])) > 1:
            ax1d = fig_ax[1].reshape(-1)
        else:
            ax1d = fig_ax[1]
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

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim, rescale=True)


def compareSpin(a=0.5, time_bin_factor=2):
    linestyleList = None
    oz_num = 0
    if a == None:
        # compare between different spins
        dirtagList = [  # TODO: include oz from old multizone
            # "100724_a0.0_n4",
            # "100724_a0.1_n4",
            # "100724_a0.3_n4",
            # "100724_a0.5_n4",
            # "100724_a0.7_n4",
            # "100724_a0.9_n4",
            # "030425_a0.0_b8n4_safe",
            "030625_a0.0_n4_013124",
            # "030425_a0.1_b8n4_safe",
            "031125_a0.1_cap_correctly",
            # "030425_a0.3_b8n4_safe",
            "031125_a0.3_cap_correctly",
            # "030425_a0.5_b8n4_safe_longtin10",
            "031125_a0.5_cap_correctly",
            # "030425_a0.7_b8n4_safe",
            "031125_a0.7_cap_correctly",
            # "030425_a0.9_b8n4_safe",
            "031125_a0.9_cap_correctly",
            # "2023/122723_n4_onezone_wks0.04/00000",
            "delta/030525_a0.0_oz",
            "delta/030525_a0.5_oz",
            "delta/030525_a0.9_oz",
            # "030325_a0.5_oz_128",
            # "030325_a0.9_oz_128",
            # "100724_a0.5_oz",
        ]
        oz_num = 3
        colorList = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(dirtagList) - oz_num))  # colors for each runs
        colorList = np.append(colorList, [colorList[0], colorList[3], colorList[-1]], axis=0)  # for onezones
        # colorList = np.append(colorList, [colorList[3], colorList[5]], axis=0)  # for onezones
        labelList = ["0", "0.1", "0.3", "0.5", "0.7", "0.9", "0.0_oz", "0.5_oz", "0.9_oz"]  #
        linestyleList = ["solid"] * (len(dirtagList) - oz_num) + ["dashed"] * oz_num
        plot_dir = "../plots/030525_different_spin"
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
    if a == 0.9:
        dirtagList = [
            # "030325_a0.9_oz_128",
            "delta/030525_a0.9_oz",
            "042125_a0.9_oz_jks",
            "031125_a0.9_cap_correctly",
            "042125_n4_a0.9_bondi_jks2",
            # "042225_n4_a0.9_bondi_jks2_clearangle",
            "delta/032025_a0.9_oz_clearangle",
            "032125_n4a0.9_toriilike",
            "041625_n4_a0.9_toriilike_jks2_smth2_reconnect",
            "042225_n4_a0.9_retrograde",
            # "042325_n4_a0.9_tl_uphi0",
            "042225_n4_a0.9_toriilike_jks2_nocap",
            # "042225_n4_a0.9_bondi_jks2_nocap"
            # "040325_n4_a0.9_torrilike_nocap"
        ]
        oz_num = len(dirtagList)  # just so that I don't have a time cap
        colorList = ["black", "tab:blue", "c", "g", "r", "tab:orange", "m", "y", "pink", "gray"]  # colors for each runs
        labelList = ["oz", "oz_jks", "mz", "mz_jks", "oz_tl", "mz_tl", "mz_tl_jks", "mz_tl_-.9", "mz_tl_nocap"]  # "mz_jks_ca", "mz_tl_0",
        plot_dir = "../plots/042125_spin_" + str(a)

    # quantityList = ["Mdot", "rho", "beta", "eta", "eta_Fl", "eta_EM", "Omega", "T"] #"Omega"["Mdot", "eta", "eta_Fl", "eta_EM"] #
    quantityList = ["beta", "eta", "phib", "Omega"]  # , "T"] #
    if a == None:
        rEH = 2  # just use a=0 rEH
    else:
        rEH = calc_rEH(a)
    xlim = (rEH, 3e4)

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim, linestyles=linestyleList, tmax=[2e5] * (len(dirtagList) - oz_num) + [None] * oz_num, rescale=True, time_bin_factor=time_bin_factor)  # tmax 4e5


def compareSpinTimeAverages(quantity="phib", tmax=None, average_factor=2, show_RN22=False):
    matplotlib_settings()
    dirtagList = [
        "030625_a0.0_n4_013124",
        "031125_a0.1_cap_correctly",
        "031125_a0.3_cap_correctly",
        "031125_a0.5_cap_correctly",
        # "031125_a0.7_cap_correctly",
        "031125_a0.9_cap_correctly",
        # "042125_n4_a0.9_bondi_jks2",
        # "041625_n4_a0.9_toriilike_jks2_smth2_reconnect",
        # "2023/122723_n4_onezone_wks0.04/00000",
        "delta/030525_a0.0_oz",
        "delta/030525_a0.5_oz",
        # "delta/030525_a0.9_oz",
        # "030425_a0.5_safe_longtin20",
        # "030725_a0.9_safe_longtin20",
    ]

    if quantity == "eta" or quantity == "phib":
        store_Mdot10 = True

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    labels = ["mz", "oz", "mz_large"]
    for i, dirtag in enumerate(dirtagList):
        print(dirtag)
        pkl_name = glob.glob("../data_products/" + dirtag + "_profiles_all.pkl")
        pkl_name = pkl_name[0]
        with open(pkl_name, "rb") as openFile:
            D = pickle.load(openFile)
            a = get_spin(D)
            rEH = calc_rEH(a)
            if quantity == "phib":
                quantity_arr, _ = readTimeSeries(D, "Phib", rEH, tmax=tmax)
            elif quantity == "eta":
                radius = 5  # 10
                Mdot, _ = readTimeSeries(D, "Mdot", radius, tmax=tmax)
                Edot, _ = readTimeSeries(D, "Edot", radius, tmax=tmax)
                quantity_arr = Mdot - Edot
            mean = np.mean(quantity_arr[int(float(len(quantity_arr)) / average_factor) :])
            if store_Mdot10:
                Mdot_save, _ = readTimeSeries(D, "Mdot", 10, tmax=tmax)
                Mdot_save = np.mean(Mdot_save[int(float(len(Mdot_save)) / average_factor) :])  # TODO: make it a function
            if quantity == "phib":
                mean /= np.sqrt(Mdot_save)
            elif quantity == "eta":
                mean /= Mdot_save
            print("spin of", a, "gets", quantity, "of ", mean)
            is_onezone = D["dump"]["driver/type"] == "kharma"
            large_scale = D["dump"]["rs"] > 100
            if is_onezone:
                marker = "ko"
                label = labels[1]
            else:
                if large_scale:
                    marker = "k+"
                    label = labels[2]
                else:
                    marker = "kx"
                    label = labels[0]
            ax.plot(a, mean, marker, label=label, ms=10)
            if show_RN22 and quantity == "eta":
                phib, _ = readTimeSeries(D, "Phib", rEH, tmax=tmax)
                phib /= np.sqrt(Mdot_save)
                mean_phib = np.mean(phib[int(float(len(phib)) / average_factor) :])
                print(mean_phib)
                if labels[is_onezone] != "__nolegend__":
                    label = r"$\eta_{BZ6}(\phi($" + labels[is_onezone] + "))"
                else:
                    label = "__nolegend__"
                ax.plot(a, eta_BZ6(a, mean_phib), marker.replace("k", "b"), alpha=0.5, ms=10, label=label)
            if labels[is_onezone] != "__nolegend__":
                labels[is_onezone] = "__nolegend__"

    # show RN22 results
    if show_RN22:
        if quantity == "phib":
            a_arr = np.linspace(-1, 1, 100)
            phi_fit = -20.2 * np.power(a_arr, 3.0) - 14.9 * np.power(a_arr, 2.0) + 34.0 * a_arr + 52.6
            ax.plot(a_arr, phi_fit, "b--", label="RN22")
            ax.plot(a_arr, phi_fit - 15, "b:", label="RN22 - 15")

    # formatting
    ax.legend()
    ax.set_xlabel(r"$a_*$")
    ylabel = variableToLabel(quantity)
    ax.set_ylabel(ylabel)
    # ax.set_xlim([0,1])
    if quantity == "phib":
        ax.set_ylim([0, 70])
    elif quantity == "eta":
        ax.set_ylim([-0.05, 0.6])  # 1.2])

    # save plot
    plot_dir = "../plots/"
    output = "../plots/compare_spin_time_averages_" + str(quantity) + ".png"
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print("saved to " + output)


def comparePrescriptions(a=0.5):
    if a == 0.5:
        dirtagList = [
            # "100724_a0.5_oz",
            "010625_a0.5_oz_128",
            # "030325_a0.5_b8n4_safe",
            # "022825_a0.5_b8n4_tchar",
            # "022525_a0.5_b8n4",
            "030325_a0.5_b2n14_safe",
            "022625_a0.5_b2n14_tchar",
            "022525_a0.5_b2n14",
            # "123024_a0.5_oz_reconnect",
            # "010225_a0.5_oz_rdepgmax5"
        ]
        labelList = ["oz_128", "b2_safe", "b2", "b2_n10"]  # "reconnect", "b8_safe", "b8", "b8_n200",
    elif a == 0.9:
        dirtagList = ["010925_a0.9_oz_128", "022625_a0.9_n4_noreconnect", "022625_a0.9_n4"]
        labelList = ["128", "b8_noreconnect", "b8"]  # "reconnect",

    colorList = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(dirtagList)))  # colors for each runs
    linestyleList = ["solid"] * (len(dirtagList))
    plot_dir = "../plots/022625_prescriptions_a" + str(a)

    quantityList = ["Mdot", "rho", "beta", "eta", "eta_Fl", "eta_EM", "u^r", "T"]  # "Omega"
    if a == None:
        rEH = 2  # just use a=0 rEH
    else:
        rEH = calc_rEH(a)
    xlim = (rEH, 3e4)

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim, linestyles=linestyleList, tmax=3e5, rescale=True)  # tmax 4e5


def compareN8Spin(a=None, time_bin_factor=2):
    matplotlib_settings()
    if a is None:
        dirtagList = [
            "030325_a0.0_safe_tchar",
            # "032025_a0.0_toriilike",
            # "030925_a0.1_safe_longtin20",
            # "030925_a0.3_safe_longtin20",
            # "032025_a0.3_toriilike",
            "030425_a0.5_safe_longtin20",
            # "031925_a0.5_toriilike",
            # "030925_a0.7_safe_longtin20",
            # "032025_a0.7_toriilike",
            "030725_a0.9_safe_longtin20",
            # "032025_a0.9_toriilike",
        ]
        labelList = [
            "0.0",
            # "0.1",
            # "0.3",
            "0.5",
            # "0.7",
            "0.9",
        ]
    elif a == 0.5:
        dirtagList = [
            "030425_a0.5_safe_longtin20",
            "031125_a0.5_dirichlet",
            "031125_a0.5_longtin10",
            "031425_a0.5_beta100_rot",
            "031625_a0.5_longtin4",
            "031625_a0.5_toriilike",
            "031925_a0.5_toriilike",
        ]
        labelList = ["fid", "dirichlet", "lt10", "b100", "lt4", "torii", "torii2"]
    elif a == 0.9:
        dirtagList = [
            "030725_a0.9_safe_longtin20",
            "032025_a0.9_toriilike",
        ]
        labelList = ["fid", "torii"]
    colorList = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(dirtagList)))  # colors for each runs
    linestyleList = ["solid"] * (len(dirtagList))
    plot_dir = "../plots/032025_n8spin"
    if a is not None:
        plot_dir += "_a{}".format(a)

    quantityList = ["eta"]  # ["Mdot", "rho", "beta", "eta", "eta_Fl", "eta_EM"] #, "u^r", "Omega"] #
    rEH = 2  # just use a=0 rEH
    xlim = (rEH, 1e8)

    if len(quantityList) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        compareRuns(dirtagList, ["eta_EM"], colorList, labels=["__nolegend__"] * len(dirtagList), plot_dir=plot_dir, xlim=xlim, linestyles=["dashed"] * (len(dirtagList)), tmax=None, rescale=False, time_bin_factor=time_bin_factor, passed_fig_ax=(fig, np.array([ax])))
        compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim, linestyles=linestyleList, tmax=None, rescale=False, time_bin_factor=time_bin_factor, passed_fig_ax=(fig, np.array([ax])))

        # save plot
        os.makedirs(plot_dir, exist_ok=True)
        output = plot_dir + "/" + plot_dir.split("/")[-1] + "_comparison.png"
        plt.savefig(output, bbox_inches="tight")
        plt.close()
        print("saved to " + output)
    else:
        compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim, linestyles=linestyleList, tmax=None, rescale=True, time_bin_factor=time_bin_factor)


def compareN8OldVsNew(time_bin_factor=2):
    dirtagList = [
        # "../../kharma_multizone_analysis/data_products/2023/082423_n8",
        "../../kharma_multizone_analysis/data_products/2023/110223_mhd_wks",
        # "../../kharma_multizone_analysis/data_products/production_runs/072823_beta01_128",
        "../../kharma_multizone_analysis/data_products/011424_n8_locff_smth0.03",
        # "../../kharma_multizone_analysis/data_products/022124_n8_longtin2",
        # "../../kharma_multizone_analysis/data_products/022124_n8_longtin1",
        # "../../kharma_multizone_analysis/data_products/022124_n8_longtin0.5",
        # "../../kharma_multizone_analysis/data_products/010524_a0.1",
        # "021325_a0.0_011424",
        # "022725_a0.0_safe_tchar",
        # "030325_a0.0_safe_tchar",
        # "030925_a0.0_n8_lt10_dirichlet",
        # "030925_a0.0_n8_lt4_dirichlet",
        # "032025_a0.0_toriilike",
        # "040225_a0.0_fofc",
        # "040925_a0.0_fofc_noehbuffer"
        "042125_a0.0_rB2e5_jks2",
        "042425_a0.0_rB2e5_bondi_jks2",
    ]
    labelList = [
        # "cell_082423",
        "cell_cap",
        # "cell_128",
        "cell",
        # "cell_longtin4",
        # "cell_longtin2",
        # "cell_longtin1",
        # "cell_a0.1",
        # "face_safe",
        # "face_lt50",
        # "face_safe_longtin20",
        # "face_safe_longtin10_nocap",
        # "face_safe_longtin4_nocap",
        "face_toriilike",
        # "face_fofc"
        "face_jks",
    ]  # "reconnect", "b8_safe", "b8", "b8_n200",
    colorList = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(dirtagList)))  # colors for each runs
    linestyleList = ["solid"] * (len(dirtagList))
    plot_dir = "../plots/030525_n8a0oldvsnew"

    quantityList = ["Mdot", "rho", "beta", "eta", "eta_Fl", "eta_EM", "u^r", "T"]  # "Omega"
    rEH = 2  # just use a=0 rEH
    xlim = (rEH, 1e8)

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim, linestyles=linestyleList, tmax=None, rescale=True, time_bin_factor=time_bin_factor)


def compareMdotEta(a=None, time_bin_factor=2.0, rescale=False):
    matplotlib_settings()
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    if a is None:
        dirtagList = [
            # "032025_a0.0_toriilike",
            # "032025_a0.3_toriilike",
            # "031925_a0.5_toriilike",
            # "032025_a0.7_toriilike",
            # "032025_a0.9_toriilike",
            "030325_a0.0_safe_tchar",
            "030925_a0.3_safe_longtin20",
            "030425_a0.5_safe_longtin20",
            # "030925_a0.7_safe_longtin20",
            "030725_a0.9_safe_longtin20",
            "030625_a0.0_n4_013124",
            "031125_a0.1_cap_correctly",
            "031125_a0.3_cap_correctly",
            "031125_a0.5_cap_correctly",
            # "031125_a0.7_cap_correctly",
            "031125_a0.9_cap_correctly",
            # "032325_n4a0.0_toriilike",
            # "032525_n4a0.5_toriilike",
            # "032125_n4a0.9_toriilike",
            # "032325_n4a0.9_toriilike_beta1",
            # "031325_a0.9_torus_rn22_noreconnect",
            # "031425_a0.5_torus_rn22_noreconnect",
            # "delta/030525_a0.0_oz",
            # "delta/030525_a0.5_oz",
            # "delta/030525_a0.9_oz",
        ]
    linestyleList = ["solid"] * (len(dirtagList))
    plot_dir = "../plots/032125_n8spin_comparemdoteta"
    if a is not None:
        plot_dir += "_a{}".format(a)

    quantityList = ["Mdot", "eta"]
    rEH = 2  # just use a=0 rEH
    xlim = (rEH, 1e8)

    markerList = ["x", ".", "+", "o"]
    labelList = ["oz", "small", "large"]

    for d, dirtag in enumerate(dirtagList):
        pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"

        with open(pkl_name, "rb") as openFile:
            D = pickle.load(openFile)

        a = get_spin(D)
        r_sonic = D["dump"]["rs"]
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic)[0]
        if D["dump"]["parthenon/job/problem_id"] == "torus":
            is_torus = True
        else:
            is_torus = False
        is_onezone = D["dump"]["driver/type"] == "kharma"

        tDivList, binNumList = setTimeBins(D, 1, time_bin_factor=time_bin_factor)
        mask_list = get_mask(D)

        if is_onezone:
            color = "k"
            label = labelList[0]
            labelList[0] = "__nolegend__"
        else:
            color = "b"
            if rB > 1000:
                label = labelList[2]
                labelList[2] = "__nolegend__"
            else:
                label = labelList[1]
                labelList[1] = "__nolegend__"
        if is_torus:
            marker = markerList[2]
        elif rB > 1000:
            marker = markerList[0]
        else:
            marker = markerList[1]

        for i, quantity in enumerate(quantityList):
            if is_torus and quantity == "Mdot":
                continue
            if quantity == "Mdot":
                radiiList = [5]
            else:
                radiiList = [rB]  # , 10 * rB]
            # colorList = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(radiiList)))
            # if len(colorList) == 1: colorList = [colorList]
            radii, profiles = calcFinalTimeAvg(D, tDivList, binNumList, quantity, perzone_avg_frac=1, mask_list=mask_list, rescale=rescale)
            for j, read_r in enumerate(radiiList):
                i_r = np.argmin(abs(radii[0] - read_r))
                ax[i].plot(a, profiles[0][i_r], marker=marker, markersize=10, label=label, color=color)  # colorList[j])

    # formatting
    for i, quantity in enumerate(quantityList):
        ax[i].set_yscale("log")
        ylabel = variableToLabel(quantity)
        if rescale and "Mdot" in quantity:
            ylabel = ylabel.replace("arb. units", r"$\dot{M}_B$")
        ax[i].set_ylabel(ylabel)
        ax[i].set_xlabel("a")
    ax[0].legend()
    ax[0].set_ylim([1e-2, 5e-1])
    ax[1].set_ylim([8e-3, 1])

    # save plot
    os.makedirs(plot_dir, exist_ok=True)
    output = plot_dir + "/" + plot_dir.split("/")[-1] + "_comparison.png"
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print("saved to " + output)


def compareCoords(time_bin_factor=2):
    linestyleList = None
    dirtagList = [
        "032125_n4a0.9_toriilike",
        # "041625_n4_a0.9_toriilike_jks_reconnect",
        "041625_n4_a0.9_toriilike_jks2_reconnect",
        "041625_n4_a0.9_toriilike_jks2_smth2_reconnect",
        # "041625_n4_a0.9_toriilike_jks2_smth3_reconnect",
        "041625_n4_a0.9_toriilike_jks2_smth5_reconnect",
    ]
    labelList = ["eks", "jks1", "2", "5"]  #
    # dirtagList = [
    #    "040625_a0.9_rB2e3",
    #    "041625_a0.9_rB2e3_jks2",
    #    "041625_a0.9_rB2e3_jks2_smth3",
    #    "041725_a0.9_rB2e3_jks2_smth2",
    #        ]
    # labelList = ["eks", "jks1", "3", "2"]

    colorList = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(dirtagList)))  # colors for each runs
    linestyleList = ["solid"] * (len(dirtagList))
    plot_dir = "../plots/041725_compare_coords"

    quantityList = ["Mdot", "rho", "beta", "eta", "eta_Fl", "eta_EM", "u^r", "T"]  # ["Mdot", "eta", "eta_Fl", "eta_EM"] #"Omega"
    dirtag = dirtagList[0]
    pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"
    with open(pkl_name, "rb") as openFile:
        D = pickle.load(openFile)
    a = get_spin(D)
    rEH = calc_rEH(a)
    xlim = (rEH, 3e4)

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim, linestyles=linestyleList, tmax=[None] * (len(dirtagList)), rescale=True, time_bin_factor=time_bin_factor)  # tmax 4e5


def compareRB(a=0.9, time_bin_factor=2):
    linestyleList = None
    if a == 0.0:
        dirtagList = ["042125_a0.0_rB2e3_jks2", "042125_a0.0_rB2e5_jks2"]
        labelList = ["2000", "2e5"]
        colorList = ["tab:blue", "c", "r", "tab:orange", "m", "g"]  # colors for each runs
    if a == 0.5:
        dirtagList = [
            # "042125_n4_a0.5_jks2",
            "042425_n4_a0.5_bondi_jks2",
            # "042125_a0.5_rB2e3_jks2",
            "delta/042725_a0.5_rB2e3_bondi",
            # "042125_a0.5_rB2e5_jks2",
            "delta/042725_a0.5_rB2e5_bondi",
        ]
        labelList = ["400", "2000", "2e5"]
        colorList = ["black", "tab:blue", "c", "r", "tab:orange", "m", "g"]  # colors for each runs
    if a == 0.9:
        dirtagList = [
            # "040325_n4_a0.9_torrilike_nocap",
            # "042225_n4_a0.9_toriilike_jks2_nocap",
            "042225_n4_a0.9_bondi_jks2_nocap",
            # "042125_n4_a0.9_bondi_jks2",
            # "041725_a0.9_rB2e3_jks2_smth2",
            # "041625_a0.9_rB2e3_jks2_smth3",
            "042325_a0.9_rB2e3_bondi",
            # "041825_a0.9_rB2e5_jks2_smth2"
            # "041725_a0.9_rB2e5_jks2_smth2.5",
            "042825_a0.9_rB2e4_bondi_rot",
            "042325_a0.9_rB2e5_bondi",
            # "042725_a0.9_rB2e5_bondi_rot",
        ]
        labelList = ["400", "2000", "2e4", "2e5"]
        colorList = ["black", "tab:blue", "c", "r", "tab:orange", "m", "g"]  # colors for each runs
    plot_dir = "../plots/042225_compareRB_spin_" + str(a)

    quantityList = ["eta", "beta", "Mdot", "rho"]  # ["Mdot", "rho", "beta", "eta", "eta_Fl", "eta_EM", "Omega", "T"] #"Omega"
    if a == None:
        rEH = 2  # just use a=0 rEH
    else:
        rEH = calc_rEH(a)
    xlim = (rEH, 1e8)

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim, linestyles=linestyleList, tmax=[None] * (len(dirtagList)), rescale=True, time_bin_factor=time_bin_factor)  #


def _main():
    # compareFvcVsOld()
    # compareN4Beta()
    # compareSpin(0.9, time_bin_factor=1.1) #1.5) #
    # compareN8Spin(None, time_bin_factor=2) #0.5)
    # compareN8OldVsNew(2)
    # comparePrescriptions() #0.9)
    # compareMdotEta(time_bin_factor=2, rescale=True)
    # compareCoords(time_bin_factor=2)
    compareRB(0.5, time_bin_factor=1.5)  # 1.05) #1.5) #1.8) #

    tmax = 2e5  # None #4.5e5 #
    # compareSpinTimeAverages('phib', show_RN22=True, tmax=tmax, average_factor=2)
    # compareSpinTimeAverages('eta', show_RN22=True, tmax=tmax, average_factor=1.5)


if __name__ == "__main__":
    _main()
