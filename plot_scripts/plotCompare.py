import glob

from plotProfiles import *
import oldplotProfiles as old_script
from plot_utils import *
from ylabel_dictionary import *

def compareRuns(dirtags, quantities, colors, labels=None, linestyles=None, plot_dir=None, row=None, figsize=None, xlim=None, passed_fig_ax=None, tmax=None, rescale=False):
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
            ax1d = axes.reshape(-1)
            for j, quantity in enumerate(quantities):
                fig, ax1d[j] = old_script.plotProfiles([pkl_name], quantity, zone_time_average_fraction=0.5, cycles_to_average=0, fig_ax=(fig, ax1d[j]), num_time_chunk=1, color_list=[colors[i]], label_list=[labels[i]])
            fig_ax = (fig, axes)
        else:
            fig_ax = plotProfiles(pkl_name, quantities, plot_dir=plot_dir, perzone_avg_frac=1, num_time_chunk=1, fig_ax=fig_ax, color_list=[colors[i]], label=labels[i], linestyle=linestyles[i], tmax=tmax, rescale=rescale)

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

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim, rescale=True)


def compareSpin(a=0.5):
    linestyleList = None
    if a == None:
        # compare between different spins
        dirtagList = [  # TODO: include oz from old multizone
            #"100724_a0.0_n4",
            #"100724_a0.1_n4",
            #"100724_a0.3_n4",
            #"100724_a0.5_n4",
            #"100724_a0.7_n4",
            #"100724_a0.9_n4",
            "030425_a0.0_b8n4_safe",
            "030425_a0.1_b8n4_safe",
            "030425_a0.3_b8n4_safe",
            "030425_a0.5_b8n4_safe_longtin10",
            "030425_a0.7_b8n4_safe",
            "030425_a0.9_b8n4_safe",
            "2023/122723_n4_onezone_wks0.04/00000",
            "100724_a0.5_oz",
        ]
        colorList = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(dirtagList) - 2))  # colors for each runs
        colorList = np.append(colorList, [colorList[0], colorList[3]], axis=0)  # for onezones
        labelList = ["0", "0.1", "0.3", "0.5", "0.7", "0.9", "0.0_oz", "0.5_oz"]
        linestyleList = ["solid"] * (len(dirtagList) - 2) + ["dashed"] * 2
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

    quantityList = ["Mdot", "rho", "beta", "eta", "eta_Fl", "eta_EM", "u^r", "T"] #"Omega"
    if a == None:
        rEH = 2  # just use a=0 rEH
    else:
        rEH = calc_rEH(a)
    xlim = (rEH, 3e4)

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim, linestyles=linestyleList, tmax=None, rescale=True) # tmax 4e5

def compareSpinTimeAverages(quantity='phib', tmax=None, average_factor=2, show_RN22=False):
    matplotlib_settings()
    dirtagList = [
        "100724_a0.0_n4",
        "100724_a0.1_n4",
        "100724_a0.3_n4",
        "100724_a0.5_n4",
        "100724_a0.7_n4",
        "100724_a0.9_n4",
        "2023/122723_n4_onezone_wks0.04/00000",
        "100724_a0.5_oz",
    ]
    
    if quantity == 'eta' or quantity == 'phib':
        store_Mdot10 = True
        
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    labels = ['mz', 'oz']
    for i, dirtag in enumerate(dirtagList):
        print(dirtag)
        pkl_name = glob.glob("../data_products/" + dirtag + "_profiles_all.pkl")
        pkl_name = pkl_name[0]
        with open(pkl_name, 'rb') as openFile:
            D = pickle.load(openFile)
            a = get_spin(D)
            rEH = calc_rEH(a)
            if quantity == 'phib':
                quantity_arr, _ = readTimeSeries(D, 'Phib', rEH, tmax=tmax)
            elif quantity == 'eta':
                radius = 10
                Mdot, _ = readTimeSeries(D, 'Mdot', radius, tmax=tmax)
                Edot, _ = readTimeSeries(D, 'Edot', radius, tmax=tmax)
                quantity_arr = Mdot - Edot
            mean = np.mean(quantity_arr[int(float(len(quantity_arr))/average_factor):])
            if store_Mdot10:
                Mdot_save, _ = readTimeSeries(D, "Mdot", 10, tmax=tmax)
                Mdot_save = np.mean(Mdot_save[int(float(len(Mdot_save))/average_factor):])# TODO: make it a function
            if quantity == 'phib':
                mean /= np.sqrt(Mdot_save)
            elif quantity == 'eta':
                mean /= Mdot_save
            print("spin of", a, "gets", quantity, "of ", mean)
            is_onezone = (D["dump"]["driver/type"] != "multizone")
            if is_onezone: marker = 'ko'
            else: marker = 'kx'
            ax.plot(a, mean, marker, label = labels[is_onezone], ms=10)
            if show_RN22 and quantity == 'eta':
                phib, _ = readTimeSeries(D, 'Phib', rEH, tmax=tmax)
                phib /= np.sqrt(Mdot_save)
                mean_phib = np.mean(phib[int(float(len(phib))/average_factor):])
                print(mean_phib)
                if labels[is_onezone] != '__nolegend__': label = r"$\eta_{BZ6}(\phi($"+labels[is_onezone]+"))"
                else: label = '__nolegend__'
                ax.plot(a, eta_BZ6(a,mean_phib), marker.replace('k','b'), alpha=0.5, ms=10, label=label)
            if labels[is_onezone] != '__nolegend__': labels[is_onezone] = '__nolegend__'

    # show RN22 results
    if show_RN22:
        if quantity == 'phib':
            a_arr = np.linspace(-1,1,100)
            phi_fit = -20.2 * np.power(a_arr, 3.) - 14.9 * np.power(a_arr, 2.) + 34. * a_arr + 52.6
            ax.plot(a_arr, phi_fit, 'b--', label='RN22')
            ax.plot(a_arr, phi_fit - 15, 'b:', label='RN22 - 15')

    # formatting
    ax.legend()
    ax.set_xlabel(r'$a_*$')
    ylabel = variableToLabel(quantity)
    ax.set_ylabel(ylabel)
    #ax.set_xlim([0,1])
    if quantity == 'phib':
        ax.set_ylim([0,70])

    # save plot
    plot_dir = "../plots/"
    output = "../plots/compare_spin_time_averages_" + str(quantity)  + ".png"
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print("saved to " + output)

def comparePrescriptions(a=0.5):
    if a == 0.5:
        dirtagList = [  
            #"100724_a0.5_oz",
            "010625_a0.5_oz_128",
            #"030325_a0.5_b8n4_safe",
            #"022825_a0.5_b8n4_tchar",
            #"022525_a0.5_b8n4",
            "030325_a0.5_b2n14_safe",
            "022625_a0.5_b2n14_tchar",
            "022525_a0.5_b2n14",
            #"123024_a0.5_oz_reconnect",
            #"010225_a0.5_oz_rdepgmax5"
        ]
        labelList = ["oz_128", "b2_safe", "b2", "b2_n10"] #"reconnect", "b8_safe", "b8", "b8_n200", 
    elif a == 0.9:
        dirtagList = [  
            "010925_a0.9_oz_128",
            "022625_a0.9_n4_noreconnect",
            "022625_a0.9_n4"
        ]
        labelList = ["128", "b8_noreconnect", "b8"] #"reconnect", 

    colorList = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(dirtagList)))  # colors for each runs
    linestyleList = ["solid"] * (len(dirtagList))
    plot_dir = "../plots/022625_prescriptions_a"+str(a)

    quantityList = ["Mdot", "rho", "beta", "eta", "eta_Fl", "eta_EM", "u^r", "T"] #"Omega"
    if a == None:
        rEH = 2  # just use a=0 rEH
    else:
        rEH = calc_rEH(a)
    xlim = (rEH, 3e4)

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim, linestyles=linestyleList, tmax=3e5, rescale=True) # tmax 4e5

def compareN8OldVsNew():
    dirtagList = [
        "../../kharma_multizone_analysis/data_products/2023/082423_n8",
        "../../kharma_multizone_analysis/data_products/2023/110223_mhd_wks",
        "../../kharma_multizone_analysis/data_products/production_runs/072823_beta01_128",
        "../../kharma_multizone_analysis/data_products/011424_n8_locff_smth0.03",
        #"../../kharma_multizone_analysis/data_products/022124_n8_longtin0.5",
        #"../../kharma_multizone_analysis/data_products/022124_n8_longtin1",
        "../../kharma_multizone_analysis/data_products/022124_n8_longtin2",
        "022725_a0.0_safe_tchar",
        "030325_a0.0_safe_tchar",
    ]
    labelList = ["cell_082423", "cell_110223", "cell_128", "cell_011424", "cell_022124_4", "face_safe", "face_safe_longtin20"] #"reconnect", "b8_safe", "b8", "b8_n200", 
    colorList = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(dirtagList)))  # colors for each runs
    linestyleList = ["solid"] * (len(dirtagList))
    plot_dir = "../plots/030525_n8a0oldvsnew"

    quantityList = ['Mdot', 'eta'] #["Mdot", "rho", "beta", "eta", "eta_Fl", "eta_EM", "u^r", "T"] #"Omega"
    rEH = 2  # just use a=0 rEH
    xlim = (rEH, 1e8)

    compareRuns(dirtagList, quantityList, colorList, labels=labelList, plot_dir=plot_dir, xlim=xlim, linestyles=linestyleList, tmax=None, rescale=False)

def _main():
    # compareFvcVsOld()
    #compareN4Beta()
    #compareSpin(None)
    compareN8OldVsNew()
    #comparePrescriptions() #0.9)

    #tmax=4.5e5 #None #
    #compareSpinTimeAverages('phib', show_RN22=True, tmax=tmax, average_factor=1.5)
    #compareSpinTimeAverages('eta', show_RN22=True, tmax=tmax, average_factor=1.5)

if __name__ == "__main__":
    _main()
