import os
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
from functools import partial

from matplotlib_settings import *
from ylabel_dictionary import *
from plotProfiles import readQuantity
import bondi_analytic as bondi
from plot_utils import *


def tg2tcap(t, tcap):
    return t / tcap


def tcap2tg(t, tcap):
    return t * tcap


def plotEvolution(pkl, ax_passed=None, quantity="eta", average_factor=2):
    matplotlib_settings()
    print(pkl)
    plt.rcParams.update({"font.size": 25})
    if ax_passed is None:
        fig, ax = plt.subplots(1, 1, figsize=(24, 5))  # 8
    else:
        ax = ax_passed

    Mdot_save = np.array([])
    store_Mdot10 = False
    if quantity == "eta" or quantity == "phib":
        store_Mdot10 = True

    with open(pkl, "rb") as openFile:
        D = pickle.load(openFile)
        times = D["times"]

        if quantity == "eta":
            radius = 5  # 10 #0
            quantity_arr, _ = readTimeSeries(D, "Edot", radius)
            quantity_arr2, _ = readTimeSeries(D, "Mdot", radius)
        elif quantity == "eta_EM":
            radius = 5  # 10 #0
            quantity_arr, _ = readTimeSeries(D, "Edot_EM", radius)
            quantity_arr2, _ = readTimeSeries(D, "Mdot", radius)
        elif quantity == "phib":
            try:
                a = get_spin(D)
            except:
                a = 0.5
                print("ERROR: cant find spin, now using a=0.5")
            rEH = calc_rEH(a)
            quantity_arr, _ = readTimeSeries(D, "Phib", rEH)
        else:
            radius = 10  # TODO?
            quantity_arr, _ = readTimeSeries(D, quantity, radius)

        if store_Mdot10:
            Mdot_save, _ = readTimeSeries(D, "Mdot", 10)
        dump = D["dump"]

    if store_Mdot10:  # 0: # divide by the time dependent Mdot  #
        Mdot_save = np.mean(Mdot_save[int(float(len(Mdot_save)) / average_factor) :])  # TODO: change this to time criterion by getting indices over t_half
    if quantity == "eta":
        quantity_arr = (quantity_arr2 - quantity_arr) / Mdot_save
    elif quantity == "eta_EM":
        quantity_arr = (-quantity_arr) / Mdot_save
    elif quantity == "phib":
        quantity_arr /= np.sqrt(Mdot_save)

    try:
        if dump["multizone/combine_out"]:
            base = dump["multizone/base"]
            offset = int(np.ceil(np.log(8.0) / np.log(base))) - 1
            nzones_eff = dump["Params"]["Multizone/nzones_eff"]
            rout = np.power(base, nzones_eff + offset - 1)  # dump["multizone/combine_out_radius"]
            # rB = bondi.get_quantity_for_rarr([1], 'RB', rs=dump["bondi/rs"])[0]
            tcap = np.power(rout, 3.0 / 2)  # rout / np.sqrt(1. / rout + 1. / rB)
            print("r_cap {:.5g} and t_cap {:.5g}".format(rout, tcap))
            secax = ax.secondary_xaxis("top", functions=(partial(tg2tcap, tcap=tcap), partial(tcap2tg, tcap=tcap)))
            secax.set_xlabel(r"$t/r_{\rm cap}^{3/2}$")
    except:
        print("dump file not found, not adding second x axis")

    ax.semilogy(times, quantity_arr, color="k")
    ax.semilogy(times, -quantity_arr, "k:")
    ax.set_xlim([times[0], times[-1]])
    ax.set_xlabel("t", labelpad=10)
    ylabel = variableToLabel(quantity)
    ax.set_ylabel(ylabel)

    if quantity == "eta" or quantity == "eta_EM":
        # ax.set_ylim([1e-4,0.4])
        ax.set_ylim([1e-3, 4])

    # show and print mean
    if quantity == "phib" or quantity == "eta":
        mean = np.mean(quantity_arr[int(float(len(quantity_arr)) / average_factor) :])
        print("mean of " + quantity + " is {:.5g}".format(mean))
        ax.axhline(mean, color="k", lw=3, alpha=0.2)  # horizontal line

    if ax_passed is None:
        fig.tight_layout()
        output = "../plots/plot_evolution_" + quantity + ".png"
        plt.savefig(output, bbox_inches="tight")
        print("saved to " + output)
        plt.close(fig)
    else:
        return ax


if __name__ == "__main__":
    dirtag = "100724_a0.5_n4"
    # dirtag = "100724_a0.5_n8"
    # dirtag = "100724_a0.5_oz"
    # dirtag = "2023/122723_n4_onezone_wks0.04/00000"
    dirtag = "delta/110624_a0.0_oz_128"
    # dirtag="111524_a0.5_torus"
    # dirtag="022625_a0.5_safe"
    # dirtag="022625_a0.9_n4"
    # dirtag="022725_a0.0_safe_tchar"
    # dirtag="022725_a0.0_safe_nc800"
    # dirtag="022825_a0.5_b8n4_tchar"
    # dirtag="022825_a0.5_b8n4_nc2"
    dirtag = "022825_a0.5_n8_rdepgmax"
    dirtag = "030225_a0.0_b2_tchar_normal1dw"
    dirtag = "030325_a0.5_rdepgmax_nodelrhocap"
    dirtag = "030325_a0.0_safe_tchar"
    dirtag = "030325_a0.9_oz_128"
    dirtag = "030425_a0.5_rdepgmax5"
    dirtag = "030425_a0.5_safe_longtin20"
    dirtag = "030425_a0.9_b8n4_safe"
    dirtag = "030425_a0.5_b8n4_safe_longtin10"
    dirtag = "030425_a0.0_b2n14_safe"
    dirtag = "030525_a0.0_safe_nocap"
    pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"

    average_factor = 1.2  # 1.5 #2 #
    plotEvolution(pkl_name, quantity="Mdot", average_factor=average_factor)
    plotEvolution(pkl_name, quantity="phib", average_factor=average_factor)
    plotEvolution(pkl_name, quantity="eta", average_factor=average_factor)
    # plotEvolution(pkl_name, quantity='eta_EM', average_factor=average_factor)
