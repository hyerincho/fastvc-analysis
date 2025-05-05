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


def plotEvolution(pkl, ax_passed=None, quantity="eta", average_factor=2, xaxis_t=False):
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
        elif "Omega" in quantity:
            temp = quantity.replace("Omega", "")
            if len(temp) > 0:
                radius = float(temp)
            else:
                radius = 5  # 10 #50
            quantity_arr, _ = readTimeSeries(D, "Omega", radius)
            quantity_arr *= np.power(radius, 3.0 / 2)
        else:
            radius = 5  # 10  # TODO?
            quantity_arr, _ = readTimeSeries(D, quantity, radius)

        if store_Mdot10:
            Mdot_save, _ = readTimeSeries(D, "Mdot", 10)
        dump = D["dump"]
        innermost = np.array(D["zones"]) == 0  # <= 1 #

    if 0:  # store_Mdot10:  #  divide by the time dependent Mdot  #
        if not xaxis_t:
            Mdot_save = Mdot_save[innermost]
        Mdot_save = np.mean(Mdot_save[int(float(len(Mdot_save)) / average_factor) :])  # TODO: change this to time criterion by getting indices over t_half
    if quantity == "eta":
        quantity_arr = (quantity_arr2 - quantity_arr) / Mdot_save
    elif quantity == "eta_EM":
        quantity_arr = (-quantity_arr) / Mdot_save
    elif quantity == "phib":
        quantity_arr /= np.sqrt(Mdot_save)

    try:
        if xaxis_t:
            base = dump["multizone/base"]
            offset = int(np.ceil(np.log(8.0) / np.log(base))) - 1
            nzones_eff = dump["Params"]["Multizone/nzones_eff"]
            rout = np.power(base, nzones_eff + offset - 1)  # dump["multizone/combine_out_radius"]
            rB = bondi.get_quantity_for_rarr([1], "RB", rs=dump["bondi/rs"])[0]
            tcap = rout / np.sqrt(1.0 / rout + 1.0 / rB)
            print("r_cap {:.5g} and t_cap {:.5g}".format(rout, tcap))
            secax = ax.secondary_xaxis("top", functions=(partial(tg2tcap, tcap=tcap), partial(tcap2tg, tcap=tcap)))
            secax.set_xlabel(r"$t/r_{\rm cap}^{3/2}$")
    except:
        print("dump file not found, not adding second x axis")
    if xaxis_t:
        xaxis = times
        ax.semilogy(xaxis, quantity_arr, color="k")
        ax.semilogy(xaxis, -quantity_arr, "k:")
        ax.set_xlabel("t", labelpad=10)
    else:
        xaxis = np.arange(len(quantity_arr))
        ax.semilogy(xaxis[innermost], quantity_arr[innermost], "k.")
        ax.set_xlabel("output #", labelpad=10)
    ax.set_xlim([xaxis[0], xaxis[-1]])
    ylabel = variableToLabel(quantity)
    ax.set_ylabel(ylabel)

    if quantity == "eta" or quantity == "eta_EM":
        # ax.set_ylim([1e-4,0.4])
        ax.axhline(1, color="k", linestyle=":")
        ax.set_ylim([1e-3, 4])

    # show and print mean
    if quantity == "phib" or quantity == "eta":
        if not xaxis_t:
            quantity_arr = quantity_arr[innermost]
        mean = np.mean(quantity_arr[int(float(len(quantity_arr)) / average_factor) :])
        print("mean of " + quantity + " is {:.5g}".format(mean))
        ax.axhline(mean, color="k", lw=3, alpha=0.2)  # horizontal line
        # ax.set_yscale('linear')
        # ax.set_ylim([0, 3 * mean])

    if ax_passed is None:
        fig.tight_layout()
        output = "../plots/plot_evolution_" + quantity + ".png"
        plt.savefig(output, bbox_inches="tight")
        print("saved to " + output)
        plt.close(fig)
    else:
        return ax


def plotEvolutionMultipanel(pkl, quantities=["Mdot", "eta", "phib"], average_factor=1.2, xaxis_t=True):
    plt.rcParams.update({"font.size": 25})
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


if __name__ == "__main__":
    dirtag = "100724_a0.5_n4"
    # dirtag = "100724_a0.5_n8"
    # dirtag = "100724_a0.5_oz"
    # dirtag = "2023/122723_n4_onezone_wks0.04/00000"
    dirtag = "030325_a0.9_oz_128"
    dirtag = "delta/030525_a0.5_oz"
    # dirtag = "030425_a0.5_safe_longtin20"
    # dirtag = "030625_a0.0_n4_013124"
    dirtag = "030725_a0.9_safe_longtin20"
    # dirtag = "030925_a0.7_safe_longtin20"
    # dirtag = "031125_a0.5_dirichlet"
    dirtag = "032025_a0.9_toriilike"
    # dirtag = "delta/032025_a0.5_oz_clearangle"
    # dirtag="032125_torus_tegan"
    # dirtag="033025_a0.9_fofc"
    # dirtag="032425_torus"
    # dirtag="040225_n4_a0.0_fofc"
    # dirtag="040325_a0.9_rB1e4"
    # dirtag="040325_n4_a0.9_torrilike_nocap"
    # dirtag="040325_a0.9_n8_torus"
    # dirtag="040625_a0.9_rB2e3"
    # dirtag="040725_a0.9_rB2e3_cap"
    # dirtag="040825_n4_a0.9_torrilike_nocap_nc8000"
    dirtag = "041825_a0.9_rB2e5_jks2_smth2"
    # dirtag="042125_n4_a0.5_jks2"
    # dirtag="042125_a0.9_oz_jks"
    # dirtag="042225_n4_a0.9_toriilike_jks2_nocap"
    # dirtag="042325_n4_a0.9_tl_uphi0"
    dirtag = "042325_a0.9_rB2e3_bondi"
    # dirtag="042325_a0.9_rB2e3_capRB"
    # dirtag="042425_a0.0_rB2e5_bondi_jks2"
    # dirtag="042725_a0.9_rB2e5_bondi_rot"
    # dirtag="042825_a0.9_rB2e4_bondi_rot"
    # dirtag="043025_n4_a0.9_bondi_bflux0"
    # dirtag="043025_a0.9_rB2e3_bondi_eks"
    # dirtag="043025_n4_a0.9_bondi_rot-"
    # dirtag="delta/042725_a0.5_rB2e5_bondi"
    pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"

    average_factor = 1.5  # 2 #
    quantities = ["Mdot", "eta", "phib"]  # ["Mdot", 'eta', 'Omega5', 'Omega50', 'Omega500', 'phib'] #
    plotEvolutionMultipanel(pkl_name, quantities=quantities, average_factor=average_factor, xaxis_t=False)  # True) #
    # plotEvolution(pkl_name, quantity="Mdot", average_factor=average_factor)
    # plotEvolution(pkl_name, quantity="phib", average_factor=average_factor)
    # plotEvolution(pkl_name, quantity="eta", average_factor=average_factor)
    # plotEvolution(pkl_name, quantity='eta_EM', average_factor=average_factor)
