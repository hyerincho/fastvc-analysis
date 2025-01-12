import os
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle

from matplotlib_settings import *
from ylabel_dictionary import *
from plotProfiles import readQuantity
from plot_utils import *

def plotEvolution(pkl, ax_passed=None, quantity='eta', average_factor=2):
    matplotlib_settings()
    plt.rcParams.update({'font.size': 25})
    if ax_passed is None:
        fig, ax = plt.subplots(1,1,figsize=(24,5)) # 8
    else:
        ax = ax_passed

    Mdot_save = np.array([])
    store_Mdot10 = False
    if quantity == 'eta' or quantity == 'phib':
        store_Mdot10 = True
    
    with open(pkl, 'rb') as openFile:
        D = pickle.load(openFile)
        times = D["times"]
        
        if quantity == 'eta':
            radius = 5 #10 #0
            quantity_arr, _ = readTimeSeries(D, 'Edot', radius)
            quantity_arr2, _ = readTimeSeries(D, 'Mdot', radius)
        elif quantity == 'phib':
            try: a = get_spin(D)
            except:
                a = 0.5
                print("ERROR: cant find spin, now using a=0.5")
            rEH = calc_rEH(a)
            quantity_arr, _ = readTimeSeries(D, 'Phib', rEH)
        else:
            radius = 10 # TODO?
            quantity_arr, _ = readTimeSeries(D, quantity, radius)

        if store_Mdot10:
            Mdot_save, _ = readTimeSeries(D, "Mdot", 10)

    if store_Mdot10:
        Mdot_save = np.mean(Mdot_save[int(float(len(Mdot_save))/average_factor):]) # TODO: change this to time criterion by getting indices over t_half
    if quantity == 'eta':
        quantity_arr = (quantity_arr2 - quantity_arr) / Mdot_save
    elif quantity == 'phib':
        quantity_arr /= np.sqrt(Mdot_save)

    ax.semilogy(times, quantity_arr, color='k')
    ax.semilogy(times, -quantity_arr, 'k:')
    ax.set_xlim([times[0], times[-1]])
    ax.set_xlabel('t', labelpad=10)
    ylabel = variableToLabel(quantity)
    ax.set_ylabel(ylabel)

    if quantity == 'eta':
        #ax.set_ylim([1e-4,0.4])
        ax.set_ylim([1e-3,4])

    # show and print mean
    if quantity == 'phib' or quantity == 'eta':
        mean = np.mean(quantity_arr[int(float(len(quantity_arr))/average_factor):])
        print("mean of "+quantity + " is {:.5g}".format(mean))
        ax.axhline(mean, color='k', lw=3, alpha=0.2) # horizontal line
    
    if ax_passed is None:
        fig.tight_layout()
        output = "../plots/plot_evolution_"+quantity+".png"
        plt.savefig(output, bbox_inches='tight')
        print("saved to " + output)
        plt.close(fig)
    else:
        return ax

if __name__ == "__main__":
    dirtag = "100724_a0.5_n4"
    #dirtag = "100724_a0.5_n8"
    #dirtag = "100724_a0.5_oz"
    #dirtag = "2023/122723_n4_onezone_wks0.04/00000"
    #dirtag = "100724_a0.5_opposite_spin"
    #dirtag = "101524_a0.5_beta10_rot"
    #dirtag="110424_a0.9_oz"
    #dirtag="100724_a0.0_n8/nordepgmax_bflux0_tchar_combine2out" #nordepgmax_ncycle4000_combine2out" #
    dirtag = "delta/110424_a0.9_oz" #110624_a0.5_n8_nordepgmax" #110624_a0.9_n4_rot0.1" #
    #dirtag="111524_a0.5_torus"
    #dirtag="120824_a0.5_n4_betaflr"
    dirtag="121024_a0.9_n4_reconnect"
    dirtag="123024_a0.5_oz_reconnect"
    dirtag="010525_a0.5_n8_ncycle200"
    #dirtag="010625_a0.5_n8_ncycle200_capped"
    dirtag="010825_a0.5_n8_ncycle400_capped"
    #dirtag="010825_a0.5_n8_ncycle100_capped_rdepgmax"
    #dirtag="010825_a0.5_n8_tchar"
    #dirtag="010925_a0.5_b2n26_reconnect"
    pkl_name = "../data_products/"+dirtag+"_profiles_all.pkl"

    plotEvolution(pkl_name, quantity='Mdot')
    plotEvolution(pkl_name, quantity='phib')
    plotEvolution(pkl_name, quantity='eta')
