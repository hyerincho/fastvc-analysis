import os
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle

from matplotlib_settings import *
from ylabel_dictionary import *
from plotProfiles import readQuantity

def plotEvolution(pkl, ax_passed=None, quantity='eta', radius=100, average_factor=2):
    matplotlib_settings()
    plt.rcParams.update({'font.size': 25})
    if ax_passed is None:
        fig, ax = plt.subplots(1,1,figsize=(24,5)) # 8
    else:
        ax = ax_passed

    quantity_arr = np.array([])
    quantity_arr2 = np.array([]) # for eta, this would be Mdot(radius)
    Mdot_save = np.array([])
    if quantity == 'eta' or quantity == 'phib':
        store_Mdot10 = True
    
    with open(pkl, 'rb') as openFile:
        D = pickle.load(openFile)
        radii = D["radii"]
        times = D["times"]
        
        if quantity == 'eta':
            i100 = np.argmin(abs(radii-100))
            profiles, _ = readQuantity(D, 'Edot')
        else:
            profiles, _ = readQuantity(D, quantity)

        if store_Mdot10:
            i10 = np.argmin(abs(radii-10))
            profiles_mdot, _ = readQuantity(D, 'Mdot')

        for i, profile in enumerate(profiles):
            if quantity == 'eta':
                quantity_arr = np.concatenate((quantity_arr,[np.array(profile)[i100]]))
                quantity_arr2 = np.concatenate((quantity_arr2,[np.array(profiles_mdot[i])[i100]]))
                
            if store_Mdot10:
                Mdot_save = np.concatenate((Mdot_save,[np.array(profiles_mdot[i])[i10]]))
    
    if store_Mdot10:
        Mdot_save = np.mean(Mdot_save[int(float(len(Mdot_save))/average_factor):]) # TODO: change this to time criterion by getting indices over t_half
    if quantity == 'eta':
        quantity_arr = (quantity_arr2 - quantity_arr) / Mdot_save

    ax.semilogy(times, quantity_arr, color='k')
    ax.semilogy(times, -quantity_arr, 'k:')
    ax.set_xlim([times[0], times[-1]])
    ax.set_xlabel('t', labelpad=10)

    if quantity == 'eta':
        #ax.set_ylim([1e-4,0.4])
        ax.set_ylim([1e-3,4])
        ax.axhline(2e-2,color='k', lw=3, alpha=0.2) # horizontal line to show 2%
    
    if ax_passed is None:
        fig.tight_layout()
        plt.savefig("../plots/plot_evolution_"+quantity+".png",bbox_inches='tight')
        plt.close(fig)
    else:
        return ax

if __name__ == "__main__":
    dirtag = "100724_a0.5_n4"
    dirtag = "100724_a0.5_opposite_spin"
    #dirtag = "101524_a0.5_beta10_rot"
    pkl_name = "../data_products/"+dirtag+"_profiles_all.pkl"

    plotEvolution(pkl_name) #, quantity='phib')
