# look at a single snapshot profile
import os
from plot_utils import *

def plotSingleTimeProfile(pkl_name, quantity, fnum=None, time=None, cycle=None, formatting=True, nfiles=1):
    matplotlib_settings()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # read file
    with open(pkl_name, "rb") as openFile:
        D = pickle.load(openFile)
    radii = D["radii"]
    zones = D["zones"]
    times = np.array(D["times"])
    cycles = np.array(D["cycles"])
    try:
        rEH = D["dump"]["r_eh"]
    except:
        a = 0  # for now
        rEH = 1.0 + np.sqrt(1.0 - a**2)
    
    if quantity == "eta":
        profilesEdot, invert = readQuantity(D, "Edot")
        profilesMdot, _ = readQuantity(D, "Mdot")
        profilesEdot = np.array(profilesEdot)
        profilesMdot = np.array(profilesMdot)
        i10 = np.argmin(abs(radii - 10))
        profiles = (profilesMdot - profilesEdot) / profilesMdot[:,i10][Ellipsis,np.newaxis]
    else:
        profiles, invert = readQuantity(D, quantity)

    if fnum is not None:
        i = fnum
    elif time is not None:
        i = np.argmin(abs(times) - time)
    elif cycle is not None:
        i = np.where(cycles == cycle)[0][0]

    colors = plt.cm.gnuplot(np.linspace(0.9, 0.3, nfiles))
    for n in range(nfiles):
        print("fnum {}, n={}, i_zone {}".format(i + n, cycles[i+n], zones[i+n]))
        ax.plot(radii, profiles[i + n], color=colors[n])
    
    # Formatting
    if formatting:
        ax.set_xlabel("Radius [$r_g$]")
        ylabel = variableToLabel(quantity)
        ax.set_ylabel(ylabel)
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        xlim = (rEH, ax.get_xlim()[-1])
        ax.set_xlim(xlim)
        if "eta" in quantity and quantity != "beta" and quantity != "etaMdot":
            ax.set_ylim([1e-3, 4])
        elif quantity == "beta":
            ax.set_ylim([1e-3, 10])
        elif quantity == "phib":
            ax.set_ylim([10, 1e6])
        elif quantity == "Mdot":
            ax.set_ylim([3e-4, 5])


    plot_dir = "../plots/single_time_profile"  # common directory
    os.makedirs(plot_dir, exist_ok=True)
    output = plot_dir + "/profile_" + quantity + ".png"  # pdf"
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print("saved to " + output)

def plotTimeProfilesFromDump(dirtag, quantity, fnum, nfiles=1):
    matplotlib_settings()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = plt.cm.gnuplot(np.linspace(0.9, 0.3, nfiles))
    for n in range(nfiles):
        #fname = glob.glob("../data/" + dirtag + "/high_cadence/kastaun_drift_bsqoveru100_fafout/*{:05d}.phdf".format(fnum+n*80))[0]
        fname = glob.glob("../data/" + dirtag + "/*{:05d}.phdf".format(fnum+n))[0]
        dump = pyharm.load_dump(fname, ghost_zones=False)
        print(dump["n_step"])

        if quantity == "etaMdot":
            profile = pyharm.shell_sum(dump, "FE_norho")
        elif quantity == "eta":
            i5 = np.argmin(abs(dump["r1d"] - 5))
            profile = pyharm.shell_sum(dump, "FE_norho")
            profile /= -pyharm.shell_sum(dump, "FM")[i5]

        ax.plot(dump["r1d"], profile, color=colors[n])

    # Formatting
    ax.set_xlabel("Radius [$r_g$]")
    ylabel = variableToLabel(quantity)
    ax.set_ylabel(ylabel)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    xlim = (dump["r_eh"], ax.get_xlim()[-1])
    ax.set_xlim(xlim)
    if "eta" in quantity and quantity != "beta" and quantity != "etaMdot":
        ax.set_ylim([1e-3, 4])
    elif quantity == "beta":
        ax.set_ylim([1e-3, 10])
    elif quantity == "phib":
        ax.set_ylim([10, 1e6])
    elif quantity == "Mdot":
        ax.set_ylim([3e-4, 5])


    plot_dir = "../plots/single_time_profile"  # common directory
    os.makedirs(plot_dir, exist_ok=True)
    output = plot_dir + "/profile_" + quantity + ".png"  # pdf"
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print("saved to " + output)

if __name__ == "__main__":
    dirtag = "052825_a0.9_rB2e5_bondi_eks_largerout"
    #dirtag = "073125_a0.9_rB2e5_drift_bsqoveru100" #diffbetaflr"
    #dirtag = "080425_a0.9_rB2e5_sigma10"
    dirtag = "080425_a0.9_rB2e5_avgneighbor"
    dirtag = "092525_a0.9_rB2e5_mom_cons_test"
    pkl_name = "../data_products/" + dirtag + "_profiles_all.pkl"
    #plotSingleTimeProfile(pkl_name, 'etaMdot', fnum=4753, nfiles=1)
    #plotTimeProfilesFromDump(dirtag, 'eta', fnum=4762, nfiles=10)
    #plotTimeProfilesFromDump(dirtag, 'eta', fnum=580, nfiles=10)
    plotTimeProfilesFromDump(dirtag, 'eta', fnum=830, nfiles=7)
