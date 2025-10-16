from plot_utils import *

def getThProfile(dirtag, quantity, radius=None, zones_to_av=1, tmax=None, average_factor=1.25):
    matplotlib_settings()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    files = sorted(glob.glob("../data/" + dirtag + "/*out*.phdf"))[::-1]
    
    # basic information
    dump = pyharm.load_dump(files[0])
    oz = (dump["driver/type"] == "kharma")
    ncycle_per_zone = dump["multizone/ncycle_per_zone"]
    long_t_in = dump["multizone/long_t_in"]
    base = dump["multizone/base"]
    nzeff = dump["Params"]["Multizone/nzones_eff"]
    r_sonic = dump["rs"]
    mdot = dump["mdot"]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot)[0]
    tB = np.power(rB, 3./2)

    
    # defaults
    sum_signed=0; save_Mdot=False; rescale=False; negate=False
    if radius is None:
        radius = rB # Bondi radius
    if tmax is None:
        tmax = dump["t"] / tB
        print("tmax set to {:.5g}".format(tmax))
    if quantity == "eta":
        q_label = "FE_norho"
        save_Mdot = True
    elif quantity == "eta_K":
        q_label = "FE_PAKE"
        save_Mdot = True
    elif quantity == "eta_T":
        q_label = "FE_EN"
        save_Mdot = True
    elif "Mdot" in quantity:
        q_label = "FM"
        rescale = True
        if quantity == "Mdot_out":
            sum_signed = 1 # sum only positive
        elif quantity == "Mdot_in":
            sum_signed = -1
            negate = True
        else:
            negate = True
    zone_save = int(np.floor(np.log(radius)/np.log(base) - 0.5)) # zone number corresponding to the radius
    i5 = np.argmin(abs(dump["r1d"] - 5))
    irB = np.argmin(abs(dump["r1d"] - rB))
    isave = np.argmin(abs(dump["r1d"] - radius))
    i_slice = (slice(isave, isave + zones_to_av), slice(None), slice(None))
    print("averaging from {:.3g} to {:.3g}".format(dump["r1d"][isave], dump["r1d"][isave+zones_to_av]))

    # initialization
    avged = 0.
    rho_save = 0.
    num_sum = 0
    first_time = -1
    last_time = tmax * tB
    
    for fname in files:
        dump = pyharm.load_dump(fname, ghost_zones=False)
        iwv = dump["Params"]["Multizone/i_within_vcycle"]
        i_zone = abs(iwv - (nzeff - 1))

        if dump["t"] < tmax * tB / average_factor:
            break
        elif dump["t"] <= tmax * tB:
            if i_zone == zone_save:
                to_sum = np.copy(dump[q_label])
                if sum_signed > 0:
                    to_sum[to_sum < 0] = 0
                elif sum_signed < 0:
                    to_sum[to_sum > 0] = 0
                temp = np.sum((to_sum * dump["gdet"] * dump["dx3"])[i_slice], axis=(0,2))

                # any rescaling or normalizations
                if save_Mdot:
                    Mdot5 = -pyharm.shell_sum(dump, "FM")[i5]
                    temp /= Mdot5
                if rescale:
                    rho_save += thphi_average(dump, "rho", mass_weight=False, pole_pad=1)[irB] # the method used in Cho+24
                if negate:
                    temp *= -1
                
                if first_time < 0 or dump["t"] < first_time:
                    first_time = dump["t"]

                # add to averaged
                avged += temp
                num_sum += 1
    
    # take the mean
    avged /= (num_sum * zones_to_av)
    if rescale:
        rho_save /= num_sum
        Mdot_analytic = bondi.get_quantity_for_rarr([rB], "Mdot", rs=r_sonic, mdot=mdot)[0]
        rho_analytic = bondi.get_quantity_for_rarr([100 * rB], "rho", rs=r_sonic, mdot=mdot)[0]
        print(rho_analytic / (Mdot_analytic * rho_save))
        avged *= rho_analytic / (Mdot_analytic * rho_save)
    if quantity == "eta":
        print("Final sum of eta is {:.5g}".format(np.sum(avged * dump["dx2"])))
    elif "Mdot" in quantity:
        print("Final sum of " + quantity + "/MdotB is {:.5g}".format(np.sum(avged * dump["dx2"])))

    print(first_time / tB, last_time / tB)

    ax.plot(dump["th1d"], avged, 'k')
    if np.any(avged < 0.0):
        ax.plot(dump["th1d"], -avged, 'k:')
    ax.set_xlabel(r'$\theta$')
    ylabel = variableToLabel(quantity)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')

    # save plot
    output = "../plots/th_" + quantity + ".png"
    plt.savefig(output,bbox_inches='tight')
    print("saved to " + output)
    plt.close()

    # save file
    output = output.replace(".png", ".txt")
    np.savetxt(output, np.array([dump["th1d"], avged]).T)

def combine_all_th_profile():
    matplotlib_settings()
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    quantities = ["eta", "eta_T", "eta_K", "Mdot_out", "Mdot_in", "Mdot"]
    colors = ['k', 'm', 'g', 'r', 'b', 'k']
    list_of_arrays = [None for _ in range(len(quantities))]
    for i, q in enumerate(quantities):
        fname = "../plots/th_" + q + ".txt"
        tharr, qarr = (np.loadtxt(fname)).T
        nx2 = len(tharr)
        list_of_arrays[i] = (qarr[:nx2//2] + np.flip(qarr[nx2//2:]))/2.
        if "eta" in q: j = 0
        elif "Mdot" in q: j = 1
        ax[j].plot(tharr[:nx2 //2], list_of_arrays[i], colors[i], label=q)

    for j in range(2):
        ax[j].set_yscale('log')
        ax[j].set_xlabel(r'$\theta$')
        ax[j].legend()
    ax[0].set_title(r'$\frac{dE}{\dot{M}\,dt\,d\theta}$')
    ax[1].set_title(r'$\frac{dM}{\dot{M}_B\,dt\,d\theta}$')
    
    # save plot
    output = "../plots/th_profiles.png"
    plt.savefig(output, bbox_inches='tight')
    print("saved to " + output)
    plt.close()
    
    # save file
    output = output.replace(".png", ".txt")
    list_of_arrays = np.array(list_of_arrays).T
    np.savetxt(output, np.c_[tharr[:nx2//2], list_of_arrays], header="theta,"+",".join(quantities))
    


if __name__ == "__main__":
    dirtag = "052825_a0.9_rB2e5_bondi_eks_largerout"
    #dirtag = "043025_a0.9_rB2e3_bondi_eks"
    #dirtag = "091525_a0.9_rB2e3_normal-recovery"
    #dirtag = "092525_a0.9_rB2e3_mom_cons"
    dirtag = "092525_a0.9_rB2e5_mom_cons_test"
    #combine_all_th_profile()
    for q in ["eta", "eta_T", "eta_K", "Mdot_out", "Mdot_in", "Mdot"]:
        getThProfile(dirtag, q,average_factor=1.25,zones_to_av=1) #, tmax=700) #, radius=5) #, tmax=140)
