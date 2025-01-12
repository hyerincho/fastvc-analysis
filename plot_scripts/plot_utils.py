import numpy as np
import pdb

def calc_rEH(a):
    return 1.0 + np.sqrt(1.0 - a**2)

def get_spin(D, verbose=False):
    try:
        dump = D["dump"]
        a = dump["a"]
        if verbose: print("Found spin of", D["dump"]["a"])
    except:
        print("ERROR: cant find spin, now using a=0.5")
        a = 0.5  # for now
    return a

def eta_BZ6(a, phib):
    #kappa = 0.053 # split-monopole
    #kappa = 0.044 # parabolic
    kappa = 0.03
    rEH = calc_rEH(a)
    Omega = a / (2 * rEH)
    return kappa * (4. * np.pi) * np.power(phib * Omega, 2.) * (1. + 1.38 * Omega**2 - 9.2 * Omega**4) / 100 # from percentage to decimals

def readQuantity(dictionary, quantity):
    invert = False
    if quantity == "beta":
        if "inv_beta" in dictionary["quantities"]:
            quantity = "inv_beta"
            invert = True
        else:
            print("inv_beta doesn't exist, so we will stick with beta.")
        quantity_index = dictionary["quantities"].index(quantity)
        profiles = [list[quantity_index] for list in dictionary["profiles"]]
    elif quantity == "Pg":
        if "Pg" in dictionary["quantities"]:
            quantity_index = dictionary["quantities"].index("Pg")
            profiles = [list[quantity_index] for list in dictionary["profiles"]]
        else:
            try:
                gam = dictionary["gam"]
            except:
                gam = 5.0 / 3.0
            quantity_index = dictionary["quantities"].index("u")
            profiles = [np.array(list[quantity_index]) * (gam - 1.0) for list in dictionary["profiles"]]
    elif quantity == "Pb":
        quantity_index = dictionary["quantities"].index("b")
        profiles = [np.array(list[quantity_index]) ** 2 / 2.0 for list in dictionary["profiles"]]
    else:
        # just reading the pre-calculated quantities
        quantity_index = dictionary["quantities"].index(quantity)
        profiles = [list[quantity_index] for list in dictionary["profiles"]]
    return profiles, invert

def readTimeSeries(D, quantity='eta', radius=100, tmax=None):
    quantity_arr = np.array([])
    radii = D["radii"]
    times = D["times"]
    iRead = np.argmin(abs(radii - radius))
    profiles, _ = readQuantity(D, quantity)
    for i, profile in enumerate(profiles):
        quantity_arr = np.concatenate((quantity_arr,[np.array(profile)[iRead]]))
    if tmax is None:
        return quantity_arr, times
    else:
        times = np.array(times)
        if times[-1] <= tmax: print("the time series not reached tmax of {:.3g} yet".format(tmax))
        i_keep = (times < tmax)
        return quantity_arr[i_keep], times[i_keep]
