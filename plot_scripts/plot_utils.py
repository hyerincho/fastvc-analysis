import numpy as np
import pdb
import pickle
import matplotlib.pyplot as plt


def calc_rEH(a):
    return 1.0 + np.sqrt(1.0 - a**2)


def get_spin(D, verbose=False):
    try:
        dump = D["dump"]
        a = dump["a"]
        if verbose:
            print("Found spin of", D["dump"]["a"])
    except:
        print("ERROR: cant find spin, now using a=0.5")
        a = 0.5  # for now
    return a


def eta_BZ6(a, phib, kappa=0.03):
    # kappa = 0.053 # split-monopole
    # kappa = 0.044 # parabolic
    # kappa = 0.03
    rEH = calc_rEH(a)
    Omega = a / (2 * rEH)
    return 0.01 * kappa * (4.0 * np.pi) * np.power(phib * Omega, 2.0) * (1.0 + 1.38 * Omega**2 - 9.2 * Omega**4)  # from percentage to decimals


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


def readTimeSeries(D, quantity="eta", radius=100, tmax=None):
    quantity_arr = np.array([])
    radii = D["radii"]
    times = D["times"]
    iRead = np.argmin(abs(radii - radius))
    profiles, _ = readQuantity(D, quantity)
    for i, profile in enumerate(profiles):
        quantity_arr = np.concatenate((quantity_arr, [np.array(profile)[iRead]]))
    if tmax is None:
        return quantity_arr, times
    else:
        times = np.array(times)
        if times[-1] <= tmax:
            print("the time series not reached tmax of {:.3g} yet".format(tmax))
        i_keep = times < tmax
        return quantity_arr[i_keep], times[i_keep]


def plot_shell_summed(ax, dump, x, var, color="k", lw=5, j_slice=slice(None), label=None, alpha=1):
    var = np.squeeze(np.sum((var * dump["gdet"] * dump["dx2"] * dump["dx3"])[:, j_slice, :], axis=(1, 2)))

    if label is None:
        label = "__nolegend__"
    ax.plot(x, var, color=color, lw=lw, label=label, alpha=alpha)
    ax.plot(x, -var, color=color, lw=lw, ls=":", alpha=alpha)
    return var


def extractQuantity(D, quantity, average_factor=2.0):
    store_Mdot10 = False
    if quantity == "eta" or quantity == "phib":
        store_Mdot10 = True

    if quantity == "eta":
        radius = 5
        quantity_arr, _ = readTimeSeries(D, "Edot", radius)
        quantity_arr2, _ = readTimeSeries(D, "Mdot", radius)
    elif quantity == "eta_EM":
        radius = 5
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
        radius = 5
        quantity_arr, _ = readTimeSeries(D, quantity, radius)

    if store_Mdot10:
        Mdot_save, _ = readTimeSeries(D, "Mdot", 10)
    innermost = np.array(D["zones"]) == 0  # <= 1 #

    if store_Mdot10:  # 0: # divide by the time dependent Mdot  #
        Mdot_save = Mdot_save[innermost]
        Mdot_save = np.mean(Mdot_save[int(float(len(Mdot_save)) / average_factor) :])  # TODO: change this to time criterion by getting indices over t_half
    if quantity == "eta":
        quantity_arr = (quantity_arr2 - quantity_arr) / Mdot_save
    elif quantity == "eta_EM":
        quantity_arr = (-quantity_arr) / Mdot_save
    elif quantity == "phib":
        quantity_arr /= np.sqrt(Mdot_save)

    quantity_arr = quantity_arr[innermost]
    mean = np.mean(quantity_arr[int(float(len(quantity_arr)) / average_factor) :])
    return mean
