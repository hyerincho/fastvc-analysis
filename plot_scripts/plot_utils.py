import numpy as np
import pdb
import pickle
import matplotlib.pyplot as plt
import glob
from astropy import units as u
from astropy import constants as const
import pyharm

from matplotlib_settings import *
import bondi_analytic as bondi
from ylabel_dictionary import *

c = const.c
G = const.G

def rg2pc(r,M=6.5e9*u.Msun):
    rg = G*M/c**2
    return (r*rg).to('pc').value

def pc2rg(R,M=6.5e9*u.Msun):
    rg = G*M/c**2
    return (R*u.pc/rg).to('')

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


def eta_BZ6(a, phib, kappa=0.05):
    # kappa = 0.053 # split-monopole
    # kappa = 0.044 # parabolic
    # kappa = 0.03
    rEH = calc_rEH(a)
    Omega = a / (2 * rEH)
    return kappa / (4.0 * np.pi) * np.power(phib * Omega, 2.0) * (1.0 + 1.38 * Omega**2 - 9.2 * Omega**4)  # from percentage to decimals

def thphi_average(dump,quantity,sum_instead=False,mass_weight=True, hemisphere=None, pole_pad=0):
  if isinstance(quantity,str):
    to_average=np.copy(dump[quantity])
  else:
    to_average=np.copy(quantity)
  
  if mass_weight:
    to_average *= dump['rho']

  if hemisphere is None:
      j_slice = slice(pole_pad, dump["n2"] - pole_pad)
  elif hemisphere == "n":
      j_slice = slice(pole_pad, dump["n2"] // 2)
  elif hemisphere == "s":
      j_slice = slice(dump["n2"] // 2, dump["n2"] - pole_pad)

  if sum_instead:
      return pyharm.shell_sum(dump,to_average, j_slice=j_slice)
  else:
      if mass_weight:
          return pyharm.shell_sum(dump,to_average, j_slice=j_slice)/pyharm.shell_sum(dump,dump["rho"], j_slice=j_slice)
      else:
          if dump['n3']>1: #3d
              return pyharm.shell_avg(dump,to_average, j_slice=j_slice)
          else:
              return np.mean(to_average[:,j_slice,:],axis=1)

def phi_average(dump,quantity,sum_instead=False,mass_weight=True):
  if isinstance(quantity,str):
    to_average=np.copy(dump[quantity])
  else:
    to_average=np.copy(quantity)
  
  if mass_weight: weight = dump["rho"]
  else: weight = dump["1"]

  if sum_instead:
      return np.sum(to_average, axis=2)
  else:
      if dump['n3']>1:
        return np.sum(to_average * weight * np.sqrt(dump['gcov'][3,3]) * dump['dx3'], axis=2) / np.sum(weight * np.sqrt(dump['gcov'][3,3]) * dump['dx3'],axis=2)
      else:
        return to_average

def phi_dispersion(dump,quantity):
  if isinstance(quantity,str):
    to_average=np.copy(dump[quantity])
  else:
    to_average=np.copy(quantity)
  
  if dump['n3']>1:
    return np.std(to_average, axis=2)

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
    elif quantity == "etaMdot":
        quantity_index = dictionary["quantities"].index("Edot")
        quantity_index2 = dictionary["quantities"].index("Mdot")
        profiles = [(np.array(list[quantity_index2]) - np.array(list[quantity_index])) for list in dictionary["profiles"]]
    elif quantity == "eta":
        quantity_index = dictionary["quantities"].index("Edot")
        quantity_index2 = dictionary["quantities"].index("Mdot")
        i5 = np.argmin(abs(dictionary["radii"] - 5))
        Mdot_normalize = np.array([list[quantity_index2] for list in dictionary["profiles"]])
        zones = np.array(dictionary["zones"])
        Mdot_normalize = [Mdot_normalize[np.argwhere(zones[:i] == 0)[-1,0]] if zones[i]!=0 and i > zones[0] else Mdot_normalize[i] for i in range(len(Mdot_normalize))]
        #for i, zone in enumerate(dictionary["zones"]):
        #    if zone != 0 and i > dictionary["zones"][0]:
        #        pdb.set_trace()
        #        print(i, np.argwhere(np.array(dictionary["zones"])[:i] == 0)[-1,0])

        profiles = [(np.array(list[quantity_index2]) - np.array(list[quantity_index])) / np.array(list[quantity_index2])[i5] for list in dictionary["profiles"]]
    elif quantity == "phib":
        quantity_index = dictionary["quantities"].index("Phib")
        quantity_index2 = dictionary["quantities"].index("Mdot")
        i5 = np.argmin(abs(dictionary["radii"] - 5))
        Mdot_normalize = np.array([list[quantity_index2] for list in dictionary["profiles"]])
        zones = np.array(dictionary["zones"])
        Mdot_normalize = [Mdot_normalize[np.argwhere(zones[:i] == 0)[-1,0]] if zones[i]!=0 and i > zones[0] else Mdot_normalize[i] for i in range(len(Mdot_normalize))]
        profiles = [np.array(list[quantity_index]) / np.sqrt(np.array(list[quantity_index2])[i5]) for list in dictionary["profiles"]]
    else:
        # just reading the pre-calculated quantities
        quantity_index = dictionary["quantities"].index(quantity)
        profiles = [list[quantity_index] for list in dictionary["profiles"]]
    return profiles, invert


def readTimeSeries(D, quantity, radius=100, tmax=None):
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
        r_sonic = D["dump"]["rs"]
        mdot = D["dump"]["mdot"]
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot)[0]
        tB = np.power(rB, 3./2)
        if times[-1] <= tmax * tB:
            print("the time series not reached tmax of {:.3g} yet".format(tmax * tB))
        i_keep = times < tmax * tB
        return quantity_arr[i_keep], times[i_keep]

def processTimeSeries(D, quantity, use_Mdot_mean=True, average_factor=2., rescale=False, tmax=None, radius=None):
    store_Mdot10 = False
    if quantity == "eta" or quantity == "phib" or quantity == "eta_EM":
        store_Mdot10 = True

    # set radius
    if radius is None:
        if quantity in ["phib", "Mdot"]:
            try:
                a = get_spin(D)
            except:
                a = 0.5
                print("ERROR: cant find spin, now using a=0.5")
            radius = calc_rEH(a)
        elif "Omega" in quantity:
            temp = quantity.replace("Omega", "")
            if len(temp) > 0:
                radius = float(temp)
            else:
                radius = 5  # 10 #50
        elif "u^r" in quantity or "u^th" in quantity or "u^phi" in quantity: radius = 10
        else: radius = 5

    if quantity == "eta":
        quantity_arr, times = readTimeSeries(D, "Edot", radius, tmax)
        quantity_arr2, _ = readTimeSeries(D, "Mdot", radius, tmax)
    elif quantity == "etaB":
        quantity_arr, times = readTimeSeries(D, "Edot", radius, tmax)
        quantity_arr2, _ = readTimeSeries(D, "Mdot", radius, tmax)
    elif quantity == "eta_EM":
        quantity_arr, times = readTimeSeries(D, "Edot_EM", radius, tmax)
        quantity_arr2, _ = readTimeSeries(D, "Mdot", radius, tmax)
    elif quantity == "phib":
        quantity_arr, times = readTimeSeries(D, "Phib", radius, tmax)
    elif "Omega" in quantity:
        quantity_arr, times = readTimeSeries(D, "Omega", radius, tmax)
        quantity_arr *= np.power(radius, 3.0 / 2)
    else:
        quantity_arr, times = readTimeSeries(D, quantity, radius, tmax)

    if store_Mdot10:
        #Mdot_save, _ = readTimeSeries(D, "Mdot", 10, tmax)
        Mdot_save, _ = readTimeSeries(D, "Mdot", 5, tmax)
    innermost = np.array(D["zones"]) == 0  # <= 1 #

    r_sonic = D["dump"]["rs"]
    mdot = D["dump"]["mdot"]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot)[0]
    tB = np.power(rB, 3./2)
    if tmax is None: last_time = times[-1]
    else: last_time = tmax * tB

    if rescale and (quantity == "Mdot" or quantity == "etaB"):
        print("t={:.5g}-{:.5g}".format(last_time/average_factor, last_time))
        Mdot_analytic = bondi.get_quantity_for_rarr([rB], "Mdot", rs=r_sonic, mdot=mdot)[0]
        rho_analytic = bondi.get_quantity_for_rarr([100 * rB], "rho", rs=r_sonic, mdot=mdot)[0]
        rho_save, _ = readTimeSeries(D, "rho", rB, tmax)
        zones = np.array(D["zones"][:len(rho_save)])
        rB_zone = int(np.floor(np.log(rB) / np.log(8) - 0.5))
        i_keep = np.argwhere((times < last_time) & (times > last_time / average_factor) & (zones == rB_zone))
        rho_save = np.mean(rho_save[i_keep])
        print("rho_save={:.5g}, factor={:.5g}".format(rho_save, rho_analytic / (Mdot_analytic * rho_save)))
        Mdot_analytic *= rho_save / rho_analytic
        if quantity == "Mdot": quantity_arr /= Mdot_analytic

    if store_Mdot10 and use_Mdot_mean:
        i_keep = np.argwhere((times < last_time) & (times > last_time / average_factor) & (innermost[:len(times)]))
        Mdot_save = Mdot_save[i_keep]
        Mdot_save = np.mean(Mdot_save)
        print(Mdot_save)
    if quantity == "eta":
        quantity_arr = (quantity_arr2 - quantity_arr) / Mdot_save
    elif quantity == "eta_EM":
        quantity_arr = (-quantity_arr) / Mdot_save
    elif quantity == "etaB":
        quantity_arr = (quantity_arr2 - quantity_arr) / Mdot_analytic
        if rescale:
            quantity_arr
    elif quantity == "phib":
        quantity_arr /= np.sqrt(Mdot_save)

    return quantity_arr, times

def plot_shell_summed(ax, dump, x, var, color="k", lw=5, j_slice=slice(None), label=None, alpha=1, normalize=None, inverse=False):
    if normalize is not None:
        if np.shape(normalize) == np.shape(var):
            normalize = np.squeeze(np.sum((normalize * dump["gdet"] * dump["dx2"] * dump["dx3"])[:, j_slice, :], axis=(1, 2)))
    var = np.squeeze(np.sum((var * dump["gdet"] * dump["dx2"] * dump["dx3"])[:, j_slice, :], axis=(1, 2)))
    if normalize is not None:
        var /= normalize
    if inverse: var = 1. / var

    if label is None:
        label = "__nolegend__"
    ax.plot(x, var, color=color, lw=lw, label=label, alpha=alpha)
    ax.plot(x, -var, color=color, lw=lw, ls=":", alpha=alpha)
    return var

def extractQuantity(D, quantity, tmax=None, average_factor=2.0, return_mean=True, use_Mdot_mean=True, verbose=False):
    # extract steady state of the quantity
    r_sonic = D["dump"]["rs"]
    mdot = D["dump"]["mdot"]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot)[0]
    tB = np.power(rB, 3./2)
    
    innermost = np.array(D["zones"]) == 0  # <= 1 #
    times = np.array(D["times"])[innermost]
    i_keep = None
    if tmax is not None:
        if times[-1] <= tmax * tB:
            print("the time series not reached tmax of {:.3g} yet".format(tmax * tB))
        else:
            i_keep = times < tmax * tB
            times = times[i_keep]
    times = times[int(float(len(times)) / average_factor) :]
    if quantity == "time":
        return times / tB

    quantity_arr, _ = processTimeSeries(D, quantity, use_Mdot_mean=use_Mdot_mean, average_factor=average_factor)
    quantity_arr = quantity_arr[innermost]
    if tmax is not None and i_keep is not None:
        quantity_arr = quantity_arr[i_keep]
    
    quantity_arr = quantity_arr[int(float(len(quantity_arr)) / average_factor) :]
    if verbose: 
        print("t={:.5g}-{:.5g}".format(times[0], times[-1]))
    if return_mean: return np.mean(quantity_arr)
    else: return quantity_arr

def extract_rth_info(fnames, quantity, radii, num_files=-1, which="phiav"):
    # each dumps midplane slice
    #fnames = sorted(glob.glob('../data/' + dirtag + '/*out0.*.phdf'))
    if num_files == -1: num_files = len(fnames) // 2
    if which == 'phiweight': dtype = 'complex_'
    else: dtype = float
    quantity_arr = np.zeros((len(radii), num_files), dtype=dtype)
    for j, fname in enumerate(fnames[-num_files:]): # -1000
        dump = pyharm.load_dump(fname, ghost_zones=False)
        if which == "phiav": temp = phi_average(dump, quantity)
        elif which == "phistd": temp = phi_dispersion(dump, quantity)
        elif which == "max": temp = np.max(dump[quantity], axis=(1,2))
        elif which == "avg": temp = thphi_average(dump, quantity)
        elif which == "signedavg":
            temp = np.copy(dump[quantity])
            temp[:,:np.shape(temp)[1] // 2,:] *= -1 # north hemisphere is multiplied a negative sign
            temp = thphi_average(dump, temp) #, mass_weight=False)
        elif which == "phiweight":
            temp = np.copy(dump[quantity])
            temp = temp * np.exp(1j * dump["phi"])
            temp = thphi_average(dump, temp, hemisphere='n') # only focus on north hemisphere for now
            #temp = np.abs(temp) # look at abs value for now
        for i, radius in enumerate(radii):
            i_r = np.argmin(abs(dump["r1d"] - radius))
            if "phi" in which and which != "phiweight": quantity_arr[i][j] = temp[i_r, dump["nx2"]//2]
            else: quantity_arr[i][j] = temp[i_r]
            if quantity == "Omega": quantity_arr[i][j] *= np.power(dump["r1d"][i_r], 3./2)
    return quantity_arr

def extract_shellsum(fnames, quantity, radii, num_files=-1):
    # shell sum directly from each dumps
    if num_files == -1: num_files = len(fnames) // 2
    quantity_arr = np.zeros((len(radii), num_files))

    Mdot_save = False
    if ("eta" in quantity and quantity != "beta") or quantity == "phib":
        Mdot_save = True
    for j, fname in enumerate(fnames[-num_files:]): # -1000
        dump = pyharm.load_dump(fname, ghost_zones=False)
        if Mdot_save: Mdot = -pyharm.shell_sum(dump, "FM")
        if quantity == "eta":
            Edot = -pyharm.shell_sum(dump, "FE")
            summed = (Mdot - Edot) / Mdot
        else:
            print("WARNING, not supported")
        for i, radius in enumerate(radii):
            i_r = np.argmin(abs(dump["r1d"] - radius))
            quantity_arr[i][j] = summed[i_r]
    return quantity_arr

def corr(t1, t2):
    # t1 and t2 are time series
    f1 = np.fft.fftshift(np.fft.fft(t1))
    f2 = np.fft.fftshift(np.fft.fft(t2))

    p_f = np.conj(f1) * f2
    p_t = np.fft.ifft(np.fft.fftshift(p_f))
    return p_t

if __name__ == "__main__":
    print(eta_BZ6(0.9375, 50.18, 0.05))
