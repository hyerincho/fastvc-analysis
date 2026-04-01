import numpy as np
import pdb
import pickle
import matplotlib.pyplot as plt
import glob
from astropy import units as u
from astropy import constants as const
import pyharm
from scipy.optimize import curve_fit
from functools import partial

from matplotlib_settings import *
import bondi_analytic as bondi
from ylabel_dictionary import *

c = const.c
G = const.G

gdirtags = [
    "012026_n4_a0_bondi_nocap_momcons",
    "012026_a0_rB2e3_momcons",
    "012026_a0_rB2e4_momcons",
    "012026_a0_rB2e5_mom_cons_test",
    "012026_a0_rB2e6_momcons",
    #"012026_a0_rB2e7_momcons",
    "121325_n4_a0.1_bondi_nocap_momcons",
    "121325_n4_a0.3_bondi_nocap_momcons",
    "121325_n4_a0.5_bondi_nocap_momcons",
    "121325_n4_a0.7_bondi_nocap_momcons",
    "121325_n4_a0.9_bondi_nocap_momcons",
    "delta/102825_a0.1_rB2e3_momcons",
    "delta/102825_a0.3_rB2e3_momcons",
    "delta/102825_a0.5_rB2e3_momcons",
    "delta/102825_a0.7_rB2e3_momcons",
    "092525_a0.9_rB2e3_mom_cons",
    "102925_a0.97_rB2e3",
    "delta/102825_a0.1_rB2e4_momcons",
    "delta/102825_a0.3_rB2e4_momcons",
    "delta/102825_a0.5_rB2e4_momcons",
    "delta/102825_a0.7_rB2e4_momcons",
    "delta/092525_a0.9_rB2e4_mom_cons",
    "delta/092425_a0.1_rB2e5_momcons",
    "delta/092425_a0.3_rB2e5_momcons",
    "delta/092425_a0.5_rB2e5_momcons",
    "delta/092425_a0.7_rB2e5_momcons",
    "092525_a0.9_rB2e5_mom_cons_test",
    "110725_a0.1_rB2e6_momcons",
    "110725_a0.3_rB2e6_momcons",
    "110725_a0.5_rB2e6_momcons",
    "110725_a0.7_rB2e6_momcons",
    "092525_a0.9_rB2e6_momcons",
        ]

def lin_func(x, a, b):
    return a * x + b


def rg2pc(r,M=6.5e9*u.Msun):
    rg = G*M/c**2
    return (np.array(r)*rg).to('pc').value

def pc2rg(R,M=6.5e9*u.Msun):
    rg = G*M/c**2
    return (np.array(R)*u.pc/rg).to('')

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

def get_lcorr(quantity_arr):
    mean = np.mean(quantity_arr)
    corrarr = np.correlate(quantity_arr-mean,quantity_arr-mean,'same')
    n = np.linspace(0,len(corrarr)-1,len(corrarr))
    n -= n[len(n)//2]
    i_search = abs(n)<4
    n_search = n[i_search]
    corr_search = corrarr[i_search]
    n_corr = abs(n_search[np.argmin(np.abs(corr_search - np.max(corr_search)/2.))]) + 1
    return n_corr

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
    elif quantity == "eta_EM":
        quantity_index = dictionary["quantities"].index("Edot_EM")
        quantity_index2 = dictionary["quantities"].index("Mdot")
        i5 = np.argmin(abs(dictionary["radii"] - 5))
        profiles = [- (np.array(list[quantity_index])) / np.array(list[quantity_index2])[i5] for list in dictionary["profiles"]]
    elif quantity == "phib":
        quantity_index = dictionary["quantities"].index("Phib")
        quantity_index2 = dictionary["quantities"].index("Mdot")
        i5 = np.argmin(abs(dictionary["radii"] - 5))
        Mdot_normalize = np.array([list[quantity_index2] for list in dictionary["profiles"]])
        zones = np.array(dictionary["zones"])
        Mdot_normalize = [Mdot_normalize[np.argwhere(zones[:i] == 0)[-1,0]] if zones[i]!=0 and i > zones[0] else Mdot_normalize[i] for i in range(len(Mdot_normalize))]
        profiles = [np.array(list[quantity_index]) / np.sqrt(np.array(list[quantity_index2])[i5]) for list in dictionary["profiles"]]
    elif quantity == "norm_Edot":
        quantity_index = dictionary["quantities"].index("Edot")
        quantity_index2 = dictionary["quantities"].index("Mdot")
        i5 = np.argmin(abs(dictionary["radii"] - 5))
        profiles = [(np.array(list[quantity_index])) / np.array(list[quantity_index2])[i5] for list in dictionary["profiles"]]
    elif quantity == "norm_Ldot":
        quantity_index = dictionary["quantities"].index("Ldot")
        quantity_index2 = dictionary["quantities"].index("Mdot")
        i5 = np.argmin(abs(dictionary["radii"] - 5))
        #profiles = [(-np.array(list[quantity_index])) / np.array(list[quantity_index2])[i5] for list in dictionary["profiles"]]
        profiles = [-(np.array(list[quantity_index])) / np.array(list[quantity_index2])[i5] for list in dictionary["profiles"]]
    elif quantity == "eta_KE":
        quantity_index = dictionary["quantities"].index("Edot_KE")
        quantity_index2 = dictionary["quantities"].index("Mdot")
        i5 = np.argmin(abs(dictionary["radii"] - 5))
        profiles = [(- np.array(list[quantity_index])) / np.array(list[quantity_index2])[i5] for list in dictionary["profiles"]]
    elif quantity == "eta_TE":
        quantity_index = dictionary["quantities"].index("Edot_TE")
        quantity_index2 = dictionary["quantities"].index("Mdot")
        i5 = np.argmin(abs(dictionary["radii"] - 5))
        profiles = [(- np.array(list[quantity_index])) / np.array(list[quantity_index2])[i5] for list in dictionary["profiles"]]
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
        gam = D["dump"]["gam"]
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
        tB = np.power(rB, 3./2)
        if times[-1] <= tmax * tB:
            print("the time series not reached tmax of {:.3g} yet".format(tmax * tB))
        i_keep = times < tmax * tB
        return quantity_arr[i_keep], times[i_keep]

def processTimeSeries(D, quantity, use_Mdot_mean=True, average_factor=2., rescale=False, tmax=None, radius=None):
    from plotProfiles import setTimeBins, get_mask, calcFinalTimeAvg, plotProfileQuantity
    store_Mdot10 = False
    if quantity == "eta" or quantity == "phib" or quantity == "eta_EM" or quantity == "s_EM":
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
    elif quantity == "s":
        quantity_arr, times = readTimeSeries(D, "norm_Edot", radius, tmax)
        quantity_arr2, _ = readTimeSeries(D, "norm_Ldot", radius, tmax)
    elif quantity == "s_EM":
        quantity_arr, times = readTimeSeries(D, "Edot_EM", radius, tmax)
    else:
        quantity_arr, times = readTimeSeries(D, quantity, radius, tmax)

    if store_Mdot10:
        #Mdot_save, _ = readTimeSeries(D, "Mdot", 10, tmax)
        Mdot_save, _ = readTimeSeries(D, "Mdot", 5, tmax)
    innermost = np.array(D["zones"]) == 0  # <= 1 #

    r_sonic = D["dump"]["rs"]
    mdot = D["dump"]["mdot"]
    gam = D["dump"]["gam"]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
    tB = np.power(rB, 3./2)
    if tmax is None: last_time = times[-1]
    else: last_time = tmax * tB
    a = D["dump"]["a"]

    Mdot_analytic = bondi.get_quantity_for_rarr([rB], "Mdot", rs=r_sonic, mdot=mdot, gam=gam)[0]
    if rescale and (quantity == "Mdot" or quantity == "etaB"):
        print("t={:.5g}-{:.5g}".format(last_time/average_factor, last_time))
        rho_analytic = bondi.get_quantity_for_rarr([100 * rB], "rho", rs=r_sonic, mdot=mdot, gam=gam)[0]
        rho_save, _ = readTimeSeries(D, "rho", rB, tmax) # Cho+24 method
        #tDivList, binNumList = setTimeBins(D, 1, time_bin_factor=average_factor, tmax=tmax)
        #mask_list = get_mask(D, prioritize_inner=False) #True)
        #radii, profiles = calcFinalTimeAvg(D, tDivList, binNumList, "rho", perzone_avg_frac=0.05, mask_list=mask_list)
        #rho_save = min(profiles[0][radii[0] < 5 * rB])
        zones = np.array(D["zones"][:len(quantity_arr)])
        rB_zone = int(np.floor(np.log(rB) / np.log(8) - 0.5))
        if 1: #use_Mdot_mean:
            i_keep = np.argwhere((times < last_time) & (times > last_time / average_factor) & (zones == rB_zone))
            rho_save = np.mean(rho_save[i_keep])
            print("rho_save={:.5g}, factor={:.5g}".format(rho_save, rho_analytic / (Mdot_analytic * rho_save)))
        Mdot_analytic = Mdot_analytic * rho_save / rho_analytic
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
    elif quantity == "s":
        quantity_arr = quantity_arr2 - 2. * quantity_arr * a
    elif quantity == "s_EM":
        rEH = calc_rEH(a)
        OmegaH = a / (2 * rEH)
        k = 0.35
        quantity_arr = quantity_arr / Mdot_save * (1./(k*OmegaH) - 2.* a)

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
    gam = D["dump"]["gam"]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
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
    f1 = (np.fft.fft(t1))
    f2 = (np.fft.fft(t2))

    p_f = np.conj(f1) * f2
    p_t = np.fft.ifft((p_f))
    return p_t

if __name__ == "__main__":
    print(eta_BZ6(0.9375, 50.18, 0.05))
