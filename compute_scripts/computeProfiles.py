import os
import numpy as np
import glob
import h5py
import pdb
import pickle
import pyharm


def shellAverage(dump, quantity, imin=0, density_weight=True, pole_pad=1):
    """
    Starting with a pyharm dump object, average some quantity with respect to phi and theta.
    Includes some smart parsing of quantities that are not keys in the dump object.
    """

    # Many quantities we want are already keys, but some are not.  For those, we must come up with our own formula.

    ## quantities to be summed over all theta
    if quantity == "Mdot":
        return -pyharm.shell_sum(dump, "FM")
    elif quantity == "Mdot_in":
        to_sum = np.copy(dump["FM"])
        to_sum[to_sum > 0] = 0
        return -pyharm.shell_sum(dump, to_sum)
    elif quantity == "Mdot_out":
        to_sum = np.copy(dump["FM"])
        to_sum[to_sum < 0] = 0
        return -pyharm.shell_sum(dump, to_sum)
    elif quantity == "Edot":
        return -pyharm.shell_sum(dump, "FE")
    elif quantity == "Edot_Fl":
        return -pyharm.shell_sum(dump, "FE_Fl")
    elif quantity == "Edot_EM":
        return -pyharm.shell_sum(dump, "FE_EM")
    elif quantity == "pdot":
        return pyharm.shell_sum(dump, "Fp")
    elif quantity == "Phib":
        return 0.5 * pyharm.shell_sum(dump, "abs_B1") * np.sqrt(4.0 * np.pi)
    elif quantity == "Etot":
        return pyharm.shell_sum(dump, "JE0")
    if quantity == "Ldot":
        return pyharm.shell_sum(dump, "FL")

    ## quantities to be averaged over all theta except near the poles
    if quantity == "T":
        to_average = dump["Theta"]  # dump['u'] / dump['rho'] * (dump['gam']-1)
    else:
        to_average = dump[quantity]

    ## TODO: quantities to be only averaged over phi

    # Weighting for the average.
    volumetric_weight = dump["gdet"]
    if density_weight and quantity != "rho":
        density = dump["rho"]
    else:
        density = dump["1"]

    if dump["n3"] > 1:  # 3d
        if pole_pad > 1:
            print("using pole_pad ", pole_pad)
        return np.sum((to_average * volumetric_weight * density)[imin:, pole_pad:-pole_pad, :], axis=(1, 2)) / np.sum((volumetric_weight * density)[imin:, pole_pad:-pole_pad, :], axis=(1, 2))
    else:
        return np.sum(to_average[imin:, :] * volumetric_weight * density, axis=1) / np.sum(volumetric_weight * density, axis=1)


def computeProfileSet(dump, quantities=["Mdot", "rho", "u", "T", "u^r", "u^phi"], imin=0, density_weight=True, pole_pad=1):
    """
    Open one dump, then compute various profiles from it.  Return a list of profiles.
    """

    output = []
    for quantity in quantities:
        print(f"   {quantity}")
        try:
            output.append(shellAverage(dump, quantity, imin=imin, density_weight=density_weight, pole_pad=pole_pad))
        except:
            continue

    return output


def computeAllProfiles(runName, outPickleName, quantities=["Mdot", "rho", "u", "T", "u^r", "u^phi"], density_weight=True):
    """
    Loop through every file of a given run.  Compute profiles, then save a dictionary to a pickle.
    """

    print("calculating " + runName)

    allFiles = glob.glob(os.path.join(runName, "*.phdf"))
    runIndices = np.array([int(fname.split(".")[-2]) for fname in allFiles])
    order = np.argsort(runIndices)
    allFiles = np.array(allFiles)[order]
    allFiles_calc = np.copy(allFiles)

    # zone-independent information from h5py
    f = h5py.File(allFiles[0], "r")
    dump = pyharm.load_dump(allFiles[0])
    oz = dump["driver/type"] != "multizone"
    if oz:
        nzones = 1
        nzones_eff = 1
        base = np.sqrt(dump["r_out"] / dump["r_in"])
        ncycle_per_zone = -1
    else:
        try:
            nzones = f["Params"].attrs["Multizone/nzones"]
        except:
            nzones = dump["nzone"]
        try:
            base = float(dump["base"])
        except:
            base = 8.0
        try:
            nzones_eff = f["Params"].attrs["Multizone/nzones_eff"]
        except:
            nzones_eff = nzones
        ncycle_per_zone = f["Params"].attrs["Multizone/ncycle_per_zone"]
    f.close()

    # initialization of lists
    listOfProfiles = []
    listOfTimes = []
    listOfCycles = []
    listOfZones = []
    listOfn0 = []
    listOft0 = []
    listOfActiveRange = [None for _ in range(nzones_eff)]

    # if file exists, don't do the whole calculation but continue from the last computed dump
    if os.path.isfile(outPickleName):
        # here it is assumed that no phdf files are deleted
        with open(outPickleName, "rb") as openFile:
            D_read = pickle.load(openFile)

        # initialize the list with previously calculated data
        listOfProfiles = D_read["profiles"]
        listOfTimes = D_read["times"]
        listOfCycles = D_read["cycles"]
        listOfZones = D_read["zones"]
        listOfn0 = D_read["n0_zone"]
        listOft0 = D_read["t0_zone"]
        listOfActiveRange = D_read["active_range"]

        # make a shorter list of dumps to be calculated
        num_saved = len(listOfTimes)
        allFiles_calc = allFiles_calc[num_saved:]

        # check the assumption
        dump = pyharm.load_dump(allFiles[num_saved - 1])
        if listOfCycles[-1] != dump["n_step"]:
            print("WARNING! There has been a change of list of output dumps! Please check the list of dumps again.")
            return
        print("Calculation exists and starting from dump # {}".format(num_saved))

    for file in allFiles_calc:
        f = h5py.File(file, "r")
        dump = pyharm.load_dump(file)

        # basic multizone information
        if oz:
            i_zone = 0
            n0_zone = 0
            t0_zone = 0
        else:
            i_within_vcycle = f["Params"].attrs["Multizone/i_within_vcycle"]
            i_vcycle = f["Params"].attrs["Multizone/i_vcycle"]
            i_zone = abs(i_within_vcycle - (nzones_eff - 1))
            print("Vcycle #: {:d}, zone #: {:d}".format(i_vcycle, i_zone))
            n0_zone = f["Params"].attrs["Multizone/n0_zone"]
            t0_zone = f["Params"].attrs["Multizone/t0_zone"]

        listOfProfiles.append(computeProfileSet(dump, quantities=quantities, density_weight=density_weight))
        listOfTimes.append(dump["t"])
        listOfCycles.append(dump["n_step"])
        listOfZones.append(i_zone)
        listOfn0.append(n0_zone)
        listOft0.append(t0_zone)
        if listOfActiveRange[i_zone] is None:
            if oz:
                active_rin = dump["r_in"]
                active_rout = dump["r_out"]
            else:
                try:
                    active_rin = f["Params"].attrs["Multizone/active_rin"]
                except:
                    active_rin = int(base ** (i_zone))
                try:
                    active_rout = f["Params"].attrs["Multizone/active_rout"]
                except:
                    active_rout = int(base ** (i_zone + 2))
            listOfActiveRange[i_zone] = [active_rin, active_rout]

        f.close()

    # save
    D = {}
    D["runName"] = runName
    D["quantities"] = quantities
    ## zone-independent quantities
    D["nzones"] = nzones
    D["nzones_eff"] = nzones_eff
    D["radii"] = dump["r1d"]
    D["gam"] = dump["gam"]
    D["base"] = base
    D["ncycle_per_zone"] = ncycle_per_zone
    ## zone-dependent quantities
    D["profiles"] = listOfProfiles
    D["times"] = listOfTimes
    D["cycles"] = listOfCycles
    D["zones"] = listOfZones
    D["n0_zone"] = listOfn0
    D["t0_zone"] = listOft0
    D["active_range"] = listOfActiveRange
    # D['runIndices'] = runIndices
    # D['folders'] = subFolders
    # D['r_sonic'] = r_sonic
    # D['spin'] = spin

    save_path = "/".join(outPickleName.split("/")[:-1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(outPickleName, "wb") as openFile:
        pickle.dump(D, openFile, protocol=2)
    print(f"Output saved to {outPickleName}.")


if __name__ == "__main__":
    # Input and output locations.
    grmhdLocation = "../data"
    dataLocation = "../data_products"

    # For example...
    # python computeProfiles.py bondi_multizone_121322_gizmo_3d_ext_g
    import sys

    run = sys.argv[1]

    inName = os.path.join(grmhdLocation, run)
    outName = os.path.join(dataLocation, run + "_profiles_all.pkl")
    quantityList = [
        "Ldot",
        "Edot",
        "Edot_Fl",
        "Edot_EM",
        "Mdot",
        "Mdot_in",
        "Mdot_out",
        "rho",
        "u",
        "T",
        "abs_u^r",
        "u^phi",
        "u^th",
        "u^r",
        "abs_u^th",
        "abs_u^phi",
        "b",
        "inv_beta",
        "beta",
        "Omega",
        "abs_Omega",
        "K",
        "Phib",
    ]  #'Etot', 'u^t',
    computeAllProfiles(inName, outName, quantities=quantityList)
