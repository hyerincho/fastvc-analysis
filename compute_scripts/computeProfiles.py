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
    if quantity == 'Mdot':
        return -pyharm.shell_sum(dump, 'FM')
    elif quantity == 'Mdot_in':
        to_sum = np.copy(dump["FM"])
        to_sum[to_sum>0] = 0
        return -pyharm.shell_sum(dump, to_sum)
    elif quantity == 'Mdot_out':
        to_sum = np.copy(dump["FM"])
        to_sum[to_sum<0] = 0
        return -pyharm.shell_sum(dump, to_sum)
    elif quantity == 'Edot':
        return -pyharm.shell_sum(dump, 'FE')
    elif quantity == 'pdot':
        return pyharm.shell_sum(dump, 'Fp')
    elif quantity == 'Phib':
        return 0.5 * pyharm.shell_sum(dump, 'abs_B1') * np.sqrt(4.*np.pi)
    elif quantity == 'Etot':
        return pyharm.shell_sum(dump, 'JE0')
    if quantity == 'Ldot':
        return pyharm.shell_sum(dump, 'FL')

    ## quantities to be averaged over all theta except near the poles
    if quantity == 'T':
        to_average = dump["Theta"] #dump['u'] / dump['rho'] * (dump['gam']-1)
    else:
        to_average = dump[quantity]

    # Weighting for the average.
    volumetric_weight = dump['gdet']
    if density_weight and quantity != 'rho':
        density = dump['rho']
    else:
        density = dump['1']

    if dump['n3'] > 1: # 3d
        if pole_pad > 1: print("using pole_pad ", pole_pad)
        return np.sum((to_average * volumetric_weight * density)[imin:,pole_pad:-pole_pad,:], axis=(1,2)) / np.sum((volumetric_weight * density)[imin:,pole_pad:-pole_pad,:], axis=(1,2))
    else:
        return np.sum(to_average[imin:,:] * volumetric_weight * density, axis=1) / np.sum(volumetric_weight * density, axis=1)

def computeProfileSet(dump, quantities=['Mdot', 'rho', 'u', 'T', 'u^r', 'u^phi'], imin=0, density_weight=True, pole_pad=1):
    """
    Open one dump, then compute various profiles from it.  Return a list of profiles.
    """
  
    output = []
    for quantity in quantities:
        print(f"   {quantity}")
        try: output.append(shellAverage(dump, quantity, imin=imin, density_weight=density_weight, pole_pad=pole_pad))
        except: continue
  
    return output

def computeAllProfiles(runName, outPickleName, quantities=['Mdot', 'rho', 'u', 'T', 'u^r', 'u^phi'], density_weight=True):
    """
    Loop through every file of a given run.  Compute profiles, then save a dictionary to a pickle.
    """

    allFiles = sorted(glob.glob(os.path.join(runName, '*.phdf')))
    
    # zone-independent information from h5py
    f = h5py.File(allFiles[0], "r")
    nzones = f['Params'].attrs["Multizone/nzones"]
    nzones_eff = f['Params'].attrs["Multizone/nzones_eff"]
    try: base = float(dump['base'])
    except: base = 8.
    ncycle_per_zone = f['Params'].attrs["Multizone/ncycle_per_zone"]
    f.close()

    # initialization of lists
    listOfProfiles = []
    listOfTimes = []
    listOfCycles = []
    listOfZones = []
    listOfn0 = []
    listOft0 = []
    listOfActiveRange = [None for _ in range(nzones_eff)]
    
    for file in allFiles:
        f = h5py.File(file, "r")
        dump = pyharm.load_dump(file)

        # basic multizone information
        i_within_vcycle = f['Params'].attrs["Multizone/i_within_vcycle"]
        i_vcycle = f['Params'].attrs["Multizone/i_vcycle"]
        i_zone = abs(i_within_vcycle - (nzones_eff - 1))
        print(i_vcycle, i_zone)
        n0_zone = f['Params'].attrs["Multizone/n0_zone"]
        t0_zone = f['Params'].attrs["Multizone/t0_zone"]
        
        listOfProfiles.append(computeProfileSet(dump, quantities=quantities, density_weight=density_weight))
        listOfTimes.append(dump["t"])
        listOfCycles.append(dump["n_step"])
        listOfZones.append(i_zone)
        listOfn0.append(n0_zone)
        listOft0.append(t0_zone)
        if listOfActiveRange[i_zone] is None:
            active_rin = f['Params'].attrs["Multizone/active_rin"]
            active_rout = f['Params'].attrs["Multizone/active_rout"]
            listOfActiveRange[i_zone] = [active_rin, active_rout]

        f.close()

    # save
    D = {}
    D['runName'] = runName
    D['quantities'] = quantities
    ## zone-independent quantities
    D['nzones'] = nzones
    D['nzones_eff'] = nzones_eff
    D['radii'] = dump['r1d']
    D['gam'] = dump['gam']
    D['base'] = base
    D['ncycle_per_zone'] = ncycle_per_zone
    D['n0_zone'] = listOfn0
    D['t0_zone'] = listOft0
    ## zone-dependent quantities
    D['profiles'] = listOfProfiles
    D['times'] = listOfTimes
    D['cycles'] = listOfCycles
    D['zones'] = listOfZones
    D['active_range'] = listOfActiveRange
    #D['runIndices'] = runIndices
    #D['folders'] = subFolders
    #D['r_sonic'] = r_sonic
    #D['spin'] = spin

    save_path = '/'.join(outPickleName.split('/')[:-1])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(outPickleName, 'wb') as openFile:
        pickle.dump(D, openFile, protocol=2)
    print(f"Output saved to {outPickleName}.")


if __name__ == '__main__':
    # Input and output locations.
    grmhdLocation = '../data'
    dataLocation = '../data_products'

    # For example...
    # python computeProfiles.py bondi_multizone_121322_gizmo_3d_ext_g
    import sys
    run = sys.argv[1]
  
    inName = os.path.join(grmhdLocation, run)
    outName = os.path.join(dataLocation, run + '_profiles_all.pkl')
    quantityList = ['Ldot', 'Edot', 'Mdot', 'Mdot_in','Mdot_out', 'rho', 'u', 'T', 'abs_u^r', 'u^phi', 'u^th', 'u^r','abs_u^th', 'abs_u^phi', 'b', 'inv_beta', 'beta', 'Omega', 'abs_Omega', 'K', 'Phib'] #'Etot', 'u^t', 
    computeAllProfiles(inName, outName, quantities=quantityList)
