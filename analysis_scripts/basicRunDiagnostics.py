import numpy as np
import h5py
import glob
import pdb
import subprocess
from astropy import units as u

def count_cycles(dirtag):
    # count cycles for each zone
    
    # find a log file
    log_fname = glob.glob("../data/" + dirtag + "/out*txt")
    if len(log_fname) != 1:
        print("ERROR: log file not unique!")
        log_fname = sorted(log_fname)[-1]
        #return
    else: log_fname = log_fname[0]

    # basic parsing
    fname = glob.glob("../data/" + dirtag + "/*rhdf")[0]
    f = h5py.File(fname, "r")
    n_zones = f["Params"].attrs["Multizone/nzones_eff"]
    f.close()

    # initialization
    i_zone = n_zones - 1
    ncycle_at_switch = 0
    ncycle_at_zone = np.full(n_zones, 0)
    nvisit_at_zone = np.full(n_zones, 0)
    previous_line = None

    #with open(log_fname, 'r') as f:
        #for i, line in enumerate(f):
    lines = head(log_fname, 10000000)
    for i,line in enumerate(lines):
        if 1:
            if i % 100000 == 0: print(i)
            if "i_within_vcycle" in line:
                zone_temp = int(line.split()[2])
                if zone_temp != i_zone:
                    ncycle_at_switch_temp = int(previous_line.split()[0][6:])
                    dncycle = ncycle_at_switch_temp - ncycle_at_switch
                    ncycle_at_zone[i_zone] += dncycle
                    nvisit_at_zone[i_zone] += 1
                    if i % 100 == 0: print(i_zone, dncycle)
                    i_zone = zone_temp
                    ncycle_at_switch = ncycle_at_switch_temp
            previous_line = line
    print("nvisit at zone ", nvisit_at_zone)
    avg_ncycle_per_zone =  ncycle_at_zone / nvisit_at_zone
    return avg_ncycle_per_zone

def head(f, n):
    proc = subprocess.Popen(['head', '-n', str(n), f], stdout=subprocess.PIPE)
    lines = proc.stdout.readlines()
    return [x.decode("utf-8") for x in lines]

def tail(f, n):
    proc = subprocess.Popen(['tail', '-n', str(n), f], stdout=subprocess.PIPE)
    lines = proc.stdout.readlines()
    return [x.decode("utf-8") for x in lines]

def print_walltime_summary(dirtags):
    for dirtag in dirtags:
        logfiles = sorted(glob.glob("../data/" + dirtag + "/*out-*"))
        last_ncycle = 0; last_time = 0.; walltime = 0.
        for fname in logfiles:
            lines = tail(fname, 10000)
            for line in lines[::-1]:
                if "wsec_total" in line:
                    splitted = line.split(' ')
                    ncycle = int(splitted[0].split('=')[-1])
                    if ncycle > last_ncycle: last_ncycle = ncycle
                    time = float(splitted[1].split('=')[-1])
                    if time > last_time: last_time = time
                    walltime += float(splitted[-2].split('=')[-1])
                    break
        print(dirtag + ", total walltime {:.3g} for {} ncycles ({:.3g} cycles/s) and {:.3g} tg runtime ({:.3g} tg/s).".format((walltime * u.s).to('d'), last_ncycle, (last_ncycle/(walltime * u.s)).to('1/s').value, last_time, (last_time/(walltime * u.s)).to('1/s').value))
    #return last_walltime, time

def _main():
    #print_walltime_summary(["delta/081224_oz_a0.5_128", "081524_a0.5_oz", "081424_a0.5_bfluxc_moverin", "081524_a0.5_bflux0_moverin", "081924_a0.5_consistentB", "082624_a0.5_consistentB", "082124_a0.5_consistentB_kastaun", "082724_a0.5_rdepgmax", "090124_a0.5_rdepgmax_flr", "090324_a0.5_rdepgmax_flr_1dw", "090424_a0.5_rdepgmax_ctop", "080724_fastvc_consistentB/sg07", "090524_a0_oz_ismr", "061724_fastvc/combineout_ismr", "090624_test_runtime_tchar", "091124_a0.5_production", "092224_a0.5_production_gmax2", "092624_a0.5_noismr", "092624_a0.5_sigmaceildown", "092324_a0.5_rdepgamx_test", "100224_a0.5_production", "100224_a0.5_oz", "100724_a0.5_oz", "100724_a0.5_n4", "100724_a0.5_beta100_rot", "101524_a0.5_beta10_rot"])
    if 1: # count average cycles for each zone for tchar runs
        dirtag = "061724_fastvc/combineout_restructured" #"100724_a0.0_n8/nordepgmax_bflux0_tchar" #_capped_ncycle"_ncycle200 #
        dirtag = "100724_a0.5_n8"
        avg_ncycles = count_cycles(dirtag)
        print(avg_ncycles)
        print("ratio: ", avg_ncycles / avg_ncycles[-1])

if __name__ == "__main__":
    _main()
