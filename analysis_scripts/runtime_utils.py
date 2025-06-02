import numpy as np
from astropy import units as u
import subprocess
import pdb
import glob

import pyharm
import bondi_analytic as bondi

def head(f, n):
    proc = subprocess.Popen(['head', '-n', str(n), f], stdout=subprocess.PIPE)
    lines = proc.stdout.readlines()
    return [x.decode("utf-8") for x in lines]

def tail(f, n):
    proc = subprocess.Popen(['tail', '-n', str(n), f], stdout=subprocess.PIPE)
    lines = proc.stdout.readlines()
    return [x.decode("utf-8") for x in lines]

def read_runtime(fname):
    # for onezone, if there's no walltime used summary given
    lines = head(fname, 150)
    start_time = 0.
    for line in lines:
        if "wsec_total" in line:
            start_time = float(line.split(' ')[1].split('=')[-1])
            break
    
    lines = tail(fname, 50)
    last_walltime = 0; last_time = 0
    for line in lines:
        if "wsec_total" in line:
            walltime = float(line.split(' ')[-2].split('=')[-1])
            if walltime > last_walltime: 
                last_walltime = walltime
            time = float(line.split(' ')[1].split('=')[-1])
            if time > last_time: 
                last_time = time
    return last_walltime, (time - start_time)


if __name__ == "__main__":
    fname = "../data/042125_a0.9_oz_jks/out-12828997.txt"
    #fname = "../data/delta/030525_a0.9_oz/out-7912820.txt"
    fname = "../data/042125_n4_a0.9_bondi_jks2/out-12829527.txt"
    wt,t = read_runtime(fname)
    fname_dump = glob.glob('/'.join(fname.split('/')[:-1] + ['*out0.00000.phdf']))[0]
    dump = pyharm.load_dump(fname_dump,ghost_zones=False)
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=dump["rs"])[0]
    t /= np.power(rB, 3./2)
    wt = (wt * u.s).to('d')
    rate = (wt / t).to('d')
    print(wt, t, rate)
