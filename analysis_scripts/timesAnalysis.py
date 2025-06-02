import numpy as np
import matplotlib.pyplot as plt
import pyharm
import glob
import h5py
import pdb

from matplotlib_settings import *
from runtime_utils import *


def calc_vchar(dump):
    r = dump["r1d"]
    cs2 = 27.0 * dump["gam"] / (80.0 * dump["rs"] ** 2)
    vchar = np.sqrt(1.0 / r + cs2)
    return vchar


def max_velocities(dump, ax_passed=None):
    if ax_passed is None:
        matplotlib_settings()
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        ax = ax_passed

    r = dump["r1d"]
    nx2 = dump["nx2"]

    ax.loglog(r, calc_vchar(dump), "k:", label=r"$v_{\rm char}$")
    ax.loglog(r, np.max((dump["u^r"]), axis=(1, 2)), "k--", label=r"$u^r_{\rm max}$")
    ax.plot(r, np.max((dump["u^r"][:, nx2 // 2, :]), axis=(1)), "b:", label=r"$u^r_{\rm max,mid}$")
    ax.plot(r, np.max((dump["u^r"][:, 0, :]), axis=(1)), "r:", label=r"$u^r_{\rm max,north}$")
    ax.plot(r, np.max((dump["u^r"][:, -1, :]), axis=(1)), "r-", label=r"$u^r_{\rm max,south}$")

    # formatting
    ax.set_xlabel(pyharm.pretty("r"))
    ax.set_ylabel(pyharm.pretty("v"))
    ax.set_ylim([1e-7, 3])
    ax.legend()

    if ax_passed is None:
        ax.set_title("t={:.5g}, n={:d}".format(dump["t"], dump["n_step"]))
        plt.savefig("../plots/velocities.png", bbox_inches="tight")
    else:
        return ax


def betaGamma(dump, ax_passed=None, norm_vchar=False):
    if ax_passed is None:
        matplotlib_settings()
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        ax = ax_passed

    r = dump["r1d"]
    nx2 = dump["nx2"]

    vchar = calc_vchar(dump)
    bg = np.sqrt(np.power(dump["Gamma"], 2.0) - 1.0)
    max_bg = np.max(bg, axis=(1, 2))
    if norm_vchar:
        max_bg /= vchar

    ax.plot(r, max_bg, "k:")

    # formatting
    ax.set_xlabel(pyharm.pretty("r"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    if norm_vchar:
        ax.set_ylabel(r"$(\beta\gamma)_{\rm max} / v_{\rm char}$")
    else:
        ax.set_ylabel(r"$(\beta\gamma)_{\rm max}$")
    ax.legend()

    if ax_passed is None:
        ax.set_title("t={:.5g}, n={:d}".format(dump["t"], dump["n_step"]))
        plt.savefig("../plots/betaGamma.png", bbox_inches="tight")
    else:
        return ax


def Gamma(dump, ax_passed=None):
    # test each lorentz factors
    if ax_passed is None:
        matplotlib_settings()
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        ax = ax_passed

    r = dump["r1d"]

    ax.plot(r, np.max(dump["Gamma"], axis=(1, 2)), "k:")

    # ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(pyharm.pretty("r"))
    ax.set_ylabel(r"$\gamma_{\rm max}$")
    if ax_passed is None:
        plt.savefig("../plots/Gamma.png", bbox_inches="tight")
    else:
        return ax


def timestep(dirtag, num_files=10, ax_passed=None, color="k", label="__nolegend__"):
    if ax_passed is None:
        matplotlib_settings()
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        ax = ax_passed

    files = sorted(glob.glob("../data/" + dirtag + "/*.out0*.phdf"))
    files = files[::-1]
    for fname in files[:num_files]:
        f = h5py.File(fname, "r")
        r = f["Params"].attrs["Multizone/active_rin"]
        dt = f["Info"].attrs["dt"]

        ax.plot(r, dt, color=color, marker=".", label=label)
        if label != "__nolegend__":
            label = "__nolegend__"

    # ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(pyharm.pretty("r"))
    ax.set_ylabel(r"$dt$")
    ax.legend()
    if ax_passed is None:
        output = "../plots/timestep.png"
        plt.savefig(output, bbox_inches="tight")
        plt.close()
        print("saved to " + output)
    else:
        return ax

def compare_speed(dirtags):
    matplotlib_settings()                 
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    oz_save = [None, None]
    for dirtag in dirtags:
        logs = sorted(glob.glob("../data/" + dirtag + "/out-*"))

        # get rB
        fname_dump = glob.glob('/'.join(logs[0].split('/')[:-1] + ['*out0.00000.phdf']))[0]
        dump = pyharm.load_dump(fname_dump,ghost_zones=False)
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=dump["rs"])[0]
        is_onezone = dump["driver/type"] == "kharma"
        for fname in logs[::-1]:
            wt, t = read_runtime(fname)
            t /= np.power(rB, 3./2)
            wt = (wt * u.s).to('d')
            rate = (wt / t).to('d')
            if wt > 1. * u.d:
                print(wt, t)
                break
        if is_onezone and oz_save[0] is None: 
            oz_save[0] = rB
            oz_save[1] = rate
        
        # plot
        marker = ['ko', 'kx'][~is_onezone]
        ax.loglog(rB, rate, marker) #, label=label)
        
    # plot oz estimate
    xlim = ax.get_xlim()
    rB_xaxis = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 10)
    rate_estimate = np.power(rB_xaxis/oz_save[0], 3./2) * oz_save[1]
    ax.plot(rB_xaxis, rate_estimate, color='gray', ls='--')
    
    # labels
    ax.set_xlabel(r'$R_B$ [$r_g$]')
    ax.set_ylabel(r'rate [day/$t_B$]')

    # save figure
    savefig_name='/compare_speed.png'
    plt.savefig("../plots/"+savefig_name,bbox_inches='tight') 


if __name__ == "__main__":
    dirname = "021425_a0.5_rB1e6_reconnect"
    dirname = "021825_a0.5_rB1e6_reconnect_nocool_hse"
    dirname = "022625_a0.5_safe"
    dirname = "022825_a0.5_n8_rdepgmax"
    dirname = "030425_a0.5_rdepgmax5"
    dirname = "041625_n4_a0.9_toriilike_jks2_smth5_reconnect"
    fnum = 57  # 450  # 2000 #5000
    fname = glob.glob("../data/" + dirname + "/*.out0.{:05d}.phdf".format(fnum))[0]
    # fname = sorted(glob.glob("../data/"+dirname+"/*.out0.{:05d}.phdf".format(fnum)))[-1]
    compare_speed(["042125_a0.9_oz_jks", "050625_a0.9_rB2e5_oz_test", "042225_n4_a0.9_bondi_jks2_nocap", "042325_a0.9_rB2e3_bondi", "042325_a0.9_rB2e5_bondi"]) #"042125_n4_a0.9_bondi_jks2", "042825_a0.9_rB2e4_bondi_rot", , "042725_a0.9_rB2e5_bondi_rot"
    if 0:
        dirtags = [
            "032125_n4a0.9_toriilike",
            "041625_n4_a0.9_toriilike_jks2_reconnect",
            "041625_n4_a0.9_toriilike_jks2_smth2_reconnect",
            "041625_n4_a0.9_toriilike_jks2_smth3_reconnect",
            "041625_n4_a0.9_toriilike_jks2_smth5_reconnect",
        ]
        labels = ["eks", "jks1", "2", "3", "5"]
        dirtags = ["040625_a0.9_rB2e3", "041625_a0.9_rB2e3_jks2", "041625_a0.9_rB2e3_jks2_smth3"]
        labels = ["eks", "jks1", "3"]
        dirtags = ["040925_a0.9_fofc_noehbuffer", "041825_a0.9_rB2e5_jks2_smth2", "041725_a0.9_rB2e5_jks2_smth2.5", "042125_a0.9_rB2e5_jks2_smth2.7"]
        labels = ["eks", "jks2", "2.5", "2.7"]
        colors = plt.cm.gnuplot(np.linspace(0.0, 0.9, len(dirtags)))
        matplotlib_settings()
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for i, dirtag in enumerate(dirtags):
            timestep(dirtag, 40, ax, colors[i], labels[i])
        output = "../plots/timestep.png"
        plt.savefig(output, bbox_inches="tight")
        plt.close()
        print("saved to " + output)

    dump = pyharm.load_dump(fname, ghost_zones=False)
    # max_velocities(dump)
    # betaGamma(dump, norm_vchar=True)
    # Gamma(dump)
