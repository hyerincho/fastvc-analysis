import numpy as np
import matplotlib.pyplot as plt
import pyharm
import glob

from matplotlib_settings import *


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


if __name__ == "__main__":
    dirname = "021425_a0.5_rB1e6_reconnect"
    dirname = "021825_a0.5_rB1e6_reconnect_nocool_hse"
    dirname = "022625_a0.5_safe"
    dirname = "022825_a0.5_n8_rdepgmax"
    fnum = 30  # 60 #2000 #5000
    fname = glob.glob("../data/" + dirname + "/*.out0.{:05d}.phdf".format(fnum))[0]
    # fname = sorted(glob.glob("../data/"+dirname+"/*.out0.{:05d}.phdf".format(fnum)))[-1]

    dump = pyharm.load_dump(fname, ghost_zones=False)
    max_velocities(dump)
    betaGamma(dump, norm_vchar=True)
    Gamma(dump)
