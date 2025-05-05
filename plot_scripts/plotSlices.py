import os
import glob
import h5py
import pyharm
from pyharm.plots.plot_dumps import plot_xz

from matplotlib_settings import *
from ylabel_dictionary import *
from plot_utils import *
import bondi_analytic as bondi


def FE_slice(dirtag, num_files=-1, native=True, sum_each_ann=False):
    matplotlib_settings()
    if native:
        fig, ax = plt.subplots(2, 3, figsize=(16, 6), sharex=native, sharey="row", gridspec_kw={"height_ratios": [1, 2]})
        plt.subplots_adjust(hspace=0.4, wspace=0.01)
    else:
        fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharex=native, sharey="row", gridspec_kw={"height_ratios": [1, 3]})
        plt.subplots_adjust(hspace=0.3, wspace=0.01)

    files = sorted(glob.glob("../data/" + dirtag + "/*out*.phdf"))[::-1]
    # files = sorted(glob.glob("../data/"+dirtag+"/*out*.rhdf"))[::-1]
    # files = files[(len(files)//4)*3:] # temporary
    # nzones = f["Params"].attrs["Multizone/nzones_eff"] # TODO: sum for each ann
    FE = 0
    FM = 0
    FE_Fl = 0
    FE_KE = 0
    FE_EN = 0
    FE_EM = 0
    num_sum = 0
    first_time = -1
    last_time = 0

    if num_files < 0:
        num_files = len(files) // 2
    for i in range(num_files):
        save = True
        f = h5py.File(files[i], "r")
        oz = f["Params"].attrs["Driver/name"] == "kharma"
        ncycle_per_zone = f["Params"].attrs["Multizone/ncycle_per_zone"]
        long_t_in = f["Params"].attrs["Multizone/long_t_in"]
        n0_zone = f["Params"].attrs["Multizone/n0_zone"]

        if oz:
            active_rin = 1
        else:
            active_rin = f["Params"].attrs["Multizone/active_rin"]
        # if f["Info"].attrs["NCycle"] == n0_zone: pdb.set_trace()
        # if ncycle_per_zone > 0 and f["Info"].attrs["NCycle"] < n0_zone + (ncycle_per_zone * long_t_in) // 2: # TODO test
        #    save = False
        if active_rin < 2 and save:  # temporary, only add when the whole domain is active
            dump = pyharm.load_dump(files[i], ghost_zones=False)
            FM += dump["FM"]
            FE += dump["FE_norho"]
            FE_Fl += dump["FE_Fl_norho"]
            FE_KE += dump["FE_PAKE"]
            FE_EN += dump["FE_EN"]
            FE_EM += dump["FE_EM"]
            num_sum += 1
            if dump["t"] > last_time:
                last_time = dump["t"]
            if first_time < 0 or dump["t"] < first_time:
                first_time = dump["t"]

    FE /= num_sum
    FM /= num_sum
    FE_Fl /= num_sum
    FE_KE /= num_sum
    FE_EN /= num_sum
    FE_EM /= num_sum

    radii = dump["r1d"]
    Mdot = np.squeeze(np.sum(-FM * dump["gdet"] * dump["dx2"] * dump["dx3"], axis=(1, 2)))
    i10 = np.argmin(abs(radii - 10))
    Mdot_save = Mdot[i10]

    # plot settings
    if dump["a"] > 0.3 or (not native):
        vmin = -1
    else:
        vmin = -1e-1
    vmax = -vmin
    lw = 2
    labels = [r"$\eta^{\rm tot}$", r"$\eta^{\rm fl}$", r"$\eta^{\rm kin}$", r"$\eta^{\rm th}$", r"$\eta^{\rm em}$"]

    # th, phi summed radial profiles
    plot_shell_summed(ax[0, 0], dump, np.log10(radii), FE / Mdot_save, color="k", lw=lw, label=labels[0])
    plot_shell_summed(ax[0, 1], dump, np.log10(radii), FE_Fl / Mdot_save, color="b", lw=lw, label=labels[1])
    plot_shell_summed(ax[0, 1], dump, np.log10(radii), FE_KE / Mdot_save, color="g", lw=1, label=labels[2])
    plot_shell_summed(ax[0, 1], dump, np.log10(radii), FE_EN / Mdot_save, color="r", lw=1, label=labels[3])
    plot_shell_summed(ax[0, 2], dump, np.log10(radii), FE_EM / Mdot_save, color="c", lw=lw, label=labels[4])

    # r, th slice
    if native:
        window = (np.log10(dump["r_eh"]), np.log10(dump["r_out"]), 0, np.pi)
    else:
        window = None
    plot_xz(ax[1, 0], dump, FE * dump["gdet"] / Mdot_save, native=native, log_r=(~native), symlog=True, vmin=vmin, vmax=vmax, cbar=0, shading="flat", window=window, average=1, xlabel=native, ylabel=native)
    plot_xz(ax[1, 1], dump, FE_Fl * dump["gdet"] / Mdot_save, native=native, log_r=(~native), symlog=True, vmin=vmin, vmax=vmax, cbar=0, shading="flat", window=window, average=1, xlabel=native, ylabel=native)
    plot_xz(ax[1, 2], dump, FE_EM * dump["gdet"] / Mdot_save, native=native, log_r=(~native), symlog=True, vmin=vmin, vmax=vmax, cbar=0, shading="flat", window=window, average=1, xlabel=native, ylabel=native)

    # label
    ax[1, 0].set_title(r"Total $F_E/\overline{\dot{M}}_{10}$")
    ax[1, 1].set_title(r"Fluid $F_E^{fl}/\overline{\dot{M}}_{10}$")
    ax[1, 2].set_title(r"EM $F_E^{EM}/\overline{\dot{M}}_{10}$")
    ax[0, 0].set_title(dirtag, fontsize=13)
    ax[0, -1].set_title("t = {:.3g}-{:.3g} num_sum = {:d}".format(first_time, last_time, num_sum), fontsize=13)

    # configure
    for j in range(ax.shape[1]):
        ax[0, j].set_yscale("log")
        ax[0, j].set_xlim([np.log10(dump["r_eh"]), np.log10(dump["r_out"])])
        ax[0, j].set_ylim([vmax / 1e2, vmax])
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=dump["rs"])[0]
        ax[0, j].axvline(np.log10(rB), color="grey", lw=2, alpha=1, ls="--")  # show R_B
    for ax_temp in (ax[:, 1:]).flatten():
        ax_temp.set_ylabel("")
    lines_labels = [ax.get_legend_handles_labels() for ax in ax[0, :]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, ncol=5, bbox_to_anchor=(0.65, 1.01))

    output = "../plots/FE_slice.png"  # pdf"
    if not native:
        output = output.replace(".png", "_logr.png")
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print("saved to " + output)


if __name__ == "__main__":
    dirtag = "010625_a0.5_oz_128"
    dirtag = "010925_a0.9_oz_128"
    # dirtag = "022525_a0.5_b8n4"
    # dirtag = "022525_a0.5_b2n14"
    # dirtag = "022625_a0.5_b2n14_tchar"
    # dirtag = "022625_a0.9_n4/noreconnect"
    # dirtag = "022725_a0.0_safe_tchar"
    dirtag = "022825_a0.5_b8n4_tchar"
    # dirtag = "022825_a0.5_b8n4_nc2"
    # dirtag="022025_a0.5_n8_nc40"
    # dirtag="021325_a0.0_011424"
    # dirtag="022425_a0.0_b2_tchar"
    # dirtag="022425_a0.0_n8_ncycle8000"
    dirtag = "022625_a0.5_safe"
    # dirtag = "022825_a0.5_n8_rdepgmax"
    dirtag = "030325_a0.0_safe_tchar"
    # dirtag = "030425_a0.5_b8n4_safe_longtin10"
    dirtag = "030425_a0.5_safe_longtin20"
    # dirtag = "030425_a0.9_b8n4_safe"
    # dirtag="030725_a0.9_safe_longtin20"
    # dirtag="030925_a0.7_safe_longtin20"
    # dirtag="030925_a0.0_n8_lt10_dirichlet"
    # dirtag="031125_a0.5_longtin10"
    # dirtag="031625_a0.5_toriilike"
    # dirtag="031625_a0.5_longtin4"
    dirtag = "031925_a0.5_toriilike"
    # dirtag="032025_a0.0_toriilike"

    FE_slice(dirtag, num_files=-1, native=True)
