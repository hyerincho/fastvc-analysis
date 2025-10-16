import os
import glob
import h5py
import pyharm
from pyharm.plots.plot_dumps import plot_xz
from pyharm.ana.analyses import omega_bz_advanced

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
    plot_xz(ax[1, 2], dump, FE_EM * dump["gdet"] / Mdot_save, native=native, log_r=(~native), symlog=True, vmin=vmin, vmax=vmax, cbar=1, shading="flat", window=window, average=1, xlabel=native, ylabel=native)

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
        rB = bondi.get_quantity_for_rarr([1], "RB", rs=dump["rs"], mdot=dump["mdot"])[0]
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

def tavgedSlice(dirtag, quantity, num_files=-1, native=True, sum_each_ann=False, rscale=None, only_sum=0):
    matplotlib_settings()
    if native:
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=native, sharey="row", gridspec_kw={"height_ratios": [1, 2]})
        plt.subplots_adjust(hspace=0.4, wspace=0.01)
    else:
        fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=native, sharey="row", gridspec_kw={"height_ratios": [1, 3]})
        plt.subplots_adjust(hspace=0.3, wspace=0.01)
    
    symlog = False
    log = False
    inverse = False
    if 'symlog' in quantity:
        symlog = True
        quantity = quantity.replace('symlog_','')
    elif 'log' in quantity:
        log = True
        quantity = quantity.replace('log_', '')
    if 'FE' in quantity or 'FM' in quantity or "rho" in quantity: density_weight = False
    else: density_weight = True
    if quantity == "beta":
        quantity = "inv_beta"
        inverse = True

    files = sorted(glob.glob("../data/" + dirtag + "/*out*.phdf"))[::-1]
    quantity_tavg = 0
    weight_tavg = 0
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
        if active_rin < 2 and save:  # temporary, only add when the whole domain is active
            dump = pyharm.load_dump(files[i], ghost_zones=False)
            if density_weight: weight = dump["rho"]
            else: weight = dump["1"]
            temp = dump[quantity] * weight
            if only_sum > 0: temp[temp<0] = 0.
            quantity_tavg += temp
            weight_tavg += weight
            num_sum += 1
            if dump["t"] > last_time:
                last_time = dump["t"]
            if first_time < 0 or dump["t"] < first_time:
                first_time = dump["t"]


    radii = dump["r1d"]

    # plot settings
    if quantity == "K":
        vmin = 1e0; vmax = 1e4
    elif quantity == "rho":
        vmin = 1e-9; vmax = 1e-3
    elif quantity == "betagamma":
        vmin = 1e-3; vmax = 4
    elif quantity == "Gamma":
        vmin = 1; vmax = 1.3
    elif quantity == "sigma":
        vmin = 1e-5; vmax = 50
    elif quantity == "inv_beta":
        quantity = "beta"
        vmin = 5e-2; vmax = 1e1
    elif "u^r" in quantity:
        vmin = 1e-5; vmax = 1e0
    elif "Be" in quantity:
        vmin = 1e-5; vmax = 2
    elif quantity == "b":
        vmin = 1e-5; vmax = 1e-1
    elif quantity == "FE_EM_A":
        vmin = 1e-5; vmax = 1e-3
    elif quantity == "FM_A":
        vmin = 1e-5; vmax = 1e-1
    if np.any(quantity_tavg < 0.0):
        symlog = True
        #vmax = np.log10(vmax)
        vmin = -vmax
    lw = 2
    if native:
        window = (np.log10(dump["r_eh"]), np.log10(dump["r_out"]), 0, np.pi)
    else:
        window = None

    # th, phi summed radial profiles
    plot_shell_summed(ax[0], dump, np.log10(radii), quantity_tavg, color="k", lw=lw, label=quantity, normalize=weight_tavg, inverse=inverse)
    
    # configure
    if log or symlog: ax[0].set_yscale("log")
    ax[0].set_xlim([np.log10(dump["r_eh"]), np.log10(dump["r_out"])])
    if not symlog: ax[0].set_ylim([vmin, vmax])
    else: ax[0].set_ylim([vmax/1e2, vmax])
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=dump["rs"], mdot=dump["mdot"])[0]
    ax[0].axvline(np.log10(rB), color="grey", lw=2, alpha=1, ls="--")  # show R_B
    ylabel = variableToLabel(quantity)
    ax[0].set_ylabel(ylabel)

    # r, th slice
    phi_averaged = phi_average(dump,quantity_tavg,0,0)[Ellipsis,np.newaxis]
    phi_averaged /= phi_average(dump,weight_tavg,0,0)[Ellipsis,np.newaxis]
    if inverse: phi_averaged = 1. / phi_averaged
    if rscale is not None:
        phi_averaged *= np.power(radii[:, np.newaxis, np.newaxis], rscale)
        vmin = vmax / 100
    plot_xz(ax[1], dump, phi_averaged, native=native, log_r=(~native), log=log, symlog=symlog, vmin=vmin, vmax=vmax, cbar=1, shading="flat", window=window, average=0, xlabel=native, ylabel=native, half_cut=True)

    # label
    ax[0].set_title(dirtag, fontsize=13)
    ax[1].set_title("t = {:.3g}-{:.3g} num_sum = {:d}".format(first_time, last_time, num_sum), fontsize=13)


    output = "../plots/tavged_slice_"+quantity+".png"
    if not native:
        output = output.replace(".png", "_logr.png")
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print("saved to " + output)

def plotOmegaFieldvsTh(dirtag, ax_passed=None, num_files=-1):
    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    if ax_passed is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))  # 8
    else:
        ax = ax_passed
    
    F01 = 0.
    F13 = 0.
    num_sum = 0
    fnames = sorted(glob.glob('../data/' + dirtag + '/*out0.*.phdf'))
    dump = pyharm.load_dump(fnames[0], ghost_zones=False)
    rEH = dump["r_eh"] #* 10
    i_rEH = np.argmin(abs(dump["r1d"] - rEH))
    
    if num_files == -1: num_files = len(fnames)//2
    for fname in fnames[-num_files:]: #-1000 #fnames[200:200+num_files]: #
        dump = pyharm.load_dump(fname, ghost_zones=False)
        file_num = int(fname.split('/')[-1].split('.')[-2])
        F01 += phi_average(dump, 'F_0_1', mass_weight=False)#[i_rEH]
        F13 += phi_average(dump, 'F_1_3', mass_weight=False)#[i_rEH]
        num_sum += 1
    
    #omegaF = pyharm.ana.reductions.theta_profile(dump, 'F_0_1', i_rEH, 5) / pyharm.ana.reductions.theta_profile(dump, 'F_1_3', i_rEH, 5)
    #omegaF = (omegaF * 2. * rEH / dump["a"])#[i_rEH, 0]
    #out = {}
    #omega_bz_advanced(dump, out, **{'do_tavgs': True})
    #pdb.set_trace()
    #omegaF = out['rhth/omega_alt'][i_rEH]
    #omegaF *= 2. * rEH / dump["a"]
    omegaF = ((F01 / F13) * 2. * rEH / dump["a"])#[i_rEH, 0]
    colors=['k', 'b', 'g', 'c']
    for i, radius in enumerate([rEH]): #, 3*rEH]): #, 10*rEH, 100*rEH]):
        i_r = np.argmin(abs(dump["r1d"] - radius))
        ax.plot(dump["th1d"], omegaF[i_r], marker='.', color=colors[i])
        #ax.plot(dump["th1d"][:len(dump["th1d"])//2], omegaF, marker='.', color=colors[i])
    ax.axhline(0.5, color='k', ls=':')

    ax.set_ylim([0, 1])
    ax.set_xlabel(r"$\theta$") #, labelpad=10)
    ylabel = variableToLabel(r"$\Omega_F/\Omega_H$")
    ax.set_ylabel(ylabel)
    plt.suptitle(dirtag)

    if ax_passed is None:
        fig.tight_layout()
        output = "../plots/plot_omega_field_vs_th.png"
        plt.savefig(output, bbox_inches="tight")
        print("saved to " + output)
        plt.close(fig)
    else:
        return ax

if __name__ == "__main__":
    dirtag = "010625_a0.5_oz_128"
    dirtag = "010925_a0.9_oz_128"
    # dirtag = "022525_a0.5_b8n4"
    # dirtag="031625_a0.5_longtin4"
    dirtag = "031925_a0.5_toriilike"
    dirtag = "delta/032025_a0.5_oz_clearangle"
    # dirtag="032025_a0.0_toriilike"
    #dirtag="040325_torus_noehbuffer"
    #dirtag="041625_n4_a0.9_toriilike_jks2_smth2_reconnect"
    #dirtag = "042325_a0.9_rB2e5_bondi"
    #dirtag="043025_a0.9_rB2e5_bondi_eks"
    dirtag="031425_a0.5_torus_rn22_noreconnect"
    #dirtag="031325_a0.9_torus_rn22_noreconnect"
    #dirtag="050825_torus_noehbuffer_noismr"
    #dirtag="052125_torus_noehbuffer_noismr_a0.5"
    #dirtag="052525_torus_noehbuffer_noismr_a0.5_nofofc"
    dirtag="052725_torus_noehbuffer_noismr_a0.5_diffflr"
    dirtag="052825_a0.9_rB2e5_bondi_eks_largerout"
    #dirtag="053025_torus_noehbuffer_noismr_a0.5_lowx1res"
    dirtag="080625_a0.9_rB2e5_96"
    #dirtag="092525_a0.9_rB2e5_mom_cons_test"

    #FE_slice(dirtag, num_files=-1, native=False, sum_each_ann=True)
    tavgedSlice(dirtag, 'log_K', num_files=-1, native=False) #, only_sum=1) #, rscale=1)
    #plotOmegaFieldvsTh(dirtag, num_files=100)
