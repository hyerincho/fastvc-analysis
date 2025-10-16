import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib_settings import *
import pdb
import pyharm


# schematics for spinning BH run multi-zone
def schematic(n_zones=8, n_zones_eff=None, base=8, fake=True, rs=np.sqrt(1e5), gam=5.0 / 3):
    matplotlib_settings()
    # fig, ax = plt.subplots(1, 1, figsize=(12,6))
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if n_zones_eff is None:
        n_zones_eff = n_zones
    log_r_in = np.log10(np.array([base**i for i in range(n_zones_eff)]))  # array of log_r_ins
    t_run = 0
    zone_order = [n_zones_eff - 1 - i for i in range(n_zones_eff)] + [i + 1 for i in range(n_zones_eff - 1)]  # one V cycle
    colors = plt.cm.gnuplot(np.linspace(0.3, 0.9, n_zones_eff))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    y = [(i + 1) for i in range(n_zones + 1)]
    yticks = [r"$10^{:d}$".format(i) for i in y]
    r_B = 80.0 * rs**2 / (27 * gam)
    cmap = plt.cm.twilight_shifted  # plt.cm.coolwarm #plt.cm.cubehelix_r #plt.cm.terrain_r
    r_out_true = base ** 10

    for nz in range(len(zone_order)):
        zone = zone_order[nz]
        if fake:
            tchar = np.power(1.5, zone)  # this is the fake part
        else:
            r_out = np.power(10.0, log_r_in[zone]) * base**2
            vff2 = 1.0 / r_out
            vcs2 = 1.0 / r_B
            tchar = r_out / np.sqrt(vff2 + vcs2)  # np.power(np.power(10,log_r_in[zone]),3./2)
        edgecolor = "k"
        lw = None
        zorder = None
        if nz == len(zone_order) - 2 or nz == 1 or nz == len(zone_order) - 3:
            log_r_in_temp = log_r_in[n_zones_eff - 2]
            if nz == 1:
                t_save = t_run + tchar
        y_length = np.log10(r_out_true) - log_r_in[zone]
        rect = matplotlib.patches.Rectangle((t_run, log_r_in[zone]), tchar, y_length, facecolor="none", edgecolor=edgecolor, lw=lw, zorder=zorder)
        ax.add_patch(rect)
        t_run += tchar
        if zone == 0 or zone == n_zones_eff - 1:
            ax.text(t_run - tchar / 2.0 - 4, 9.5, "zone-{}".format(zone))
            ax.plot([t_run - tchar / 2.0] * 2, [8, 9.4], ls=":", color="gray")
    t_total = t_run
    t_run = 0.0
    for nz in range(len(zone_order)):
        zone = zone_order[nz]
        if fake:
            tchar = np.power(1.5, zone)  # this is the fake part
        if nz < n_zones_eff:
            fc = "w"
            r_base = 0.0
            t_len = tchar
        else:
            r_base = log_r_in[zone - 1]
            t_len = t_total - t_run
        rect = matplotlib.patches.Rectangle((t_run, r_base), t_len, log_r_in[zone] - r_base, facecolor=fc, edgecolor=None, lw=0, zorder=-100)
        if nz >= n_zones_eff - 1:
            fc = cmap((t_run + tchar) / t_total)
        ax.add_patch(rect)
        t_run += tchar

    ax.imshow([np.linspace(0, 1, 100), np.linspace(0, 1, 100)], extent=(0.0, t_run, 0.0, np.log10(r_out_true)), cmap=cmap, interpolation="bicubic", aspect="auto", zorder=-101)

    xmin = 0
    xmax = t_total + 5
    ymin = 0
    ymax = np.log10(r_out_true)
    ax.set_yticks(y, yticks)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, 9.4])
    ax.set_xticks([])
    fs = 23
    ax.set_xlabel("t", fontsize=fs, labelpad=15)
    ax.set_ylabel(r"$r$ [$r_g$]", fontsize=fs)
    ax.text(t_total + 1, 4, "...", fontsize=30)
    ax.arrow(t_total / 4.0, 9.5, 5, 0, width=0.05, head_width=0.2, head_length=1, zorder=100, clip_on=False, color="k")
    ax.arrow(t_total / 3.0 * 2.0, 9.5, 5, 0, width=0.05, head_width=0.2, head_length=1, zorder=100, clip_on=False, color="k")
    ax.axhline(np.log10(r_B), color="grey", lw=5, alpha=0.5)  # , ls='--')
    ax.text(1, np.log10(r_B) - 0.7, r"$R_B$", fontsize=20, color="grey")
    ax.text(t_total * 0.68, 2.0, "frozen until", color="grey")
    ax.text(t_total * 0.65, 1.3, "next activated", color="grey")

    # manual arrowhead width and length
    hw = 1.0 / 40.0 * (ymax - ymin)
    hl = 1.0 / 40.0 * (xmax - xmin)
    lw = 0.5  # axis line width
    ohg = 0.3  # arrow overhang
    ax.arrow(xmin, 0, xmax - xmin, 0.0, fc="k", ec="k", lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)

    plt.savefig("../plots/schematic.png", bbox_inches="tight")

def bflux_schematic():
    # For the Appendix on bflux
    matplotlib_settings()
    plt.rc('text', usetex=True)
    plt.rc('font', size=30)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    figsize = (13, 6)
    nxval = 4 #5
    nyval = 2
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # grid
    for i in range(nxval):
        ax.axvline(i, color='gray')
    for j in range(nyval):
        ax.axhline(j, color='gray')
    
    ax.axvline(1, color='m', alpha=0.3, lw=10)
    ax.text(nxval / 2 - 0.3, nyval - 0.3, 'physical')
    ax.text(0.2, nyval - 0.3, 'ghost')

    # Omega
    ax.text(1.1, 1.1, r"$\Omega_{i_0+1/2, j_0+1/2}=\begin{cases} 0, & \text{\texttt{bflux0}}\\ \text{(constant)}, & \text{\texttt{bflux-const}} \end{cases}$", bbox=dict(facecolor='white', edgecolor='white',alpha=0.8))
    ax.plot(1, 1, 'k.', ms=30)

    # B field
    ax.arrow(1, 0.5, 0.4, 0., fc='k', ec='k', head_width=0.05)
    ax.text(1.1, 0.3, r"$B^x_{i_0+1/2,j_0}$")

    # x,y axes
    ax.arrow(0.1, 0.1, 0, 0.4, fc='k', ec='k', head_width=0.05)
    ax.arrow(0.1, 0.1, 0.4, 0, fc='k', ec='k', head_width=0.05)
    ax.text(0.6, 0.05, 'x')
    ax.text(0.05, 0.65, 'y')

    #ax.set_frame_on(False)
    ax.set_ylim([-0.5, nyval - 0.5])
    ax.set_xlim([-0.5, nxval - 0.5])
    ax.axis('off')
    
    plt.savefig('../plots/bflux_schematic.png', bbox_inches='tight')

def _main():
    schematic(n_zones_eff=7)
    #bflux_schematic()

if __name__ == "__main__":
    _main()
