import numpy as np
import pdb
import matplotlib.pyplot as plt
import glob
from astropy import units as u
from astropy import constants as const
import pyharm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def matplotlib_settings():
    """
    Makes some modifications to the default matplotlib settings.
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({"font.size": 25})  # , "text.usetex": True})

def vertical_zoomin():
    matplotlib_settings()
    
    fig = plt.figure(figsize=(12,36))
    spec_snp = GridSpec(1, 1)[0]
    plt.subplots_adjust(hspace=0.01)
    
    n_zones = 6
    gs = GridSpecFromSubplotSpec(n_zones, 1, subplot_spec = spec_snp, hspace=0.02, wspace=0.02)
    ax = []
    for cell in gs: ax += [plt.subplot(cell)]

    plotrc={}
    plotrc.update({'xlabel': False, 'ylabel': False,'xticks': [], 'yticks': [],'cbar': False, 'frame': False, 'no_title': True, 'shading': 'flat'})
    inwards = -1 # (if < 0, zoom out) jk, inwards > 0 not implemented yet haha

    # read file
    dirtag = "092525_a0.9_rB2e5_mom_cons_test"
    fnum = 5551
    dirtag = "102125_a0.9_rB2e5_mom_cons_96"
    fnum = 5572
    fn = glob.glob("/n/holylfs05/LABS/bhi/Users/hyerincho/grmhd/data/" + dirtag + "/*{:05d}*.phdf".format(fnum))[0]
    dump = pyharm.load_dump(fn,ghost_zones=False)
    print(dump["n_step"])
    r_sonic = dump["rs"]
    mdot = dump["mdot"]
    gam = dump["gam"]
    #rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot, gam=gam)[0]
    iwv = dump["Params"]["Multizone/i_within_vcycle"]
    nzeff = dump["Params"]["Multizone/nzones_eff"]
    i_zone = abs(iwv - (nzeff - 1))
    #print("fnum {} t={:.5g} tg / {:.5g} tB izone{}".format(fnum, dump["t"], dump["t"] / np.power(rB, 3./2), i_zone))
    patch_sz = np.zeros(n_zones)

    axes = ax
    for i in range(np.shape(axes)[0]):
        if i < n_zones: 
            sz = 8**(i+1.8)
            window = (-sz, sz, -sz, sz)
            if inwards > 0:
                iax = n_zones - i -1
            else: iax = i
            im1 = pyharm.plots.plot_xz(axes[iax], dump, "log_Theta", window=window, vmin=8e-6, vmax=1, cmap='gist_heat', **plotrc)
            
            scale = np.power(10,np.floor(np.log10(sz)))
            c= 'white'
            scalebar = AnchoredSizeBar(axes[iax].transData, scale, r'$10^{:d}\, r_g$'.format(int(np.log10(scale))), 'lower left', pad=0.5, color=c, frameon=False, size_vertical=sz/8**2)
            axes[iax].add_artist(scalebar)
            axes[iax].title.set_visible(False)
            
            patch_sz[i] = sz
    
    for i in range(np.shape(axes)[0]):
        if i+(inwards>0) < n_zones and i+inwards >= 0: # and i < n_zones:
            rect = patches.Rectangle((-patch_sz[i+inwards],-patch_sz[i+inwards]), 2*patch_sz[i+inwards], 2*patch_sz[i+inwards], linewidth=3, edgecolor='w', facecolor='none')
            axes[i].add_patch(rect)
            print(patch_sz[i+inwards])
            con1 = patches.ConnectionPatch(xyA=(-inwards*patch_sz[i+inwards], patch_sz[i+inwards]), xyB=(-inwards*patch_sz[i+inwards],-patch_sz[i+inwards]), coordsA="data", coordsB="data", axesA=axes[i], axesB=axes[i+inwards], color='w',ls=':', lw=2)
            con2 = patches.ConnectionPatch(xyA=(inwards*patch_sz[i+inwards], patch_sz[i+inwards]), xyB=(inwards*patch_sz[i+inwards],-patch_sz[i+inwards]), coordsA="data", coordsB="data", axesA=axes[i], axesB=axes[i+inwards], color='w',ls=':', lw=2)

            fig.add_artist(con1)
            fig.add_artist(con2)
            
    
    output = "../plots/vertical_zoomin.png"
    plt.savefig(output,bbox_inches='tight')
    print("saved to " + output)
    plt.close()

if __name__ == "__main__":
    vertical_zoomin()
