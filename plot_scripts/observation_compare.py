from plot_utils import *
from matplotlib import transforms
import matplotlib.image as mpimg
from functools import partial

def Xray_approx(dump):
    # synthetic image (Lalakos+22)
    sz = 2e6
    window = (-sz, sz, -sz, sz)
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    pyharm.plots.plot_xz(ax, dump, dump["rho"]**2 * np.sqrt(dump["Theta"]), log=True, cbar=1, shading="flat", window=window, average=0, cmap='twilight_shifted')

    output = "../plots/Xray_approx.png"
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print("saved to " + output)

def shock(dump):
    # show shock
    sz = 2e6
    window = (-sz, sz, -sz, sz)
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    pyharm.plots.plot_xz(ax, dump, dump["Gamma"] / dump["cs"], log=True, cbar=1, shading="flat", window=window)

    output = "../plots/shock.png"
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print("saved to " + output)

def observation_compare_with_M87(fnum, show_full_sb=False):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from pyharm.plots.overlays import overlay_field
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.ticker import FixedLocator

    matplotlib_settings()
    plt.rcParams.update({"font.size": 25})
    
    n_zones = 3 #6 
    if show_full_sb: 
        fig = plt.figure(figsize=(18.6,6.8))
        hr = [1,3]
    else: 
        fig = plt.figure(figsize=(18.6,6.2)) #, constrained_layout=True)
        hr = [1,5]
    gs = fig.add_gridspec(nrows=2, ncols=n_zones + 1, height_ratios=hr, wspace=0.02, hspace=0.2)
    ax_sb = fig.add_subplot(gs[0, :n_zones])
    ax = []
    for j in range(n_zones + 1): ax += [fig.add_subplot(gs[1,j])]

    plotrc={}
    plotrc.update({'xlabel': False, 'ylabel': False,'xticks': [], 'yticks': [],'cbar': False, 'frame': False, 'no_title': True, 'shading': 'flat'})
    inwards = -1 # 1

    # read file
    dirtag = "052825_a0.9_rB2e5_bondi_eks_largerout" #"043025_a0.9_rB2e5_bondi_eks"
    fn = glob.glob("../data/" + dirtag + "/*{:05d}*.phdf".format(fnum))[0]
    dump = pyharm.load_dump(fn,ghost_zones=False)
    print(dump["n_step"])
    r_sonic = dump["rs"]
    mdot = dump["mdot"]
    rB = bondi.get_quantity_for_rarr([1], "RB", rs=r_sonic, mdot=mdot)[0]
    iwv = dump["Params"]["Multizone/i_within_vcycle"]
    nzeff = dump["Params"]["Multizone/nzones_eff"]
    i_zone = abs(iwv - (nzeff - 1))
    print("fnum {} t={:.5g} tg / {:.5g} tB izone{}".format(fnum, dump["t"], dump["t"] / np.power(rB, 3./2), i_zone))
    patch_sz = np.zeros(n_zones)

    for i in range(n_zones):
        sz = 300**(i+0.55) #8**(i+1.8)
        window = (-sz, sz, -sz, sz)
        im = pyharm.plots.plot_xz(ax[i], dump, "log_Theta", window=window, vmin=8e-6, vmax=1, cmap='gist_heat', **plotrc)
        
        c= 'white'
        if show_full_sb:
            sz_pc = rg2pc(sz, M=2.5e9*u.Msun)#.value()
            scale = np.power(10,np.floor(np.log10(sz_pc)))
            scale_rg = pc2rg(scale, M=2.5e9*u.Msun)
            #pdb.set_trace()
            if int(np.log10(scale)) == -3: label = r'${\rm mpc}$'
            elif int(np.log10(scale)) == -1: label = r'$0.1{\rm pc}$'
            elif int(np.log10(scale)) == 2: label = r'$0.1{\rm kpc}$'
            scalebar = AnchoredSizeBar(ax[i].transData, scale_rg, label, 'upper right', pad=0.5, color=c, frameon=False, size_vertical=sz/8**2)
        else:
            scale = np.power(10,np.floor(np.log10(sz)))
            scale_rg = scale
            scalebar = AnchoredSizeBar(ax[i].transData, scale_rg, r'$10^{:d}\, r_g$'.format(int(np.log10(scale))), 'upper right', pad=0.5, color=c, frameon=False, size_vertical=sz/8**2)
        ax[i].add_artist(scalebar)
        ax[i].title.set_visible(False)
        
        if i==0:
            ax[i].text(-sz*0.9, sz*0.7, r'$T$', color='w', fontsize=40)

            #cb=fig.colorbar(im, cax=ax[i].inset_axes((-0.1, 0.15, 0.05, 0.7)))
            #cb.ax.tick_params(labelleft=True, labelright=False)

        patch_sz[i] = sz
        # connect
        c_connect = 'k' #'magenta'
        point, = ax_sb.plot(scale_rg, 0.02, marker="o", color=c_connect, ms=10)
        con1 = patches.ConnectionPatch(xyA=(scale_rg, 0.02), xyB=(0.65*sz,0.9*sz), coordsA="data", coordsB="data", axesA=ax_sb, axesB=ax[i], color=c_connect, lw=2)
        fig.add_artist(con1)


    # observation
    img = mpimg.imread('../plots/hercules_A.jpg')
    imgsz = 1560
    tr = transforms.Affine2D().rotate_deg(-65)
    im2 = ax[n_zones].imshow(img, extent=(-0.5*imgsz,0.5*imgsz,-0.5*imgsz,0.5*imgsz), transform=tr + ax[n_zones].transData)
    ax[n_zones].set_xticks([])
    ax[n_zones].set_yticks([])
    ax[n_zones].set_title('Observations \n(Hercules A)')
    sz = imgsz /2.6
    ax[n_zones].set_xlim([-sz, sz])
    ax[n_zones].set_ylim([-sz, sz])
    if show_full_sb: ymax=0.8
    else: ymax=0.88
    line = plt.Line2D((.71,.71),(.1,ymax), color="gray", linewidth=10) #, ls='--')
    fig.add_artist(line)
    line = plt.Line2D((.71,.9),(ymax,ymax), color="gray", linewidth=10) #, ls='--')
    fig.add_artist(line)
    line = plt.Line2D((.9,.9),(.1,ymax), color="gray", linewidth=10) #, ls='--')
    fig.add_artist(line)
    line = plt.Line2D((.71,.9),(.1,.1), color="gray", linewidth=10) #, ls='--')
    fig.add_artist(line)
    #scale = np.power(10,np.floor(np.log10(sz)))
    #scalebar = AnchoredSizeBar(ax[n_zones].transData, scale, r'$10^{:d}\, r_g$'.format(int(np.log10(scale))), 'lower right', pad=0.5, color='k', frameon=False, size_vertical=sz/8**2)
    #ax[n_zones].add_artist(scalebar)

    # scalebar
    #ax_sb.semilogx(np.array([1,1e7]),np.array([0,0]))
    ax_sb.set_xlim([1,1e10])
    ax_sb.set_ylim([0,1])
    if 1:
        ax_sb.axvline(2e5,color='gray',ls='--') # Bondi radius
        ax_sb.text(3e5, 0.7, r'$R_B$', color='gray')
    if 1:
        for i, l in enumerate(['(a)','(b)','(c)','(d)']):
            ax[i].text(0.05, 0.05, l, color='w', transform=ax[i].transAxes)
    #ax_sb.set_xlabel(r'r [$r_g$]')
    ax_sb.set_yticks([])
    ax_sb.tick_params(axis='x', bottom=False, labelbottom=False)
    secax = ax_sb.secondary_xaxis('top', functions=(partial(rg2pc, M=2.5e9*u.Msun), partial(pc2rg, M=2.5e9*u.Msun)))
    #secax.xaxis.set_major_locator(FixedLocator([1e-3, 1, 1e3]))
    secax.set_xlabel(r'$R$ [pc]', labelpad=7)
    ax_sb.set_xscale('log')
    # Cosmosim
    x = np.logspace(5.5, 10, 100)
    y = np.linspace(0, 1, 2)
    X,Y=np.meshgrid(x,y)
    z = np.linspace(0,1,100)
    Z=np.vstack((z,z))
    ax_sb.pcolor(X,Y,Z, cmap=matplotlib.cm.Blues,zorder=-1) #, alpha=0.5)
    if show_full_sb:
        ax_sb.text(5e6, 0.4, 'Galaxy Simulations', fontsize=22)
    # GRMHD
    gradient = np.linspace(0.6, 0, 100).reshape(1, -1)
    if show_full_sb: 
        ymax = 0.5
        yloc = 0.13
        extent=[1, 100, 0.48, 1.]
        ax_sb.imshow(gradient, aspect='auto', cmap=matplotlib.cm.RdPu, extent=extent, vmin=0, vmax=1, alpha=0.7)
        ax_sb.text(1.3, 0.6, 'all GRMHD', fontsize=20)
    else: 
        ymax = 1
        yloc = 0.4
    extent=[1, 1e6, 0, ymax]
    ax_sb.imshow(gradient, aspect='auto', cmap=matplotlib.cm.Reds, extent=extent, vmin=0, vmax=1, alpha=0.5)
    ax_sb.text(2, yloc, 'GRMHD w/ multizone (Cho+23)', fontsize=26) #, color='purple')

            
    
    for i in range(n_zones):
        if i+(inwards>0) < n_zones and i+inwards >= 0: # and i < n_zones:
            rect = patches.Rectangle((-patch_sz[i+inwards],-patch_sz[i+inwards]), 2*patch_sz[i+inwards], 2*patch_sz[i+inwards], linewidth=3, edgecolor='w', facecolor='none')
            ax[i].add_patch(rect)
            print(patch_sz[i+inwards])
            con1 = patches.ConnectionPatch(xyA=(inwards*patch_sz[i+inwards], -patch_sz[i+inwards]), xyB=(-inwards*patch_sz[i+inwards],-patch_sz[i+inwards]), coordsA="data", coordsB="data", axesA=ax[i], axesB=ax[i+inwards], color='w',ls=':', lw=2)
            con2 = patches.ConnectionPatch(xyA=(inwards*patch_sz[i+inwards], patch_sz[i+inwards]), xyB=(-inwards*patch_sz[i+inwards],patch_sz[i+inwards]), coordsA="data", coordsB="data", axesA=ax[i], axesB=ax[i+inwards], color='w',ls=':', lw=2)

            fig.add_artist(con1)
            fig.add_artist(con2)
        
            
    output = "../plots/observation_compare_with_M87.png"
    plt.savefig(output,bbox_inches='tight')
    print("saved to " + output)
    plt.close()

if __name__ == "__main__":
    fname = "../data/052825_a0.9_rB2e5_bondi_eks_largerout/bondi.out0.04783.phdf"
    dump = pyharm.load_dump(fname)
    #Xray_approx(dump)
    #shock(dump)
    observation_compare_with_M87(4783, show_full_sb=True)

    
