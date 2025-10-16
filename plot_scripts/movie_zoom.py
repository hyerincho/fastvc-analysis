import os
from plot_utils import *
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib import transforms
from astropy import units as u
from astropy import constants as const

G = const.G
M = 6.5e9 * u.Msun
c = const.c

def multiscale_obs(fname):
    matplotlib_settings()
    fig, ax = plt.subplots(1, 2, figsize=(16,8), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    dump = pyharm.load_dump(fname, ghost_zones=False)

    sz = 5e2 #4
    plotrc = {'window':[-sz, sz, -sz, sz], 'vmax':-1, 'vmin':-5.096910013008056,  'cmap':'gist_heat', 'cbar':False, 'xlabel':False, 'ylabel':False, 'xticks':[], 'yticks':[]}
    pyharm.plots.plot_xz(ax[0], dump, dump["log_Theta"], **plotrc)

    imgszs = [15.7, 260, 1560, 2100, 2.4e4, 1.38e6]
    if 1:
        # Chandra (Snios+19)
        img = mpimg.imread('../plots/m87_observations/m87_5.jpg')
        imgsz = imgszs[5]
        tr = transforms.Affine2D().rotate_deg(90)
        im6 = ax[1].imshow(img, extent=(-0.4*imgsz,3.6*imgsz,-imgsz,imgsz), transform=tr + ax[1].transData)
        
        # EVN at 170 mm
        img = mpimg.imread('../plots/m87_observations/m87_6.jpg')
        imgsz = imgszs[4]
        tr = transforms.Affine2D().rotate_deg(65)
        im5 = ax[1].imshow(img, extent=(-1.1*imgsz,4.9*imgsz,-0.2*imgsz,1.8*imgsz), transform=tr + ax[1].transData)
    
        # EAVN at 22 GHz
        img = mpimg.imread('../plots/m87_observations/m87_4.png')
        imgsz = imgszs[3]
        tr = transforms.Affine2D().rotate_deg(70)
        im4 = ax[1].imshow(img, extent=(-0.3*imgsz,3.7*imgsz,-0.25*imgsz,1.75*imgsz), transform=tr + ax[1].transData)
        
        # VLBA at 43 GHz
        img = mpimg.imread('../plots/m87_observations/m87_3.jpg')
        imgsz = imgszs[2]
        tr = transforms.Affine2D().rotate_deg(70)
        im3 = ax[1].imshow(img, extent=(-0.76*imgsz,3.24*imgsz,-0.455*imgsz,1.545*imgsz), transform=tr + ax[1].transData)
        
        # GMVA+ALMA+GLT image at 86 GHz
        img = mpimg.imread('../plots/m87_observations/m87_2.png')
        imgsz = imgszs[1]
        tr = transforms.Affine2D().rotate_deg(75)
        im2 = ax[1].imshow(img, extent=(-imgsz*0.3,imgsz,-imgsz*0.4,imgsz*0.9), transform=tr + ax[1].transData)

        # EHT
        img = mpimg.imread('../plots/m87_observations/m87_1.jpg')
        imgsz = imgszs[0]
        im1 = ax[1].imshow(img, extent=(-imgsz,imgsz,-imgsz,imgsz))

    ax[0].set_title('Simulation')
    labels = ['EHT', 'GMVA+ALMA+GLT', 'VLBA', 'EAVN', 'EVN', 'Chandra']
    ims = [im1, im2, im3, im4, im5, im6]
    imgszs_cpy = np.copy(np.array(imgszs))
    #imgszs_cpy[1:] *= 2
    for i, sz in enumerate(np.logspace(np.log10(6), np.log10(3e6), 1000)):
        if i % 100 == 0:
            print(i)
        ax[0].set_xlim([-sz, sz])
        ax[0].set_ylim([-sz, sz])
        
        ilabel = np.where(np.array(imgszs_cpy)*3-sz >0)[0]
        if len(ilabel) > 0:
            ilabel = ilabel[0]
        else:
            ilabel = -1
        for itemp in range(ilabel):
            if ims[itemp] is not None:
                ims[itemp].remove()
            ims[itemp] = None

        ax[1].set_title('Observation ('+labels[ilabel]+')')
        
        scale = np.power(10,np.floor(np.log10(sz)))
        scalebar = AnchoredSizeBar(ax[0].transData, scale, r'$10^{:d}\, r_g$'.format(int(np.log10(scale))), 'lower left', color='w', frameon=False, bbox_to_anchor=(0.05, 0.01), bbox_transform=ax[0].transAxes, pad=1, size_vertical=sz/100) #
        ax[0].add_artist(scalebar)
        sz_phys = (sz * G * M / c**2).to('pc').value
        scale = np.power(10,np.floor(np.log10(sz_phys)))
        scalebar = AnchoredSizeBar(ax[1].transData, (scale * u.pc / G / M * c**2).to(''), r'${}\, $pc'.format(scale), 'lower left', color='c', frameon=False, bbox_to_anchor=(0.05, 0.01), bbox_transform=ax[1].transAxes, pad=1, size_vertical=sz/100) #
        ax[1].add_artist(scalebar)
        plt.savefig('../plots/m87_movie/i{:05d}.png'.format(i), bbox_inches='tight')
        ax[0].artists[0].remove()
        ax[1].artists[0].remove()

if __name__ == "__main__":
    fname = "../data/052825_a0.9_rB2e5_bondi_eks_largerout/bondi.out0.04783.phdf"
    multiscale_obs(fname)
