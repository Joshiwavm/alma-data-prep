import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt

import casatools

# Add the path for CASA analysis utilities (override with CASA_ANALYSIS_SCRIPTS env var)
sys.path.append(os.environ.get(
    "CASA_ANALYSIS_SCRIPTS",
    "/almastorage/allegro/lib/jao-mirror/AIV/science/analysis_scripts/",
))
import analysisUtils as au  # CASA utilities

# Speed of light in meters per second
C = 299792458

def arcsec_to_uvdist(arcsec=1.65 * 60):
    """Converts angular scale in arcseconds to uv-distance in kilolambda."""
    return 1 / np.deg2rad(arcsec / 3600) / 1e3

def uvdist_to_arcsec(uvdist):
    """Converts uv-distance in kilolambda to angular scale in arcseconds."""
    return np.rad2deg(1 / (uvdist * 1e3)) * 3600

def l_to_arcsec(ell):
    """Converts spatial scale in wavenumber l to arcseconds."""
    return 180 / ell * 3600

def arcsec_to_l(arcsec):
    """Converts arcseconds to spatial wavenumber l."""
    return 180 / (arcsec / 3600)

def l_to_uvdist(ell):
    """Converts spatial scale in wavenumber l to uv-distance in kilolambda."""
    return arcsec_to_uvdist(l_to_arcsec(ell))

def uvdist_to_l(uvdist):
    """Converts uv-distance in kilolambda to spatial wavenumber l."""
    return arcsec_to_l(uvdist_to_arcsec(uvdist))

def get_npy_files(directory):
    """Returns a list of .npy files in the given directory."""
    return [f for f in os.listdir(directory) if f.endswith(".npy")]

def uvload(vis):
    """Loads uv distances and weights from a measurement set (MS)."""
    uvwghts = np.empty(0)
    uvdists = np.empty(0)
    msmd = casatools.msmetadata()
    
    # Open MS metadata and retrieve field IDs and science spws
    msmd.open(vis)
    fields = msmd.fieldsforintent("*OBSERVE_TARGET*", False)  # False returns field IDs
    spws = au.getScienceSpws(vis, intent='OBSERVE_TARGET#ON_SOURCE', returnString=False)
    msmd.close()
    
    for field in fields:
        for spw in spws:
            ms = casatools.ms()
            ms.open(vis)
            ms.selectinit(reset=True)
            ms.selectinit(datadescid=int(spw))
            ms.select({'field_id': int(field)})
            
            rec = ms.getdata(['u', 'v', 'weight'])
            uvwght = 4.0 / (1.0 / rec['weight'][0] + 1.0 / rec['weight'][1])
            
            u = rec['u']
            v = rec['v']
            freqs = ms.range('chan_freq')['chan_freq'][:, 0]
            
            ms.close()
            
            uwave = (u.reshape(-1, 1) * freqs / C)
            vwave = (v.reshape(-1, 1) * freqs / C)
            
            uwave = np.swapaxes(uwave, 0, 1)
            vwave = np.swapaxes(vwave, 0, 1)
            shapes = np.ones_like(uwave)
            
            uwave = uwave.flatten()
            vwave = vwave.flatten()
            
            uvwght = (shapes * uvwght.reshape(1, -1)).flatten()
            uvwghts = np.append(uvwghts, uvwght)
            uvdists = np.append(uvdists, np.sqrt(uwave ** 2 + vwave ** 2))
    
    return uvwghts.flatten(), uvdists.flatten()

def getWeightDistribution(vis, bins=np.logspace(np.log10(0.1), np.log10(150), 31), savename=None):
    """Computes white noise sensitivity from visibility weights in a measurement set."""

    wgts, uvdists = uvload(vis)
    uvdists/=1e3
    
    if savename is None:
        return (1/ np.nansum(wgts[wgts>0]))**0.5 
    
    if not os.path.exists(os.path.dirname(savename)):
        os.makedirs(os.path.dirname(savename))
    
    std_binned = []
    for i in range(len(bins) - 1):
        mask = (uvdists > bins[i]) & (uvdists <= bins[i + 1])  
        wgts_sum = np.nansum(wgts[mask])
        std_binned.append((1 / wgts_sum) ** 0.5 if wgts_sum > 0 else np.nan)
    
    bin_centers = 10 ** ((np.log10(bins[1:]) + np.log10(bins[:-1])) / 2)
    np.save(savename, (bin_centers, np.array(std_binned)), allow_pickle=True)
    return 1 / (np.sum(wgts[wgts>0])) ** 0.5

def plotWeightDistribution(target, savename):

    # If saving, ensure output directory exists
    if not os.path.exists(os.path.dirname(savename)):
        os.makedirs(os.path.dirname(savename))

    fig, ax = plt.subplots(constrained_layout=True)

    for i, f in enumerate(get_npy_files(f'../../output/{target}')):
        bin_centers, std_binned = np.load(f'../../output/{target}/{f}')

        labelname = 'ACA, ' if '7m' in f else '12m-array, '
        labelname += f.split('GHz')[0][-2:] + 'GHz'
        
        ax.plot(bin_centers, std_binned*1e6, c=f'C{i}', label = labelname)
        ax.axhline((1/(np.nansum(1/std_binned**2)))**0.5*1e6, c=f'C{i}', ls='--')

    ax.text(0.03, 0.97, f'{target}', transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
    
    ax.set_xlabel(r'uv-distance [k$\lambda$]', fontsize=12)
    ax.set_ylabel(r'$\sigma$ [$\mu$Jy]', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis = 'x', which = 'both', top=False)
    ax.axis(xmin = 0.2, xmax=5e2, ymin = 1)
    
    secax = ax.secondary_xaxis('top', functions=(arcsec_to_uvdist, uvdist_to_arcsec))
    secax.set_xlabel('Spatial scale ["]', fontsize=12)

    plt.legend(frameon=True, loc = 1)
    plt.savefig(savename, dpi = 300)
    plt.close()