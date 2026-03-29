"""
visualization.py

Centralizes Matplotlib logic for UVJ diagrams, IRAC wedges, and SED plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_galaxy_sed(wavelengths, fluxes, name, template_set, ax=None, filters=None):
    """
    Plots an SED, optionally with filter passbands overlaid.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(wavelengths, fluxes, color='k', linewidth=1.5, label=f'SED: {name}')
    
    if filters:
        # Scale filters for visualization
        sc = np.max(fluxes) * 0.1
        for f_name, pb in filters.items():
            ax.fill_between(pb.wavelength, 0, sc * pb.transmission, alpha=0.3, label=f_name)
            
    ax.set_xlabel('Wavelength (Angstroms)')
    ax.set_ylabel('Flux (erg/s/cm^2/Angstrom)')
    ax.set_title(f'SED of: {name} ({template_set})')
    ax.legend()
    return ax

def plot_uvj_diagram(vj, uv, classifications=None, ax=None, title="UVJ Colour-Colour Diagram", 
                     show_density=False, show_arrows=None, avg_error=None):
    """
    Advanced UVJ plotting with classifications, density (KDE), and vector arrows.
    show_arrows: Tuple of (vj_initial, uv_initial) if you want to draw arrows to (vj, uv).
    avg_error: Tuple of (vj_err, uv_err) to plot a representative error bar.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
    vj = np.asarray(vj)
    uv = np.asarray(uv)
    
    if show_density:
        sns.kdeplot(x=vj, y=uv, fill=True, cmap='cividis', levels=5, alpha=0.3, ax=ax)

    if classifications is not None:
        ax.scatter(vj[classifications == 0], uv[classifications == 0], c='r', s=15, label='Quiescent', alpha=0.6)
        ax.scatter(vj[classifications == 1], uv[classifications == 1], c='b', s=15, label='Star-forming', alpha=0.6)
        ax.scatter(vj[classifications == 2], uv[classifications == 2], c='g', s=15, label='Dusty', alpha=0.6)
    else:
        ax.scatter(vj, uv, c='k', s=10, alpha=0.4)

    if show_arrows:
        vj_init, uv_init = show_arrows
        for i in range(len(vj)):
            if not np.isnan(vj[i]) and not np.isnan(vj_init[i]):
                ax.arrow(vj_init[i], uv_init[i], vj[i] - vj_init[i], uv[i] - uv_init[i], 
                         head_width=0.03, head_length=0.05, fc='k', ec='k', alpha=0.3)

    if avg_error:
        vj_err, uv_err = avg_error
        ax.errorbar(1.8, 0.3, xerr=vj_err, yerr=uv_err, fmt='o', markersize=5, 
                    color='k', capsize=3, label='Avg Error')

    # Styling and Regions
    ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([0, 2.5])
    ax.set_xlabel('Restframe V - J')
    ax.set_ylabel('Restframe U - V')
    ax.set_title(title)
    
    # Selection Patches
    path_q = [[-0.5, 1.3], [0.85, 1.3], [1.6, 1.95], [1.6, 2.5], [-0.5, 2.5], [-0.5, 1.3]]
    ax.add_patch(plt.Polygon(path_q, closed=True, fill=True, facecolor='red', alpha=0.05, edgecolor='k', lw=1.5))
    
    ax.plot([1.2, 1.2], [0, 1.6], 'k--', lw=1) # Dusty/SF boundary line
    
    ax.text(-0.3, 2.3, 'Quiescent', fontsize=12, fontweight='bold')
    ax.text(-0.3, 0.2, 'Star-forming', fontsize=12, fontweight='bold')
    ax.text(1.8, 2.3, 'Dusty', fontsize=12, fontweight='bold')
    
    if classifications is not None or avg_error:
        ax.legend(loc='lower right')
        
    return ax
