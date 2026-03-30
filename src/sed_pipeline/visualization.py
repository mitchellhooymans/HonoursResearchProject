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

def plot_ugr_diagram(gr, ug, ax=None, title="ugr Colour-Colour Diagram"):
    """
    Plots the ugr diagram with the U-dropout selection wedge.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
    gr = np.asarray(gr)
    ug = np.asarray(ug)
    
    ax.scatter(gr, ug, c='k', s=10, alpha=0.4)
    
    ax.set_xlim([-1, 4])
    ax.set_ylim([-1, 8.5])
    ax.set_xlabel('g - r')
    ax.set_ylabel('u - g')
    ax.set_title(title)
    
    # U-dropout selection wedge
    u_rule = [[1.2, 9], [1.2, 2.2], [0.6, 1.6], [-3, 1.6], [-3, 9], [1.2, 9]]
    ax.add_patch(plt.Polygon(u_rule, closed=True, fill=True, facecolor=(1, 0, 0, 0.05), edgecolor='k', lw=1.5, label='U-dropout (2.6 < z < 3.6)'))
    
    # Do not call ax.legend() blindly if no scatter has labels, but here we labelled the wedge
    ax.legend(loc='upper right')
    return ax

def plot_irac_diagram(f5836, f8045, ax=None, title="IRAC Colour-Colour Diagram"):
    """
    Plots the IRAC colour-colour diagram with the Lacy AGN selection wedge.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
    x = np.asarray(f5836)
    y = np.asarray(f8045)
    
    ax.scatter(x, y, c='k', s=10, alpha=0.4)
    
    xmax = 0.8
    ymax = 1.2
    xmin = -0.6
    ymin = -0.8
    
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel('log(f$_{5.8}$/f$_{3.6}$)')
    ax.set_ylabel('log(f$_{8.0}$/f$_{4.5}$)')
    ax.set_title(title)
    
    # Lacy wedge boundaries: x > -0.1, y > -0.2, y < 0.8*x + 0.5
    # Find minimum valid x for plotting the wedge nicely
    lacy_selection_condition = (x > -0.1) & (y > -0.2) & (y < 0.8 * x + 0.5)
    min_x = np.nanmin(x[lacy_selection_condition]) if np.any(lacy_selection_condition) else -0.1
    
    # Clamp min_x to not go below -0.1 to maintain wedge shape
    min_x = max(min_x, -0.1)
    
    wedge_vertices = [
        (xmax, -0.2),
        (-0.1, -0.2),
        (-0.1, 0.8 * min_x + 0.5),
        (((ymax - 0.5) / 0.8), ymax),
        (xmax, ymax), # Close it nicely on the right edge
        (xmax, -0.2)
    ]
    ax.add_patch(plt.Polygon(wedge_vertices, closed=True, fill=True, facecolor=(1, 0, 0, 0.05), edgecolor='r', lw=1.5, label='Lacy Wedge'))
    
    ax.legend(loc='lower right')
    return ax

def plot_composite_sed_progression(composite_data, filters, population_names=['Star-forming', 'Quiescent', 'Dusty'], 
                                   alphas_to_label=[0.0, 0.5, 1.0], output_path=None):
    """
    Recreates Fig 6: 1D SED progression for each population type.
    composite_data: Dictionary {pop_name: {alpha: SED DataFrame}}
    filters: Dictionary {filter_name: Passband object}
    """
    fig, axes = plt.subplots(len(population_names), 1, figsize=(12, 12), sharex=True)
    if len(population_names) == 1:
        axes = [axes]
        
    for i, pop in enumerate(population_names):
        ax = axes[i]
        pop_seds = composite_data.get(pop, {})
        
        # Use a colormap for alpha
        cmap = plt.cm.plasma
        
        sorted_alphas = sorted(pop_seds.keys())
        for alpha in sorted_alphas:
            sed_df = pop_seds[alpha]
            # Label only specific alphas to keep legend clean
            label = f'Alpha = {int(alpha*100)}%' if any(np.isclose(alpha, a) for a in alphas_to_label) else None
            ax.loglog(sed_df['lambda (Angstroms)'], sed_df['Total Flux (erg/s/cm^2/Angstrom)'], 
                      color=cmap(alpha), alpha=0.7, label=label)
        
        # Overlay filters (U, V, J)
        # Scale transmission for visualization (sc=1e-2 as per honours script)
        sc = 1e-2
        for f_name in ['U', 'V', 'J']:
            if f_name in filters:
                pb = filters[f_name]
                # Normalize transmission for the plot if it isn't
                trans = pb.transmission / np.max(pb.transmission) if np.max(pb.transmission) > 0 else pb.transmission
                ax.loglog(pb.wavelength, sc * trans, lw=2, label=f_name if i==0 else None)
                # Plot the filter letter
                ax.text(np.mean(pb.wavelength), sc * 0.1, f_name, fontsize=14, fontweight='bold', ha='center')

        ax.set_ylabel('Flux (erg/s/cm$^2$/$\AA$)')
        ax.set_title(f'Composite SED Progression: {pop}', fontweight='bold')
        
        ax.set_xlim(1e2, 3.4e5)
        ax.set_ylim(1e-5, 1e3)
        
        if i == 0:
            ax.legend(loc='upper right', ncol=2, fontsize=10)

    axes[-1].set_xlabel('Wavelength ($\AA$)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig, axes
