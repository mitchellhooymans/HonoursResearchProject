"""
recreate_cigale_results.py

Recreates Figures 9, 10, and 11 from the paper using the CIGALE SED 
decomposition results for the ZFOURGE survey.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from glass import visualization, photometry, analysis

def recreate_cigale_decomposed_results():
    print("--- Recreating CIGALE Population Analysis ---")
    
    # 1. Load the comprehensive summary table
    summary_csv = os.path.join('datasets', 'full_zfourge_decomposed', 'zfourge_full_final.csv')
    if not os.path.exists(summary_csv):
        print(f"Error: {summary_csv} not found. Ensure honours datasets are available.")
        return
    
    df = pd.read_csv(summary_csv)
    
    # Identify columns
    z_col = 'zpk_x' if 'zpk_x' in df.columns else 'zpk'
    
    # 2. Figure 9: Side-by-side Full vs Decomposed
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    
    visualization.plot_uvj_diagram(df['VJ_Full'], df['UV_Full'], ax=axes[0], title="Full Galaxy (Host + AGN)")
    visualization.plot_uvj_diagram(df['VJ_Decomposed'], df['UV_Decomposed'], ax=axes[1], title="Decomposed Host (AGN Removed)")
    
    os.makedirs(os.path.join('outputs', 'ResultPlots'), exist_ok=True)
    fig9_path = os.path.join('outputs', 'ResultPlots', 'UVJ_CIGALE_SideBySide.png')
    plt.savefig(fig9_path, dpi=300)
    print(f"Saved Figure 9: {fig9_path}")
    plt.close()

    # 3. Figure 10: Comparative Shift
    fig, ax = plt.subplots(figsize=(8, 8))
    visualization.plot_uvj_diagram([], [], ax=ax, title="Population Migration (AGN -> Host)")
    
    # Plot background points
    ax.scatter(df['VJ_Full'], df['UV_Full'], c='blue', s=2, alpha=0.05, label='Full')
    ax.scatter(df['VJ_Decomposed'], df['UV_Decomposed'], c='red', s=2, alpha=0.05, label='Decomposed')
    
    # Plot a representative subset of individual movement arrows
    # Only for galaxies that significantly moved (cutoff by error)
    avg_err_vj = df['eVJ_Full'].mean() if 'eVJ_Full' in df.columns else 0.05
    avg_err_uv = df['eUV_Full'].mean() if 'eUV_Full' in df.columns else 0.05
    cutoff = np.sqrt(avg_err_vj**2 + avg_err_uv**2)
    
    dist = np.sqrt((df['VJ_Full'] - df['VJ_Decomposed'])**2 + (df['UV_Full'] - df['UV_Decomposed'])**2)
    df_moved = df[dist > cutoff]
    
    sample = df_moved.sample(min(150, len(df_moved)))
    for _, row in sample.iterrows():
        ax.annotate('', xy=(row['VJ_Decomposed'], row['UV_Decomposed']), 
                    xytext=(row['VJ_Full'], row['UV_Full']),
                    arrowprops=dict(arrowstyle="->", color='black', lw=0.5, alpha=0.2))
        
    # Plot population means for the "movers"
    mean_full = [df_moved['VJ_Full'].mean(), df_moved['UV_Full'].mean()]
    mean_dec = [df_moved['VJ_Decomposed'].mean(), df_moved['UV_Decomposed'].mean()]
    ax.annotate('Mean Shift', xy=(mean_dec[0], mean_dec[1]), xytext=(mean_full[0], mean_full[1]),
                arrowprops=dict(arrowstyle="->", color='blue', lw=4, alpha=0.8),
                fontsize=12, fontweight='bold', color='blue')
    
    ax.legend(loc='lower right')
    fig10_path = os.path.join('outputs', 'ResultPlots', 'UVJ_CIGALE_Migration_Arrows.png')
    plt.savefig(fig10_path, dpi=300)
    print(f"Saved Figure 10: {fig10_path}")
    plt.close()

    # 4. Figure 11: Redshift-Binned Divergence
    bins = [[0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0], [2.0, 2.5], [2.5, 3.5]]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    visualization.plot_uvj_diagram([], [], ax=ax, title="UVJ Evolution: Full (Circles) vs Decomposed (Triangles)")
    
    cmap = plt.cm.viridis
    for i, (z_min, z_max) in enumerate(bins):
        mask = (df[z_col] >= z_min) & (df[z_col] < z_max)
        subset = df[mask]
        if len(subset) < 5: continue
        
        m_f_vj, m_f_uv = subset['VJ_Full'].mean(), subset['UV_Full'].mean()
        m_d_vj, m_d_uv = subset['VJ_Decomposed'].mean(), subset['UV_Decomposed'].mean()
        
        color = cmap(i / len(bins))
        ax.plot(m_f_vj, m_f_uv, 'o', color=color, markersize=10, label=f'z=[{z_min},{z_max}]')
        ax.plot(m_d_vj, m_d_uv, '^', color=color, markersize=10)
        ax.annotate('', xy=(m_d_vj, m_d_uv), xytext=(m_f_vj, m_f_uv),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2, alpha=0.7))

    ax.legend(loc='lower right', title="Redshift Bins", fontsize='small')
    fig11_path = os.path.join('outputs', 'ResultPlots', 'UVJ_CIGALE_Redshift_Bins.png')
    plt.savefig(fig11_path, dpi=300)
    print(f"Saved Figure 11: {fig11_path}")
    plt.close()

if __name__ == "__main__":
    recreate_cigale_decomposed_results()
