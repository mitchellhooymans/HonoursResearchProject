"""
recreate_observational_results.py

Recreates Figure 7, Figure 8, and Table 4 from the paper using the 
observational ZFOURGE composite data generated during the honours phase.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sed_pipeline import config, visualization, photometry, analysis

def recreate_observational_uvj_results(agn_type='Type1'):
    print(f"--- Recreating Observational UVJ Results for {agn_type} AGN ---")
    
    # 1. Load the pre-generated observational composite data
    csv_path = os.path.join('outputs', 'composite_seds', f'ZFOURGE_obsevational_composites_fluxes{agn_type}AGN.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please ensure honours outputs are available.")
        return
    
    df = pd.read_csv(csv_path, index_col=0)
    
    # 2. Re-classify the base galaxies (Alpha=0)
    # The magnitudes are already in AB as per the CSV creation logic
    uv_0 = df['U_0'] - df['V_0']
    vj_0 = df['V_0'] - df['J_0']
    classifications = photometry.classify_uvj(vj_0, uv_0)
    df['Classification'] = classifications

    # 3. Figure 7: Density Plot of Composites at a specific Alpha (e.g. 50%)
    alpha_target = 50
    uv_alpha = df[f'U_{alpha_target}'] - df[f'V_{alpha_target}']
    vj_alpha = df[f'V_{alpha_target}'] - df[f'J_{alpha_target}']
    
    # Use the visualization module
    visualization.plot_uvj_diagram(vj_alpha, uv_alpha, classifications=classifications, 
                                   show_density=True, title=f"ZFOURGE UVJ Density (Alpha={alpha_target}%)")
    
    os.makedirs(os.path.join('outputs', 'ThesisPlots'), exist_ok=True)
    fig7_path = os.path.join('outputs', 'ThesisPlots', f'ZFOURGE_UVJ_Density_Alpha{alpha_target}.png')
    plt.savefig(fig7_path, dpi=300)
    print(f"Saved Figure 7: {fig7_path}")
    plt.close()

    # 4. Figure 8: Average Galaxy tracks
    alphas = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    regions = {0: 'Quiescent', 1: 'Star-forming', 2: 'Dusty'}
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    
    fig, ax = plt.subplots(figsize=(8, 8))
    visualization.plot_uvj_diagram([], [], ax=ax, title="ZFOURGE Average Region Tracks")
    
    for class_id, name in regions.items():
        subset = df[df['Classification'] == class_id]
        if len(subset) == 0:
            continue
            
        track_vj = []
        track_uv = []
        for a in alphas:
            uv_a = subset[f'U_{a}'] - subset[f'V_{a}']
            vj_a = subset[f'V_{a}'] - subset[f'J_{a}']
            track_uv.append(uv_a.mean())
            track_vj.append(vj_a.mean())
            
        ax.plot(track_vj, track_uv, marker='o', markersize=8, color=colors[class_id], label=name, lw=2)
        # Add arrow for direction (from alpha=0 to alpha=100)
        ax.annotate('', xy=(track_vj[-1], track_uv[-1]), xytext=(track_vj[0], track_uv[0]),
                    arrowprops=dict(arrowstyle="->", color=colors[class_id], lw=1.5, alpha=0.5))

    ax.legend(loc='lower right')
    fig8_path = os.path.join('outputs', 'ThesisPlots', 'ZFOURGE_Average_Region_Tracks.png')
    plt.savefig(fig8_path, dpi=300)
    print(f"Saved Figure 8: {fig8_path}")
    plt.close()

    # 5. Table 4: Mean Vector Offsets (dex)
    table_data = []
    for a in alphas:
        row = {'Alpha (%)': a}
        for class_id, name in regions.items():
            subset = df[df['Classification'] == class_id]
            if len(subset) == 0:
                row[name] = 0.0
                continue
                
            uv_0 = subset['U_0'] - subset['V_0']
            vj_0 = subset['V_0'] - subset['J_0']
            uv_a = subset[f'U_{a}'] - subset[f'V_{a}']
            vj_a = subset[f'V_{a}'] - subset[f'J_{a}']
            
            # Use analysis module for vector offset calculation
            offset = analysis.calculate_mean_vector_offset(vj_a, uv_a, vj_0, uv_0)
            row[name] = round(offset, 3)
        table_data.append(row)
    
    offset_df = pd.DataFrame(table_data)
    print("\nTable 4: Mean Vector Offsets per Region")
    print(offset_df)
    
    table4_path = os.path.join('outputs', 'ThesisPlots', 'ZFOURGE_Population_VectorOffsets.csv')
    offset_df.to_csv(table4_path, index=False)
    print(f"Saved Table 4: {table4_path}")

if __name__ == "__main__":
    recreate_observational_uvj_results('Type1')
