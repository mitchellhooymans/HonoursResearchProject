# %% [markdown]
# # Master Results Notebook for Paper Production
# 
# This notebook serves as the primary driver for generating the final, publication-quality figures and tables for the paper. It leverages the consolidated logic in `src/glass/` to ensure zero data loss and reproducible results.

# %%
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Ensure the project root is in the path to import the local package
sys.path.append(os.path.abspath('..'))

from src.glass import config, data_io, composite_math, photometry, visualization, analysis

# PASA publication styling
plt.style.use('default')
visualization.apply_pasa_style()
os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Load Configurations and Base Models
# 
# We start by loading the necessary filters (UVJ, ugr, IRAC) and the base Type 1 and Type 2 AGN SKIRTOR models using parameters defined in the project's modelling methodology.

# %%
# Load Filters
filters = photometry.load_passbands(config.FILTER_PATHS)

# Set specific paths for templates
skirtor_dir = os.path.join(config.RAW_DATA_DIR, 'Templates', 'Skirtor')
brown_dir = os.path.join(config.RAW_DATA_DIR, 'Templates', 'Brown', '2014', 'Rest')

# Load AGN Models
agn_type1 = data_io.read_skirtor_model(skirtor_dir, **config.SKIRTOR_TYPE1_PARAMS)
agn_type2 = data_io.read_skirtor_model(skirtor_dir, **config.SKIRTOR_TYPE2_PARAMS)

# Load GALSEDATLAS (Brown) Templates
brown_templates, brown_names = data_io.read_brown_galaxy_templates(brown_dir)

# %% [markdown]
# ## 2. Generate Composite SEDs and Calculate Colours
# 
# We will iterate over the alpha values, calculate composite SEDs for all templates and both AGN types, and compute their colours. For `ugr` and `IRAC` diagrams, specific artificial redshifting is applied to reflect survey selections.

# %%
results = []
alphas = config.ALPHA_VALUES

# Target redshifts for artificial redshifting
z_ugr = 3.0 # Within the U-dropout target range of 2.6 < z < 3.6
z_irac = 0.0 # Restframe for standard Lacy wedge demo

for t_idx, gal_sed in enumerate(brown_templates):
    gal_name = brown_names[t_idx]
    print(f"Processing template {t_idx+1}/{len(brown_templates)}: {gal_name}")
    
    for agn_name, agn_model in [('Type1', agn_type1), ('Type2', agn_type2)]:
        for alpha in alphas:
            comp_sed = composite_math.create_composite_sed(agn_model, gal_sed, alpha)
            
            # Calculate UVJ (restframe)
            uv, vj = photometry.calculate_UVJ_colours(comp_sed, filters['U'], filters['V'], filters['J'])
            
            # Calculate ugr (redshifted to z=3.0)
            ug, gr = photometry.calculate_ugr_colours(comp_sed, filters['u'], filters['g'], filters['r'], redshift=z_ugr)
            
            # Calculate IRAC (restframe)
            f5836, f8045 = photometry.calculate_irac_colours(comp_sed, filters['3.6'], filters['4.5'], filters['5.8'], filters['8.0'], redshift=z_irac)
            
            results.append({
                'Template': gal_name,
                'AGN_Type': agn_name,
                'Alpha': alpha,
                'U-V': uv,
                'V-J': vj,
                'u-g': ug,
                'g-r': gr,
                'f5836': f5836,
                'f8045': f8045
            })

df_results = pd.DataFrame(results)
print("Processed all composite combinations.")

# %% [markdown]
# ## 3. Colour Evolution Evolution Plots
# 
# These plots show how individual galaxies move through colour-colour space as the AGN contribution ($\alpha$) increases from 0 to 1.

# %%
def plot_evolution_tracks(df, agn_type, x_col, y_col, plot_func, title, output_name):
    fig, ax = plt.subplots(figsize=visualization.PASA_SQ)
    subset = df[df['AGN_Type'] == agn_type]

    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=1)

    for template in subset['Template'].unique():
        temp_df = subset[subset['Template'] == template].sort_values('Alpha')
        xs = temp_df[x_col].values
        ys = temp_df[y_col].values
        for i in range(len(temp_df) - 1):
            ax.plot(xs[i:i+2], ys[i:i+2],
                    color=cmap(norm(temp_df['Alpha'].iloc[i])),
                    linewidth=2.5, alpha=0.72, solid_capstyle='round')
        ax.scatter(xs[0],  ys[0],  color='royalblue', s=20, zorder=5, alpha=0.5)
        ax.scatter(xs[-1], ys[-1], color='firebrick',  s=20, zorder=5, alpha=0.5)

    plot_func([], [], ax=ax, title=title)

    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', markersize=5, label=r'Pure galaxy ($\alpha=0$)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='firebrick',  markersize=5, label=r'Max AGN ($\alpha=1$)'),
    ]
    ax.legend(handles=custom_lines, loc='lower right')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'AGN Contribution ($\alpha$)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(config.PROCESSED_DATA_DIR, output_name), dpi=300, bbox_inches='tight')
    return fig, ax

plot_evolution_tracks(df_results, 'Type1', 'V-J', 'U-V', visualization.plot_uvj_diagram,
                      '', 'Paper_UVJ_Evolution_Type1.pdf')
plot_evolution_tracks(df_results, 'Type2', 'V-J', 'U-V', visualization.plot_uvj_diagram,
                      '', 'Paper_UVJ_Evolution_Type2.pdf')

plot_evolution_tracks(df_results, 'Type1', 'g-r', 'u-g', visualization.plot_ugr_diagram,
                      '', 'Paper_ugr_Evolution_Type1.pdf')
plot_evolution_tracks(df_results, 'Type2', 'g-r', 'u-g', visualization.plot_ugr_diagram,
                      '', 'Paper_ugr_Evolution_Type2.pdf')

plot_evolution_tracks(df_results, 'Type1', 'f5836', 'f8045', visualization.plot_irac_diagram,
                      '', 'Paper_IRAC_Evolution_Type1.pdf')
plot_evolution_tracks(df_results, 'Type2', 'f5836', 'f8045', visualization.plot_irac_diagram,
                      '', 'Paper_IRAC_Evolution_Type2.pdf')

plt.show()

# %% [markdown]
# ## 3b. SWIRE Template Evolution (Fig 3)
# 
# Generating initial composite modelling using the SWIRE template library.

# %%
# Load SWIRE Templates
swire_dir = os.path.join(config.RAW_DATA_DIR, 'Templates', 'SWIRE')
swire_templates, swire_names = data_io.read_swire_templates(swire_dir)

swire_results = []

for t_idx, gal_sed in enumerate(swire_templates):
    gal_name = swire_names[t_idx]
    for agn_name, agn_model in [('Type1', agn_type1), ('Type2', agn_type2)]:
        for alpha in alphas:
            comp_sed = composite_math.create_composite_sed(agn_model, gal_sed, alpha)
            uv, vj = photometry.calculate_UVJ_colours(comp_sed, filters['U'], filters['V'], filters['J'])
            swire_results.append({'Template': gal_name, 'AGN_Type': agn_name, 'Alpha': alpha, 'U-V': uv, 'V-J': vj})

df_swire_results = pd.DataFrame(swire_results)

plot_evolution_tracks(df_swire_results, 'Type1', 'V-J', 'U-V', visualization.plot_uvj_diagram, '', 'SWIREType1AGNUVJ.pdf')
plot_evolution_tracks(df_swire_results, 'Type2', 'V-J', 'U-V', visualization.plot_uvj_diagram, '', 'SWIREType2AGNUVJ.pdf')
plt.show()

# %% [markdown]
# ## 4. Selection Completeness Analysis
# 
# Evaluating how well the standard selection wedges (UVJ, ugr dropout, Lacy IRAC) identify their targets as AGN contamination increases.

# %%
# 1. ugr Dropout Completeness (from redshift grid data)
grid_file = os.path.join(config.PROCESSED_DATA_DIR, 'ugr_completeness_grid.csv')
if os.path.exists(grid_file):
    df_grid = pd.read_csv(grid_file)
    df_comp_ugr = analysis.calculate_completeness_from_df(df_grid)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for agn_type in df_comp_ugr['AGN_Type'].unique():
        subset = df_comp_ugr[df_comp_ugr['AGN_Type'] == agn_type]
        ax.plot(subset['Alpha'], subset['Completeness'], marker='o', label=f'{agn_type} AGN')
        
    ax.set_xlabel(r'AGN Contribution ($\alpha$)')
    ax.set_ylabel('Completeness')
    ax.set_title('ugr Dropout Selection Completeness')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.savefig(os.path.join(config.PROCESSED_DATA_DIR, 'Paper_ugr_Completeness.pdf'), bbox_inches='tight')
    plt.show()

# 2. IRAC Lacy Wedge Completeness (calculated from df_results)
def calculate_irac_completeness(df):
    res = []
    for agn_type in df['AGN_Type'].unique():
        subset = df[df['AGN_Type'] == agn_type]
        for alpha in subset['Alpha'].unique():
            alpha_df = subset[subset['Alpha'] == alpha]
            in_wedge = analysis.is_in_lacy_wedge(alpha_df['f5836'], alpha_df['f8045'])
            # For IRAC, completeness is fraction identified as AGN
            completeness = np.sum(in_wedge) / len(alpha_df)
            res.append({'AGN_Type': agn_type, 'Alpha': alpha, 'Completeness': completeness})
    return pd.DataFrame(res)

df_comp_irac = calculate_irac_completeness(df_results)

fig, ax = plt.subplots(figsize=(8, 6))
for agn_type in df_comp_irac['AGN_Type'].unique():
    subset = df_comp_irac[df_comp_irac['AGN_Type'] == agn_type]
    ax.plot(subset['Alpha'], subset['Completeness'], marker='s', label=f'{agn_type} AGN')
    
ax.set_xlabel(r'AGN Contribution ($\alpha$)')
ax.set_ylabel('AGN Selection Fraction')
ax.set_title('IRAC Lacy Wedge AGN Identification')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
fig.savefig(os.path.join(config.PROCESSED_DATA_DIR, 'Paper_IRAC_Completeness.pdf'), bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Statistical Tables (LaTeX Exports)
# 
# Exporting the final results to LaTeX for inclusion in the paper.

# %%
def calculate_uvj_offsets(df, agn_type):
    subset = df[df['AGN_Type'] == agn_type]
    offsets = []
    for alpha in subset['Alpha'].unique():
        alpha_df = subset[subset['Alpha'] == alpha]
        initial_df = subset[subset['Alpha'] == 0.0]
        
        vj_alpha = alpha_df['V-J'].values
        uv_alpha = alpha_df['U-V'].values
        vj_init = initial_df['V-J'].values
        uv_init = initial_df['U-V'].values
        
        mean_offset = analysis.calculate_mean_vector_offset(vj_alpha, uv_alpha, vj_init, uv_init)
        offsets.append({'Alpha': alpha, 'Mean_Offset': mean_offset})
    return pd.DataFrame(offsets)

off1 = calculate_uvj_offsets(df_results, 'Type1')
off2 = calculate_uvj_offsets(df_results, 'Type2')

table_df = pd.DataFrame({
    r'AGN Contribution ($\\alpha$)': [f"{int(a*100)}\\%" for a in alphas],
    'Type 1 Offset (dex)': off1['Mean_Offset'].round(3),
    'Type 2 Offset (dex)': off2['Mean_Offset'].round(3)
})

display(table_df)
table_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, 'UVJ_vectoroffset.csv'), index=False)
latex_table = table_df.to_latex(index=False, caption=r"Mean vector offset for Type 1 and Type 2 AGN composites in the UVJ diagram.", label="tab:UVJ_vectoroffset")
with open(os.path.join(config.PROCESSED_DATA_DIR, 'UVJ_vectoroffset.tex'), 'w') as f:
    f.write(latex_table)

print("Exported UVJ Vector Offset table.")

# %% [markdown]
# ## 6. UVJ Population Fraction Analysis
# 
# This section analyzes how AGN contamination causes galaxies to migrate between the Star-forming, Quiescent, and Dusty regions of the UVJ diagram.

# %%
def calculate_uvj_fractions(df, agn_type):
    subset = df[df['AGN_Type'] == agn_type]
    alphas_sorted = sorted(subset['Alpha'].unique())
    q_fracs, sf_fracs, d_fracs = [], [], []
    for alpha in alphas_sorted:
        alpha_df = subset[subset['Alpha'] == alpha]
        classifications = analysis.classify_uvj(alpha_df['V-J'], alpha_df['U-V'])
        q, sf, d = analysis.calculate_population_fractions(classifications)
        q_fracs.append(q)
        sf_fracs.append(sf)
        d_fracs.append(d)
    return alphas_sorted, q_fracs, sf_fracs, d_fracs

fig, ax = plt.subplots(figsize=(3.31, 2.8))

pop_colors = {'Quiescent': '#CC2929', 'Star-forming': '#1A6FB5', 'Dusty': '#E07B00'}

for agn_type, ls in [('Type1', '-'), ('Type2', '--')]:
    alphas_sorted, q_fracs, sf_fracs, d_fracs = calculate_uvj_fractions(df_results, agn_type)
    type_label = 'Type 1' if agn_type == 'Type1' else 'Type 2'
    ax.plot(alphas_sorted, q_fracs,  color=pop_colors['Quiescent'],    linestyle=ls, marker='o', markersize=3, label=f'Quiescent ({type_label})')
    ax.plot(alphas_sorted, sf_fracs, color=pop_colors['Star-forming'], linestyle=ls, marker='s', markersize=3, label=f'Star-forming ({type_label})')
    ax.plot(alphas_sorted, d_fracs,  color=pop_colors['Dusty'],        linestyle=ls, marker='^', markersize=3, label=f'Dusty ({type_label})')

ax.set_xlabel(r'AGN Contribution ($\alpha$)')
ax.set_ylabel('Population Fraction')
ax.legend(fontsize=6, loc='center right', ncol=1)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim([0, 1.1])

plt.tight_layout()
fig.savefig(os.path.join(config.PROCESSED_DATA_DIR, 'Paper_UVJ_Fractions_Combined.pdf'), dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Composite SED Progression (Fig 6)
# 
# Recreating Figure 6: 1D SED progression for representative Star-forming, Quiescent, and Dusty galaxies with increasing Type 1 AGN contribution.

# %%
# Representative templates from Brown (GALSEDATLAS) set
# Star-forming: NGC_0337, Quiescent: NGC_4552, Dusty: IC_4553 (Arp 220)
rep_templates = {
    'Star-forming': 'NGC_0337',
    'Quiescent': 'NGC_4552',
    'Dusty': 'IC_4553'
}

comp_progression_data = {}

for pop_name, gal_name in rep_templates.items():
    if gal_name in brown_names:
        t_idx = brown_names.index(gal_name)
        gal_sed = brown_templates[t_idx]
        comp_progression_data[pop_name] = {}
        for alpha in alphas:
            comp_sed = composite_math.create_composite_sed(agn_type1, gal_sed, alpha)
            comp_progression_data[pop_name][alpha] = comp_sed

visualization.plot_composite_sed_progression(
    comp_progression_data, filters,
    show_title=True,
    orientation='horizontal',
    figsize=visualization.PASA_SED_WIDE,
    output_path=os.path.join(config.PROCESSED_DATA_DIR, 'CompositeSEDs_UVJ.pdf')
)
plt.show()

# %% [markdown]
# ## 8. Observational ZFOURGE Composites (Fig 7, 8, Table 4)
# 
# This section confirms theoretical trends using observational ZFOURGE data by tracking how the average galaxy in each UVJ region evolves with increasing Type 1 AGN contribution.

# %%
# Load the pre-generated observational composite data (calculated during Honours phase)
obs_csv_path = os.path.join('..\outputs', 'composite_seds', 'ZFOURGE_obsevational_composites_fluxesType1AGN.csv')
if os.path.exists(obs_csv_path):
    df_obs = pd.read_csv(obs_csv_path, index_col=0)

    # Re-classify Alpha=0 galaxies
    df_obs['Classification'] = photometry.classify_uvj(df_obs['V_0'] - df_obs['J_0'], df_obs['U_0'] - df_obs['V_0'])

    # Fig 7: UVJ Density at Alpha=0%
    uv_0 = df_obs['U_0'] - df_obs['V_0']
    vj_0 = df_obs['V_0'] - df_obs['J_0']
    fig, ax = plt.subplots(figsize=visualization.PASA_SQ)
    visualization.plot_uvj_diagram(vj_0, uv_0, classifications=df_obs['Classification'],
                                   show_density=True, ax=ax, title="",
                                   scatter_colors=('#E03030', '#1A85D0', '#E08800'))
    plt.tight_layout()
    fig.savefig(os.path.join(config.PROCESSED_DATA_DIR, 'ZFOURGE_UVJ_Density_alpha0.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

    # Fig 8: Average Region Tracks — use vibrant print-safe colours matching scatter
    regions = {0: 'Quiescent', 1: 'Star-forming', 2: 'Dusty'}
    colors  = {0: '#CC2929',   1: '#1A6FB5',       2: '#E07B00'}
    fig, ax = plt.subplots(figsize=visualization.PASA_SQ)
    visualization.plot_uvj_diagram([], [], ax=ax, title="")

    for cid, name in regions.items():
        sub = df_obs[df_obs['Classification'] == cid]
        if len(sub) == 0: continue
        tvj = [sub[f'V_{a}'].mean() - sub[f'J_{a}'].mean() for a in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
        tuv = [sub[f'U_{a}'].mean() - sub[f'V_{a}'].mean() for a in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
        ax.plot(tvj, tuv, marker='o', color=colors[cid], label=name, lw=2, markersize=4)
        ax.annotate('', xy=(tvj[-1], tuv[-1]), xytext=(tvj[0], tuv[0]),
                    arrowprops=dict(arrowstyle="->", color=colors[cid], lw=1.5))
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(config.PROCESSED_DATA_DIR, 'ZFOURGE_UVJ_RegionTracks.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
else:
    print(f"Observational composite file not found at: {obs_csv_path}")

# %%
# --- Type 1 AGN UVJ Density Evolution (4-panel, ZFOURGE observational data) ---
alpha_cols   = [0, 30, 70, 100]
alpha_labels = [r'$\alpha=0\%$', r'$\alpha=30\%$', r'$\alpha=70\%$', r'$\alpha=100\%$']

fig, axes = plt.subplots(1, 4, figsize=(6.85, 2.5), sharex=True, sharey=True)

for ax, a, label in zip(axes, alpha_cols, alpha_labels):
    vj = df_obs[f'V_{a}'] - df_obs[f'J_{a}']
    uv = df_obs[f'U_{a}'] - df_obs[f'V_{a}']
    valid = vj.notna() & uv.notna()
    sns.kdeplot(x=vj[valid], y=uv[valid],
                fill=True, cmap='Blues', levels=5, alpha=0.75, ax=ax)
    visualization.plot_uvj_diagram([], [], ax=ax, title=label)
    for txt in ax.texts:
        txt.set_fontsize(6.5)
        if 'Star-forming' in txt.get_text():
            txt.set_position((txt.get_position()[0], 0.4))
    ax.set_xlabel('')
    ax.set_ylabel('')

axes[0].set_ylabel('Restframe U - V')
fig.text(0.5, -0.01, 'Restframe V - J', ha='center', fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(config.PROCESSED_DATA_DIR, 'UVJ_Type1_Density_Evolution.pdf'), dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9. CIGALE Decomposed Population Analysis (Fig 9, 10, 11)
# 
# Final observational support: analyzing the UVJ shift of the entire ZFOURGE population when AGN light is removed via CIGALE SED decomposition.

# %%
# Load CIGALE summary table
cigale_csv = os.path.join('..\datasets', 'full_zfourge_decomposed', 'zfourge_full_final.csv')
if os.path.exists(cigale_csv):
    df_cig = pd.read_csv(cigale_csv)
    z_col = 'zpk_x' if 'zpk_x' in df_cig.columns else 'zpk'

    # Fig 9: Combined full+decomposed with per-galaxy faint arrows and bold mean vectors
    fig, ax = plt.subplots(figsize=visualization.PASA_SQ)
    visualization.plot_uvj_diagram([], [], ax=ax, title="")

    # Faint per-galaxy arrows: full → decomposed
    for _, row in df_cig.dropna(subset=['VJ_Full', 'UV_Full', 'VJ_Decomposed', 'UV_Decomposed']).iterrows():
        ax.annotate('', xy=(row['VJ_Decomposed'], row['UV_Decomposed']),
                    xytext=(row['VJ_Full'], row['UV_Full']),
                    arrowprops=dict(arrowstyle='->', color='#888888', alpha=0.12, lw=0.5))

    # Scatter both populations
    ax.scatter(df_cig['VJ_Full'],       df_cig['UV_Full'],       c='#444444',   s=6, alpha=0.3, label='Full (Host+AGN)',  zorder=3)
    ax.scatter(df_cig['VJ_Decomposed'], df_cig['UV_Decomposed'], c='darkorange', s=6, alpha=0.3, label='Decomposed host', zorder=3)

    # Bold mean vectors per UVJ region (classify by full-galaxy position)
    classifications_full = photometry.classify_uvj(df_cig['VJ_Full'], df_cig['UV_Full'])
    region_colors = {0: '#CC2929', 1: '#1A6FB5', 2: '#E07B00'}
    region_names  = {0: 'Quiescent', 1: 'Star-forming', 2: 'Dusty'}
    for cid in [0, 1, 2]:
        mask = classifications_full == cid
        if mask.sum() < 3: continue
        mvj_f = df_cig.loc[mask, 'VJ_Full'].mean()
        muv_f = df_cig.loc[mask, 'UV_Full'].mean()
        mvj_d = df_cig.loc[mask, 'VJ_Decomposed'].mean()
        muv_d = df_cig.loc[mask, 'UV_Decomposed'].mean()
        ax.annotate('', xy=(mvj_d, muv_d), xytext=(mvj_f, muv_f),
                    arrowprops=dict(arrowstyle='->', color=region_colors[cid], lw=2.5))
        ax.plot(mvj_f, muv_f, 'o', color=region_colors[cid], markersize=7, zorder=6)
        ax.plot(mvj_d, muv_d, '^', color=region_colors[cid], markersize=7, zorder=6,
                label=f'{region_names[cid]} shift')

    ax.legend(loc='lower right', fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(config.PROCESSED_DATA_DIR, 'UVJ_CIGALE_FullDecomp_Vectors.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

    # Fig 11: Redshift Divergence — mean full vs decomposed positions per redshift bin
    bins = [[0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0], [2.0, 2.5], [2.5, 3.5]]
    fig, ax = plt.subplots(figsize=visualization.PASA_SQ)
    visualization.plot_uvj_diagram([], [], ax=ax, title="")
    cmap = plt.cm.viridis
    for i, (z_min, z_max) in enumerate(bins):
        sub = df_cig[(df_cig[z_col] >= z_min) & (df_cig[z_col] < z_max)]
        if len(sub) < 5: continue
        mvj_f, muv_f = sub['VJ_Full'].mean(), sub['UV_Full'].mean()
        mvj_d, muv_d = sub['VJ_Decomposed'].mean(), sub['UV_Decomposed'].mean()
        c = cmap(i / len(bins))
        ax.plot(mvj_f, muv_f, 'o', color=c, markersize=6, label=f'z=[{z_min},{z_max}]')
        ax.plot(mvj_d, muv_d, '^', color=c, markersize=6)
        ax.annotate('', xy=(mvj_d, muv_d), xytext=(mvj_f, muv_f),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2, alpha=0.7))
    ax.legend(loc='lower right', title="Redshift bins", title_fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(config.PROCESSED_DATA_DIR, 'UVJ_agn_evolution_CIGALE_ZFOURGE_comparison_redshift.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("CIGALE summary table not found.")


