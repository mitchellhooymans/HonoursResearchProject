import json
import os

nb_path = 'notebooks/Paper_Results_Master.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

markdown_cell = {
   'cell_type': 'markdown',
   'metadata': {},
   'source': [
    '## 9. CIGALE Decomposed Population Analysis (Fig 9, 10, 11)\n',
    '\n',
    'Final observational support: analyzing the UVJ shift of the entire ZFOURGE population when AGN light is removed via CIGALE SED decomposition.'
   ]
}

code_cell = {
   'cell_type': 'code',
   'execution_count': None,
   'metadata': {},
   'outputs': [],
   'source': [
    '# Load CIGALE summary table\n',
    'cigale_csv = os.path.join(\'..\\datasets\', \'full_zfourge_decomposed\', \'zfourge_full_final.csv\')\n',
    'if os.path.exists(cigale_csv):\n',
    '    df_cig = pd.read_csv(cigale_csv)\n',
    '    z_col = \'zpk_x\' if \'zpk_x\' in df_cig.columns else \'zpk\'\n',
    '\n',
    '    # Fig 9: Side-by-side\n',
    '    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)\n',
    '    visualization.plot_uvj_diagram(df_cig[\'VJ_Full\'], df_cig[\'UV_Full\'], ax=axes[0], title=\"Full Galaxy (Host + AGN)\")\n',
    '    visualization.plot_uvj_diagram(df_cig[\'VJ_Decomposed\'], df_cig[\'UV_Decomposed\'], ax=axes[1], title=\"Decomposed Host (AGN Removed)\")\n',
    '    plt.show()\n',
    '\n',
    '    # Fig 11: Redshift Divergence\n',
    '    bins = [[0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0], [2.0, 2.5], [2.5, 3.5]]\n',
    '    fig, ax = plt.subplots(figsize=(8, 8))\n',
    '    visualization.plot_uvj_diagram([], [], ax=ax, title=\"UVJ Divergence by Redshift\")\n',
    '    cmap = plt.cm.viridis\n',
    '    for i, (z_min, z_max) in enumerate(bins):\n',
    '        sub = df_cig[(df_cig[z_col] >= z_min) & (df_cig[z_col] < z_max)]\n',
    '        if len(sub) < 5: continue\n',
    '        mvj_f, muv_f = sub[\'VJ_Full\'].mean(), sub[\'UV_Full\'].mean()\n',
    '        mvj_d, muv_d = sub[\'VJ_Decomposed\'].mean(), sub[\'UV_Decomposed\'].mean()\n',
    '        c = cmap(i / len(bins))\n',
    '        ax.plot(mvj_f, muv_f, \'o\', color=c, markersize=10, label=f\'z=[{z_min},{z_max}]\')\n',
    '        ax.plot(mvj_d, muv_d, \'^\', color=c, markersize=10)\n',
    '        ax.annotate(\'\', xy=(mvj_d, muv_d), xytext=(mvj_f, muv_f),\n',
    '                    arrowprops=dict(arrowstyle=\"->\", color=c, lw=2, alpha=0.7))\n',
    '    ax.legend(loc=\'lower right\', title=\"Redshift Bins\")\n',
    '    plt.show()\n',
    'else:\n',
    '    print(\"CIGALE summary table not found.\")\n'
   ]
}

nb['cells'].append(markdown_cell)
nb['cells'].append(code_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Notebook patched successfully.')
