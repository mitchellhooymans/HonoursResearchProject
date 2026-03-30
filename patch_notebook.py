import json
import os

nb_path = 'notebooks/Paper_Results_Master.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

markdown_cell = {
   'cell_type': 'markdown',
   'metadata': {},
   'source': [
    '## 8. Observational ZFOURGE Composites (Fig 7, 8, Table 4)\n',
    '\n',
    'This section confirms theoretical trends using observational ZFOURGE data by tracking how the average galaxy in each UVJ region evolves with increasing Type 1 AGN contribution.'
   ]
}

code_cell = {
   'cell_type': 'code',
   'execution_count': None,
   'metadata': {},
   'outputs': [],
   'source': [
    '# Load the pre-generated observational composite data (calculated during Honours phase)\n',
    '# Note: Using relative path from notebook location\n',
    'obs_csv_path = os.path.join(\'..\\outputs\', \'composite_seds\', \'ZFOURGE_obsevational_composites_fluxesType1AGN.csv\')\n',
    'if os.path.exists(obs_csv_path):\n',
    '    df_obs = pd.read_csv(obs_csv_path, index_col=0)\n',
    '    \n',
    '    # Re-classify Alpha=0 galaxies\n',
    '    df_obs[\'Classification\'] = photometry.classify_uvj(df_obs[\'V_0\'] - df_obs[\'J_0\'], df_obs[\'U_0\'] - df_obs[\'V_0\'])\n',
    '\n',
    '    # Fig 7: UVJ Density at Alpha=50%\n',
    '    uv_50 = df_obs[\'U_50\'] - df_obs[\'V_50\']\n',
    '    vj_50 = df_obs[\'V_50\'] - df_obs[\'J_50\']\n',
    '    visualization.plot_uvj_diagram(vj_50, uv_50, classifications=df_obs[\'Classification\'], \n',
    '                                   show_density=True, title=\"ZFOURGE UVJ Density (Alpha=50%)\")\n',
    '    plt.show()\n',
    '\n',
    '    # Fig 8: Average Region Tracks\n',
    '    regions = {0: \'Quiescent\', 1: \'Star-forming\', 2: \'Dusty\'}\n',
    '    colors = {0: \'red\', 1: \'blue\', 2: \'green\'}\n',
    '    fig, ax = plt.subplots(figsize=(8, 8))\n',
    '    visualization.plot_uvj_diagram([], [], ax=ax, title=\"ZFOURGE Average Region Tracks\")\n',
    '    \n',
    '    for cid, name in regions.items():\n',
    '        sub = df_obs[df_obs[\'Classification\'] == cid]\n',
    '        if len(sub) == 0: continue\n',
    '        tvj = [sub[f\'V_{a}\'].mean() - sub[f\'J_{a}\'].mean() for a in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]\n',
    '        tuv = [sub[f\'U_{a}\'].mean() - sub[f\'V_{a}\'].mean() for a in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]\n',
    '        ax.plot(tvj, tuv, marker=\'o\', color=colors[cid], label=name, lw=2)\n',
    '        ax.annotate(\'\', xy=(tvj[-1], tuv[-1]), xytext=(tvj[0], tuv[0]),\n',
    '                    arrowprops=dict(arrowstyle=\"->\", color=colors[cid], lw=1.5, alpha=0.5))\n',
    '    ax.legend(loc=\'lower right\')\n',
    '    plt.show()\n',
    'else:\n',
    '    print(f\"Observational composite file not found at: {obs_csv_path}\")\n'
   ]
}

nb['cells'].append(markdown_cell)
nb['cells'].append(code_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Notebook patched successfully.')
