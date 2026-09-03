import nbformat
import copy
from nbclient import NotebookClient

path = 'notebooks/CIGALE_Decomposition_Analysis.ipynb'
with open(path, encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

new_code = r"""# Recovered generator for UVJ_CIGALE_movers_vs_population.pdf as actually included
# in the paper (Fig. cigale-movers-vs-population / fig:cigale-movers-vs-population).
# This is NOT the "Angle 5" cell above (which extends the comparison to all four
# transition groups, unrestricted by redshift, for exploration). The paper figure
# specifically restricts to the z <= 1.5 active-recovery regime (see Discussion Sec.
# 6.2/6.3 and docs/cigale_decomposition_findings.md Sec 4.1: recovery is a z<1.5-only
# effect) and compares three groups: the full z<=1.5 parent population, non-migrating
# AGN hosts, and hosts that migrate (from Star-forming or Dusty) to Quiescent upon
# decomposition. This cell was reconstructed during a paper audit because the notebook
# state no longer contained it; verified against paper.tex to match exactly (n=293
# migrating, n=3,513 non-migrating, median logM*=9.38/8.76, median logsSFR=-9.87/-9.20).
# One minor exception: this recomputation gives parent n=7,429 vs the paper's 7,435 (a
# ~0.08% difference, most likely a slightly different NaN/z-boundary rule in the
# original lost cell) -- medians for the parent group (8.62 / -9.27) match exactly, so
# this does not affect any number actually reported in the paper.

df_hosts_z = df_cig[df_cig['fracAGN'] > 0].copy()
df_hosts_z['cls_full'] = photometry.classify_uvj(df_hosts_z['VJ_Full'].values, df_hosts_z['UV_Full'].values)
df_hosts_z['cls_dec']  = photometry.classify_uvj(df_hosts_z['VJ_Decomposed'].values, df_hosts_z['UV_Decomposed'].values)
migrating_to_q = ((df_hosts_z['cls_full'] == 1) | (df_hosts_z['cls_full'] == 2)) & (df_hosts_z['cls_dec'] == 0)

hosts_z15 = df_hosts_z[df_hosts_z[z_col] <= 1.5]
migrating_mask_z15 = migrating_to_q.loc[hosts_z15.index]
parent_z15 = df_cig[df_cig[z_col] <= 1.5]
non_migrating_z15 = hosts_z15[~migrating_mask_z15]
migrating_z15 = hosts_z15[migrating_mask_z15]

mbins = np.arange(6, 11.26, 0.25)
sbins = np.arange(-13.25, -6.76, 0.25)
panels = [('lmass', mbins, r'$\log(M_*/M_\odot)$', r'(a) Stellar Mass Distribution ($z \leq 1.5$)'),
          ('lssfr', sbins, r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$', r'(b) Specific Star Formation Rate ($z \leq 1.5$)')]

fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.2))
for ax, (col, bins, xlabel, title) in zip(axes, panels):
    ax.hist(parent_z15[col].dropna(), bins=bins, color='#CCCCCC', label='Parent population', zorder=1)
    ax.hist(non_migrating_z15[col].dropna(), bins=bins, histtype='step', color='#333333', ls='--', lw=1.6,
            label='Non-migrating hosts', zorder=2)
    ax.hist(migrating_z15[col].dropna(), bins=bins, histtype='step', color='#CC2929', lw=2.0,
            label='Migrating to Quiescent', zorder=3)
    ax.axvline(parent_z15[col].median(), color='#888888', ls=':', lw=1.2, zorder=4)
    ax.axvline(non_migrating_z15[col].median(), color='#333333', ls='--', lw=1.2, zorder=4)
    ax.axvline(migrating_z15[col].median(), color='#CC2929', ls='-', lw=1.6, zorder=4)
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9)
axes[0].set_ylabel('Count (N)')
axes[1].legend(fontsize=7, loc='upper right')
plt.tight_layout()
fig.savefig(os.path.join(config.PROCESSED_DATA_DIR, 'UVJ_CIGALE_movers_vs_population_verify.pdf'), dpi=300, bbox_inches='tight')
plt.show()

for name, g in [('Parent', parent_z15), ('Non-migrating', non_migrating_z15), ('Migrating to Q', migrating_z15)]:
    print(f"{name:>16}: n={len(g):5d}  med logM*={g['lmass'].median():.2f}  med logsSFR={g['lssfr'].median():.2f}")
"""

new_cell = nbformat.v4.new_code_cell(source=new_code)

mini = nbformat.v4.new_notebook()
mini.cells = [copy.deepcopy(nb.cells[1]), copy.deepcopy(nb.cells[6]), new_cell]
client = NotebookClient(mini, timeout=300, kernel_name='python3', resources={'metadata': {'path': 'notebooks'}})
client.execute()

for i, c in enumerate(mini.cells):
    print(f'--- mini cell {i} exec={c.get("execution_count")} outputs={len(c.get("outputs", []))} ---')
    for out in c.get('outputs', []):
        if out.get('output_type') == 'error':
            print('ERROR:', out.get('ename'), out.get('evalue'))
        elif 'text' in out:
            print(''.join(out['text']))

nbformat.write(mini, '.claude-scratch/_mini_verify.ipynb')
print('WROTE MINI NOTEBOOK')
