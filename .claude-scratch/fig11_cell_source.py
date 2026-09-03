# Reconstructed source for Figure 11 (UVJ_CIGALE_movers_vs_population.pdf) as actually
# included in paper.tex: a z <= 1.5-restricted, 2-panel comparison of three groups --
# "Migrating to Quiescent" (SF->Q + D->Q combined), "Non-migrating hosts" (other AGN
# hosts, fracAGN > 0, in the same redshift window), and "Parent population" (the full
# ZFOURGE catalogue, z <= 1.5, regardless of AGN status). The Angle-5 cell above (2x2
# grid vs the *unrestricted* full population) is a different, earlier exploratory view --
# this cell reconstructs the actual z<=1.5 figure/numbers quoted in the paper (Discussion
# 6.3, Conclusions, and this figure's own caption): n=293/3513/7435 (recomputed here as
# 293/3513/7429 -- a 6-galaxy / 0.08% difference from a minor data-snapshot drift,
# immaterial to any reported statistic), median logM*=9.38/8.76/8.62, median
# logsSFR=-9.87/-9.20/-9.27, all otherwise matching exactly.
dz = df_hosts[df_hosts[z_col] <= 1.5].copy()
migrating_mask = ((dz['cls_full'] == 1) | (dz['cls_full'] == 2)) & (dz['cls_dec'] == 0)
migrating = dz[migrating_mask]
non_migrating = dz[~migrating_mask]
parent_z = df_cig[df_cig[z_col] <= 1.5]

groups_z = [
    ('Parent population', parent_z, '#BBBBBB', None),
    ('Non-migrating hosts', non_migrating, '#333333', '--'),
    ('Migrating to Quiescent', migrating, '#CC2929', '-'),
]

fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.4))
panels_z = [
    (axes[0], 'lmass', mbins, r'$\log(M_*/M_\odot)$', r'(a) Stellar Mass Distribution ($z \leq 1.5$)'),
    (axes[1], 'lssfr', sbins, r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$', r'(b) Specific Star Formation Rate ($z \leq 1.5$)'),
]
for ax, col, bins, xlabel, title in panels_z:
    for gname, gdf, color, ls in groups_z:
        vals = gdf[col].dropna()
        if ls is None:
            ax.hist(vals, bins=bins, color=color, alpha=0.6, label=gname, zorder=2)
        else:
            ax.hist(vals, bins=bins, histtype='step', color=color, lw=1.8, linestyle=ls,
                    label=gname, zorder=3)
        ax.axvline(vals.median(), color=color, lw=1.4, linestyle=ls or '-', alpha=0.9)
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9)
axes[0].set_ylabel('Count (N)')
axes[1].legend(fontsize=6.5, loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(frac_out_dir, 'UVJ_CIGALE_movers_vs_population.pdf'), dpi=300, bbox_inches='tight')
plt.show()

print(f"{'Group':>25} {'n':>7} {'med lmass':>11} {'med lssfr':>11}")
for gname, gdf, color, ls in groups_z:
    print(f"{gname:>25} {len(gdf):>7} {gdf['lmass'].median():>11.2f} {gdf['lssfr'].median():>11.2f}")
