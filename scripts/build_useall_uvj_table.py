"""Recompute CIGALE full and host-only (decomposed) rest-frame UVJ colours for every ZFOURGE Use==1 galaxy.

Reads each galaxy's CIGALE best-fit FITS file (datasets/full_zfourge_decomposed/<field>_best_models_fits/<Seq>_best_model.fits):
  - full SED      = L_lambda_total
  - host-only SED = glass.analysis.decompose_cigale_sed(target='host')
and runs both through the glass U, V, J passbands (glass.photometry.calculate_UVJ_colours), the same functions used for the
paper's theoretical grid. Also records the AGN fraction, inclination and luminosities from the FITS header and the catalogue
redshift / stellar mass / sSFR (matched on Seq).

Output: datasets/full_zfourge_decomposed/zfourge_useall_uvj.csv (one row per galaxy). Work is done in chunks written to
outputs/useall_uvj_chunks/ so an interrupted run resumes where it stopped.

Usage:  python scripts/build_useall_uvj_table.py [--workers N] [--chunk 500] [--limit N]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, 'datasets', 'full_zfourge_decomposed')
ZF = os.path.join(ROOT, 'datasets', 'zfourge')
CHUNK_DIR = os.path.join(ROOT, 'outputs', 'useall_uvj_chunks')
OUT_CSV = os.path.join(DATA, 'zfourge_useall_uvj.csv')

_filters = None


def _init():
    global _filters
    import warnings
    warnings.filterwarnings('ignore')
    from src import config
    from glass import photometry
    _filters = photometry.load_passbands(config.FILTER_PATHS)


def process(row):
    """row: (field, seq, z). Returns a dict (NaNs where a step failed)."""
    from astropy.io import fits
    from glass import data_io, photometry, analysis
    field, seq, z = row
    out = {'ID': f'{field.upper()}_{seq}', 'field': field.upper(), 'Seq': seq, 'z_used': z}
    path = os.path.join(DATA, f'{field}_best_models_fits', f'{seq}_best_model.fits')
    try:
        h = fits.getheader(path, 1)
        out.update({'fracAGN': float(h['agn.fracAGN']), 'agn_i': float(h['agn.i']),
                    'L_AGN': float(h['agn.luminosity']), 'L_6um': float(h['agn.L_6um']),
                    'z_cigale': float(h['universe.redshift'])})
    except Exception as e:  # noqa: BLE001
        out['error'] = f'header: {e}'
        return out
    try:
        full = data_io.read_cigale_best_model(path, redshift=z, restframe=True)
        full_t = full[['lambda (Angstroms)']].copy()
        full_t['Total Flux (erg/s/cm^2/Angstrom)'] = full['L_lambda_total'].values
        host = analysis.decompose_cigale_sed(full, target='host')
        uvf, vjf = photometry.calculate_UVJ_colours(full_t, _filters['U'], _filters['V'], _filters['J'])
        uvd, vjd = photometry.calculate_UVJ_colours(host, _filters['U'], _filters['V'], _filters['J'])
        out.update({'UV_Full': uvf, 'VJ_Full': vjf, 'UV_Decomposed': uvd, 'VJ_Decomposed': vjd})
    except Exception as e:  # noqa: BLE001
        out['error'] = f'sed: {type(e).__name__}: {e}'
    return out


def build_catalogue():
    """Use==1 galaxies with catalogue redshift, mass and sSFR, matched on Seq."""
    from astropy.io import fits
    rows = []
    for f in ['cdfs', 'cosmos', 'uds']:
        main = fits.open(os.path.join(ZF, f'zf_{f}.fits'))[1].data
        sfr = fits.open(os.path.join(ZF, f'zf_{f}_sfr.fits'))[1].data
        eaz = fits.open(os.path.join(ZF, f'zf_{f}_eazy.fits'))[1].data
        m = pd.DataFrame({'Seq': np.asarray(main['Seq']).astype(int), 'Use': np.asarray(main['Use'])})
        s = pd.DataFrame({'Seq': np.asarray(sfr['Seq']).astype(int), 'lmass': np.asarray(sfr['lmass']).astype(float),
                          'lssfr': np.asarray(sfr['lssfr']).astype(float)})
        e = pd.DataFrame({'Seq': np.asarray(eaz['Seq']).astype(int), 'zpk': np.asarray(eaz['zpk']).astype(float)})
        t = m.merge(s, on='Seq', how='left').merge(e, on='Seq', how='left')
        t = t[t.Use == 1].copy()
        t['field'] = f
        t['has_fits'] = [os.path.exists(os.path.join(DATA, f'{f}_best_models_fits', f'{i}_best_model.fits')) for i in t.Seq]
        rows.append(t)
    cat = pd.concat(rows, ignore_index=True)
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    ap.add_argument('--chunk', type=int, default=500)
    ap.add_argument('--limit', type=int, default=0, help='only the first N galaxies (pilot)')
    a = ap.parse_args()

    os.makedirs(CHUNK_DIR, exist_ok=True)
    cat = build_catalogue()
    print(f'Use==1 galaxies: {len(cat)}; with a FITS file: {int(cat.has_fits.sum())}', flush=True)
    todo = cat[cat.has_fits & cat.zpk.notna()].reset_index(drop=True)
    if a.limit:
        todo = todo.sample(a.limit, random_state=0).reset_index(drop=True)
    jobs = list(zip(todo.field, todo.Seq, todo.zpk))
    chunks = [jobs[i:i + a.chunk] for i in range(0, len(jobs), a.chunk)]
    tag = f'limit{a.limit}_' if a.limit else ''
    print(f'{len(jobs)} galaxies in {len(chunks)} chunks, {a.workers} workers', flush=True)

    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        for k, ch in enumerate(chunks):
            fn = os.path.join(CHUNK_DIR, f'{tag}chunk_{k:04d}.csv')
            if os.path.exists(fn):
                continue
            res = pool.map(process, ch, chunksize=20)
            pd.DataFrame(res).to_csv(fn, index=False)
            done = (k + 1) * a.chunk
            print(f'chunk {k + 1}/{len(chunks)} done | {time.time() - t0:.0f}s elapsed', flush=True)

    parts = [pd.read_csv(os.path.join(CHUNK_DIR, f)) for f in sorted(os.listdir(CHUNK_DIR)) if f.startswith(f'{tag}chunk_')]
    res = pd.concat(parts, ignore_index=True)
    res = res.merge(todo[['field', 'Seq', 'lmass', 'lssfr']].assign(field=lambda d: d.field.str.upper()), on=['field', 'Seq'], how='left')
    res = res.rename(columns={'z_used': 'zpk'})
    out = OUT_CSV if not a.limit else OUT_CSV.replace('.csv', f'_pilot{a.limit}.csv')
    res.to_csv(out, index=False)
    ok = res[['UV_Full', 'VJ_Full', 'UV_Decomposed', 'VJ_Decomposed']].notna().all(axis=1)
    print(f'wrote {out}: {len(res)} rows, {int(ok.sum())} with all four colours, {int((~ok).sum())} failed', flush=True)
    if (~ok).any() and 'error' in res:
        print(res.loc[~ok, 'error'].value_counts().head(5).to_string(), flush=True)


if __name__ == '__main__':
    main()
