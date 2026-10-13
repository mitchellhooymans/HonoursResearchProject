"""Re-add a theoretical SKIRTOR AGN to each AGN host's CIGALE host-only SED and recompute the UVJ colours.

For every galaxy in zfourge_useall_uvj.csv with fracAGN > 0 (and all four colours), this adds back, with
alpha = f_AGN / (1 - f_AGN) taken from the CIGALE fracAGN (not fitted):
  Type1   : config.SKIRTOR_TYPE1_PARAMS (i = 30 deg)
  Type2   : config.SKIRTOR_TYPE2_PARAMS (i = 70 deg)
  Matched : CIGALE's own torus shape (p=1, q=1) with the galaxy's own fitted inclination (agn.i)
and records the recombined rest-frame (U-V, V-J) for each.

Output: datasets/full_zfourge_decomposed/zfourge_useall_recombined.csv. Chunked + resumable
(chunks in outputs/useall_recomb_chunks/).

Usage: python scripts/build_useall_recombination.py [--workers N] [--chunk 500] [--limit N]
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
IN_CSV = os.path.join(DATA, 'zfourge_useall_uvj.csv')
OUT_CSV = os.path.join(DATA, 'zfourge_useall_recombined.csv')
CHUNK_DIR = os.path.join(ROOT, 'outputs', 'useall_recomb_chunks')

_g = {}


def _init():
    import warnings
    warnings.filterwarnings('ignore')
    from src import config
    from glass import data_io, photometry
    _g['filters'] = photometry.load_passbands(config.FILTER_PATHS)
    d = config.SKIRTOR_DIR
    matched = {'optical_depth': 7, 'p': 1, 'q': 1, 'opening_angle': 40, 'radius_ratio': 20}
    _g['templates'] = {
        'Type1': data_io.read_skirtor_model(d, **config.SKIRTOR_TYPE1_PARAMS),
        'Type2': data_io.read_skirtor_model(d, **config.SKIRTOR_TYPE2_PARAMS),
        'Matched30': data_io.read_skirtor_model(d, inclination=30, **matched),
        'Matched70': data_io.read_skirtor_model(d, inclination=70, **matched),
    }


def process(row):
    from glass import data_io, photometry, analysis, composite_math
    field, seq, z, f, agn_i = row
    out = {'ID': f'{field.upper()}_{seq}'}
    path = os.path.join(DATA, f'{field.lower()}_best_models_fits', f'{seq}_best_model.fits')
    try:
        full = data_io.read_cigale_best_model(path, redshift=z, restframe=True)
        host = analysis.decompose_cigale_sed(full, target='host')
        alpha = f / (1.0 - f)
        T = _g['templates']
        fl = _g['filters']
        methods = {'Type1': T['Type1'], 'Type2': T['Type2'], 'Matched': T['Matched30'] if int(round(agn_i)) == 30 else T['Matched70']}
        for name, tmpl in methods.items():
            try:
                comp = composite_math.create_composite_sed(tmpl, host, alpha)
                uv, vj = photometry.calculate_UVJ_colours(comp, fl['U'], fl['V'], fl['J'])
            except Exception:  # noqa: BLE001
                uv, vj = np.nan, np.nan
            out[f'UV_rec_{name}'] = uv
            out[f'VJ_rec_{name}'] = vj
    except Exception as e:  # noqa: BLE001
        out['error'] = f'{type(e).__name__}: {e}'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=max(1, min(5, (os.cpu_count() or 2) - 2)))
    ap.add_argument('--chunk', type=int, default=500)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(CHUNK_DIR, exist_ok=True)

    t = pd.read_csv(IN_CSV)
    t = t.dropna(subset=['UV_Full', 'VJ_Full', 'UV_Decomposed', 'VJ_Decomposed'])
    t = t[(t.fracAGN > 0) & (t.fracAGN < 1)].reset_index(drop=True)
    if a.limit:
        t = t.sample(a.limit, random_state=0).reset_index(drop=True)
    jobs = list(zip(t.field, t.Seq, t.zpk, t.fracAGN, t.agn_i))
    chunks = [jobs[i:i + a.chunk] for i in range(0, len(jobs), a.chunk)]
    tag = f'limit{a.limit}_' if a.limit else ''
    print(f'{len(jobs)} AGN hosts in {len(chunks)} chunks, {a.workers} workers', flush=True)

    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        for k, ch in enumerate(chunks):
            fn = os.path.join(CHUNK_DIR, f'{tag}chunk_{k:04d}.csv')
            if os.path.exists(fn):
                continue
            pd.DataFrame(pool.map(process, ch, chunksize=10)).to_csv(fn, index=False)
            print(f'chunk {k + 1}/{len(chunks)} done | {time.time() - t0:.0f}s elapsed', flush=True)

    parts = [pd.read_csv(os.path.join(CHUNK_DIR, f)) for f in sorted(os.listdir(CHUNK_DIR)) if f.startswith(f'{tag}chunk_')]
    res = pd.concat(parts, ignore_index=True)
    out = OUT_CSV if not a.limit else OUT_CSV.replace('.csv', f'_pilot{a.limit}.csv')
    res.to_csv(out, index=False)
    ok = res[[c for c in res.columns if c.startswith(('UV_rec', 'VJ_rec'))]].notna().all(axis=1)
    print(f'wrote {out}: {len(res)} rows, {int(ok.sum())} complete, {int((~ok).sum())} incomplete', flush=True)


if __name__ == '__main__':
    main()
