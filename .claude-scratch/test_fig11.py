import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('notebooks/..'))
os.chdir('notebooks')
from src import config
from glass import photometry, visualization

plt.style.use('default')
visualization.apply_pasa_style()
os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

cigale_csv = os.path.join('..', 'datasets', 'full_zfourge_decomposed', 'zfourge_full_final.csv')
agn_frac_csv = os.path.join('..', 'datasets', 'full_zfourge_decomposed', 'agn_fractions.csv')
df_cig = pd.read_csv(cigale_csv, low_memory=False)
z_col = 'zpk_x' if 'zpk_x' in df_cig.columns else 'zpk'
df_cig = df_cig.merge(pd.read_csv(agn_frac_csv), on='ID', how='left')

frac_out_dir = os.path.join(config.PROCESSED_DATA_DIR, 'fracAGN_diagnostics')
os.makedirs(frac_out_dir, exist_ok=True)

df_hosts = df_cig[df_cig['fracAGN'] > 0].copy()
df_hosts['cls_full'] = photometry.classify_uvj(df_hosts['VJ_Full'].values, df_hosts['UV_Full'].values)
df_hosts['cls_dec']  = photometry.classify_uvj(df_hosts['VJ_Decomposed'].values, df_hosts['UV_Decomposed'].values)

mbins = np.arange(7.5, 12.01, 0.25)
sbins = np.arange(-13.0, -7.01, 0.25)

exec(open('../.claude-scratch/fig11_cell_source.py').read())
