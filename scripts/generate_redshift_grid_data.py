"""
generate_redshift_grid_data.py

Batch script to calculate ugr completeness across a grid of redshifts (0.0 to 4.0).
Implements the project's redshift-grid completeness methodology.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sed_pipeline import config, data_io, composite_math, photometry, analysis

def main():
    print("Loading Filters...")
    filters = photometry.load_passbands(config.FILTER_PATHS)
    
    skirtor_dir = os.path.join(config.RAW_DATA_DIR, 'Templates', 'Skirtor')
    brown_dir = os.path.join(config.RAW_DATA_DIR, 'Templates', 'Brown', '2014', 'Rest')
    
    print("Loading AGN Models...")
    agn_type1 = data_io.read_skirtor_model(skirtor_dir, **config.SKIRTOR_TYPE1_PARAMS)
    agn_type2 = data_io.read_skirtor_model(skirtor_dir, **config.SKIRTOR_TYPE2_PARAMS)
    
    print("Loading Galaxy Templates...")
    brown_templates, brown_names = data_io.read_brown_galaxy_templates(brown_dir)
    
    alphas = config.ALPHA_VALUES
    redshifts = np.arange(0.0, 4.1, 0.1)
    
    results = []
    
    total_templates = len(brown_templates)
    
    for t_idx, gal_sed in enumerate(brown_templates):
        gal_name = brown_names[t_idx]
        print(f"Processing template {t_idx+1}/{total_templates}: {gal_name}")
        
        for agn_name, agn_model in [('Type1', agn_type1), ('Type2', agn_type2)]:
            for alpha in alphas:
                comp_sed = composite_math.create_composite_sed(agn_model, gal_sed, alpha)
                
                for z in redshifts:
                    # Calculate ugr (redshifted to z)
                    ug, gr = photometry.calculate_ugr_colours(comp_sed, filters['u'], filters['g'], filters['r'], redshift=z)
                    
                    classification = analysis.classify_ugr_selection(ug, gr, true_redshift=z)
                    
                    results.append({
                        'Template': gal_name,
                        'AGN_Type': agn_name,
                        'Alpha': alpha,
                        'Redshift': np.round(z, 2),
                        'u-g': ug,
                        'g-r': gr,
                        'Classification': classification
                    })
                    
    df_results = pd.DataFrame(results)
    
    output_path = os.path.join(config.PROCESSED_DATA_DIR, 'ugr_completeness_grid.csv')
    df_results.to_csv(output_path, index=False)
    print(f"Successfully saved completeness grid to {output_path}")

if __name__ == "__main__":
    main()
