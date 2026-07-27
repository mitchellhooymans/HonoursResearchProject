# AGN Synthetic SED Modelling Pipeline

Research code investigating how Active Galactic Nuclei (AGN) contamination biases
rest-frame colour classification of galaxies — specifically the UVJ diagram widely
used to separate quiescent, star-forming, and dusty galaxy populations.

The project combines two complementary approaches:

1. **Theoretical modelling** — synthetic composite SEDs built by mixing SKIRTOR AGN
   torus models with galaxy templates (Brown 2014, SWIRE, GALSEDATLAS) across a grid
   of AGN fractions and redshifts, to predict how AGN light should shift a host
   galaxy's observed colours.
2. **Observational validation** — the theoretical predictions are tested against real
   galaxies from the ZFOURGE survey, using CIGALE SED decomposition to separate each
   galaxy's AGN and host-galaxy light and measure the actual colour shift.

## Repository structure

```
src/sed_pipeline/     Core pipeline package (see below)
notebooks/            Analysis notebooks, one per stage of the modelling/validation
scripts/              Standalone scripts that regenerate specific outputs
datasets/             Input catalogs, filter curves, and SED templates
outputs/              Generated figures, tables, and intermediate CSVs
docs/                 Written findings that go beyond what fits in a notebook
```

### `src/sed_pipeline`

The reusable pipeline logic, factored out of the original one-off analysis scripts:

| Module              | Responsibility                                                          |
|----------------------|--------------------------------------------------------------------------|
| `config.py`          | Cosmology, model grid, file paths, filter/SKIRTOR parameter definitions |
| `data_io.py`         | Reading SKIRTOR/template models and catalogs into DataFrames           |
| `composite_math.py`  | Core SED physics: alignment, interpolation, flux scaling, composites   |
| `photometry.py`      | Synthetic photometry, UVJ/Lacy wedge classification, mag conversions   |
| `analysis.py`        | Error propagation, vector offsets, completeness and population stats   |
| `visualization.py`   | Matplotlib styling and plotting helpers (PASA-format figure sizes)     |

### `notebooks`

| Notebook                                  | Covers                                                                  |
|---------------------------------------------|--------------------------------------------------------------------------|
| `Model_Validation_via_IRAC.ipynb`          | Validating composite AGN+galaxy models against IRAC colour space       |
| `UVJ_Colour_Evolution.ipynb`               | How UVJ colours evolve with AGN fraction and redshift across templates |
| `Observational_Validation_ZFOURGE.ipynb`   | Testing model predictions against ZFOURGE + CIGALE-decomposed galaxies |
| `Paper_Results_Master.ipynb`               | Consolidated notebook producing the full set of results in one place   |

`notebooks/old_scripts/` holds the pre-refactor scripts these notebooks replaced —
kept for reference during the migration.

### `scripts`

Command-line entry points that exercise the pipeline outside a notebook:

- `run_pipeline.py` — minimal smoke test wiring the modules together.
- `generate_redshift_grid_data.py` — batches the UGR-completeness calculation across
  a redshift grid.
- `recreate_theoretical_results.py`, `recreate_observational_results.py`,
  `recreate_cigale_results.py` — regenerate specific results from pre-computed
  outputs without re-running a full notebook.

### `docs`

- `cigale_decomposition_findings.md` — the AGN-fraction analysis behind the
  observational validation: why population-averaged colour shifts looked
  negligible, and the follow-up discovery that AGN contamination hides quiescent
  galaxies at low redshift.

## Getting started

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

This installs the pinned dependencies from `uv.lock` (NumPy, pandas, Matplotlib,
Astropy, SciPy, Seaborn) into `.venv`. `astLib` is vendored directly in the repo
(used for its `astSED` passband/SED handling) rather than pulled from PyPI.

## Data

`datasets/` and `outputs/` hold catalogs, templates, and generated results. The
largest raw files (ZFOURGE FITS catalogs, EAZY `.h5` template sets, per-galaxy CIGALE
best-fit models) are excluded from version control via `.gitignore` — obtain these
separately and place them under the paths referenced in `src/sed_pipeline/config.py`
before running the observational-validation notebooks/scripts.

## Running the analysis

From an activated environment (`uv run jupyter lab`, or point your IDE's kernel at
`.venv`), open a notebook under `notebooks/` and run it top to bottom — each is
self-contained and documented section by section. To regenerate a specific result
without notebooks, run the matching script under `scripts/` from the repository
root, e.g.:

```bash
uv run python scripts/recreate_observational_results.py
```
