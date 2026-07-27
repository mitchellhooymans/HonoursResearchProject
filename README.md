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

## How the modelling works

**Building a composite SED.** Each theoretical data point starts as two spectra on a
common wavelength grid (Angstroms, flux in erg/s/cm²/Å): a SKIRTOR AGN torus model
(Type 1 = face-on, inclination 0°; Type 2 = edge-on, inclination 90°; fixed torus
optical depth, opening angle, and radial structure) and a host-galaxy template (either
an empirical Brown 2014 GALSEDATLAS spectrum or a SWIRE template). The two spectra are
interpolated onto their overlapping wavelength range, then the AGN spectrum is scaled
so its integrated flux matches the galaxy's:

```
scaling_factor = ∫ F_galaxy dλ / ∫ F_AGN dλ
composite(λ)   = F_galaxy(λ) + alpha × scaling_factor × F_AGN(λ)
```

`alpha` (`config.ALPHA_VALUES`, 11 steps from 0 to 1) is therefore a direct "fractional
AGN contribution" knob: alpha = 0 is a pure galaxy, alpha = 1 means the AGN contributes
as much integrated flux as the host itself. Sweeping alpha at a fixed redshift produces
a track showing how progressively brighter AGN light drags a galaxy's colours away from
its intrinsic, AGN-free position.

**Turning spectra into colours.** Each composite spectrum is convolved with real filter
passbands (via `astLib.astSED`) to get synthetic AB-magnitude colours in three
diagnostic spaces:

- **UVJ** (rest-frame U−V vs. V−J) — the standard diagram for separating quiescent,
  star-forming, and dusty galaxies. A galaxy is classified quiescent if it falls inside
  a fixed polygon in UVJ space (`photometry.classify_uvj`); otherwise it's dusty if
  V−J > 1.2, or star-forming otherwise.
- **ugr** (u−g vs. g−r) and **IRAC** (log f₅.₈/f₃.₆ vs. log f₈.₀/f₄.₅, the Lacy AGN
  wedge) — computed at a chosen observed redshift by redshifting the composite
  spectrum before convolving with the filters, so both intrinsic (rest-frame) and
  redshift-dependent contamination effects can be tracked separately.

**Checking it against real galaxies.** The same colour-shift signature should be
recoverable in real data. ZFOURGE gives rest-frame U, V, J fluxes for ~10,000 galaxies
across three fields; CIGALE SED fitting independently decomposes each galaxy's best-fit
model into its AGN and host-galaxy flux components (`data_io.read_cigale_best_model`),
giving an observational analogue of "subtracting off alpha's worth of AGN light" without
relying on the theoretical templates at all. Comparing a galaxy's classification with
and without its fitted AGN component isolates how much of its UVJ position was an
artefact of AGN contamination rather than the host's actual stellar population — see
`docs/cigale_decomposition_findings.md` for the resulting findings.

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
