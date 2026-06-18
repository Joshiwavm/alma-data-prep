# alma-data-prep

Utilities for organizing, concatenating, and exporting ALMA interferometric data
with CASA. Walks an ALMA archive tree, groups `*.ms.split.cal` measurement sets by
project/science-goal/group, computes metadata (on-source time, resolution, MRS,
white-noise sensitivity), concatenates per target, and exports continuum UV/FITS
products and spectral-line cubes with automatic line detection.

## Components

| Module | Class / API | Purpose |
|--------|-------------|---------|
| `organizer.py`   | `ProjectDataOrganizer` | Discover MS dirs, compute metadata, `mstransform`+`concat`, export CSV/LaTeX tables |
| `export.py`      | `Export`, `ExportConfig` | Per-field/SPW split, UV → compressed NPZ, continuum `tclean` → FITS |
| `export_cube.py` | `ExportCube`, `parse_line_freq` | Spectral-cube `tclean`, moment-8 map, greedy 4σ-ellipse line detection |
| `analysis_utils.py` | weight/sensitivity helpers | White-noise sensitivity from visibility weights |

## Requirements

PyPI dependencies (installed automatically): `numpy`, `astropy`, `matplotlib`, `pandas`.

**External prerequisites (not installable via pip):**

- **CASA** — provides `casatasks` and `casatools`. Run inside the CASA Python
  environment (e.g. `casa` or `python` from a modular CASA install).
- **analysisUtils** — the CASA `analysis_scripts` package. Point the package at
  your local copy with the `CASA_ANALYSIS_SCRIPTS` environment variable:

  ```bash
  export CASA_ANALYSIS_SCRIPTS=/path/to/analysis_scripts/
  ```

  If unset, it defaults to the Allegro cluster path
  `/almastorage/allegro/lib/jao-mirror/AIV/science/analysis_scripts/`.

## Install

```bash
git clone https://github.com/Joshiwavm/alma-data-prep.git
cd alma-data-prep
pip install -e .
```

## Usage

```python
from alma_data_prep import ProjectDataOrganizer

DF = ProjectDataOrganizer('/path/to/archive')
DF.mstransform_and_concat()   # transform + concat; computes white-noise sensitivity
DF.display_structure()
DF.export_to_csv()            # writes project_data.csv and project_data.tex

# Continuum export (UV NPZ + FITS per field/SPW)
DF.export()

# Spectral cube + line detection for a named line at a redshift
DF.export_cube(line_name='CO(3-2)', redshift=2.0)
```

`ProjectDataOrganizer(...)` computes on-source time, minimum resolution, and
maximum recoverable scale at construction. White-noise sensitivity is filled in
during `mstransform_and_concat()`.

## License

MIT
