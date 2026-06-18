"""
Export class scaffold with basic execution

Encapsulates logic from scripts/exports/export-*.py using metadata that Organizer
already computed. Uses the final concatenated MS path (concatvis) and derives
array/imaging defaults from dish size and frequency. Methods implement:
- splitting per field/SPW and saving UV to compressed NPZ
- running tclean and exporting FITS products, with PB mask cleanup
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import os
import shutil
import numpy as np
from astropy.constants import c
from astropy.io import fits

# Ensure CASA tasks are available when this module is imported (not execfile)
from casatasks import split, tclean, exportfits
from casatools import ms as ms_tool
from casatools import msmetadata as msmd_tool


@dataclass
class ExportConfig:
    """Static configuration for export processing (overridable)."""

    timebin: str = "0s"
    weighting: str = "natural"
    gridder: str = "standard"
    specmode: str = "mfs"
    niter: int = 0
    pblimit: float = 0.0
    # Optional explicit overrides
    fields: Optional[List[str]] = None
    spws: Optional[List[str]] = None
    imsize_12m: int = 1024
    imcell_12m: str = "0.3arcsec"
    imsize_7m: int = 512
    imcell_7m: str = "1.5arcsec"
    band_name: Optional[str] = None  # default derived from frequency; fallback "band1"


class Export:
    """Export class that keeps track of data and filenames and can run the pipeline.

    Parameters
    - target: science target name
    - dish_size_m: 7 or 12 typically; used to infer array/paths
    - median_freq_ghz: central frequency in GHz used to derive band name
    - concatvis: path to concatenated MS (produced by Organizer)
    - output_dir: directory where derived products will be saved
    - config: CASA/tclean parameters and optional overrides
    """

    def __init__(
        self,
        target: str,
        dish_size_m: Optional[int],
        median_freq_ghz: Optional[float],
        concatvis: Optional[str],
        output_dir: Optional[str] = None,
        config: Optional[ExportConfig] = None,
    ) -> None:
        self.target = target
        self.dish_size_m = dish_size_m
        self.median_freq_ghz = median_freq_ghz
        self.concatvis = Path(concatvis) if concatvis else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.config = config or ExportConfig()

        # Derived, human-friendly base name matching existing convention
        # e.g., ACT-CL_J0123.5-0428_12m_39GHz
        if self.target and self.dish_size_m and self.median_freq_ghz:
            self.basename = f"{self.target}_{int(self.dish_size_m)}m_{int(self.median_freq_ghz)}GHz"
        else:
            self.basename = self.target or "unknown"

        # Derived array label and defaults
        self.array_label = self._derive_array_label()
        self.band_label = self._derive_band_label()
        self.imsize, self.imcell = self._derive_imaging_params()

    # ---- Factory helpers -------------------------------------------------
    @classmethod
    def from_group_record(cls, group_record: Dict[str, Any], output_root: str ) -> "Export":
        """Create an Export from a single group's data dict from Organizer.projects.

        Expected keys in group_record:
        - target: str
        - dish_sizes: List[float] or List[int]
        - median_frequency: float (GHz)
        - concatvis: str path to concatenated MS
        """
        target = group_record.get("target")
        dish_sizes: Optional[List[float]] = group_record.get("dish_sizes")
        dish_size_m = int(min(dish_sizes)) if dish_sizes else None
        median_freq = group_record.get("median_frequency")
        concatvis = group_record.get("concatvis")

        out_dir = Path(output_root) / (target.replace(" ", "_") if target else "unknown")
        return cls(
            target=target,
            dish_size_m=dish_size_m,
            median_freq_ghz=median_freq,
            concatvis=concatvis,
            output_dir=str(out_dir),
        )

    # ---- Derivations -----------------------------------------------------
    def _derive_array_label(self) -> str:
        if self.dish_size_m == 12:
            return "com12m"
        if self.dish_size_m == 7:
            return "com07m"
        # Fallback
        return "unknown"

    def _derive_band_label(self) -> str:
        # Minimal mapping: default to band1 for ~39 GHz if not provided
        if self.config.band_name:
            return self.config.band_name
        if self.median_freq_ghz is None:
            return "band1"
        # Simple ALMA band heuristics (not exhaustive)
        f = self.median_freq_ghz
        if 35 <= f <= 50:
            return "band1"
        # Extend here if needed
        return "band1"

    def _derive_imaging_params(self) -> Tuple[int, str]:
        if self.dish_size_m == 12:
            return self.config.imsize_12m, self.config.imcell_12m
        if self.dish_size_m == 7:
            return self.config.imsize_7m, self.config.imcell_7m
        # Fallback
        return self.config.imsize_12m, self.config.imcell_12m

    def _ensure_output_dir(self) -> Path:
        assert self.output_dir is not None, "output_dir must be set"
        out = self.output_dir / self.array_label
        out.mkdir(parents=True, exist_ok=True)
        return out

    # ---- Discovery -------------------------------------------------------
    def _discover_fields_spws(self) -> Tuple[List[str], List[str]]:
        """Discover field IDs and SPW IDs from the concatenated MS using CASA tools.
        Assumes a valid CASA environment (no fallbacks).
        """
        if self.config.fields is not None and self.config.spws is not None:
            return list(self.config.fields), list(self.config.spws)

        assert self.concatvis is not None, "concatvis must be set"

        msmd = msmd_tool()
        msmd.open(str(self.concatvis))
        fids = msmd.fieldsforintent("*OBSERVE_TARGET*", False)
        if getattr(fids, "size", 0) == 0:
            names = list(msmd.fieldnames())
            fields = [str(i) for i in range(len(names))]
        else:
            fields = [str(int(x)) for x in fids]

        spw_ids = msmd.spwsforintent("OBSERVE_TARGET#ON_SOURCE")
        spws = [str(int(x)) for x in spw_ids]
        msmd.close()

        return fields, spws

    def _build_binvis_prefix(self) -> str:
        out_root = self._ensure_output_dir()
        # e.g., ../output/<target>/<array>/output_<band>_<array>.ms.field-fid.spw-sid
        return str(out_root / f"output_{self.band_label}_{self.array_label}.ms.field-fid.spw-sid")

    # ---- Pipeline steps --------------------------------------------------
    def split_to_per_field_spw(self) -> None:
        """Split the concatenated MS per field and SPW, then save UV arrays to NPZ and cleanup split MS."""
        
        assert self.concatvis is not None, "concatvis must be set"
        
        fields, spws = self._discover_fields_spws()
        binvis = self._build_binvis_prefix()

        for field in fields:
            print(f"Processing field {field}")
            for spw in spws:
                print(f"- Spectral window {spw}")
        
                outvis = binvis.replace("-fid", f"-{field}").replace("-sid", f"-{spw}")

                # Inspect channel frequencies to set width
                _ms = ms_tool()
                _ms.open(str(self.concatvis), nomodify=True)
                try:
                    _ms.selectinit(int(spw))
                    _ms.select({"field_id": int(field)})
                    freqs = _ms.range("chan_freq")["chan_freq"][:, 0]
                finally:
                    _ms.close()
                print()
                print(str(self.concatvis))
                print(outvis)
                print()

                split(
                    vis=str(self.concatvis),
                    outputvis=outvis,
                    datacolumn="data",
                    keepflags=False,
                    field=field,
                    spw=spw,
                    width=len(freqs),
                    timebin=self.config.timebin,
                )

                # Save to compressed numpy array
                _ms2 = ms_tool()
                _ms2.open(outvis, nomodify=True)
                try:
                    _ms2.selectinit(0)
                    _ms2.select({"field_id": 0})
                    freqs2 = _ms2.range("chan_freq")["chan_freq"][:, 0]
                    rec = _ms2.getdata(["u", "v", "data", "weight", "flag"])
                finally:
                    _ms2.close()

                flags = np.logical_and(~rec["flag"][0][0], ~rec["flag"][1][0])
                uwave = (rec["u"])[flags] * freqs2[0] / c.value
                vwave = (rec["v"])[flags] * freqs2[0] / c.value
                uvreal = (rec["data"][0][0][flags].real + rec["data"][1][0][flags].real) / 2.0
                uvimag = (rec["data"][0][0][flags].imag + rec["data"][1][0][flags].imag) / 2.0
                uvwght = 4.0 / (1.0 / rec["weight"][0][flags] + 1.0 / rec["weight"][1][flags])
                uvfreq = np.ones(np.shape(uwave)) * freqs2[0]

                uvdata = np.array([uwave, vwave, uvreal, uvimag, uvwght, uvfreq])
                np.savez_compressed(f"{outvis.replace('.ms.', '.im.')}.data", uvdata)
                # Cleanup split MS
                if os.path.exists(outvis):
                    shutil.rmtree(outvis, ignore_errors=True)

    def tclean_and_export_fits(self) -> None:
        """Run tclean for each field/SPW pair and export FITS products; then cleanup CASA images and adjust PB FITS."""
        assert self.concatvis is not None, "concatvis must be set"
        fields, spws = self._discover_fields_spws()
        binvis = self._build_binvis_prefix()

        for field in fields:
            for spw in spws:
                outvis = binvis.replace("-fid", f"-{field}").replace("-sid", f"-{spw}")
                imagename = outvis.replace(".ms", ".im")

                tclean(
                    vis=str(self.concatvis),
                    imagename=imagename,
                    field=field,
                    spw=spw,
                    niter=self.config.niter,
                    pblimit=self.config.pblimit,
                    imsize=self.imsize,
                    cell=self.imcell,
                    gridder=self.config.gridder,
                    weighting=self.config.weighting,
                    specmode=self.config.specmode,
                    parallel=False,
                )

                exportfits(f"{imagename}.pb", f"{imagename}.pbeam.fits", overwrite=True)
                exportfits(f"{imagename}.psf", f"{imagename}.psf.fits", overwrite=True)
                exportfits(f"{imagename}.image", f"{imagename}.image.fits", overwrite=True)

                # Cleanup CASA images
                for suffix in [".pb", ".psf", ".image", ".model", ".sumwt", ".weight", ".residual"]:
                    p = f"{imagename}{suffix}"
                    if os.path.exists(p):
                        shutil.rmtree(p, ignore_errors=True)

                # Adjust PB FITS (masking negatives/NaNs)
                pbeam = f"{imagename}.pbeam.fits"
                if os.path.exists(pbeam):
                    hdu = fits.open(pbeam)[0]
                    hdu.data[np.isnan(hdu.data)] = 0.0

                    data = np.copy(hdu.data[0, 0])
                    diff0a = np.ones(np.shape(data))
                    diff0b = np.ones(np.shape(data))
                    diff1a = np.ones(np.shape(data))
                    diff1b = np.ones(np.shape(data))

                    diff0a[1:, :] = np.diff(data, axis=0)
                    diff0a[int(diff0a.shape[0] / 2) + 1 :, :] *= -1.0
                    diff0b[:-1, :] = np.diff(data, axis=0)
                    diff0b[int(diff0b.shape[0] / 2) :, :] *= -1.0
                    diff1a[:, 1:] = np.diff(data, axis=1)
                    diff1a[:, int(diff1a.shape[1] / 2) + 1 :] *= -1.0
                    diff1b[:, :-1] = np.diff(data, axis=1)
                    diff1b[:, int(diff1b.shape[1] / 2) :] *= -1.0

                    mask0 = np.logical_or(diff0a < 0, diff0b < 0)
                    mask1 = np.logical_or(diff1a < 0, diff1b < 0)
                    mask = np.logical_or(mask0, mask1)
                    mask = np.logical_or(mask, data == 0)
                    data[mask] = np.nan

                    hdu.data[0, 0] = np.copy(data)
                    hdu.writeto(pbeam, overwrite=True)

    def run_all(self) -> None:
        """Run the full export pipeline for all fields/SPWs using derived parameters."""

        self.split_to_per_field_spw()
        self.tclean_and_export_fits()
        print(f"[Export] run_all() finished for {self.basename}")

    def __repr__(self) -> str:  # pragma: no cover - simple representation
        return (
            "Export("
            f"target={self.target!r}, dish_size_m={self.dish_size_m!r}, "
            f"median_freq_ghz={self.median_freq_ghz!r}, concatvis={str(self.concatvis)!r}, "
            f"output_dir={str(self.output_dir)!r})"
        )
