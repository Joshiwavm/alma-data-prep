"""Line subtraction via clean-in-mask + ft + uvsub.

After detections are identified, clean the line emission inside the same 4σ
ellipse masks used for spectral extraction (down to nsigma=1), then subtract the
resulting clean model from the *non-continuum-subtracted* visibilities with the
CASA tasks ``ft`` and ``uvsub``. The output is written to
``<ms>.replace('.ms', '_linesubtracted.ms')``.

Workflow
--------
1. Build a CRTF mask: one spectral-ranged ellipse per detection (combined).
2. ``tclean`` (specmode='cube', usemask='user', niter large, nsigma=1) on the
   continuum-subtracted MS -> a pure-line clean model.
3. For each non-contsub target MS: copy -> ``_linesubtracted.ms``, ``ft`` the
   model into MODEL_DATA, then ``uvsub`` (CORRECTED = CORRECTED - MODEL).

The subtracted visibilities live in the CORRECTED_DATA column of the output MS.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any

import os
import shutil
import subprocess

from casatasks import tclean, exportfits, ft, uvsub, clearcal, split

C_KMS = 2.99792458e5


def _safe_rmtree(path: str) -> None:
    """Remove a directory tree robustly, handling CASA table directories."""
    try:
        shutil.rmtree(path)
    except OSError:
        subprocess.run(["rm", "-rf", path], check=True)


def build_line_mask_crtf(detections: Sequence[Dict[str, Any]], outfile: str,
                         n_fwhm: float = 2.0) -> str:
    """Write a CRTF mask with one spectral-ranged ellipse per detection.

    Each detection dict must provide ``RA_deg``, ``Dec_deg``, ``a_arcsec``,
    ``b_arcsec``, ``pa_deg``, ``center_GHz_f`` and ``fwhm_kms_f``. The spectral
    ``range`` spans center ± ``n_fwhm`` × FWHM (conservatively wide by default).
    """
    lines = ["#CRTFv0"]
    for d in detections:
        ra   = float(d["RA_deg"]);  dec = float(d["Dec_deg"])
        a    = float(d["a_arcsec"]); b  = float(d["b_arcsec"]); pa = float(d["pa_deg"])
        cen  = float(d["center_GHz_f"]); fwhm = float(d["fwhm_kms_f"])
        fwhm_ghz = fwhm / C_KMS * cen
        f_lo = cen - n_fwhm * fwhm_ghz
        f_hi = cen + n_fwhm * fwhm_ghz
        lines.append(
            f"ellipse[[{ra:.6f}deg, {dec:+.6f}deg], "
            f"[{a:.3f}arcsec, {b:.3f}arcsec], {pa:.2f}deg], "
            f"range=[{f_lo:.6f}GHz, {f_hi:.6f}GHz]"
        )
    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    with open(outfile, "w") as f:
        f.write("\n".join(lines) + "\n")
    return outfile


class LineSubtractor:
    """Clean line emission inside detection masks and subtract it from the visibilities."""

    _IMG_SUFFIXES = [".image", ".pb", ".psf", ".model", ".sumwt",
                     ".weight", ".residual", ".mask"]

    def __init__(self, output_dir: str, imsize: int, cell: str,
                 weighting: str = "natural", gridder: str = "standard",
                 pblimit: float = 0.0, restfreq: str = "", width_kms: str = "",
                 nsigma: float = 1.0, niter: int = 100000, n_fwhm: float = 2.0,
                 overwrite: bool = False):
        self.output_dir = Path(output_dir)
        self.imsize     = imsize
        self.cell       = cell
        self.weighting  = weighting
        self.gridder    = gridder
        self.pblimit    = pblimit
        self.restfreq   = restfreq
        self.width_kms  = width_kms
        self.nsigma     = nsigma
        self.niter      = niter
        self.n_fwhm     = n_fwhm
        self.overwrite  = overwrite
        self.model_image_fits = None   # restored clean image (Jy/beam)
        self.model_cube_fits  = None   # clean components cube (Jy/pixel)

    def _clean_line_model(self, line_vis: List[str],
                          detections: Sequence[Dict[str, Any]],
                          imagename_full: str) -> str:
        """tclean the line emission inside the CRTF mask; return the .model path."""
        mask = build_line_mask_crtf(detections, imagename_full + ".lines.crtf", self.n_fwhm)

        if self.overwrite:
            for suffix in self._IMG_SUFFIXES:
                p = f"{imagename_full}{suffix}"
                if os.path.exists(p):
                    _safe_rmtree(p)

        tclean(
            vis=line_vis if len(line_vis) > 1 else line_vis[0],
            imagename=imagename_full,
            specmode="cube",
            imsize=self.imsize,
            cell=self.cell,
            weighting=self.weighting,
            gridder=self.gridder,
            restfreq=self.restfreq or "",
            width=self.width_kms or "",
            pblimit=self.pblimit,
            niter=self.niter,
            nsigma=self.nsigma,
            usemask="user",
            mask=mask,
            datacolumn="data",
            parallel=False,
        )
        exportfits(f"{imagename_full}.image", f"{imagename_full}.image.fits", overwrite=True)
        exportfits(f"{imagename_full}.model", f"{imagename_full}.model.fits", overwrite=True)
        # Remember FITS products (clean components cube used by the validation plot)
        self.model_image_fits = f"{imagename_full}.image.fits"
        self.model_cube_fits  = f"{imagename_full}.model.fits"
        return f"{imagename_full}.model"

    def run(self, line_vis, target_vis, detections, imagename: Optional[str] = None) -> List[str]:
        """Clean the line model from ``line_vis`` and subtract it from each ``target_vis``.

        Parameters
        ----------
        line_vis : str | list[str]
            Continuum-subtracted MS(es) the line model is cleaned from.
        target_vis : str | list[str]
            Non-continuum-subtracted MS(es) to subtract the model from.
        detections : list[dict]
            Kept-detection records (see ``build_line_mask_crtf``).
        imagename : str, optional
            Base name for the clean model image.
        """
        if not detections:
            print("[LineSubtractor] No detections; skipping line subtraction.")
            return []

        line_vis   = [line_vis]   if isinstance(line_vis, str)   else list(line_vis)
        target_vis = [target_vis] if isinstance(target_vis, str) else list(target_vis)

        fits_dir = self.output_dir / "fits"
        fits_dir.mkdir(parents=True, exist_ok=True)
        imagename = imagename or "line_clean"
        imagename_full = str(fits_dir / f"{imagename}_lineclean")

        model = self._clean_line_model(line_vis, detections, imagename_full)

        outputs = []
        for ms in target_vis:
            out = ms.replace(".ms", "_linesubtracted.ms")
            if os.path.exists(out):
                if self.overwrite:
                    print(f"[LineSubtractor]   Removing existing {out}")
                    _safe_rmtree(out)
                else:
                    print(f"[LineSubtractor]   Skipping {out} (exists, overwrite=False).")
                    outputs.append(out)
                    continue
            print(f"[LineSubtractor]   {ms} -> {out}")
            tmp = ms.replace(".ms", "_linesub_tmp.ms")
            if os.path.exists(tmp):
                _safe_rmtree(tmp)
            shutil.copytree(ms, tmp)
            clearcal(vis=tmp, addmodel=True)   # create CORRECTED (=DATA) and MODEL columns
            ft(vis=tmp, model=model, usescratch=True)
            uvsub(vis=tmp)                     # CORRECTED = CORRECTED - MODEL (line removed)
            # Materialize line-removed CORRECTED into DATA so downstream Export/
            # ExportCube (which read the DATA column) use the subtracted visibilities.
            split(vis=tmp, outputvis=out, datacolumn="corrected")
            _safe_rmtree(tmp)
            outputs.append(out)

        # Cleanup the clean-model CASA images (FITS copies are kept)
        for suffix in self._IMG_SUFFIXES:
            p = f"{imagename_full}{suffix}"
            if os.path.exists(p):
                _safe_rmtree(p)

        print(f"[LineSubtractor] Done; wrote {len(outputs)} line-subtracted MS.")
        return outputs
