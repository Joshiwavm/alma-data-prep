from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import os
import csv
import shutil
import subprocess
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import models as _m, fitting as _f

from casatasks import tclean, exportfits, uvcontsub as casa_uvcontsub
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as MplEllipse


def _safe_rmtree(path: str) -> None:
    """Remove a directory tree robustly, handling CASA table directories."""
    try:
        shutil.rmtree(path)
    except OSError:
        subprocess.run(["rm", "-rf", path], check=True)


def parse_line_freq(line_name: str, redshift: float) -> tuple[str, float]:
    """Look up a line name and return (obs_freq_str, rest_freq_hz)."""
    key = line_name
    if key not in LINE_REST_FREQ_GHZ:
        lower_map = {k.lower(): k for k in LINE_REST_FREQ_GHZ}
        canonical = lower_map.get(key.lower())
        if canonical is None:
            raise ValueError(
                f"Unknown line {line_name!r}. Valid: {', '.join(sorted(LINE_REST_FREQ_GHZ))}"
            )
        key = canonical
    rest_ghz = LINE_REST_FREQ_GHZ[key]
    obs_ghz  = rest_ghz / (1 + redshift)
    return f"{obs_ghz:.6f}GHz", rest_ghz * 1e9


LINE_REST_FREQ_GHZ: dict[str, float] = {
    "CO(1-0)":   115.271203,  "CO(2-1)":   230.538000,  "CO(3-2)":   345.795990,
    "CO(4-3)":   461.040768,  "CO(5-4)":   576.267931,  "CO(6-5)":   691.473076,
    "CO(7-6)":   806.651806,
    "[CI](1-0)": 492.160651,  "[CI](2-1)": 809.341970,
    "[CII]":    1900.536900,  "[NII]205":  1461.133800,  "[OIII]88": 3393.006240,
    "H2O(1-1)":  556.936002,  "H2O(2-1)":  752.033227,  "H2O(3-1)": 1113.342964,
    "HCN(1-0)":   88.631847,  "HCN(2-1)":  177.261111,
    "HCN(3-2)":  265.886434,  "HCN(4-3)":  354.505477,
    "HCO+(1-0)":  89.188526,  "HCO+(3-2)": 267.557625,  "HCO+(4-3)": 356.734288,
    "CS(2-1)":    97.980953,  "CS(7-6)":   342.882857,
}

C_KMS = 2.99792458e5


@dataclass
class ExportConfig:
    """tclean parameters for spectral cube imaging."""
    timebin: str = "0s"
    weighting: str = "natural"
    gridder: str = "standard"
    specmode: str = "cube"
    niter: int = 0
    pblimit: float = 0.0
    imsize: Optional[int] = None
    cell: Optional[str] = None
    spw: Optional[str] = None
    field: Optional[str] = None
    restfreq: Optional[str] = None
    outframe: Optional[str] = None
    width_kms: str = ""


class ExportCube:
    """Produce and analyze a spectral cube FITS from one or more MS using CASA tclean."""

    def __init__(self, concatvis, output_dir: str, config: Optional[ExportConfig] = None,
                 target: Optional[str] = None, overwrite: bool = False):
        if isinstance(concatvis, str):
            self.concatvis = [s.strip() for s in concatvis.split(",")] if "," in concatvis else [concatvis]
        elif isinstance(concatvis, list):
            self.concatvis = concatvis
        else:
            raise ValueError("concatvis must be a string or list of strings")
        self.output_dir = Path(output_dir)
        self.config     = config or ExportConfig()
        self.target     = target
        self.overwrite  = overwrite
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"ExportCube(concatvis={self.concatvis!r}, output_dir={str(self.output_dir)!r})"

    # ---- Static helpers --------------------------------------------------

    @staticmethod
    def _get_freq_axis(header, nchan):
        """Return frequency axis [Hz] from FITS WCS keywords."""
        crval = header.get("CRVAL3", 0)
        cdelt = header.get("CDELT3", 1)
        crpix = header.get("CRPIX3", 1)
        return crval + (np.arange(nchan) - (crpix - 1)) * cdelt

    @staticmethod
    def _get_beam_deg(header, hdul):
        """Return (bmaj_deg, bmin_deg, bpa_deg) from FITS header or BEAMS table."""
        bmaj_deg = header.get("BMAJ")
        bmin_deg = header.get("BMIN")
        bpa_deg  = header.get("BPA") or 0.0
        if bmaj_deg is None or bmin_deg is None:
            try:
                bmaj_deg = hdul[1].data['BMAJ'].mean() / 3600
                bmin_deg = hdul[1].data['BMIN'].mean() / 3600
                bpa_deg  = float(hdul[1].data['BPA'].mean()) \
                           if 'BPA' in hdul[1].data.names else 0.0
            except Exception:
                raise ValueError("[ExportCube] Header missing BMAJ/BMIN.")
        return bmaj_deg, bmin_deg, bpa_deg

    @staticmethod
    def _beam_area_pixels(header, hdul):
        """Return synthesised beam area in pixels² for Jy/beam -> Jy/pixel conversion."""
        bmaj_deg, bmin_deg, _ = ExportCube._get_beam_deg(header, hdul)
        return ((np.pi / (4.0 * np.log(2.0))) * bmaj_deg * bmin_deg
                / (abs(header.get("CDELT1", 1)) * abs(header.get("CDELT2", 1))))

    @staticmethod
    def _ellipse_mask(shape, cy, cx, a_pix, b_pix, pa_deg=0.0):
        """Return 2D boolean mask for an ellipse centred at (cy, cx)."""
        ny, nx         = shape
        yy, xx         = np.mgrid[:ny, :nx].astype(float)
        dy, dx         = yy - cy, xx - cx
        pa_rad         = np.deg2rad(pa_deg)
        cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)
        u = dy * cos_pa + dx * sin_pa
        v = -dy * sin_pa + dx * cos_pa
        return (u / max(a_pix, 0.5)) ** 2 + (v / max(b_pix, 0.5)) ** 2 <= 1.0

    @staticmethod
    def _fit_gaussian_line(freq_hz, spectrum_jy, peak_ch, bad_channels=None, fit_half_width=15):
        """Fit a 1D Gaussian to the spectrum near peak_ch."""
        nchan   = len(spectrum_jy)
        bad_set = set(bad_channels or [])
        lo = max(0, peak_ch - fit_half_width)
        hi = min(nchan, peak_ch + fit_half_width + 1)
        x, y  = freq_hz[lo:hi], spectrum_jy[lo:hi]
        valid = np.array(
            [not np.isnan(y[i]) and (lo + i) not in bad_set for i in range(len(y))],
            dtype=bool,
        )
        xv, yv  = x[valid], y[valid]
        chan_hz  = abs(float(freq_hz[1] - freq_hz[0])) if nchan > 1 else 1e6
        g0 = _m.Gaussian1D(
            amplitude=float(spectrum_jy[peak_ch]),
            mean=float(freq_hz[peak_ch]),
            stddev=2.0 * chan_hz,
        )
        g0.amplitude.bounds = (0.0, None)
        g_fit        = _f.LevMarLSQFitter()(g0, xv, yv, maxiter=500)
        center_hz    = float(g_fit.mean.value)
        amplitude_jy = float(g_fit.amplitude.value)
        stddev_hz    = abs(float(g_fit.stddev.value))
        integral_mjy_kms = (amplitude_jy * stddev_hz * np.sqrt(2.0 * np.pi)
                            * C_KMS / center_hz * 1e3) if center_hz > 0 else np.nan
        return center_hz, amplitude_jy, stddev_hz, integral_mjy_kms, g_fit

    @staticmethod
    def _save_detection_overview_plot(moment8_map, header, regions,
                                      a_pix, b_pix, bpa_deg, sigma, threshold, outfile,
                                      skipped_regions=None,
                                      continuum_map=None, continuum_std=None):
        """Save moment-8 image with 4σ beam ellipses.

        Kept detections drawn with colored solid ellipses; skipped detections
        (failed FWHM cut) drawn with grey dashed ellipses. If a continuum map
        and its std are supplied, overlay contours at [-5,-3,3,5,7]×std
        (negative dashed, positive solid).
        """
        skipped_regions = skipped_regions or []
        wcs2d = WCS(header).celestial
        fig   = plt.figure(figsize=(8, 7))
        ax    = fig.add_subplot(111, projection=wcs2d)
        ax.set_xlabel('Right Ascension (J2000)', fontsize=10)
        ax.set_ylabel('Declination (J2000)', fontsize=10)
        px_transform = ax.get_transform('pixel')

        im   = ax.imshow(moment8_map, origin='lower', cmap='inferno',
                         interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Peak SNR', shrink=0.85)
        ax.contour(moment8_map, levels=[threshold],
                   colors=['white'], linewidths=0.8, linestyles='--', alpha=0.7)

        # Continuum contours at [-5,-3,3,5,7]×std (negative dashed, positive solid)
        if (continuum_map is not None and continuum_std
                and np.isfinite(continuum_std) and continuum_std > 0):
            if continuum_map.shape == moment8_map.shape:
                neg = [m * continuum_std for m in (-5, -3)]
                pos = [m * continuum_std for m in (3, 5, 7)]
                ax.contour(continuum_map, levels=neg, colors='cyan',
                           linewidths=0.7, linestyles='dashed', alpha=0.9)
                ax.contour(continuum_map, levels=pos, colors='cyan',
                           linewidths=0.7, linestyles='solid', alpha=0.9)
                ax.plot([], [], color='cyan', lw=0.7,
                        label=r'continuum $\pm$[3,5,7]$\sigma$')
                ax.legend(fontsize=8, loc='lower right')
            else:
                print(f"[ExportCube] Continuum shape {continuum_map.shape} != "
                      f"moment-8 shape {moment8_map.shape}; skipping contours.")

        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, (cy, cx, _ell) in enumerate(regions):
            col   = colors[i % 10]
            patch = MplEllipse(
                (cx, cy), width=2 * b_pix, height=2 * a_pix, angle=bpa_deg,
                fill=False, edgecolor=col, linewidth=1.5, transform=px_transform,
            )
            ax.add_patch(patch)
            ax.text(cx + b_pix * 0.7, cy + a_pix * 0.7, str(i + 1),
                    color=col, fontsize=8, fontweight='bold', transform=px_transform)

        for cy, cx, _ell in skipped_regions:
            patch = MplEllipse(
                (cx, cy), width=2 * b_pix, height=2 * a_pix, angle=bpa_deg,
                fill=False, edgecolor='grey', linewidth=1.0, linestyle='--',
                alpha=0.5, transform=px_transform,
            )
            ax.add_patch(patch)

        n_kept    = len(regions)
        n_skipped = len(skipped_regions)
        ax.set_title(
            f"Moment-8 (SNR)  —  {n_kept} detection(s) above SNR = {sigma}"
            + (f"  ({n_skipped} skipped, grey dashed)" if n_skipped else "")
            + "\n(ellipse = 4σ beam)",
            fontsize=10,
        )
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
        plt.savefig(outfile, dpi=150)
        plt.close()

    # ---- Static analysis -------------------------------------------------

    @staticmethod
    def load_fits_cube(fits_path):
        """Load FITS cube; squeezes stokes axis -> (nchan, ny, nx). Returns (data, header, hdul)."""
        hdul   = fits.open(fits_path)
        raw    = hdul[0].data
        header = hdul[0].header
        data   = raw.squeeze()
        if data.ndim == 4:
            data = data[0]
        if data.ndim != 3:
            raise ValueError(f"[ExportCube] Unexpected cube shape: {data.shape}")
        return data, header, hdul

    @staticmethod
    def std_per_channel(cube):
        """Return per-channel spatial std, shape (nchan,)."""
        return np.nanstd(cube, axis=(1, 2))

    @staticmethod
    def find_bad_channels(stds, sigma=4, flag_high_rms=False):
        """Return channel indices with zero or NaN std.

        If flag_high_rms=True (legacy), also flags channels where std exceeds
        mean + sigma * std across all channels.
        """
        bad_mask = np.isnan(stds) | (stds == 0)
        if flag_high_rms:
            mean = np.nanmean(stds)
            std  = np.nanstd(stds)
            bad_mask |= (stds > mean + sigma * std)
        return np.where(bad_mask)[0].tolist()

    @staticmethod
    def plot_channel_rms(stds, freq_axis, bad_channels, sigma, outfile, title=None):
        """Plot per-channel RMS with color-coded bad channels (high-RMS, NaN, zero) and save."""
        nchan     = len(stds)
        freq_ghz  = freq_axis / 1e9
        stds_mjy  = stds * 1e3
        nan_mask  = np.isnan(stds_mjy)
        zero_mask = (~nan_mask) & (stds_mjy == 0)
        bad_set   = set(bad_channels)
        high_mask = np.array([(i in bad_set) and not nan_mask[i] and not zero_mask[i]
                               for i in range(nchan)])
        good_mask = ~nan_mask & ~zero_mask & ~high_mask
        mean_mjy  = float(np.nanmean(stds_mjy))
        thr_mjy   = mean_mjy + sigma * float(np.nanstd(stds_mjy))

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(freq_ghz, np.where(nan_mask, np.nan, stds_mjy), color="steelblue", lw=0.8, zorder=1)
        ax.plot(freq_ghz[good_mask], stds_mjy[good_mask], "o", color="steelblue", ms=3, zorder=2,
                label=f"Good ({good_mask.sum()})")
        if high_mask.any():
            ax.plot(freq_ghz[high_mask], stds_mjy[high_mask], "o", color="crimson", ms=6, zorder=4,
                    label=f"High RMS ({high_mask.sum()})")
        if nan_mask.any():
            ax.plot(freq_ghz[nan_mask], np.zeros(nan_mask.sum()), "x", color="darkorange",
                    ms=7, mew=1.5, zorder=5, label=f"NaN std ({nan_mask.sum()})")
        if zero_mask.any():
            ax.plot(freq_ghz[zero_mask], stds_mjy[zero_mask], "^", color="goldenrod", ms=6, zorder=3,
                    label=f"Zero std ({zero_mask.sum()})")
        ax.axhline(thr_mjy,  color="crimson", ls="--", lw=1.0, label=f"mean+{sigma}σ = {thr_mjy:.3f} mJy/beam")
        ax.axhline(mean_mjy, color="gray",    ls=":",  lw=1.0, label=f"mean = {mean_mjy:.3f} mJy/beam")
        ax.set_xlabel("Frequency [GHz]", fontsize=11)
        ax.set_ylabel("RMS [mJy/beam]", fontsize=11)
        ax.set_title(title or "Per-channel RMS", fontsize=12)
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        ax.set_xlim(freq_ghz.min(), freq_ghz.max())
        ax2 = ax.twiny(); ax2.set_xlim(0, nchan - 1); ax2.set_xlabel("Channel index", fontsize=9)
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
        plt.savefig(outfile, dpi=150); plt.close()

    @staticmethod
    def plot_sample_spectrum(cube, stds, header, outfile, bad_channels=None, title=None):
        """Plot center-pixel spectrum in SNR units (flux / per-channel RMS)."""
        nchan, ny, nx = cube.shape
        freq_ghz      = ExportCube._get_freq_axis(header, nchan) / 1e9
        cy, cx        = ny // 2, nx // 2
        rms           = stds.astype(float).copy()
        rms[(rms == 0) | np.isnan(rms)] = np.nan
        spectrum_snr  = cube[:, cy, cx] / rms

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(freq_ghz, spectrum_snr, color="steelblue", lw=0.9, drawstyle="steps-mid")
        ax.axhline(0, color="gray", lw=0.6, ls="--")
        ax.set_xlabel("Frequency [GHz]", fontsize=11)
        ax.set_ylabel("SNR", fontsize=11)
        ax.set_title(title or f"Sample spectrum (SNR) — center pixel ({cy}, {cx})", fontsize=12)
        ax.set_xlim(freq_ghz.min(), freq_ghz.max())

        ax2 = ax.twiny(); ax2.set_xlim(0, nchan - 1); ax2.set_xlabel("Channel index", fontsize=9)
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
        plt.savefig(outfile, dpi=150); plt.close()

    @staticmethod
    def make_moment8_map(cube, stds):
        """Return peak-SNR map (ny, nx) in signal-to-noise units.

        Each channel is divided by its per-channel RMS (stds).  Channels with
        zero or NaN RMS are masked to NaN before taking the max, so they cannot
        dominate the peak map regardless of their actual flux.
        """
        rms = stds.astype(float).copy()
        rms[(rms == 0) | np.isnan(rms)] = np.nan
        snr_cube = cube / rms[:, np.newaxis, np.newaxis]
        return np.nanmax(snr_cube, axis=0)

    @staticmethod
    def save_moment8_fits(map_data, header, filename):
        """Write the moment-8 map as a 2-D FITS file with a clean 2-D WCS header."""
        hdr2d = header.copy()
        for key in ["NAXIS3","CRVAL3","CDELT3","CRPIX3","CTYPE3","CUNIT3",
                    "NAXIS4","CRVAL4","CDELT4","CRPIX4","CTYPE4","CUNIT4",
                    "PC3_1","PC3_2","PC1_3","PC2_3","PC3_3","PC4_4"]:
            hdr2d.remove(key, ignore_missing=True)
        hdr2d["NAXIS"] = 2
        hdr2d["BUNIT"] = "SNR"
        fits.writeto(filename, map_data.astype(np.float32), hdr2d, overwrite=True)

    @staticmethod
    def extract_and_plot_profiles(cube, moment8_map, header, hdul, output_dir,
                                  sigma, line_freq_hz=None, bad_channels=None,
                                  continuum_map=None, continuum_std=None):
        """Greedy 4σ-ellipse extraction of spectral line profiles from the moment-8 map.

        Algorithm
        ---------
        1. Find all pixels above sigma × std in the moment-8 map; sort by brightness.
        2. Take the brightest unmasked pixel → place a 4σ beam ellipse there.
        3. Convert cube to Jy/pixel and integrate over the ellipse → spectrum [Jy].
        4. Mark all pixels inside the ellipse as used; repeat from step 2.
        5. Fit a 1D Gaussian to each spectrum, estimate redshift and line integral.
        6. Save per-detection spectrum plots and a WCS diagnostic overview.

        Parameters
        ----------
        line_freq_hz : float, optional
            REST frequency [Hz] used to compute z = ν_rest / ν_obs − 1.
        bad_channels : list of int, optional
            Channel indices masked before fitting.
        continuum_map : 2D array, optional
            Continuum image (same pixel grid as moment-8) for contour overlay.
        continuum_std : float, optional
            Std of the continuum map; contours drawn at [-5,-3,3,5,7]×std.
        """
        os.makedirs(output_dir, exist_ok=True)
        bad_set       = set(bad_channels or [])
        nchan, ny, nx = cube.shape
        freq_axis     = ExportCube._get_freq_axis(header, nchan)
        freq_ghz      = freq_axis / 1e9

        bmaj_deg, bmin_deg, bpa_deg = ExportCube._get_beam_deg(header, hdul)
        cdelt2_deg = abs(header.get("CDELT2", 1))
        SIGMA_MULT = 4.0 / 2.3548   # FWHM = 2.3548σ → 4σ semi-axis
        a_pix = bmaj_deg / cdelt2_deg * SIGMA_MULT
        b_pix = bmin_deg / cdelt2_deg * SIGMA_MULT

        cube_jy_pix = cube / ExportCube._beam_area_pixels(header, hdul)

        threshold  = float(sigma)
        above_mask = np.isfinite(moment8_map) & (moment8_map > threshold)
        above_yx   = np.argwhere(above_mask)

        # Greedy ellipse extraction (may yield empty regions)
        regions = []
        if len(above_yx) > 0:
            above_yx = above_yx[np.argsort(-moment8_map[above_mask])]
            used = np.zeros((ny, nx), dtype=bool)
            for py, px in above_yx:
                py, px = int(py), int(px)
                if used[py, px]:
                    continue
                ell = ExportCube._ellipse_mask((ny, nx), py, px, a_pix, b_pix, bpa_deg)
                used |= ell
                regions.append((py, px, ell))

        print(f"[ExportCube] {len(regions)} detection(s) above SNR = {sigma}.")

        diag_dir = os.path.join(os.path.dirname(os.path.abspath(output_dir)), "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)

        if not regions:
            ExportCube._save_detection_overview_plot(
                moment8_map, header, [], a_pix, b_pix, bpa_deg, sigma, threshold,
                outfile=os.path.join(diag_dir, "moment8_detections.png"),
                continuum_map=continuum_map, continuum_std=continuum_std,
            )
            return

        # Per-detection spectrum extraction and plotting
        HALF_BW_GHZ  = 0.75   # show ±0.75 GHz (= 1.5 GHz total) around the line
        cube_2d      = cube_jy_pix.reshape(nchan, ny * nx)
        kept_regions    = []
        skipped_regions = []
        detections      = []   # per-kept-detection stats for CSV
        wcs2d           = WCS(header).celestial

        for idx, (py, px, ell) in enumerate(regions, start=1):
            ell_flat    = ell.ravel()
            n_pix       = int(ell_flat.sum())
            spectrum_jy = np.nansum(cube_2d[:, ell_flat], axis=1).astype(float)
            for ch in bad_set:
                if 0 <= ch < nchan:
                    spectrum_jy[ch] = np.nan

            peak_ch       = int(np.nanargmax(spectrum_jy))
            peak_freq_ghz = float(freq_axis[peak_ch]) / 1e9
            peak_flux_jy  = float(spectrum_jy[peak_ch])

            center_hz, amp_jy, stddev_hz, integral, g_fit = ExportCube._fit_gaussian_line(
                freq_axis, spectrum_jy, peak_ch, bad_channels=list(bad_set)
            )
            center_ghz = center_hz / 1e9
            fwhm_kms   = stddev_hz * 2.3548 * C_KMS / center_hz
            z_est      = line_freq_hz / center_hz - 1.0 if line_freq_hz else None
            print(f"[ExportCube]   Gaussian: center={center_ghz:.6f} GHz,  "
                  f"FWHM={fwhm_kms:.1f} km/s,  integral={integral:.1f} mJy km/s")
            if fwhm_kms < 70.0:
                print(f"[ExportCube]   Skipping detection {idx} — FWHM={fwhm_kms:.1f} km/s < 70 km/s.")
                skipped_regions.append((py, px, ell))
                continue
            if fwhm_kms > 700.0:
                print(f"[ExportCube]   Skipping detection {idx} — FWHM={fwhm_kms:.1f} km/s > 700 km/s.")
                skipped_regions.append((py, px, ell))
                continue
            kept_regions.append((py, px, ell))

            # Sky coordinates (J2000) of the detection peak pixel
            sky      = wcs2d.pixel_to_world(px, py)
            ra_deg   = float(sky.icrs.ra.deg)
            dec_deg  = float(sky.icrs.dec.deg)
            radec_hd = sky.icrs.to_string("hmsdms", sep=":", precision=2)
            ra_str, dec_str = radec_hd.split(" ")
            detections.append({
                "detection":      len(kept_regions),
                "RA_J2000":       ra_str,
                "Dec_J2000":      dec_str,
                "RA_deg":         f"{ra_deg:.6f}",
                "Dec_deg":        f"{dec_deg:.6f}",
                "pix_x":          px,
                "pix_y":          py,
                "n_pix":          n_pix,
                "center_GHz":     f"{center_ghz:.6f}",
                "peak_GHz":       f"{peak_freq_ghz:.6f}",
                "redshift":       f"{z_est:.6f}" if z_est is not None else "",
                "FWHM_kms":       f"{fwhm_kms:.1f}",
                "amplitude_mJy":  f"{amp_jy * 1e3:.3f}",
                "peak_flux_mJy":  f"{peak_flux_jy * 1e3:.3f}",
                "integral_mJy_kms": f"{integral:.1f}",
            })

            if z_est is not None:
                print(f"[ExportCube]   Redshift: z = {z_est:.6f}  (v = {z_est * C_KMS:.0f} km/s)")

            # Frequency window: ±0.75 GHz around Gaussian center, clipped to coverage
            plot_lo = max(freq_ghz.min(), center_ghz - HALF_BW_GHZ)
            plot_hi = min(freq_ghz.max(), center_ghz + HALF_BW_GHZ)

            # Corresponding channel-index range for the top axis
            ch_scale = (nchan - 1) / (freq_ghz[-1] - freq_ghz[0])
            ch_lo    = (plot_lo - freq_ghz[0]) * ch_scale
            ch_hi    = (plot_hi - freq_ghz[0]) * ch_scale

            # Y-axis: min/max of visible data excluding bad channels
            vis_mask = (freq_ghz >= plot_lo) & (freq_ghz <= plot_hi)
            for ch in bad_set:
                if 0 <= ch < nchan:
                    vis_mask[ch] = False
            vis_spec = spectrum_jy[vis_mask] * 1e3   # mJy
            if vis_spec.size > 0 and np.any(np.isfinite(vis_spec)):
                ylo    = float(np.nanmin(vis_spec))
                yhi    = float(np.nanmax(vis_spec))
                margin = 0.15 * max(yhi - ylo, 1e-6)
            else:
                ylo, yhi, margin = -1.0, 1.0, 0.1

            spectrum_mjy = spectrum_jy * 1e3
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(freq_ghz, spectrum_mjy, color="steelblue", lw=1.0, drawstyle="steps-mid", zorder=2)

            if bad_set:
                cdelt_ghz = abs((freq_ghz[-1] - freq_ghz[0]) / max(nchan - 1, 1))
                for ch in sorted(bad_set):
                    if 0 <= ch < nchan:
                        ax.axvspan(freq_ghz[ch] - 0.5 * cdelt_ghz,
                                   freq_ghz[ch] + 0.5 * cdelt_ghz,
                                   color="gray", alpha=0.15, lw=0, zorder=0)

            ax.axvline(peak_freq_ghz, color="crimson", ls="--", lw=0.9, zorder=3,
                       label=f"Peak channel  {peak_freq_ghz:.5f} GHz\n"
                             f"{peak_flux_jy*1e3:.2f} mJy  ({n_pix} px)")

            plot_x = freq_axis[max(0, peak_ch - 20) : min(nchan, peak_ch + 21)]
            z_lbl  = f"  z={z_est:.2f}" if z_est is not None else ""
            ax.plot(plot_x / 1e9, g_fit(plot_x) * 1e3, color="darkorange", lw=1.8, alpha=0.7, zorder=4,
                    label=f"Gaussian  {center_ghz:.5f} GHz{z_lbl}\nFWHM={round(fwhm_kms):.0f} km/s  I={round(integral):.0f} mJy km/s")
            ax.axvline(center_ghz, color="darkorange", ls=":", lw=0.9, zorder=3)

            ax.axhline(0, color="gray", lw=0.6, ls="--", zorder=1)
            ax.set_xlabel("Frequency [GHz]", fontsize=11)
            ax.set_ylabel("Flux density [mJy]", fontsize=11)
            z_title = f"  z={z_est:.2f}" if z_est is not None else ""
            ax.set_title(f"Detection {idx} — pixel ({py},{px})  "
                         f"[{n_pix} px, 4σ ellipse]{z_title}", fontsize=10)
            ax.legend(fontsize=9, loc="upper right")
            ax.set_xlim(plot_lo, plot_hi)
            ax.set_ylim(ylo - margin, yhi + margin)
            ax2 = ax.twiny(); ax2.set_xlim(ch_lo, ch_hi); ax2.set_xlabel("Channel index", fontsize=9)
            plt.tight_layout()

            z_tag   = f"_z{z_est:.4f}".replace(".", "p") if z_est is not None else ""
            outname = os.path.join(output_dir, f"spectrum_det{idx:02d}_pix{py}_{px}{z_tag}.png")
            plt.savefig(outname, dpi=150)
            plt.close()

            # Zoomed plot: ±5× FWHM around Gaussian center
            fwhm_ghz   = fwhm_kms / C_KMS * center_ghz
            zoom_lo    = max(freq_ghz.min(), center_ghz - 5.0 * fwhm_ghz)
            zoom_hi    = min(freq_ghz.max(), center_ghz + 5.0 * fwhm_ghz)
            zoom_mask  = (freq_ghz >= zoom_lo) & (freq_ghz <= zoom_hi)
            zoom_spec  = spectrum_mjy[zoom_mask]
            if zoom_spec.size > 0 and np.any(np.isfinite(zoom_spec)):
                zylo    = float(np.nanmin(zoom_spec))
                zyhi    = float(np.nanmax(zoom_spec))
                zmargin = 0.15 * max(zyhi - zylo, 1e-6)
            else:
                zylo, zyhi, zmargin = -1.0, 1.0, 0.1

            zoom_x = freq_axis[(freq_axis / 1e9 >= zoom_lo) & (freq_axis / 1e9 <= zoom_hi)]

            fig2, ax2z = plt.subplots(figsize=(5, 5))
            ax2z.plot(freq_ghz, spectrum_mjy, color="steelblue", lw=1.0, drawstyle="steps-mid", zorder=2)
            ax2z.plot(zoom_x / 1e9, g_fit(zoom_x) * 1e3, color="darkorange", lw=1.8, alpha=0.85, zorder=4,
                      label=f"Gaussian  {center_ghz:.5f} GHz{z_lbl}\nFWHM={round(fwhm_kms):.0f} km/s  I={round(integral):.0f} mJy km/s")
            ax2z.axhline(0, color="gray", lw=0.6, ls="--", zorder=1)
            ax2z.set_xlabel("Frequency [GHz]", fontsize=11)
            ax2z.set_ylabel("Flux density [mJy]", fontsize=11)
            ax2z.legend(fontsize=9, loc="upper right")
            ax2z.set_xlim(zoom_lo, zoom_hi)
            ax2z.set_ylim(zylo - zmargin, zyhi + zmargin)
            plt.tight_layout()
            outname_zoom = os.path.join(output_dir, f"spectrum_det{idx:02d}_pix{py}_{px}{z_tag}_zoom.png")
            plt.savefig(outname_zoom, dpi=150)
            plt.close()

        ExportCube._save_detection_overview_plot(
            moment8_map, header, kept_regions,
            a_pix, b_pix, bpa_deg, sigma, threshold,
            outfile=os.path.join(diag_dir, "moment8_detections.png"),
            skipped_regions=skipped_regions,
            continuum_map=continuum_map, continuum_std=continuum_std,
        )

        # Write Gaussian statistics of kept detections to CSV
        if detections:
            csv_path = os.path.join(output_dir, "detections.csv")
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(detections[0].keys()))
                writer.writeheader()
                writer.writerows(detections)
            print(f"[ExportCube] Wrote {len(detections)} detection(s) to {csv_path}")

    # ---- Pipeline --------------------------------------------------------

    def uvcontsub(self, fitspec: str = "", fitorder: int = 0) -> List[str]:
        """Run CASA uvcontsub on each MS; returns list of .contsub paths."""
        output_paths = []
        for vis in self.concatvis:
            outputvis = vis + ".contsub"
            if os.path.exists(outputvis):
                if self.overwrite:
                    print(f"[ExportCube]   Removing existing {outputvis}")
                    _safe_rmtree(outputvis)
                else:
                    print(f"[ExportCube]   Skipping uvcontsub — {outputvis} exists (overwrite=False).")
                    output_paths.append(outputvis)
                    continue
            casa_uvcontsub(vis=vis, outputvis=outputvis, fitspec=fitspec, fitorder=fitorder)
            output_paths.append(outputvis)
        return output_paths

    def make_cube(self, imagename: Optional[str] = None, vis_list: Optional[List[str]] = None) -> str:
        """Run tclean (specmode=cube) and export FITS; returns path to the FITS cube."""
        fits_dir       = self.output_dir / "fits"
        fits_dir.mkdir(parents=True, exist_ok=True)
        imagename      = imagename or f"{(self.target or 'cube').replace(' ', '_')}_cube"
        imagename_full = str(fits_dir / imagename)
        fits_path      = f"{imagename_full}.cube.fits"

        if not self.overwrite and os.path.exists(fits_path):
            print(f"[ExportCube] Skipping tclean — cube exists (overwrite=False): {fits_path}")
            return fits_path

        if self.overwrite:
            for suffix in [".image", ".pb", ".psf", ".model", ".sumwt", ".weight", ".residual", ".mask"]:
                p = f"{imagename_full}{suffix}"
                if os.path.exists(p):
                    _safe_rmtree(p)

        vis_to_use = vis_list or self.concatvis
        tclean(
            vis=vis_to_use if len(vis_to_use) > 1 else vis_to_use[0],
            imagename=imagename_full,
            imsize=self.config.imsize or 512,
            cell=self.config.cell or "1.5arcsec",
            weighting=self.config.weighting, gridder=self.config.gridder,
            niter=self.config.niter, pblimit=self.config.pblimit,
            specmode=self.config.specmode,
            spw=self.config.spw or "", field=self.config.field or "",
            restfreq=self.config.restfreq or "", outframe=self.config.outframe or "",
            width=self.config.width_kms, parallel=False,
        )
        exportfits(f"{imagename_full}.image", fits_path, overwrite=True)
        for suffix in [".image", ".pb", ".psf", ".model", ".sumwt", ".weight", ".residual", ".mask"]:
            p = f"{imagename_full}{suffix}"
            if os.path.exists(p):
                _safe_rmtree(p)

        return fits_path

    def make_continuum(self, imagename: Optional[str] = None,
                       vis_list: Optional[List[str]] = None) -> str:
        """Dirty MFS continuum image over all fields/SPWs (pre-contsub).

        Uses the same imsize/cell as the cube so the pixel grid matches the
        moment-8 map for contour overlay. niter=0 (dirty). Returns FITS path.
        """
        fits_dir       = self.output_dir / "fits"
        fits_dir.mkdir(parents=True, exist_ok=True)
        imagename      = (imagename or f"{(self.target or 'cube').replace(' ', '_')}") + "_continuum"
        imagename_full = str(fits_dir / imagename)
        fits_path      = f"{imagename_full}.cont.fits"

        if not self.overwrite and os.path.exists(fits_path):
            print(f"[ExportCube] Skipping continuum tclean — exists (overwrite=False): {fits_path}")
            return fits_path

        if self.overwrite:
            for suffix in [".image", ".pb", ".psf", ".model", ".sumwt", ".weight", ".residual", ".mask"]:
                p = f"{imagename_full}{suffix}"
                if os.path.exists(p):
                    _safe_rmtree(p)

        vis_to_use = vis_list or self.concatvis
        tclean(
            vis=vis_to_use if len(vis_to_use) > 1 else vis_to_use[0],
            imagename=imagename_full,
            imsize=self.config.imsize or 512,
            cell=self.config.cell or "1.5arcsec",
            weighting=self.config.weighting, gridder=self.config.gridder,
            niter=0, pblimit=self.config.pblimit,
            specmode="mfs",
            spw="", field="",
            parallel=False,
        )
        exportfits(f"{imagename_full}.image", fits_path, overwrite=True)
        for suffix in [".image", ".pb", ".psf", ".model", ".sumwt", ".weight", ".residual", ".mask"]:
            p = f"{imagename_full}{suffix}"
            if os.path.exists(p):
                _safe_rmtree(p)

        return fits_path

    @staticmethod
    def load_continuum_map(fits_path):
        """Load a 2-D continuum image and its std. Returns (map_2d, std)."""
        hdul = fits.open(fits_path)
        data = hdul[0].data.squeeze()
        hdul.close()
        if data.ndim == 3:
            data = data[0]
        if data.ndim != 2:
            raise ValueError(f"[ExportCube] Unexpected continuum shape: {data.shape}")
        return data, float(np.nanstd(data))

    def run_all(self, do_uvcontsub: bool, bad_channel_sigma: float, detection_sigma: float,
                imagename: Optional[str] = None, line_freq_hz: Optional[float] = None) -> None:
        """End-to-end pipeline: uvcontsub → tclean → flag channels → moment-8 → spectra."""
        out      = self.output_dir / "analysis"
        diag_dir = out / "diagnostics"
        out.mkdir(parents=True, exist_ok=True)
        diag_dir.mkdir(exist_ok=True)

        print(f"\n[ExportCube] run_all() — {self.output_dir}")
        if line_freq_hz:
            print(f"[ExportCube]   line = {line_freq_hz/1e9:.6f} GHz")

        # Dirty continuum over all fields/SPWs from the raw (pre-contsub) data
        cont_fits = self.make_continuum(imagename=imagename)
        cont_map, cont_std = ExportCube.load_continuum_map(cont_fits)
        print(f"[ExportCube] Continuum std = {cont_std*1e3:.4f} mJy/beam; "
              f"contours at [-5,-3,3,5,7]×std")

        contsub_paths = self.uvcontsub() if do_uvcontsub else None
        fits_path     = self.make_cube(imagename=imagename, vis_list=contsub_paths)

        data, header, hdul = ExportCube.load_fits_cube(fits_path)
        freq_axis = ExportCube._get_freq_axis(header, data.shape[0])
        stds      = ExportCube.std_per_channel(data)
        bad       = ExportCube.find_bad_channels(stds, sigma=bad_channel_sigma)
        print(f"[ExportCube] Bad channels ({bad_channel_sigma}σ, {len(bad)} total): {bad}")

        ExportCube.plot_channel_rms(stds, freq_axis, bad, sigma=bad_channel_sigma,
                                    outfile=str(diag_dir / "channel_rms.png"),
                                    title=f"Per-channel RMS  (σ={bad_channel_sigma}  |  bad={len(bad)})")
        ExportCube.plot_sample_spectrum(data, stds, header,
                                        outfile=str(diag_dir / "sample_spectrum_center.png"),
                                        title="Sample spectrum (SNR) — center pixel")

        m8        = ExportCube.make_moment8_map(data, stds)
        fits_stem = os.path.basename(fits_path)
        m8_name   = fits_stem.replace("_contsub_cube.cube.fits", "_contsub_moment8.fits")
        if m8_name == fits_stem:
            m8_name = fits_stem.replace(".cube.fits", "_moment8.fits")
        if m8_name == fits_stem:
            m8_name = os.path.splitext(fits_stem)[0] + "_moment8.fits"
        ExportCube.save_moment8_fits(m8, header, str(self.output_dir / "fits" / m8_name))

        ExportCube.extract_and_plot_profiles(data, m8, header, hdul,
                                             output_dir=str(out / "spectra"),
                                             sigma=detection_sigma,
                                             line_freq_hz=line_freq_hz,
                                             bad_channels=bad,
                                             continuum_map=cont_map,
                                             continuum_std=cont_std)

        for log in Path(".").glob("casa*.log"):
            log.unlink()
            print(f"[ExportCube] Removed {log}")

        print(f"[ExportCube] run_all() done — {out}")
