"""alma-data-prep: ALMA visibility organization, concatenation, and cube export.

Top-level entry points:
    ProjectDataOrganizer  -- discover, concat, and tabulate ALMA projects
    Export                -- per-field/SPW UV + continuum FITS export
    ExportCube            -- spectral-cube imaging and line detection
"""

from .organizer import ProjectDataOrganizer
from .export import Export, ExportConfig
from .export_cube import (
    ExportCube,
    ExportConfig as CubeExportConfig,
    parse_line_freq,
)

__all__ = [
    "ProjectDataOrganizer",
    "Export",
    "ExportConfig",
    "ExportCube",
    "CubeExportConfig",
    "parse_line_freq",
]

__version__ = "0.1.0"
