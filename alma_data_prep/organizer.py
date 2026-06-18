import os
import re
import sys
import csv
import shutil
import pandas as pd
from collections import defaultdict
from pathlib import Path
from casatasks import mstransform, concat, listobs, statwt

from .export import Export
from .export_cube import ExportCube, ExportConfig as CubeExportConfig, parse_line_freq

# Add the path for analysisUtils (CASA analysis_scripts). Override location with
# the CASA_ANALYSIS_SCRIPTS environment variable; falls back to the Allegro path.
sys.path.append(os.environ.get(
    "CASA_ANALYSIS_SCRIPTS",
    "/almastorage/allegro/lib/jao-mirror/AIV/science/analysis_scripts/",
))
import analysisUtils as au  # CASA utilities
from . import analysis_utils as oau  # Custom utilities for white noise sensitivity

from casatools import msmetadata  # CASA 6+ tool

class ProjectDataOrganizer:
    def __init__(self, root_dir, concatted=False, data_dir = '../../data/', plotuvdist=True):
        self.root_dir = root_dir
        self.concatted = concatted  
        self.data_dir = data_dir

        # Ensure the new directory exists
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        self.projects = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
            "files": defaultdict(list), 
            "target": None, 
            "on_source_time": None,
            "min_resolution": None,
            "MRS": None,
            "dish_sizes": None,
            "median_frequency": None,
            "concatvis": None,
            "white_noise_sensitivity": None,
            "cube_export": None,
        })))          
        
        self.msmd = msmetadata()
        self.plotuvdist = plotuvdist
        self._run()
    
    def _run(self):
        """Runs all required processing steps to find directories and targets."""
        self._find_ms_split_cal_directories()
        self._get_targets_for_groups()
        self._get_dish_sizes_and_median_frequency()
        self.get_metadata()

    def get_metadata(self):
        """Computes all necessary metadata values like dish sizes, frequency, on-source time, resolution, and MRS."""
        self._get_total_on_source_time()
        self._get_minimum_resolution()
        self._get_maximum_recoverable_scale()

    def _get_vis_path(self, ms_file, **kwargs):
        if self.concatted:
            return os.path.join(self.data_dir, ms_file)  # New transformed file structure
        else:
            return os.path.join(self.root_dir, kwargs['project_code'], kwargs['science_goal'], 
                                kwargs['group'], kwargs['member'], "calibrated", ms_file)  # Original structure

    def _extract_identifiers(self, path):
        """Extract project_code, science_goal, group_uid, member_uid from a given path."""
        parts = path.split(os.sep)

        project_code, science_goal, group_uid, member_uid = None, None, None, None
        for part in parts:
            if re.match(r"\d{4}\.\d+\.\d+\.[A-Za-z]", part): 
                project_code = part
            elif part.startswith("science_goal.uid"):
                science_goal = part
            elif part.startswith("group.uid"):
                group_uid = part
            elif part.startswith("member.uid"):
                member_uid = part
        return project_code, science_goal, group_uid, member_uid

    def _find_ms_split_cal_directories(self):
        """Find all .ms.split.cal directories and organize them by project."""
        for root, dirs, _ in os.walk(self.root_dir):
            for directory in dirs:
                if directory.endswith(".ms.split.cal"):
                    full_path = os.path.join(root, directory)
                    last_dir_name = os.path.basename(full_path)

                    # Extract project_code, science_goal, group_uid, member_uid
                    project_code, science_goal, group_uid, member_uid = self._extract_identifiers(full_path)
                    if project_code and science_goal and group_uid and member_uid:
                        self.projects[project_code][science_goal][group_uid]["files"][member_uid].append(last_dir_name)

    def _get_targets_for_groups(self):
        """Check if all member.uids in a group have the same target, and store it at the group level."""
        for project_code, science_goals in self.projects.items():
            for science_goal, groups in science_goals.items():
                for group, data in groups.items():
                    member_targets = set()
                    for member, ms_files in data["files"].items():
                        for ms_file in ms_files:
                            vis = self._get_vis_path(ms_file, 
                                                     project_code=project_code, 
                                                     science_goal=science_goal, 
                                                     group=group, 
                                                     member=member)
                            
                            if not os.path.exists(vis): 
                                print(f"Warning: Could not find {vis}, skipping...")
                                continue
                            try:
                                self.msmd.open(vis)
                                targets = self.msmd.fieldsforintent("*OBSERVE_TARGET*", True)
                                if targets.size > 0:
                                    member_targets.add(str(targets[0]))  # Store target as a string
                                self.msmd.close()
                            except Exception as e:
                                print(f"Error processing {vis}: {e}")

                    # Ensure all member.uids have the same target
                    if len(member_targets) == 1:
                        data["target"] = member_targets.pop()  # Store the consistent target
                    elif len(member_targets) > 1:
                        print(f"Warning: Multiple different targets found in {group}: {member_targets}")
                        data["target"] = next(iter(member_targets))  # Assign one, but log the discrepancy

    def _get_all_targets(self):
        """Returns a list of all unique target names found in the dataset."""
        targets = set()
        
        for project_code, science_goals in self.projects.items():
            for science_goal, groups in science_goals.items():
                for group, data in groups.items():
                    if data["target"]:
                        targets.add(data["target"])  # Store unique targets
        
        return sorted(targets)  # Return sorted list of unique targets


    def _get_total_on_source_time(self):
        """Calculates the total on-source time per group UID by summing all member.uid times."""
        for project_code, science_goals in self.projects.items():
            for science_goal, groups in science_goals.items():
                for group, data in groups.items():
                    total_time = 0.0  # Ensure total time is a float
                    group_target = data["target"]  # Get the assigned target for this group
                    for member, ms_files in data["files"].items():
                        for ms_file in ms_files:
                            vis = self._get_vis_path(ms_file, 
                                                     project_code=project_code, 
                                                     science_goal=science_goal, 
                                                     group=group, 
                                                     member=member)
                            
                            if not os.path.exists(vis):  
                                print(f"Warning: Could not open ms = {vis}, skipping...")
                                continue
                            try:
                                member_time = au.timeOnSource(vis, field=group_target, verbose=True) 
                                if isinstance(member_time, dict) and "minutes_on_science" in member_time:
                                    minutes_on_science = member_time["minutes_on_science"]
                                    if isinstance(minutes_on_science, (int, float)):
                                        total_time += minutes_on_science
                                    else:
                                        print(f"Warning: Invalid time format for {vis}: {minutes_on_science}")
                                else:
                                    print(f"Warning: Missing 'minutes_on_science' field in response for {vis}")
                            except Exception as e:
                                print(f"Error calculating time for {vis}: {e}")
                                
                    data["on_source_time"] = total_time  # Store at the group level

    def _get_minimum_resolution(self):
        """Estimates the minimum synthesized beam resolution (arcseconds) for each group UID."""
        for project_code, science_goals in self.projects.items():
            for science_goal, groups in science_goals.items():
                for group, data in groups.items():
                    min_resolution = float('inf')  # Start with an infinitely large value
                    group_target = data["target"]  # Get the assigned target for this group
                    
                    for member, ms_files in data["files"].items():
                        for ms_file in ms_files:
                            vis = self._get_vis_path(ms_file, 
                                                     project_code=project_code, 
                                                     science_goal=science_goal, 
                                                     group=group, 
                                                     member=member)
                            
                            if not os.path.exists(vis):  
                                print(f"Warning: Could not open ms = {vis}, skipping...")
                                continue
                            try:
                                arcsec = au.estimateSynthesizedBeam(vis=vis, field=group_target, verbose=False)
                                
                                if isinstance(arcsec, (int, float)):  # Ensure it's a valid number
                                    min_resolution = min(min_resolution, arcsec)
                                else:
                                    print(f"Warning: Invalid resolution format for {vis}: {arcsec}")
                            except Exception as e:
                                print(f"Error calculating resolution for {vis}: {e}")

                    data["min_resolution"] = min_resolution if min_resolution != float('inf') else None

    def _get_dish_sizes_and_median_frequency(self):
        """Computes dish size and median frequency per member UID, storing results at the group level."""
        c = 299792458  # Speed of light in m/s

        for project_code, science_goals in self.projects.items():
            for science_goal, groups in science_goals.items():
                for group, data in groups.items():
                    unique_dish_sizes = set()
                    median_frequencies = []

                    for member, ms_files in data["files"].items():
                        for ms_file in ms_files:
                            vis = self._get_vis_path(ms_file, 
                                                     project_code=project_code, 
                                                     science_goal=science_goal, 
                                                     group=group, 
                                                     member=member)
                            
                            if not os.path.exists(vis):
                                print(f"Warning: Could not open ms = {vis}, skipping...")
                                continue
                            try:
                                # Determine dish size (7m or 12m)
                                if au.sevenMeterAntennasMajority(vis):
                                    dish_size = 7.0  # 7m antenna
                                else:
                                    dish_size = 12.0  # 12m antenna
                                unique_dish_sizes.add(dish_size)

                                # Compute median observation frequency
                                median_freq = au.medianFrequencyOfIntent(vis=vis, intent='OBSERVE_TARGET#ON_SOURCE',
                                                                         verbose=False, ignoreChanAvgSpws=False)

                                if isinstance(median_freq, (int, float)):
                                    median_frequencies.append(median_freq / 1e9)  # Convert Hz to GHz
                                else:
                                    print(f"Warning: Invalid median frequency for {vis}: {median_freq}")

                            except Exception as e:
                                print(f"Error processing antenna details for {vis}: {e}")

                    # Store unique dish sizes and median frequency at the group level
                    data["dish_sizes"] = sorted(unique_dish_sizes) if unique_dish_sizes else None
                    data["median_frequency"] = (sum(median_frequencies) / len(median_frequencies)) if median_frequencies else None

    def _get_maximum_recoverable_scale(self):
        """Computes Maximum Recoverable Scale (MRS) in arcsec, ensuring dish size & frequency are available."""
        c = 299792458  # Speed of light in m/s
        rad_to_arcsec= (180 / 3.141592653589793) * 3600  # Conversion factor from radians to arcsec

        for project_code, science_goals in self.projects.items():
            for science_goal, groups in science_goals.items():
                for group, data in groups.items():
                    # Ensure dish size and median frequency are available
                    if data["dish_sizes"] is None or data["median_frequency"] is None:
                        self._get_dish_sizes_and_median_frequency()

                    # Compute Maximum Recoverable Scale (MRS) in arcsec
                    if data["median_frequency"] and data["dish_sizes"]:
                        lambda_val = c / (data["median_frequency"] * 1e9)  # Convert frequency to wavelength
                        min_dish_size = min(data["dish_sizes"])  # Use the smallest dish size
                        mrs_radians = (1.22 * lambda_val) / min_dish_size  # Compute in radians
                        data["MRS"] = mrs_radians * rad_to_arcsec  # Convert to arcseconds
                    else:
                        data["MRS"] = None  # Could not compute MRS

    def mstransform_and_concat(self):
        """Transforms and concatenates visibility data."""
        print(f"Concatinating Observation Sets")

        for project_code, science_goals in self.projects.items():
            for science_goal, groups in science_goals.items():
                for group, data in groups.items():
                    vislist = []

                    print(f"  Target: {data['target'] if data['target'] else 'Unknown'}")
                    
                    target_name = data["target"].replace(" ", "_") if data["target"] else "unknown_target"
                    
                    concatvis = os.path.join(self.data_dir, f"{target_name}/{target_name}_{str(data['dish_sizes'][0]).split('.')[0]}m_{str(data['median_frequency']).split('.')[0]}GHz_tbin30s_concatted.ms")

                    # Concatenate the transformed files
                    if not os.path.exists(os.path.dirname(concatvis)):
                        os.makedirs(os.path.dirname(concatvis))

                    if os.path.exists(concatvis):
                        print(f"Warning: ms = {concatvis} already existed, skipping...")
                        data["concatvis"]   = concatvis
                        continue

                    i = 0
                    for member, ms_files in data["files"].items():
                        print(f"  Member: {member}")
                        for ms_file in ms_files:
                            print(f"  - {ms_file}")
                            vis = self._get_vis_path(ms_file, project_code=project_code, science_goal=science_goal, group=group, member=member)
                            outputvis = os.path.join(self.data_dir, f"{target_name}_{str(data['dish_sizes'][0]).split('.')[0]}m_{str(data['median_frequency']).split('.')[0]}GHz_uid{i}_tbin30s.ms")

                            if not os.path.exists(vis):  
                                print(f"Warning: ms = {vis} already existed, skipping...")
                                continue
                            try:
                                spw = au.getScienceSpws(vis, intent='OBSERVE_TARGET#ON_SOURCE', returnString=True)
                            except Exception as e:
                                print(f"Error retrieving SPW for {vis}: {e}")
                                continue
                                                        
                            if not os.path.exists(outputvis):
                                mstransform(vis=vis,
                                        outputvis=outputvis,
                                        spw=spw,
                                        field=f'"{target_name}"',
                                        intent="*OBSERVE_TARGET*",
                                        timeaverage=True, timebin='30s',
                                        datacolumn="data")

                            vislist.append(outputvis)
                            i+=1
                    
                    if vislist:
                        concat(concatvis=concatvis, vis=vislist, freqtol="20MHz")     
                        statwt(concatvis, datacolumn='data')
                        listobs(vis=concatvis, listfile=concatvis.replace('.ms', '.ms.listobs'))
                    else:
                        print(f"Warning: No transformed visibilities found for {target_name}, skipping concatenation.")

                    data["concatvis"] = concatvis
                    
                    # Remove the temporaryor split files
                    for vis_file in vislist:
                        if os.path.exists(vis_file):
                            shutil.rmtree(vis_file)

            self.concatted = True

        self._get_white_noise_sensitivity()


    def _get_white_noise_sensitivity(self):
        """Computes and stores the white noise sensitivity for each group UID."""
        
        # Ensure data is concatenated before computing white noise sensitivity
        if not self.concatted:
            self.mstransform_and_concat()

        for project_code, science_goals in self.projects.items():
            for science_goal, groups in science_goals.items():
                for group, data in groups.items():
                    concatvis = data.get("concatvis", None)
                    if concatvis is None or not os.path.exists(concatvis):
                        print(f"Warning: No concatenated data found for {group}, skipping sensitivity calculation.")
                        continue
                    try:
                        savename=concatvis.replace('data','output')
                        savename=savename.replace('.ms','.npy')
                        
                        sensitivity = oau.getWeightDistribution(concatvis,savename=savename)
                        data["white_noise_sensitivity"] = sensitivity*1e6
                    except Exception as e:
                        print(f"Error computing white noise sensitivity for {concatvis}: {e}")

        if self.plotuvdist:
            for target in self._get_all_targets():
                oau.plotWeightDistribution(target, f'../../plots/{target}_baselinedist.pdf')

    def display_structure(self):
        """Print the structured project data with target names and total on-source time per group."""
        for project_code, science_goals in self.projects.items():
            print(f"\nProject Code: {project_code}")
            for science_goal, groups in science_goals.items():
                print(f"  Science Goal: {science_goal}")
                for group, data in groups.items():
                    print(f"    Group: {group}")
                    print(f"      Target: {data['target'] if data['target'] else 'Unknown'}")
                    
                    # Handle None values safely
                    on_source_time = f"{data['on_source_time']:.2f} min" if data["on_source_time"] is not None else "N/A"
                    dish_sizes = f"{data['dish_sizes']} m" if data["dish_sizes"] is not None else "N/A"
                    median_frequency = f"{data['median_frequency']:.1f} GHz" if data["median_frequency"] is not None else "N/A"
                    mrs = f"{data['MRS']:.1f} arcsec" if data["MRS"] is not None else "N/A"
                    min_resolution = f"{data['min_resolution']:.1f} arcsec" if data["min_resolution"] is not None else "N/A"
                    
                    print(f"      Total On-Source Time: {on_source_time}")
                    print(f"      Dish size: {dish_sizes}")
                    print(f"      Median Frequency: {median_frequency}")
                    print(f"      Maximum Recoverable Scale: {mrs}")
                    print(f"      Minimum Resolution: {min_resolution}")
                    if self.concatted: 
                        print(f"      White Noise Sensitivity: {data['white_noise_sensitivity']:.3f} mJy" if data['white_noise_sensitivity'] is not None else "N/A")
                        print(f"      Data File: {data['concatvis']}")
                        
                    else:
                        for member, ms_files in data["files"].items():
                            print(f"      Member: {member}")
                            for ms_file in ms_files:
                                print(f"        - {ms_file}")
            print()
    
    def export_to_csv(self, filename="../../output/project_data.csv"):
        """Exports group-level project data to CSV and LaTeX files."""

        output_dir = os.path.dirname(filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        columns = ["Target", "Project code", "Dish size [m]", "Time on source [min]",
                   "Frequency [GHz]", r"White noise sensitivity [$\mu$Jy]", "Min & Max recoverable scale [arcsec]"]

        data_list = []
        for project_code, science_goals in self.projects.items():
            for science_goal, groups in science_goals.items():
                for group, data in groups.items():
                    target = data['target'] if data['target'] else 'Unknown'
                    dish_size = ', '.join(map(str, data['dish_sizes'])) if data['dish_sizes'] else 'N/A'
                    total_time = f"{data['on_source_time']:.2f}" if data['on_source_time'] is not None else 'N/A'
                    median_freq = f"{data['median_frequency']:.1f}" if data['median_frequency'] is not None else 'N/A'
                    white_noise = f"{data['white_noise_sensitivity']:.2f}" if data['white_noise_sensitivity'] is not None else 'N/A'
                    min_res = f"{data['min_resolution']:.1f}" if data['min_resolution'] is not None else 'N/A'
                    max_res = f"{data['MRS']:.1f}" if data['MRS'] is not None else 'N/A'
                    res_column = f"{min_res} / {max_res}"
                    data_list.append([target, project_code, dish_size, total_time, median_freq, white_noise, res_column])

        data_list.sort(key=lambda x: x[0])

        df = pd.DataFrame(data_list, columns=columns)
        df.to_csv(filename, index=False)
        print(f"Data successfully exported to {filename}")

        tex_filename = filename.replace(".csv", ".tex")
        self._write_latex_table(data_list, tex_filename)
        print(f"Data successfully exported to {tex_filename}")

    @staticmethod
    def _write_latex_table(rows, tex_filename):
        """Write a publication-ready booktabs LaTeX table with a two-line header.

        - Underscores in target names become '~' (non-breaking space), so
          'ACT-CL_J0001.9+0112' renders as 'ACT-CL J0001.9+0112'.
        - Header is split over two rows: column name, then unit.
        Requires \\usepackage{booktabs} in the document preamble.
        """
        # (name, unit) per column, in the same order as the data rows
        header = [
            ("Target",                 ""),
            ("Project",                "code"),
            ("Dish size",              "[m]"),
            ("Time on source",         "[min]"),
            ("Frequency",              "[GHz]"),
            ("White-noise sens.",      r"[$\mu$Jy]"),
            ("Recoverable scale",      "min / max [arcsec]"),
        ]
        names = " & ".join(n for n, _ in header) + r" \\"
        units = " & ".join(u for _, u in header) + r" \\"
        colspec = "l" + "c" * (len(header) - 1)

        lines = [
            "% Requires \\usepackage{booktabs}",
            r"\begin{table}",
            r"\centering",
            r"\caption{ALMA observation summary.}",
            r"\label{tab:project_data}",
            r"\begin{tabular}{" + colspec + "}",
            r"\toprule",
            names,
            units,
            r"\midrule",
        ]
        for row in rows:
            cells = [str(row[0]).replace("_", "~")] + [str(c) for c in row[1:]]
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

        with open(tex_filename, mode="w") as f:
            f.write("\n".join(lines))

    def export(self, output_root: str = "../../output/vis") -> None:
        """Instantiate and run Export for every group; attaches instance to data["export"]."""
        if not self.concatted:
            print("Warning: Data not concatenated yet; run mstransform_and_concat() first.")

        for project_code, science_goals in self.projects.items():
            for science_goal, groups in science_goals.items():
                for group_uid, data in groups.items():
                    data["export"] = Export.from_group_record(data, output_root=output_root)
                    print(data["export"])
                    data["export"].run_all()

    def export_linesubtracted(self, output_root: str = "../../output/vis_linesub") -> None:
        """Run Export on the *_linesubtracted.ms produced by export_cube().

        Discovers, for each group, the line-removed MS derived from its
        concatvis (``<concatvis>`` -> ``*_linesubtracted.ms``) and runs the
        continuum Export pipeline on it. Skips groups whose line-removed MS
        does not exist yet.
        """
        for data in self._iter_groups():
            cv = data.get("concatvis")
            if not cv:
                continue
            linesub = cv.replace(".ms", "_linesubtracted.ms")
            if not os.path.exists(linesub):
                print(f"Warning: {linesub} not found; run export_cube() first. Skipping.")
                continue

            dish_sizes  = data.get("dish_sizes")
            target      = data.get("target")
            target_safe = target.replace(" ", "_") if target else "unknown"
            exp = Export(
                target=target,
                dish_size_m=int(min(dish_sizes)) if dish_sizes else None,
                median_freq_ghz=data.get("median_frequency"),
                concatvis=linesub,
                output_dir=str(Path(output_root) / target_safe),
            )
            data["export_linesub"] = exp
            print(exp)
            exp.run_all()

    def _iter_groups(self):
        """Yield every group data dict across all projects/science_goals/groups."""
        for sgs in self.projects.values():
            for groups in sgs.values():
                yield from groups.values()

    def export_cube(
        self,
        output_root: str = "../../output",
        line_name: str = None,         
        line_freq_hz: float = None,    
        redshift: float = 0,
        config: CubeExportConfig = None,
        do_uvcontsub: bool = True,
        overwrite: bool = False,
        imagename: str = None,
        combine_arrays: bool = False,
        bad_channel_sigma: float = 1.5,
        detection_sigma: float = 5.5,
        do_linesub: bool = True,
        validate_linesub: bool = True,
        linesub_overwrite: bool = None,
    ) -> None:

        """Run the spectral cube pipeline for every group (or jointly per target if combine_arrays=True)."""

        if not self.concatted:
            print("Warning: Data not concatenated yet; run mstransform_and_concat() first.")

        cfg = config or CubeExportConfig()
        if line_name is not None:
            obs_freq_str, rest_freq_hz = parse_line_freq(line_name, redshift)
            cfg.restfreq = obs_freq_str
            if line_freq_hz is None:
                line_freq_hz = rest_freq_hz

        run_kw = dict(line_freq_hz=line_freq_hz, do_linesub=do_linesub,
                      validate_linesub=validate_linesub,
                      linesub_overwrite=linesub_overwrite)

        if combine_arrays:
            target_data = defaultdict(lambda: {"vis_list": [], "dish_sizes": [], "groups": []})
            for data in self._iter_groups():
                cv = data.get("concatvis")
                t  = data.get("target", "unknown")
                if cv and os.path.exists(cv):
                    target_data[t]["vis_list"].append(cv)
                    target_data[t]["dish_sizes"].extend(data.get("dish_sizes") or [])
                    target_data[t]["groups"].append(data)

            for target, td in target_data.items():
                target_safe  = target.replace(" ", "_")
                dish_sizes   = sorted(set(int(d) for d in td["dish_sizes"]))
                sizes_str    = "+".join(f"{d}m" for d in dish_sizes)
                derived_name = imagename or f"{target_safe}_{sizes_str}_cube"
                ec = ExportCube(
                    concatvis=td["vis_list"],
                    output_dir=str(Path(output_root) / "cube" / target_safe),
                    config=cfg, target=target, overwrite=overwrite,
                )
                for data in td["groups"]:
                    data["cube_export"] = ec
                ec.run_all(do_uvcontsub, bad_channel_sigma, detection_sigma, imagename=derived_name, **run_kw)

        else:
            for data in self._iter_groups():
                concatvis = data.get("concatvis")
                if not concatvis or not os.path.exists(concatvis):
                    continue
                target       = data.get("target")
                target_safe  = target.replace(" ", "_") if target else "unknown"
                derived_name = imagename or Path(concatvis).stem.replace("_concatted", "_cube")
                ec = ExportCube(
                    concatvis=concatvis,
                    output_dir=str(Path(output_root) / "cube" / target_safe),
                    config=cfg, target=target, overwrite=overwrite,
                )
                data["cube_export"] = ec
                ec.run_all(do_uvcontsub, bad_channel_sigma, detection_sigma, imagename=derived_name, **run_kw)


