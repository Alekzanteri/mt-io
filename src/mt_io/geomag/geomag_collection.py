from mt_io.collection import Collection

import pathlib
from pathlib import Path
from typing import List

import pandas as pd

from .geomag02 import geomag02

class GeomagCollection(Collection):
    """
    Collection of geomag files into runs based on start and end times.

    This is based on mt_io.lemi.LEMICollection with some changes to accomodate geomag legacy files.

    Intended use case of this class is to read a folder containing files from SINGLE geomag station.

    Parameters
    ----------
    file_path : str or pathlib.Path, optional
        Full path to single station geomag directory, by default None
    file_ext : list of str, optional
        Extensions of geomag files, by default ["txt", "TXT"]
    **kwargs
        Additional keyword arguments passed to parent Collection class

    Attributes
    ----------
    station_id : str
        Station identification string, defaults to "geomag_site_001"
    survey_id : str
        Survey identification string, defaults to "mt"
    .. note:: geomag data comes with little metadata about the station or survey,
     therefore you should assign `station_id` and `survey_id`.
    """

    def __init__(
            self,
            file_path: str | pathlib.Path | None =None,
            file_ext: List[str]| None=None,
            **kwargs
    ) -> None:
        if file_ext is None:
            file_ext=['txt', 'TXT']
        super().__init__(file_path=file_path, file_ext=file_ext, **kwargs)

        self.station_id="geomag_site_001"
        self.survey_id="mt"

    def to_dataframe(
            self,
            sample_rates: int | List[int] | None = None,
            run_name_zeros: int=4,
            calibration_path: str | Path | None = None
    ) -> pd.DataFrame:
        """
        Create a data frame of the files in the given directory.
        Always uses geomag02 reader.
        Calibration path does nothing, but is included to be consistent
        with other instrument spesific collections.
        -----
        This assumes the given directory contains a single station

        Parameters
        ----------
        sample_rates : int or list of int, optional
            defaults to 10 Hz
        run_name_zeros : int, optional
            Number of zeros to assign to the run name, by default 4
        calibration_path : str or pathlib.Path, optional
            Path to calibration files, by default None

        Returns
        -------
        pd.DataFrame
            DataFrame with information of each TXT file in the given directory
        """
        if calibration_path is not None:
            self.logger.warning('Geomag reader is not able to use calibration files!')

        if sample_rates is None:
            sample_rates=[10]
        
        entries=[]

        for fn in self.get_files(self.file_ext):
            fn_path=pathlib.Path(fn)

            if fn_path.suffix in [".txt", ".TXT"]:
                geomag_obj=geomag02(fn)
                geomag_obj.read_metadata()
                instrument_id="geomag02"
                n_samples=geomag_obj.n_samples
                sample_rate=geomag_obj.sample_rate
                file_size=geomag_obj.file_size
                start=geomag_obj.run_time['start'].isoformat()
                end=geomag_obj.run_time['end'].isoformat()
                components=",".join(geomag_obj.run_metadata.channels_recorded_all)
            
            else:
                self.logger.warning(f"Unknown file extension for {fn}, skipping")
                continue
        
            if sample_rate not in sample_rates:
                continue

            entry=self.get_empty_entry_dict()
            entry = self.get_empty_entry_dict()
            entry["survey"] = self.survey_id
            entry["station"] = self.station_id
            entry["start"] = start
            entry["end"] = end
            entry["component"] = components
            entry["fn"] = fn
            entry["sample_rate"] = sample_rate
            entry["file_size"] = file_size
            entry["n_samples"] = n_samples
            entry["instrument_id"] = instrument_id

            entries.append(entry)

        if len(entries)==0:
            self.logger.warning('No entries found for geomag collection.')
            return pd.DataFrame(columns=self._columns)
        
        df=pd.DataFrame(entries)
        df.loc[:, "channel_id"] = 1
        df.loc[:, "sequence_number"] = 0

        df = self._sort_df(self._set_df_dtypes(df), run_name_zeros)
        return df
    
    def assign_run_names(self, df: pd.DataFrame, zeros: int = 4) -> pd.DataFrame:
        """
        This is same function than LemiCollection.assign_run_names.
        Run names are assigned as sr{sample_rate}_{run_number:0{zeros}}.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the appropriate columns
        zeros : int, optional
            Number of zeros in run name, by default 4

        Returns
        -------
        pd.DataFrame
            DataFrame with run names assigned

        """
        run_counts = {}
        previous_ends = {}

        for row in df.itertuples():
            sr = int(row.sample_rate)

            # Initialize for this sample rate
            if sr not in run_counts:
                run_counts[sr] = 1
                previous_ends[sr] = None

            # Check if new run (time gap detected)
            if previous_ends[sr] is not None:
                gap = (row.start - previous_ends[sr]).total_seconds()
                if gap > 1.0 / sr:  # Gap > 1 sample period = new run
                    run_counts[sr] += 1

            # Assign run name
            df.loc[row.Index, "run"] = f"sr{sr}_{run_counts[sr]:0{zeros}}"
            previous_ends[sr] = row.end

        return df

    def _set_df_dtypes(self, df):
        """
        Parent class function is overwritten to make the code working with
        mix of full second and not full second str representation of timestamp
        coming from MTTime.
        """
        df.start = pd.to_datetime(df.start, format="mixed")
        df.end = pd.to_datetime(df.end, format="mixed")
        df.instrument_id = df.instrument_id.astype(str)
        df.calibration_fn = df.calibration_fn.astype(str)
        return(df)
