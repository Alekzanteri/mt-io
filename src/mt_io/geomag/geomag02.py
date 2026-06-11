import warnings
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from typing import Any
# supress the future warning from pandas about using datetime parser.
warnings.simplefilter(action="ignore", category=FutureWarning)


import pandas as pd
from loguru import logger
from mt_metadata.common.mttime import MTime
from mt_metadata.timeseries import Auxiliary, Electric, Magnetic, Run, Station
from mt_timeseries import ChannelTS, RunTS


def geomag_date_parser(
    year: int, month: int, day: int, hour: int, minute: int, second: float
) -> pd.Series:
    """
    This function combines geomag time output into a single column.

    Parameters
    ----------
    year : int
        Year value.
    month : int
        Month value (1-12).
    day : int
        Day of the month (1-31).
    hour : int
        Hour in 24-hour format (0-23).
    minute : int
        Minutes in the hour (0-59).
    second : float
        Seconds in the minute (0-59).

    Returns
    -------
    pd.DatetimeIndex
        Combined date-time as a pandas DatetimeIndex.
    """
    dt_df = pd.DataFrame(
        {
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second,
        }
    )
    for key in ["year", "month", "day", "hour", "minute", "second"]:
        dt_df[key] = dt_df[key].astype(float)

    return pd.to_datetime(dt_df)

class geomag02:
    """
    Read geomag02 instrument data file. This is based on LEMI424 reader.

    Parameters
    ----------
    fn : str or pathlib.Path, optional
        Full path to geomag file, by default None.
    **kwargs : dict
        Additional keyword arguments for configuration.

    Attributes
    ----------
    sample_rate : float
        Sample rate of the file in samples per second, default is 10.
    file_column_names : list of str
        Column names of the geomag02 file.
    dtypes : dict
        Data types for each column.
    data_column_names : list of str
        Same as file_column_names with an added column for date.
    data : pd.DataFrame or None
        The loaded data.

    Note
    -----
    This class is only tested with limited amount of data files.
    This is not tested with data file containing less than 5 channels.
    Calibrations are not included yet, but that can be added later if found necessary.
    Aleksanteri Harjulehto 
    """

    def __init__(self, fn: str | Path | None = None, **kwargs: Any) -> None:
        self.logger = logger
        self.fn = fn
        self.sample_rate = 10
        self.data=None
        self.lat=0
        self.lon=0
        self.elev=0
        self.data_logger='No info provided'
        self.file_column_names = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "hx",
            "hy",
            "hz",
            "ex",
            "ey",
            "temperature_fluxgate",
            "temperature_logger"
        ]
        self.dtypes=dict(
            [
            ("year", int),
            ("month", int),
            ("day", int),
            ("hour", int),
            ("minute", int),
            ("second", float),
            ("hx", float),
            ("hy", float),
            ("hz", float),
            ("ex", float),
            ("ey", float),
            ("temperature_fluxgate", float),
            ("temperature_logger", float)
            ]
        )
        self.data_column_names = ["date"] + self.file_column_names[6:]
        self._has_data=False

    def __add__(self, other: "geomag02 | pd.DataFrame") -> "geomag02":
        """
        Combine multiple geomag02 objects today.

        Matching run times, site locations, instrumentations etc. are not checked,
        so the user must be careful when using this.

        Parameters
        ----------
        other : geomag02 or pd.DataFrame
            Object to append to this geomag02 instance.

        Returns
        -------
        geomag02
            New geomag02 object with combined data.

        Raises
        ------
        ValueError
            If data is None or if DataFrame columns don't match.

        """
        if not self._has_data:
            raise ValueError("Data is None cannot append to. Read file first")
        if isinstance(other, geomag02):
            new=geomag02()
            new.__dict__.update(self.__dict__)
            new.data=pd.concat([new.data, other.data])
            return new
        elif isinstance(other, pd.DataFrame):
            if not other.columns.equals(self.data.columns):
                raise ValueError("DataFrame columns are not the same.")
            new = geomag02()
            new.__dict__.update(self.__dict__)
            new.data = pd.concat([new.data, other])
            return new
        else:
            raise ValueError(f"Cannot add {type(other)} to pd.DataFrame.")

    @property
    def fn(self) -> Path | None:
        """
        Full path to geomag02 file.

        Returns
        -------
        pathlib.Path or None
            Path to the file or None if not set.

        """
        return self._fn

    @fn.setter
    def fn(self, value: str | Path | None) -> None:
        """
        Set the file path.

        Parameters
        ----------
        value : str, pathlib.Path, or None
            Path to the geomag file.

        Raises
        ------
        IOError
            If the file does not exist.

        """
        if value is not None:
            value = Path(value)
            if not value.exists():
                raise IOError(f"Could not find {value}")
        self._fn = value
    
    @property
    def run_time(self) -> dict:
        """
        The start and end times are saved to a dictionary.

        Returns
        -------
        dict
            The start time is lableled as "start" and end as "end".

        Raises
        ------
        ValueError
            If self.data is empty. 
        """
        runtime={}
        if self._has_data:
            runtime['start']=MTime(time_stamp=self.data.index[0])
            runtime['end']=MTime(time_stamp=self.data.index[-1])
        else:
            raise ValueError('No data found, so run time can not be determined.')
        return runtime

    @property
    def station_metadata(self) -> Station:
        """
        Station metadata as mt_metadata.timeseries.Station object.

        Returns
        -------
        mt_metadata.timeseries.Station
            Station metadata object.
        """
        s=Station()
        if self._has_data:
            s.location.latitude=self.lat
            s.location.longitude=self.lon
            s.location.elevation=self.elev
            s.time_period.start=self.run_time['start']
            s.time_period.end=self.run_time['end']
            s.add_run(self.run_metadata)
        return s

    @property
    def run_metadata(self) -> Run:
        """
        Run metadata as mt_metadata.timeseries.Run object.

        Returns
        -------
        mt_metadata.timeseries.Run
            Run metadata object.

        """
        r=Run()
        r.id="a"
        r.sample_rate=self.sample_rate
        r.data_logger.model=self.data_logger
        r.data_logger.manufacturer="GEOMAG"
        if self._has_data:
            r.time_period.start=self.run_time['start']
            r.time_period.end=self.run_time['end']

            for ch_aux in ["temperature_fluxgate", "temperature_logger"]:
                r.add_channel(Auxiliary(component=ch_aux))
            for ch_e in ["ex", "ey"]:
                r.add_channel(Electric(component=ch_e))
            for ch_h in ["hx", "hy", "hz"]:
                r.add_channel(Magnetic(component=ch_h))
        return r



    
    def read(self, fn: str | Path | None = None) -> None:
        """
        Read a geomag02 file to pandas.

        Parameters
        ----------
         fn : str, pathlib.Path, or None, optional
            Full path to file. Uses self.fn if not provided, by default None.
        
        Raises
        ------
        IOError
            If file cannot be found.
        ValueError:
            if data file is empty.    

        """
        st = MTime(time_stamp=None).now()
        if fn is not None:
            self.fn = fn
        if not self.fn.exists():
            msg = f"Could not find file {self.fn}"
            self.logger.error(msg)
            raise IOError(msg)
        self.read_metadata()

        # reading in chunks is not implemented.
        # By default geomag saves 1 file per day leading to reasonable file sizes.
        self.data=pd.read_csv(
            self.fn,
            delimiter=r"\s+",
            names=self.file_column_names,
            dtype=self.dtypes,
            skiprows=8)

        if self.data.empty:
                raise ValueError("File is empty or contains no valid data")
        else:
            self._has_data=True

        self.data.index=geomag_date_parser(
            self.data["year"],
            self.data["month"],
            self.data["day"],
            self.data["hour"],
            self.data["minute"],
            self.data['second']
        )
        self.data.index.name = "date"
        self.data = self.data.drop(
            columns=["year", "month", "day", "hour", "minute", "second"]
        )

        self.data['hx']=self.x_total+self.data['hx'].cumsum()
        self.data['hy']=self.y_total+self.data['hy'].cumsum()
        self.data['hz']=self.z_total+self.data['hz'].cumsum()

        et = MTime(time_stamp=None).now()
        self.logger.debug(f"Reading {self.fn.name} took {et - st:.2f} seconds")


    def read_metadata(self) -> None:
        """
        Read metadata stored on the first 9 lines of the datafile.
        """
        def _get_lat_lon_elev(line):
            parts = line.split(';')
            lat_str = parts[1].split(':')[1].strip()
            lat=_deg_min_sec_to_deg(lat_str)
            lon_str = parts[2].split(':')[1].strip()
            lon=_deg_min_sec_to_deg(lon_str)
            elevation_str = parts[3].split(':')[1].strip()
            elevation=_validate_elevation(elevation_str)
            return lat, lon, elevation

        def _deg_min_sec_to_deg(org_str):
            if org_str.endswith('N') or org_str.endswith('E'):
                hem=1
            elif org_str.endswith('S') or org_str.endswith('W'):
                hem=-1
            else:
                raise ValueError(f"Invalid hemisphere in {org_str}")
            deg=float(org_str.split(' ')[0])
            minsec=org_str.split(' ')[1]
            minu=float(minsec.split("'")[0])
            sec=float(minsec.split("'")[1].split('"')[0])
            deg+=minu/60+sec/3600
            deg*=hem
            return deg

        def _validate_elevation(elev_str):
            elev_str=elev_str.replace("-", "")
            if len(elev_str)==0:
                self.logger.warning("No elevation provided, setting to 0 m")
                elev=0
            else:
                elev=float(elev_str[:-1])
            return(elev)

        def _get_total_fields(field_str):
            parts = field_str.split(';')
            x = float(parts[1].split('=')[1].replace('nT', '').strip())
            y = float(parts[2].split('=')[1].replace('nT', '').strip())
            z = float(parts[3].split('=')[1].replace('nT', '').strip())
            return x, y, z
        
        def _get_units(unit_str):
            name_changer={
                "X":"hx",
                "Y":"hy",
                "Z":"hz",
                "Ex":"ex",
                "Ey": "ey",
                "Ts": "temperature_fluxgate",
                "Te": "temperature_logger"
            }
            parts = unit_str.split(' ')
            units = {}
            i=0
            for part in parts:
                if '[' in part and ']' in part:
                    name = part.split('[')[0]
                    if name=='':
                        name=parts[i-1]
                    unit = part.split('[')[1].replace(']', '')
                    units[name_changer[name]] = unit
                i+=1
            return(units)

        with open(self.fn, "r") as f:
            self.data_logger=f.readline().replace(";", "").strip()
            next(f)
            self.sample_rate=float(f.readline().strip().split(' ')[-2])**(-1)
            line4=f.readline().strip()
            self.lat, self.lon, self.elev = _get_lat_lon_elev(line4)
            line5=f.readline().strip()
            self.x_total, self.y_total, self.z_total=_get_total_fields(line5)
            next(f)
            line7=f.readline().strip()
            self.units=_get_units(line7)
    
    def to_run_ts(self) -> RunTS:
        """
        Create a RunTS object from the data.

        Returns
        -------
        mth5.timeseries.RunTS
            RunTS object containing the data
        """
        ch_list=[]
        for comp in self.units.keys():
            if comp[0]=="h":
                ch=ChannelTS("magnetic")
                ch.channel_metadata.units=self.units[comp]
            elif comp[0]=="e":
                ch=ChannelTS("electric")
                ch.channel_metadata.units=self.units[comp]
            else:
                ch=ChannelTS("auxilary")
                ch.channel_metadata.units=self.units[comp]
            ch.sample_rate=self.sample_rate
            ch.start=self.run_time['start']
            ch.ts=self.data[comp].values
            ch.component=comp
            ch_list.append(ch)
        return RunTS(
            array_list=ch_list,
            station_metadata=self.station_metadata,
            run_metadata=self.run_metadata
        )

def read_geomag02(fn: str | Path | list[str | Path]) -> RunTS:
    """
    Read geomag02.txt file.

    Parameters
    ----------
    fn : str or pathlib.Path or list of either
        Input file or files

    Returns
    -------
    mth5.timeseries.RunTS
        A RunTS object with appropriate metadata.
    """
    if not isinstance(fn, (list, tuple)):
        fn = [fn]
    txt_obj = geomag02(fn[0])
    txt_obj.read()

    if len(fn)>1:
        for txt_file in fn[1:]:
            other = geomag02(txt_file)
            other.read()
            txt_obj+=other
    return txt_obj.to_run_ts()