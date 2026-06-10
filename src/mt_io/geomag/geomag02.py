
from email.mime import text
import json
import warnings
from io import StringIO

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from typing import Any

import numpy as np

# supress the future warning from pandas about using datetime parser.
warnings.simplefilter(action="ignore", category=FutureWarning)


import pandas as pd
from loguru import logger
from mt_metadata.common.mttime import MTime
from mt_metadata.timeseries import Auxiliary, Electric, Magnetic, Run, Station
from mt_metadata.timeseries.filters import ChannelResponse, FrequencyResponseTableFilter
from mt_timeseries import ChannelTS, RunTS


def geomag_date_parser(
    year: int, month: int, day: int, hour: int, minute: int, second: float
) -> pd.Series:
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

    def __init__(self, fn: str | Path | None = None, **kwargs: Any) -> None:
        self.logger = logger
        self.fn = fn
        self.sample_rate = 0.1
        self.data=None
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
            "temperature_instrument"
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
            ("temperature_instrument", float)
            ]
        )
        self.data_column_names = ["date"] + self.file_column_names[6:]

    @property
    def fn(self) -> Path | None:
        """
        Full path to LEMI424 file.

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
            Path to the GEOMAG file.

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
    
    def read(self, fn: str | Path | None = None) -> None:
        st = MTime(time_stamp=None).now()
        if fn is not None:
            self.fn = fn
        if not self.fn.exists():
            msg = f"Could not find file {self.fn}"
            self.logger.error(msg)
            raise IOError(msg)
        self.read_metadata()
        self.data=pd.read_csv(
            self.fn,
            delimiter=r"\s+",
            names=self.file_column_names,
            dtype=self.dtypes,
            skiprows=8)

        if self.data.empty:
                raise ValueError("File is empty or contains no valid data")

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
        et = MTime(time_stamp=None).now()
        self.logger.debug(f"Reading {self.fn.name} took {et - st:.2f} seconds")


    def read_metadata(self) -> None:
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
                print("No elevation provided, setting to 0 m")
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
                "Te": "temperature_instrument"
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
            f.readline()
            f.readline()
            self.sample_rate=float(f.readline().strip().split(' ')[-2])
            line4=f.readline().strip()
            self.lat, self.lon, self.elev = _get_lat_lon_elev(line4)
            line5=f.readline().strip()
            self.x_total, self.y_total, self.z_total=_get_total_fields(line5)
            f.readline()
            line7=f.readline().strip()
            self.units=_get_units(line7)
    
    def to_run_ts(self) -> RunTS:
        ch_list=[]
        for comp in self.units.keys():
            if comp[0]=="b":
                ch=ChannelTS("magnetic")
                ch.channel_metadata.units=self.units[comp]
            elif comp[0]=="e":
                ch=ChannelTS("electric")
                ch.channel_metadata.units=self.units[comp]
            else:
                ch=ChannelTS("auxilary")
                ch.channel_metadata.units=self.units[comp]
            ch.sample_rate=self.sample_rate
            #ch.start=self.start
            ch.ts=self.data[comp].values
            ch.componnent=comp
            ch_list.append(ch)
        return RunTS(
            array_list=ch_list
        )

def read_geomag02(fn: str | Path | list[str | Path]) -> RunTS:
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