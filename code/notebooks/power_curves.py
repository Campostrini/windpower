import pandas as pd
import xarray as xr
import numpy as np
import GenericWindTurbinePowerCurve as GWTPC

def power_curve_from_csv(path, xs_value, xs_level):
    df_power_curves = pd.read_excel(path, sheet_name="Power_curves", index_col=[0, 1, 2, 3])
    df_power_curves = df_power_curves.transpose()
    df_power_curves = df_power_curves[:-2]
    df_power_curves.index = df_power_curves.iloc[:, 0]
    df_power_curves.index.name = "Wind speed (m/s)"
    df_power_curves = df_power_curves.iloc[:, 1:]
    column_names = df_power_curves.columns.names
    df_power_curves.columns = pd.MultiIndex.from_tuples(
        [(int(x[0]), x[1], int(x[2]), x[3]) for x in df_power_curves.columns.to_list()],
        names=column_names
    )
    df_power_curves = df_power_curves.xs(xs_value, level=xs_level, axis=1)

    def power_conversion_function(wind_speed):
        out_da = xr.zeros_like(wind_speed)
        out_da.name = "power"
        out_da.attrs["units"] = "kW"
        out_da.attrs["long_name"] = "Power"
        for t in wind_speed.time.values:
            nearest_wind_speed = df_power_curves.index[np.abs(df_power_curves.index - wind_speed.sel(time=t).values).values.astype(np.float32).argmin()]
            out_da.loc[dict(time=t)] = df_power_curves.loc[nearest_wind_speed].iloc[0]
        return out_da

    return power_conversion_function

def power_curve_from_parameters(Pnom, Drotor, wind_speed_resolution=0.5, **kwargs):
    wind_speeds = np.arange(0, 30 + wind_speed_resolution, wind_speed_resolution)
    power_output = GWTPC.GenericWindTurbinePowerCurve(
        wind_speeds,
        Pnom,
        Drotor,
        **kwargs)
    
    def power_conversion_function(wind_speed):
        out_da = xr.zeros_like(wind_speed)
        out_da.name = "power"
        out_da.attrs["units"] = "kW"
        out_da.attrs["long_name"] = "Power"
        for t in wind_speed.time.values:
            argmin = np.abs(wind_speeds - wind_speed.sel(time=t).values).astype(np.float32).argmin()
            out_da.loc[dict(time=t)] = power_output[argmin]
        return out_da
    
    return power_conversion_function
