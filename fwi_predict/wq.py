"""Utilities for water quality assessment."""
import numpy as np
import pandas as pd

from .constants import WQ_RANGES


def check_in_range(parameter: str, values: pd.Series, periods: pd.Series) -> pd.Series:
    """
    in_range if water quality parameter is within the required range.

    Parameters:
        parameter (str): The water quality parameter to check (e.g., 'do', 'ph', 'ammonia', 'turbidity').
        values (pd.Series): A series of measurement values.
        periods (pd.Series): A series of periods ('morning', 'evening', etc.).

    Returns:
        pd.Series: A boolean series indicating if each measurement is within the required range.
    """
    # Ensure the parameter is valid
    if parameter not in WQ_RANGES:
        raise ValueError(f"Invalid parameter: {parameter}. Must be one of {list(WQ_RANGES.keys())}.")
    
    # Extract the required ranges for the parameter
    required_ranges = WQ_RANGES[parameter]['required']
    
    # Handle cases where ranges are split by periods
    if isinstance(required_ranges, dict):  # Period-dependent ranges
        valid_periods = list(required_ranges.keys()) + [np.nan]
        if not periods.isin(valid_periods).all():
            invalid_periods = periods[~periods.isin(valid_periods)].unique()
            raise ValueError(f"Invalid period(s): {invalid_periods}. Must be one of {valid_periods}.")
        
        # Check in range for each period
        in_range = pd.Series(False, index=values.index)  # Initialize all False
        for period, (low, high) in required_ranges.items():
            in_range |= (periods == period) & values.between(low, high)
        in_range[values.isna() | periods.isna()] = np.nan
        return in_range
    
    # Handle cases where ranges are not split by periods
    low, high = required_ranges
    in_range = values.between(low, high)
    in_range[values.isna()] = np.nan

    return in_range