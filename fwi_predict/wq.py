"""Utilities for water quality assessment."""
from typing import List, Tuple

import numpy as np
import pandas as pd

from .constants import WQ_RANGES


def get_in_required_range(parameter: str, values: pd.Series, periods: pd.Series) -> pd.Series:
    """Checks if water quality parameter is below, within, or above the required range.

    Parameters:
        parameter (str): The water quality parameter to check (e.g., 'do', 'ph', 'ammonia', 'turbidity').
        values (pd.Series): A series of measurement values.
        periods (pd.Series): A series of periods ('morning', 'evening', etc.).

    Returns:
        pd.Series: 
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

        # Get in range for each period
        in_range = pd.Series([''] * len(values.index), index=values.index)  # Initialize all False
        for period, (low, high) in required_ranges.items():
            period_idx = periods == period
            period_values = values[period_idx]
            
            conditions = [period_values < low, period_values.between(low, high), period_values > high]
            choices = ['below', 'within', 'above']
            in_range[period_idx] = np.select(conditions, choices, default='')

        return in_range
    
    # Handle cases where ranges are not split by periods
    low, high = required_ranges
    conditions = [values < low, values.between(low, high), values > high]
    choices = ['below', 'within', 'above']
    in_range = np.select(conditions, choices, default='')

    return in_range