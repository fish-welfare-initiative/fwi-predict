"""Utilities for water quality assessment."""
import numpy as np
import pandas as pd

from .constants import WQ_RANGES


def get_in_required_range(parameter: str, values, periods=None):
    """Checks if water quality parameter is below, within, or above the required range.

    Parameters:
        parameter (str): The water quality parameter to check (e.g., 'do', 'ph', 'ammonia', 'turbidity').
        values: Array-like of measurement values (numpy array or pandas Series).
        periods: Array-like of periods ('morning', 'evening', etc.), required for period-dependent parameters.

    Returns:
        Array-like: Array of strings indicating if values are 'below', 'within', or 'above' the required range.
    """
    # Convert inputs to numpy arrays for consistent handling
    values = np.asarray(values)
    
    # Ensure the parameter is valid
    if parameter not in WQ_RANGES:
        raise ValueError(f"Invalid parameter: {parameter}. Must be one of {list(WQ_RANGES.keys())}.")
    
    required_ranges = WQ_RANGES[parameter]['required']
    
    # Handle case where ranges are split by periods
    if isinstance(required_ranges, dict):
        if periods is None:
            raise ValueError(f"Periods must be provided for parameter {parameter}")
            
        periods = np.asarray(periods)
        result = np.full(values.shape, '', dtype='U6') # Initialize output array

        # Process each period
        for period, (low, high) in required_ranges.items():
            mask = periods == period
            conditions = [values[mask] < low, (values[mask] >= low) & (values[mask] <= high), values[mask] > high]
            result[mask] = np.select(conditions, ['below', 'within', 'above'], default='')
            
        return result
    
    # Handle case where ranges are not split by periods
    low, high = required_ranges
    conditions = [values < low, (values >= low) & (values <= high), values > high]
    return np.select(conditions, ['below', 'within', 'above'], default='')