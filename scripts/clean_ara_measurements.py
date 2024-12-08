# Clean ARA data (from program start until March 31, 2024)
import numpy as np
import pandas as pd
from tqdm import tqdm

from fwi_predict.constants import TIMEZONE


yes_no_map = {'Yes': True, 'No': False}

column_map = { # Consider later moving to package with column name standard
    'Date of Data Collection': 'date',
    'Pond ID': 'pond_id',
    'Measurement Type': 'time_of_day',
    'Group': 'group',
    'Pond Type': 'treatment_group',
    'Time (sample or data collection)': 'sample_time',
    'Name': 'name', # Is this the owner?
    'Which meter are you using?': 'measure_instrument',
    'Follow-up': 'follow_up',
    'Weather': 'weather',
    'Dissolved Oxygen (mg/L)': 'do_mg_per_L',
    'pH': 'ph',
    'Turbidity (cm)': 'turbidity_cm',
    'Ammonia (mg/L)': 'ammonia_mg_per_L',
    'Water quality in the *required* range': 'in_range',
    'Parameter(s) out of range': 'params_out_of_range',
    'Temperature (°C)': 'temperature_celsius',
    'Conductivity (ms)': 'conductivity_ms',
    'TDS (ppt)': 'tds_ppt',
    'Water Color': 'water_color',
    'Corrective actions requested': 'corrective_actions_requested',
    'Amount requested': 'amount_requested',
    'Corrective actions implementation': 'corrective_actions_implementation',
    'Corrective actions implementation date': 'corrective_actions_implementation_date',
    'Corrective actions taken': 'corrective_actions_taken',
    'Non-prescribed actions taken': 'non_prescribed_actions_taken',
    'If the farmer didn\'t apply corrective actions, what were the reasons?': 'no_corrective_reasons',
    'Improvement of *targeted* water quality parameters': 'targeted_params_improvement',
    'Notes (corrective actions)': 'corrective_action_notes',
    'Individuals air gulping': 'individuals_air_gulping',
    'Individuals tail splashing': 'individuals_tail_splashing',
    'Dead fish': 'dead_fish',
    'Notes (mortalities)': 'mortalities_notes',
    'Feed amount (kg)': 'feed_amount_kg',
    'Did we help the fish?': 'did_we_help_the_fish',
    'Stocking density (fish per acre)': 'fish_per_acre',
    'Species': 'species',
    'Weight': 'weight', # Need to find units
    'Notes (additional info)': 'additional_info',
    'Any pictures you want to share?': 'pictures',
    'Data tool Sr. No.': 'serial_no_data_tool',
    'Winkler\'s Method Used for DO': 'winklers_method',
    'Feed type': 'feed_type',
    'Days without feed since the last measurement': 'days_without_feed_since_last_measurement',
    'Chlorophyll-a': 'chl-a',
    'Phycocyanin': 'phycocyanin',
    'Plankton Sample Analysis Date': 'plankton_sample_analysis_date',
    'Total n° of cells / 1L': 'cells_per_L',
    'Submission ID': 'submission_id', # Probably refer to plankton sample
    'Dead fish found by the FARMER since the last visit': 'dead_fish_since_last_visit_farmer_report',
    'Dead fish found by YOU today': 'dead_fish_found_fwi',
    'How many locations?': 'num_locations', # Is this the number of sampling locations?
    'Prescribed collection date': 'prescribed_collection_date',
    'Reason for late or no collection (if any)': 'no_or_late_collection_reason',
    'Did the farmer collect the measurement?': 'farmer_collected_measurement',
    'Turbidity (farmer measurement)': 'turbidity_farmer',
    'Temperature (farmer measurement)': 'temperature_farmer',
    'pH (farmer measurement)': 'ph_farmer',
    'Feed given today': 'feed_given_today',
    'Time (sample analysis)': 'sample_analysis_time',
    '1. Dissolved Oxygen (mg/L)': 'do_mg_per_L_1',
    '2. Dissolved Oxygen (mg/L)': 'do_mg_per_L_2',
    '3. Dissolved Oxygen (mg/L)': 'do_mg_per_L_3',
    '1. pH': 'ph_1',
    '2. pH': 'ph_2',
    '3. pH': 'ph_3',
    '1. Temperature (in °C)': 'temperature_celsius_1',
    '2. Temperature (in °C)': 'temperature_celsius_2',
    '3. Temperature (in °C)': 'temperature_celsius_3',
    'Light bottle DO (NPP)': 'light_bottle_do_npp',
    'Dark bottle DO (R)': 'dark_bottle_do_R',
    'Outcomes of corrective actions': 'corrective_actions_outcome',
    'Salinity (ppt)': 'salinity_ppt',
    'Feeding': 'feeding',
    'Air gulping': 'air_gulping',
    'Primary Productivity GPP (mg/L)': 'primary_productivity_gpp_mg_per_L',
    'Alkalinity': 'alkalinity',
    'How did the farmer find implementing the corrective actions?': 'corrective_actions_farmer_ease',
    'Tail splashing': 'tail_splashing',
    'How many fish did we help?': 'fish_helped',
    'How did we help the fish?': 'fish_help_method',
    'Readings communicated today': 'readings_communicated_today',
    'Actions taken': 'actions_taken',
    'Details': 'details',
    'Wind': 'wind',
    'Disease outbreak': 'disease_outbreak',
    'Lice infestation': 'lice_infestation',
    'Vegetation (1+ cm into the water)': 'vegetation_in_water'
} 


def is_str(x) -> bool:
    return isinstance(x, str)


def resolve_duplicates(
    df, id_cols, string_delimiter="; ", mark_column="had_duplicates"
):
    """
    Resolve duplicates in a DataFrame, marking which rows had duplicates and resolving conflicts.

    Args:
        df (pd.DataFrame): The input DataFrame containing duplicates.
        id_cols (list): List of columns defining the unique ID.
        string_delimiter (str, optional): Delimiter to concatenate strings.
        mark_column (str, optional): Name of the column to indicate duplicates.

    Returns:
        pd.DataFrame: A DataFrame with duplicates resolved and marked.
    """
    def resolve_group(group):
        resolved = {}
        for col in group.columns:
            if col in id_cols:
                # Keep ID columns as is
                resolved[col] = group[col].iloc[0]
            else:
                if pd.api.types.is_numeric_dtype(group[col]):
                    # Resolve numeric columns by mean, ignoring NaNs
                    resolved[col] = group[col].mean(skipna=True)
                elif pd.api.types.is_string_dtype(group[col]):
                    # Resolve string columns by concatenating unique values
                    unique_strings = group[col].dropna().unique()
                    resolved[col] = string_delimiter.join(unique_strings)
                else:
                    # Cast other types to strings and concatenate unique values
                    unique_values = group[col].dropna().astype(str).unique()
                    resolved[col] = string_delimiter.join(unique_values)
        
        # Mark duplicates if the group contains more than one row
        resolved[mark_column] = len(group) > 1
        return pd.Series(resolved)

    # Apply resolution to each group
    resolved_df = (
        df.groupby(id_cols)
        .apply(resolve_group, include_groups=False)
        .reset_index()
    )

    return resolved_df


if __name__ == "__main__":
    ara_raw = pd.read_excel("data/raw/All ARA Data until March 31, 2024.xlsx", sheet_name='Sheet1')

    ara = ara_raw.rename(columns=column_map)
    assert(ara.columns.isin(column_map.values()).all()) # Assert column names standardized

    # Clean sample times
    ara['sample_time'].apply(type).value_counts() 
    str_formatted = ara['sample_time'].apply(is_str)
    ara.loc[str_formatted, 'sample_time'] # Find string formatted times
    ara.loc[str_formatted, 'sample_time'] = ara.loc[str_formatted, 'sample_time'] + ':00' # Add seconds

    # Convert to strings to deal with NAs
    ara.loc[~str_formatted  & ara['sample_time'].notna(), 'sample_time'] = \
        ara.loc[~str_formatted & ara['sample_time'].notna(), 'sample_time'].apply(lambda x: x.strftime("%H:%M:%S"))
    
    ara['sample_dt'] = pd.to_datetime(ara['date'].dt.strftime("%Y-%m-%d") + ' ' + ara['sample_time'], errors='coerce')
    ara['sample_dt'] = ara['sample_dt'].dt.tz_localize(TIMEZONE)
    ara = ara.drop(columns=['date', 'sample_time'])

    ara['time_of_day'] = ara['time_of_day'].str.lower()

    # Clean badly formatted values. See eponymous notebook for data exploration.

    # Construct IDs
    ara['region'] = ara['pond_id'].str[:2]
    ara['farm_id'] = ara['pond_id'].str.replace(r"\d+$", "", regex=True)

    # Booleans
    ara['follow_up'] = ara['follow_up'].map(yes_no_map)
    ara['in_range'] = ara['in_range'].map(yes_no_map)
    ara['did_we_help_the_fish'] = ara['did_we_help_the_fish'].map(yes_no_map)
    ara['winklers_method'] = ara['winklers_method'].map(yes_no_map)
    ara['feed_given_today'] = ara['feed_given_today'].map(yes_no_map)
    ara['readings_communicated_today'] = ara['readings_communicated_today'].map(yes_no_map)
    ara['disease_outbreak'] = ara['disease_outbreak'].map(yes_no_map)
    ara['lice_infestation'] = ara['lice_infestation'].map(yes_no_map)
    ara['vegetation_in_water'] = ara['vegetation_in_water'].map(yes_no_map)

    # Turbidity
    str_formatted = ara['turbidity_cm'].apply(is_str)
    ara.loc[str_formatted, 'turbidity_cm'] = np.nan
    ara['turbidity_cm'] = ara['turbidity_cm'].astype(float)

    # Ammonia
    str_formatted = ara['ammonia_mg_per_L'].apply(is_str)
    ara.loc[str_formatted, 'ammonia_mg_per_L'] = np.nan
    ara['ammonia_mg_per_L'] = ara['ammonia_mg_per_L'].astype(float)

    # Temperature
    str_formatted = ara['temperature_celsius'].apply(is_str)
    ara.loc[str_formatted, 'temperature_celsius'] = np.nan # Str formatted temperature is illegible so setting to nan.
    ara['temperature_celsius'] = ara['temperature_celsius'].astype(float)

    # Conductivity
    str_formatted = ara['conductivity_ms'].apply(is_str)
    ara.loc[str_formatted, 'conductivity_ms'] = ara.loc[str_formatted, 'conductivity_ms'] \
        .str.replace('o', '0') \
        .str.replace('`', '') \
        .str.replace(' ', '') \
        .str.replace('\'', '.')
    ara['conductivity_ms'] = ara['conductivity_ms'].astype(float)

    # TDS
    str_formatted = ara['tds_ppt'].apply(is_str)
    ara.loc[str_formatted, 'tds_ppt'] = ara.loc[str_formatted, 'tds_ppt'] \
        .str.replace('`', '') \
        .str.replace('l', '1')
    ara['tds_ppt'] = ara['tds_ppt'].astype(float)

    # Fish per acre
    str_formatted = ara['fish_per_acre'].apply(is_str)
    ara.loc[str_formatted, 'fish_per_acre'] = np.nan
    ara['fish_per_acre'] = ara['fish_per_acre'].astype(float)

    # Days without feed since last measurement
    str_formatted = ara['days_without_feed_since_last_measurement'].apply(is_str)
    ara.loc[str_formatted, 'days_without_feed_since_last_measurement']
    ara.loc[ara['days_without_feed_since_last_measurement'].str.contains('second day') &
            ara['days_without_feed_since_last_measurement'].notna(),
            'days_without_feed_since_last_measurement'] = 2
    ara.loc[ara['days_without_feed_since_last_measurement'].str.contains('once') &
            ara['days_without_feed_since_last_measurement'].notna(),
            'days_without_feed_since_last_measurement'] = np.nan # Set 'Weekly once' to null for now
    ara['days_without_feed_since_last_measurement'] = ara['days_without_feed_since_last_measurement'].astype(float) 

    # Dead fish farmer report
    str_formatted = ara['dead_fish_since_last_visit_farmer_report'].apply(is_str)
    ara.loc[str_formatted, 'dead_fish_since_last_visit_farmer_report'] = 0
    ara['dead_fish_since_last_visit_farmer_report'] = ara['dead_fish_since_last_visit_farmer_report'].astype(float)

    # Number of locations 
    ara.loc[ara['num_locations'].str.contains('1') & ara['num_locations'].notna(), 'num_locations'] = 1
    ara['num_locations'] = ara['num_locations'].astype(float)

    # Prescribed collection date
    str_formatted = ara['prescribed_collection_date'].apply(is_str)
    ara.loc[str_formatted, 'prescribed_collection_date'] = ara.loc[str_formatted, 'prescribed_collection_date'] \
        .str.extract('(\\d{2}/\\d{2}/\\d{4})') \
        .squeeze() \
        .pipe(pd.to_datetime)
    ara['prescribed_collection_date'] = ara['prescribed_collection_date'].astype('datetime64[ns]')

    # Numbered WQ params
    str_formatted = ara['do_mg_per_L_1'].apply(is_str)
    ara.loc[str_formatted, 'do_mg_per_L_1'] = ara.loc[str_formatted, 'do_mg_per_L_1'] \
        .str.replace(' ', '') \
        .str.replace('O', '0')
    ara['do_mg_per_L_1'] = ara['do_mg_per_L_1'].astype(float)

    str_formatted = ara['temperature_celsius_1'].apply(is_str)
    ara.loc[str_formatted, 'temperature_celsius_1'] = ara.loc[str_formatted, 'temperature_celsius_1'] \
        .str.replace('..', '.') \
        .str.replace(' ', '')
    ara['temperature_celsius_1'] = ara['temperature_celsius_1'].astype(float)

    # DO instrument measurements
    str_formatted = ara['light_bottle_do_npp'].apply(is_str)
    ara.loc[str_formatted, 'light_bottle_do_npp'] = ara.loc[str_formatted, 'light_bottle_do_npp'] \
        .str.replace('`', '') \
        .replace('NC', np.nan)
    ara['light_bottle_do_npp'] = ara['light_bottle_do_npp'].astype(float)

    str_formatted = ara['dark_bottle_do_R'].apply(is_str)
    ara.loc[str_formatted, 'dark_bottle_do_R'] = ara.loc[str_formatted, 'dark_bottle_do_R'] \
        .str.replace('o', '0') \
        .replace('NC', np.nan)
    ara['dark_bottle_do_R'] = ara['dark_bottle_do_R'].astype(float)

    # Salinity
    str_formatted = ara['salinity_ppt'].apply(is_str)
    ara.loc[str_formatted, 'salinity_ppt'] = ara.loc[str_formatted, 'salinity_ppt'] \
        .str.replace(' ', '') \
        .str.replace(',', '.')
    ara['salinity_ppt'] = ara['salinity_ppt'].astype(float)

    # Primary productivity
    str_formatted = ara['primary_productivity_gpp_mg_per_L'].apply(is_str)
    ara.loc[str_formatted, 'primary_productivity_gpp_mg_per_L'] = np.nan # All unintelligible
    ara['primary_productivity_gpp_mg_per_L'] = ara['primary_productivity_gpp_mg_per_L'].astype(float)

    # Deduplicate dataframe
    # Note that this may also remove columns without sample dates.
    print('Deduplicating...')
    id_cols = ['pond_id', 'sample_dt']
    ara = resolve_duplicates(ara, id_cols)
    print('Done deduplicating.')

    # Re-order cols and save
    front_cols = ['pond_id', 'region', 'farm_id', 'group', 'treatment_group', 'sample_dt', 'time_of_day']
    ara = ara[front_cols + [col for col in ara.columns if col not in front_cols]]

    ara.to_csv("data/clean/ara_measurements_clean.csv", index=False)
