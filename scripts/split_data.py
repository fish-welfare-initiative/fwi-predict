import os
from typing import Tuple

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, StratifiedGroupKFold, train_test_split
from ydata_profiling import compare, ProfileReport


def discretize_col(col, qtiles=10):
    return pd.qcut(col, q=qtiles, labels=False, duplicates='drop')


def split_data(df: pd.DataFrame,
               split_type: str,
               test_size: float = 0.2,
               random_state: int = 42,
               group_column: str = None,
               stratify_column: str = None,
               stratify_qtiles: int = None) -> Tuple[np.ndarray, np.ndarray]:
    
    if split_type == 'group' and group_column is None:
        raise ValueError("Group column must be specified for group split.")
    
    if split_type == 'group_stratified' and group_column is None:
        raise ValueError("Group column must be specified for group stratified split.")
    
    if split_type == 'stratified' or split_type == 'group_stratified':
        if stratify_column is None:
            raise ValueError("Stratify column must be specified for stratified split.")

        strat_col = df[stratify_column]
        if strat_col.dtype.kind in 'if':  # Check if y is continuous. This is not the right way.
            strat_col = discretize_col(strat_col, qtiles=stratify_qtiles)

    if split_type == 'shuffle':
        train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=test_size, random_state=random_state)
    
    if split_type == 'group':
        groups = df[group_column]
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, groups=groups))
    
    elif split_type == 'stratified':
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, y=strat_col))
    
    elif split_type == 'group_stratified':
        groups = df[group_column]
        splitter = StratifiedGroupKFold(n_splits=round(1/test_size), shuffle=True, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, strat_col, groups=groups))

    return train_idx, test_idx


# Some of these arguments should maybe be grouped
@click.command()
@click.argument('--input-file', type=click.Path(exists=True))
@click.option('--output-path', type=click.Path(), default="./data/splits", help='Path to the output file to save indices.') # Add format validation
@click.option('--split-type', type=click.Choice(['shuffle', 'group', 'stratified', 'group_stratified'], case_sensitive=False), default='shuffle', help='Type of split to perform.')
@click.option('--test-size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
@click.option('--random-state', type=int, default=42, help='Random state for reproducibility.')
@click.option('--group-column', type=str, default=None, help='Column name for group feature.')
@click.option('--stratify-column', type=str, default=None, help='Column name for stratification feature.')
@click.option('--stratify-qtiles', type=int, default=None, help='Number of qtiles for stratification.')
@click.option('--compare-splits/--no-compare-splits', default=True, help='Compare feature distributions in train and test splits.')
def split_dataset(input_file,
                  output_path,
                  split_type,
                  test_size,
                  random_state,
                  group_column,
                  stratify_column,
                  stratify_qtiles,
                  compare_splits):
    # Load dataset
    df = pd.read_csv(input_file) # MIght need to add header specification
    
    train_idx, test_idx = split_data(df,
                                     split_type,
                                     test_size,
                                     random_state,
                                     group_column,
                                     stratify_column,
                                     stratify_qtiles)
    
    # Save data split
    # Decide better file format later

    if os.path.splitext(output_path)[1] is None:
        filename = os.path.splitext(os.path.basename(input_file))[0]
        output_path = os.path.join(output_path, filename + '_split.npz')

    np.savez(output_path,
             ds=input_file,
             split_type=split_type,
             group_column=group_column,
             stratify_column=stratify_column,
             stratify_qtiles=stratify_qtiles,
             random_state=random_state,
             train_idx=train_idx,
             test_idx=test_idx)
    
    if compare_splits:
        splits = [df.iloc[train_idx], df.iloc[test_idx]]
        reports = [ProfileReport(split) for split in splits]
        diff_report = compare(reports)

        fp = os.path.splitext(output_path)[0] + '_diff.html'
        diff_report.to_file(fp)

    # Implement missing comparison later


if __name__ == '__main__':
    split_dataset()