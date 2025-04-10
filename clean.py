#!/usr/bin/env python3
import pandas as pd
import argparse
import os

def remove_duplicates(input_file, output_file=None, keep_first=True, subset=None):
    """
    Remove duplicate rows from a CSV file and write the result to a new file.
    
    Parameters:
    - input_file (str): Path to the input CSV file
    - output_file (str): Path to the output CSV file (default: input_file_no_duplicates.csv)
    - keep_first (bool): If True, keep the first occurrence of a duplicate. If False, keep the last.
    - subset (list): List of column names to consider for identifying duplicates. 
                    If None, all columns are used.
    
    Returns:
    - int: Number of duplicate rows removed
    """
    # If no output file is specified, modify the original file directly
    if output_file is None:
        output_file = input_file
    
    print(f"Reading CSV file: {input_file}")
    # Read the CSV file
    df = pd.read_csv(input_file)
    initial_rows = len(df)
    
    # Remove duplicates
    keep_option = 'first' if keep_first else 'last'
    df_no_duplicates = df.drop_duplicates(subset=subset, keep=keep_option)
    final_rows = len(df_no_duplicates)
    duplicates_removed = initial_rows - final_rows
    
    # Write the result to the output file
    df_no_duplicates.to_csv(output_file, index=False)
    
    print(f"Processed {initial_rows} rows")
    print(f"Removed {duplicates_removed} duplicate rows")
    if input_file == output_file:
        print(f"Original file has been updated: {output_file}")
    else:
        print(f"Output file saved to: {output_file}")
    
    return duplicates_removed

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Remove duplicate rows from a CSV file.')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('-o', '--output', help='Path to the output CSV file (if not specified, will overwrite the input file)')
    parser.add_argument('--keep', choices=['first', 'last'], default='first',
                      help='Which duplicate to keep (default: first)')
    parser.add_argument('--columns', nargs='+', help='Specific columns to consider for duplicates')
    
    args = parser.parse_args()
    
    # Call the function to remove duplicates
    remove_duplicates(
        args.input_file,
        args.output,
        keep_first=(args.keep == 'first'),
        subset=args.columns
    )

if __name__ == "__main__":
    main()
