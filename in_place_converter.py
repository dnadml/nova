#!/usr/bin/env python3
"""
In-place Molecule CSV Converter

This script converts a tab/space-separated molecule file to a properly formatted
comma-separated CSV file, modifying the file in-place.

Usage:
    python in_place_converter.py filename.csv
"""

import sys
import os
import re
import tempfile

def convert_file_in_place(filename):
    """
    Convert a molecule file from tab/space-separated to comma-separated format,
    modifying the file in-place.
    
    Args:
        filename (str): Path to the file to be converted
    """
    try:
        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp()
        
        with os.fdopen(temp_fd, 'w') as temp_file:
            # Write the header
            temp_file.write("molecule,smiles\n")
            
            # Read and process each line of the input file
            with open(filename, 'r') as input_file:
                # Skip header if it exists
                first_line = input_file.readline()
                if not first_line.lower().startswith("molecule"):
                    # Process the first line if it's not a header
                    process_line(first_line, temp_file)
                
                # Process the rest of the file
                for line in input_file:
                    process_line(line, temp_file)
        
        # Replace the original file with the temporary file
        os.replace(temp_path, filename)
        print(f"Conversion complete! File {filename} has been updated.")
    
    except Exception as e:
        print(f"Error: {e}")
        # Make sure to remove the temp file if it exists
        if 'temp_path' in locals():
            os.remove(temp_path)
        return False
    
    return True

def process_line(line, output_file):
    """
    Process a single line and write the converted line to the output file.
    
    Args:
        line (str): The line to process
        output_file: The file object to write to
    """
    line = line.strip()
    if not line:
        return
    
    # Try to split by tab first
    parts = line.split('\t')
    
    # If that didn't give us at least two parts, try spaces
    if len(parts) == 1:
        # Split on first sequence of whitespace
        match = re.match(r'(\S+)\s+(.+)', line)
        if match:
            parts = [match.group(1), match.group(2)]
    
    # Ensure we have at least two parts
    if len(parts) >= 2:
        molecule_id = parts[0].strip()
        smiles = parts[1].strip()
        
        # Write to output file
        output_file.write(f"{molecule_id},{smiles}\n")
    else:
        print(f"Warning: Could not parse line: {line}")

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python in_place_converter.py filename.csv")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        sys.exit(1)
    
    success = convert_file_in_place(filename)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
