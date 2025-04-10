#!/usr/bin/env python3
"""
Script to extract molecule names without SMILES strings from molecule_archive.db
and save them to molecule_names.txt
"""

import sqlite3
import sys

def main():
    # Database path (default or from command line)
    db_path = "molecule_archive.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    output_file = "molecule_names.txt"
    
    print(f"Searching for molecules without SMILES in {db_path}")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query for molecules where smiles is NULL or empty
        cursor.execute("SELECT molecule FROM molecules WHERE smiles IS NULL OR smiles = ''")
        molecules = cursor.fetchall()
        
        # Close the connection
        conn.close()
        
        # Extract molecule names from results
        molecule_names = [row[0] for row in molecules]
        
        print(f"Found {len(molecule_names)} molecules without SMILES")
        
        # Write to file
        with open(output_file, "w") as f:
            for name in molecule_names:
                f.write(f"{name}\n")
        
        print(f"Saved molecule names to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
