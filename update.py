#!/usr/bin/env python3
"""
Script to extract molecule names without SMILES strings from molecule_archive.db,
fetch their SMILES using smiles_scraper.py and update the database.
"""
import os
import sqlite3
import sys
import subprocess
import tempfile
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("missing_smiles.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_molecules_without_smiles(db_path, output_file=None):
    """
    Extract molecules without SMILES from the database
    
    Args:
        db_path: Path to the molecule_archive.db
        output_file: Optional file to save the list of molecules
        
    Returns:
        List of molecule names without SMILES
    """
    logger.info(f"Searching for molecules without SMILES in {db_path}")
    
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
        
        logger.info(f"Found {len(molecule_names)} molecules without SMILES")
        
        # Write to file if requested
        if output_file:
            with open(output_file, "w") as f:
                for name in molecule_names:
                    f.write(f"{name}\n")
            logger.info(f"Saved molecule names to {output_file}")
        
        return molecule_names
        
    except Exception as e:
        logger.error(f"Error extracting molecules: {e}")
        return []

def fetch_smiles(molecules, max_concurrent=5):
    """
    Fetch SMILES for the list of molecules using smiles_scraper.py
    
    Args:
        molecules: List of molecule names
        max_concurrent: Maximum number of concurrent requests
        
    Returns:
        Path to CSV file with results or None if fetching failed
    """
    if not molecules:
        logger.warning("No molecules to fetch SMILES for")
        return None
    
    logger.info(f"Fetching SMILES for {len(molecules)} molecules")
    
    try:
        # Create a temporary file with the molecule names
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp:
            temp_path = temp.name
            for molecule in molecules:
                temp.write(f"{molecule}\n")
        
        # Create temp files for output
        smiles_db_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt').name
        csv_output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
        
        # Run smiles_scraper.py
        cmd = [
            "python", "smiles_scraper.py",
            "--input", temp_path,
            "--output", smiles_db_temp,
            "--csv-output", csv_output_temp,
            "--max-concurrent", str(max_concurrent)
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if successful
        if result.returncode == 0:
            logger.info("Successfully fetched SMILES")
            logger.info(result.stdout)
            
            # Clean up temp input file
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return csv_output_temp
        else:
            logger.error(f"Error fetching SMILES: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Error during SMILES fetching: {e}")
        return None

def update_database_with_smiles(db_path, csv_path):
    """
    Update the database with SMILES from the CSV file
    
    Args:
        db_path: Path to the molecule_archive.db
        csv_path: Path to the CSV file with molecule,smiles rows
        
    Returns:
        Number of molecules updated
    """
    if not csv_path or not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return 0
    
    logger.info(f"Updating database with SMILES from {csv_path}")
    
    try:
        # Read the CSV file
        molecule_to_smiles = {}
        with open(csv_path, 'r') as f:
            # Skip header
            next(f)
            for line in f:
                if ',' in line:
                    molecule, smiles = line.strip().split(',', 1)
                    if molecule and smiles:
                        molecule_to_smiles[molecule] = smiles
        
        logger.info(f"Read {len(molecule_to_smiles)} molecule-SMILES pairs from CSV")
        
        # Connect to the database and update
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Update each molecule
        updated_count = 0
        for molecule, smiles in molecule_to_smiles.items():
            cursor.execute(
                "UPDATE molecules SET smiles = ? WHERE molecule = ?",
                (smiles, molecule)
            )
            if cursor.rowcount > 0:
                updated_count += 1
        
        # Commit changes and close
        conn.commit()
        conn.close()
        
        logger.info(f"Updated SMILES for {updated_count} molecules in the database")
        return updated_count
        
    except Exception as e:
        logger.error(f"Error updating database: {e}")
        return 0

def main():
    # Database path (default or from command line)
    db_path = "molecule_archive.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    output_file = "molecule_names.txt"
    max_concurrent = 5
    
    if len(sys.argv) > 2:
        max_concurrent = int(sys.argv[2])
    
    # Extract molecules without SMILES
    molecules = extract_molecules_without_smiles(db_path, output_file)
    
    if not molecules:
        logger.info("No molecules without SMILES found. Nothing to do.")
        return
    
    # Fetch SMILES
    csv_path = fetch_smiles(molecules, max_concurrent)
    
    if not csv_path:
        logger.error("Failed to fetch SMILES. Exiting.")
        return
    
    # Update database
    updated_count = update_database_with_smiles(db_path, csv_path)
    
    # Clean up
    try:
        os.unlink(csv_path)
    except:
        pass
    
    if updated_count > 0:
        logger.info(f"Successfully updated {updated_count} molecules with SMILES")
    else:
        logger.warning("No molecules were updated with SMILES")

if __name__ == "__main__":
    main()
