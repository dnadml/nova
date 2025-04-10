#!/usr/bin/env python3
"""
NOVA Molecule SMILES Scraper

This script fetches SMILES strings for NOVA molecules by scraping the NOVA dashboard.
It handles cleaning of the SMILES strings to remove all markup.

Usage:
    python smiles_scraper.py --input molecules.txt --output smiles_db.txt --csv-output molecules.csv

Requirements:
- Python 3.6+
- requests
- beautifulsoup4
- tqdm

Install dependencies:
    pip install requests beautifulsoup4 tqdm
"""

import os
import re
import sys
import time
import random
import argparse
import logging
import threading
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("smiles_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MOLECULE_DETAILS_URL = "https://nova-dashboard-frontend.vercel.app/molecule?molecule="
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
DELAY_BETWEEN_REQUESTS = 1  # seconds

# Specific patterns for cleaning SMILES
SMILES_PREFIX = "\"children\":\""
ALT_SMILES_PREFIX = "children"
SMILES_SUFFIXES = ["\"}", "\"]", "\","]


def get_soup_from_url(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """
    Get BeautifulSoup object from a URL with retries.
    
    Args:
        url: The URL to request
        retries: Number of retries on failure
        
    Returns:
        BeautifulSoup object or None if all retries fail
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                # Add jitter to delay to avoid being detected as a bot
                jitter = random.uniform(0.5, 1.5)
                time.sleep(DELAY_BETWEEN_REQUESTS * jitter)
            else:
                logger.error(f"Failed to retrieve {url} after {retries} attempts")
                return None


def clean_smiles(raw_smiles: str) -> str:
    """
    Clean SMILES string by removing markup.
    
    Args:
        raw_smiles: Raw SMILES string potentially containing markup
        
    Returns:
        Cleaned SMILES string
    """
    if not raw_smiles:
        return ""
    
    clean_smiles = raw_smiles
    
    # Try to extract SMILES after the "children" marker
    if ALT_SMILES_PREFIX in clean_smiles:
        # Find the position of "children" and extract everything after it
        children_pos = clean_smiles.find(ALT_SMILES_PREFIX)
        if children_pos >= 0:
            # Skip past "children" and any non-SMILES characters
            start_pos = children_pos + len(ALT_SMILES_PREFIX)
            while start_pos < len(clean_smiles) and not (clean_smiles[start_pos].isalpha() or clean_smiles[start_pos] in "[]()"):
                start_pos += 1
            clean_smiles = clean_smiles[start_pos:]
            
            # Find where the SMILES string ends (at a suffix marker)
            end_pos = len(clean_smiles)
            for suffix in SMILES_SUFFIXES:
                pos = clean_smiles.find(suffix)
                if pos >= 0 and pos < end_pos:
                    end_pos = pos
            
            clean_smiles = clean_smiles[:end_pos]
    
    # Look for the old format SMILES:value
    elif "SMILES:" in clean_smiles:
        parts = clean_smiles.split("SMILES:", 1)
        if len(parts) > 1:
            clean_smiles = parts[1].strip()
    
    # Remove any trailing > character that indicates truncation
    clean_smiles = clean_smiles.rstrip('>')
    
    # Remove any remaining quotes, backslashes, and non-SMILES characters
    clean_smiles = re.sub(r'[^A-Za-z0-9\(\)\[\]\.\=\@\#\-\\\/\+]', '', clean_smiles)
    
    return clean_smiles


def fetch_smiles(molecule_id: str) -> Optional[str]:
    """
    Fetch the SMILES string for a given molecule ID from the NOVA dashboard.
    
    Args:
        molecule_id: The molecule ID to fetch SMILES for
        
    Returns:
        The cleaned SMILES string or None if it couldn't be fetched
    """
    url = f"{MOLECULE_DETAILS_URL}{molecule_id}"
    soup = get_soup_from_url(url)
    
    if not soup:
        logger.error(f"Failed to retrieve SMILES data for molecule {molecule_id}")
        return None
    
    # Extract SMILES from the page content
    html_content = str(soup)
    
    # Look for spans that might contain SMILES data
    spans = soup.find_all('span', class_=lambda c: c and 'purple' in c)
    for span in spans:
        if span.string and len(span.string) > 5:  # SMILES are typically longer than 5 chars
            smiles = clean_smiles(span.string)
            logger.info(f"Found SMILES for molecule {molecule_id}: {smiles}")
            return smiles
    
    # If we couldn't find it in spans, look in the HTML content
    # Try json "children" format
    children_match = re.search(r'"children":"([^"]+)"', html_content)
    if children_match:
        smiles = clean_smiles(children_match.group(1))
        logger.info(f"Found SMILES for molecule {molecule_id} (JSON format): {smiles}")
        return smiles
    
    # Try old format SMILES:value
    smiles_match = re.search(r'SMILES:([^\n<]+)', html_content)
    if smiles_match:
        smiles = clean_smiles(smiles_match.group(1))
        logger.info(f"Found SMILES for molecule {molecule_id} (text format): {smiles}")
        return smiles
    
    logger.warning(f"Could not find SMILES string for molecule {molecule_id}")
    return None


def fetch_smiles_batch(molecules: List[str], max_concurrent: int = 5) -> Dict[str, str]:
    """
    Fetch SMILES strings for a batch of molecules using a semaphore to limit concurrency.
    
    Args:
        molecules: List of molecule IDs
        max_concurrent: Maximum number of concurrent requests
        
    Returns:
        Dictionary mapping molecule IDs to SMILES strings
    """
    molecule_to_smiles = {}
    semaphore = threading.Semaphore(max_concurrent)
    threads = []
    
    def fetch_with_semaphore(molecule_id):
        with semaphore:
            smiles = fetch_smiles(molecule_id)
            # Make sure to store the SMILES even if it's an empty string
            molecule_to_smiles[molecule_id] = smiles if smiles else ""
            
            # Add delay between requests to avoid overloading the server
            jitter = random.uniform(0.5, 1.5)
            time.sleep(DELAY_BETWEEN_REQUESTS * jitter)
    
    logger.info(f"Fetching SMILES for {len(molecules)} molecules (max {max_concurrent} concurrent requests)...")
    pbar = tqdm(total=len(molecules), desc="Fetching SMILES")
    
    # Create and start threads for each molecule
    for molecule in molecules:
        thread = threading.Thread(target=lambda m=molecule: (fetch_with_semaphore(m), pbar.update(1)))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    pbar.close()
    
    # Count how many molecules have non-empty SMILES
    successful_fetches = sum(1 for smiles in molecule_to_smiles.values() if smiles)
    logger.info(f"Successfully fetched SMILES for {successful_fetches} out of {len(molecules)} molecules")
    
    return molecule_to_smiles


def load_existing_smiles_db(db_file: str) -> Dict[str, str]:
    """
    Load existing molecule-to-SMILES mappings from a file, cleaning SMILES as needed.
    
    Args:
        db_file: Path to the database file
        
    Returns:
        Dictionary mapping molecule IDs to SMILES strings
    """
    molecule_to_smiles = {}
    
    if os.path.exists(db_file):
        try:
            with open(db_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            molecule_id, raw_smiles = parts
                            # Clean the SMILES string when loading
                            smiles = clean_smiles(raw_smiles) if raw_smiles else ""
                            molecule_to_smiles[molecule_id] = smiles
            
            logger.info(f"Loaded {len(molecule_to_smiles)} existing SMILES mappings from {db_file}")
        except Exception as e:
            logger.error(f"Error loading SMILES database: {e}")
    else:
        logger.info(f"No existing SMILES database found at {db_file}. Creating a new one.")
    
    return molecule_to_smiles


def save_smiles_db(molecule_to_smiles: Dict[str, str], db_file: str) -> None:
    """
    Save molecule-to-SMILES mappings to a file.
    
    Args:
        molecule_to_smiles: Dictionary mapping molecule IDs to SMILES strings
        db_file: Path to the database file
    """
    try:
        with open(db_file, 'w') as f:
            for molecule_id, smiles in sorted(molecule_to_smiles.items()):
                f.write(f"{molecule_id}\t{smiles}\n")
        
        # Count non-empty SMILES for reporting
        non_empty_count = sum(1 for smiles in molecule_to_smiles.values() if smiles)
        logger.info(f"Saved {len(molecule_to_smiles)} molecule mappings to {db_file} ({non_empty_count} with valid SMILES)")
    except Exception as e:
        logger.error(f"Error saving SMILES database: {e}")


def save_molecules_as_csv(molecule_to_smiles: Dict[str, str], csv_file: str, 
                         molecules_to_include: List[str] = None) -> None:
    """
    Save molecules and their SMILES strings to a CSV file.
    
    Args:
        molecule_to_smiles: Dictionary mapping molecule IDs to SMILES strings
        csv_file: Path to the CSV file to write
        molecules_to_include: Optional list of molecules to include (if None, include all)
    """
    try:
        with open(csv_file, 'w') as f:
            # Write header
            f.write("molecule,smiles\n")
            
            # Determine which molecules to write
            to_write = molecules_to_include if molecules_to_include else molecule_to_smiles.keys()
            
            # Write data rows
            count = 0
            for molecule_id in sorted(to_write):
                if molecule_id in molecule_to_smiles and molecule_to_smiles[molecule_id]:
                    smiles = molecule_to_smiles[molecule_id]
                    # Check for commas which would break CSV format
                    if ',' in smiles:
                        logger.warning(f"SMILES for {molecule_id} contains commas, which may break CSV format")
                    f.write(f"{molecule_id},{smiles}\n")
                    count += 1
        
        logger.info(f"Saved {count} molecules with SMILES to CSV: {csv_file}")
    except Exception as e:
        logger.error(f"Error saving molecules to CSV: {e}")

def read_molecules_from_file(file_path: str) -> List[str]:
    """
    Read molecule IDs from a file.
    
    Args:
        file_path: Path to the file containing molecule IDs
        
    Returns:
        List of molecule IDs
    """
    molecules = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Handle tab-separated files (first column is molecule ID)
                if '\t' in line:
                    parts = line.split('\t', 1)
                    molecules.append(parts[0])
                else:
                    molecules.append(line)
        
        logger.info(f"Read {len(molecules)} molecules from {file_path}")
    except Exception as e:
        logger.error(f"Error reading molecules from {file_path}: {e}")
    
    return molecules


def clean_existing_db(db_file: str) -> None:
    """
    Clean an existing SMILES database file in place.
    
    Args:
        db_file: Path to the database file to clean
    """
    if not os.path.exists(db_file):
        logger.error(f"File not found: {db_file}")
        return
    
    try:
        # Load and clean in memory
        molecule_to_smiles = {}
        
        with open(db_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Get molecule ID and raw SMILES
                parts = line.split('\t', 1)
                molecule_id = parts[0]
                raw_smiles = parts[1] if len(parts) > 1 else ""
                
                # Clean the SMILES
                clean_smiles_str = clean_smiles(raw_smiles)
                molecule_to_smiles[molecule_id] = clean_smiles_str
        
        # Save back to the same file
        save_smiles_db(molecule_to_smiles, db_file)
        logger.info(f"Cleaned SMILES database: {db_file}")
        
    except Exception as e:
        logger.error(f"Error cleaning SMILES database: {e}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fetch SMILES strings for NOVA molecules")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Input file containing molecule IDs (one per line)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="smiles_db.txt",
        help="Output file for SMILES database (default: smiles_db.txt)"
    )
    parser.add_argument(
        "--csv-output", 
        type=str, 
        default="",
        help="Output CSV file with molecule,smiles format (optional)"
    )
    parser.add_argument(
        "--max-concurrent", 
        type=int, 
        default=5,
        help="Maximum number of concurrent requests (default: 5)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    parser.add_argument(
        "--update-only", 
        action="store_true",
        help="Only fetch SMILES for molecules not already in the database"
    )
    parser.add_argument(
        "--clean-only", 
        action="store_true",
        help="Just clean the existing database file without fetching new data"
    )
    return parser.parse_args()


def main():
    """Main function to fetch and clean SMILES for molecules."""
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    # Special mode to just clean an existing DB file
    if args.clean_only and os.path.exists(args.output):
        logger.info(f"Cleaning existing SMILES database: {args.output}")
        clean_existing_db(args.output)
        if args.csv_output:
            # Load the cleaned database and export to CSV
            molecule_to_smiles = load_existing_smiles_db(args.output)
            save_molecules_as_csv(molecule_to_smiles, args.csv_output)
        return
    
    # Load existing SMILES database if update-only mode is enabled
    molecule_to_smiles = load_existing_smiles_db(args.output) if args.update_only else {}
    
    # Read molecules from the input file
    molecules = read_molecules_from_file(args.input)
    
    if not molecules:
        logger.error("No molecules found in the input file.")
        return
    
    # Determine which molecules need SMILES lookup
    if args.update_only:
        molecules_to_fetch = [m for m in molecules if m not in molecule_to_smiles or not molecule_to_smiles[m]]
        logger.info(f"Found {len(molecules_to_fetch)} molecules needing SMILES lookup out of {len(molecules)} total")
    else:
        molecules_to_fetch = molecules
    
    # Fetch SMILES for molecules
    if molecules_to_fetch:
        new_smiles = fetch_smiles_batch(molecules_to_fetch, args.max_concurrent)
        molecule_to_smiles.update(new_smiles)
    
    # Save results
    save_smiles_db(molecule_to_smiles, args.output)
    
    # Save as CSV if requested
    if args.csv_output:
        save_molecules_as_csv(molecule_to_smiles, args.csv_output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("SMILES fetching interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Fatal error in SMILES fetching process: {e}", exc_info=True)
        sys.exit(1)
