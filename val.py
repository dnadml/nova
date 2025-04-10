import os
import sys
import time
import asyncio
import logging
import argparse
import traceback
from ast import literal_eval
from datetime import datetime

# Standard library imports
import pandas as pd
import requests
import sqlite3

# Bittensor imports
import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata

# Get the base directory to import utilities
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(BASE_DIR)

# Import utilities - updated to use the new imports for getting proteins
from my_utils import (
    get_smiles, 
    get_sequence_from_protein_code, 
    get_heavy_atom_count, 
    get_challenge_proteins_from_blockhash,
    upload_file_to_github
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("validator.log")
    ]
)
logger = logging.getLogger()

# ----------------------------------------------------------------------------
# SMILES SCRAPER - INTEGRATED INTO VALIDATION SCRIPT
# ----------------------------------------------------------------------------

import re
import random
import threading
from typing import List, Dict, Optional, Set, Tuple
from bs4 import BeautifulSoup
from tqdm import tqdm

# Constants for SMILES scraper
MOLECULE_DETAILS_URL = "https://nova-dashboard-frontend.vercel.app/molecule?molecule="
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
DELAY_BETWEEN_REQUESTS = 1  # seconds


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
    if "children" in clean_smiles:
        # Find the position of "children" and extract everything after it
        children_pos = clean_smiles.find("children")
        if children_pos >= 0:
            # Skip past "children" and any non-SMILES characters
            start_pos = children_pos + len("children")
            while start_pos < len(clean_smiles) and not (clean_smiles[start_pos].isalpha() or clean_smiles[start_pos] in "[]()"):
                start_pos += 1
            clean_smiles = clean_smiles[start_pos:]
            
            # Find where the SMILES string ends (at a suffix marker)
            end_pos = len(clean_smiles)
            for suffix in ["\"}", "\"]", "\","]:
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
    
    logger.info(f"Fetching SMILES for {len(molecules)} molecules in batch (max {max_concurrent} concurrent requests)...")
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
    logger.info(f"Successfully fetched SMILES for {successful_fetches} out of {len(molecules)} molecules via NOVA dashboard")
    
    return molecule_to_smiles
# ----------------------------------------------------------------------------
# SMILES VALIDATION
# ----------------------------------------------------------------------------

def is_valid_smiles(smiles_str):
    """
    Check if a SMILES string is valid by:
    1. Checking for balanced parentheses
    2. Checking for balanced square brackets
    3. Ensuring it's not empty or None
    
    Args:
        smiles_str (str): SMILES string to validate
        
    Returns:
        bool: True if SMILES appears valid, False otherwise
    """
    if not smiles_str or not isinstance(smiles_str, str):
        return False
    
    # Check for balanced parentheses
    if smiles_str.count('(') != smiles_str.count(')'):
        return False
    
    # Check for balanced square brackets
    if smiles_str.count('[') != smiles_str.count(']'):
        return False
    
    # Check for any obvious syntax errors
    for char in ['(()', '())', '[]', '][', '==']:
        if char in smiles_str:
            return False
    
    # Additional validation could be added here
    return True

# ----------------------------------------------------------------------------
# DATABASE CLASSES
# ----------------------------------------------------------------------------

class SAVILookup:
    """Fast SAVI molecule name to SMILES lookup."""
    
    def __init__(self, db_path="savi_lookup.db"):
        """Initialize the database connection."""
        self.db_path = db_path
        self.conn = None
        self.logger = logger
    
    def connect(self):
        """Connect to the database."""
        if not self.conn:
            try:
                self.conn = sqlite3.connect(self.db_path)
                self.conn.row_factory = sqlite3.Row
                self.logger.info("Connected to database")
            except Exception as e:
                self.logger.error(f"Error connecting to database: {e}")
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def lookup_smiles_batch(self, molecule_names):
        """
        Look up SMILES for multiple molecules at once using exact matches.
        
        Args:
            molecule_names: List of molecule names
            
        Returns:
            Dictionary mapping molecule names to SMILES strings
        """
        self.logger.info(f"Looking up SMILES for {len(molecule_names)} molecules in SAVI database")
        
        if not os.path.exists(self.db_path):
            self.logger.error(f"Database file not found: {self.db_path}")
            return {}
            
        self.connect()
        if not self.conn:
            self.logger.error("Database connection failed")
            return {}
            
        result = {}
        try:
            # Use placeholders for SQL query with IN clause
            placeholders = ','.join(['?'] * len(molecule_names))
            query = f"SELECT name as molecule, smiles FROM molecules WHERE name IN ({placeholders})"
            
            cursor = self.conn.cursor()
            cursor.execute(query, molecule_names)
            
            rows = cursor.fetchall()
            
            for row in rows:
                result[row['molecule']] = row['smiles']
            
            found_count = len(result)
            self.logger.info(f"Found SMILES for {found_count}/{len(molecule_names)} molecules in SAVI database ({found_count/len(molecule_names)*100:.1f}%)")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in batch SMILES lookup: {e}")
            traceback.print_exc()
            return {}


class MoleculeDB:
    """Database for storing molecules and their SMILES strings."""
    
    def __init__(self, db_path="molecule_archive.db"):
        """Initialize the database connection."""
        self.db_path = db_path
        self.conn = None
        self.logger = logger
    
    def connect(self):
        """Connect to the SQLite database."""
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def init_db(self):
        """Initialize the molecules table with existing schema."""
        self.connect()
        cursor = self.conn.cursor()
        # Create table if it doesn't exist (with the current schema)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS molecules (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            molecule TEXT UNIQUE, 
            smiles TEXT, 
            first_seen_block INTEGER, 
            first_seen_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        self.conn.commit()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_molecules_molecule ON molecules (molecule)")
        self.conn.commit()
        self.logger.info(f"Initialized molecule database at {self.db_path}")
    
    def molecule_exists(self, molecule):
        """Check if a molecule already exists in the database."""
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM molecules WHERE molecule = ?", (molecule,))
        return cursor.fetchone() is not None
    
    def add_molecule(self, molecule, smiles, block=0):
        """Add a single molecule entry to the database."""
        self.connect()
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO molecules (molecule, smiles, first_seen_block) VALUES (?, ?, ?)",
                (molecule, smiles, block)
            )
            self.conn.commit()
            self.logger.info(f"Added molecule '{molecule}' with SMILES: {smiles if smiles else 'NULL'}")
            return True
        except sqlite3.IntegrityError:
            self.logger.info(f"Molecule '{molecule}' already exists in the database")
            return False
    
    def add_molecules_batch(self, molecules_data):
        """
        Add multiple molecules to the database in a batch operation.
        
        Args:
            molecules_data: List of dicts with 'molecule_name', 'smiles', and 'block' keys
            
        Returns:
            int: Number of molecules added
        """
        self.connect()
        cursor = self.conn.cursor()
        added_count = 0
        
        for data in molecules_data:
            molecule = data['molecule_name']
            smiles = data.get('smiles', None)  # SMILES can be None
            block = data.get('block', 0)
            
            if not self.molecule_exists(molecule):
                try:
                    cursor.execute(
                        "INSERT INTO molecules (molecule, smiles, first_seen_block) VALUES (?, ?, ?)",
                        (molecule, smiles, block)
                    )
                    added_count += 1
                except sqlite3.IntegrityError:
                    self.logger.info(f"Molecule '{molecule}' already exists in the database (during batch)")
        
        self.conn.commit()
        self.logger.info(f"Added {added_count} new molecule(s) to the archive database")
        return added_count
# ----------------------------------------------------------------------------
# BITTENSOR FUNCTIONS
# ----------------------------------------------------------------------------

async def setup_bittensor(network):
    """Setup Bittensor connection."""
    logger.info(f"Connecting to {network}...")
    try:
        subtensor = bt.async_subtensor(network=network)
        await subtensor.initialize()
        logger.info("Connected to Bittensor")
        return subtensor
    except Exception as e:
        logger.error(f"Error connecting to {network}: {e}")
        traceback.print_exc()
        return None


async def get_subnet_info(subtensor, netuid):
    """Get subnet information including epoch length."""
    try:
        # Get epoch length
        epoch_length = (await subtensor.substrate.query(
            module="SubtensorModule",
            storage_function="Tempo",
            params=[netuid]
        )).value
        
        # Get current block
        current_block = await subtensor.get_current_block()
        current_epoch = current_block // epoch_length
        
        return {
            "netuid": netuid,
            "current_block": current_block,
            "epoch_length": epoch_length,
            "current_epoch": current_epoch,
            "blocks_until_next_epoch": epoch_length - (current_block % epoch_length),
            "last_epoch_boundary": (current_epoch * epoch_length)
        }
    except Exception as e:
        logger.error(f"Error getting subnet info: {e}")
        traceback.print_exc()
        return {
            "netuid": netuid,
            "error": str(e)
        }


async def get_commitments(subtensor, metagraph, block_hash, netuid):
    """
    Retrieve commitments for all miners on a given subnet (netuid) at a specific block.
    """
    try:
        # Gather commitment queries for all validators (hotkeys) concurrently.
        commits_tasks = []
        for hotkey in metagraph.hotkeys:
            task = subtensor.substrate.query(
                module="Commitments",
                storage_function="CommitmentOf",
                params=[netuid, hotkey],
                block_hash=block_hash,
            )
            commits_tasks.append(task)
        
        commits = await asyncio.gather(*commits_tasks)

        # Process the results and build a dictionary with additional metadata.
        result = {}
        for uid, hotkey in enumerate(metagraph.hotkeys):
            commit = commits[uid]
            if commit:
                try:
                    metadata = decode_metadata(commit)
                    if metadata is not None:  # Skip empty commitments
                        result[hotkey] = {
                            'uid': uid,
                            'hotkey': hotkey,
                            'block': commit['block'],
                            'stake': float(metagraph.S[uid]),
                            'data': metadata
                        }
                except Exception as e:
                    logger.error(f"Error decoding metadata for UID {uid}: {e}")
        
        return result
    except Exception as e:
        logger.error(f"Error getting commitments: {e}")
        traceback.print_exc()
        return {}


def decrypt_submissions(current_commitments, headers={"Range": "bytes=0-1024"}):
    """
    Decrypts submissions from validators by:
    1. Fetching encrypted content from GitHub URLs
    2. Decrypting them to get the molecule names
    
    Returns:
        tuple: (decrypted_submissions, submission_stats)
        - decrypted_submissions is a dict mapping UIDs to molecule names
        - submission_stats is a dict with statistics about the process
    """
    # Track statistics
    stats = {
        "total_url_commitments": 0,
        "fetch_success": 0,
        "fetch_404": 0,
        "fetch_other_error": 0,
        "parse_success": 0,
        "parse_error": 0,
        "valid_hash_format": 0,
        "decrypted_success": 0
    }
    
    try:
        # Try to initialize the decryption tool
        from btdr import QuicknetBittensorDrandTimelock
        btd = QuicknetBittensorDrandTimelock()
        
        # Fetch encrypted content from GitHub URLs
        encrypted_submissions = {}
        
        # Count total URL commitments
        for commit in current_commitments.values():
            if '/' in commit['data']:  # Filter only url submissions
                stats["total_url_commitments"] += 1
                
                try:
                    full_url = f"https://raw.githubusercontent.com/{commit['data']}"
                    logger.info(f"Fetching from URL: {full_url}")
                    response = requests.get(full_url, headers=headers, timeout=10)
                    
                    if response.status_code == 404:
                        stats["fetch_404"] += 1
                        logger.error(f"404 error fetching submission: {full_url}")
                        continue
                    elif response.status_code in [200, 206]:
                        stats["fetch_success"] += 1
                        try:
                            # Parse the content
                            encrypted_content = response.content
                            encrypted_content = encrypted_content.decode('utf-8', errors='replace')
                            
                            # Check if it's a hash format (tuple with two elements)
                            try:
                                encrypted_content = literal_eval(encrypted_content)
                                if type(encrypted_content) == tuple and len(encrypted_content) == 2:
                                    stats["valid_hash_format"] += 1
                                    encrypted_submissions[commit['uid']] = (encrypted_content[0], encrypted_content[1])
                                    stats["parse_success"] += 1
                                    logger.info(f"Successfully parsed encrypted content for UID {commit['uid']}")
                                else:
                                    logger.error(f"Invalid format for {commit['uid']}: not a 2-element tuple")
                                    stats["parse_error"] += 1
                            except Exception as e:
                                logger.error(f"Error parsing encrypted content for {commit['uid']}: {e}")
                                stats["parse_error"] += 1
                                
                        except Exception as e:
                            logger.error(f"Error processing content for {commit['uid']}: {e}")
                            stats["parse_error"] += 1
                    else:
                        stats["fetch_other_error"] += 1
                        logger.error(f"Error fetching submission ({response.status_code}): {full_url}")
                except Exception as e:
                    stats["fetch_other_error"] += 1
                    logger.error(f"Error processing URL for UID {commit['uid']}: {e}")
        
        logger.info(f"Fetched {len(encrypted_submissions)} encrypted submissions to decrypt")
        
        # Attempt to decrypt the submissions
        try:
            decrypted_submissions = btd.decrypt_dict(encrypted_submissions)
            stats["decrypted_success"] = len(decrypted_submissions)
            logger.info(f"Successfully decrypted {len(decrypted_submissions)} submissions")
            return decrypted_submissions, stats
        except Exception as e:
            logger.error(f"Error decrypting submissions: {e}")
            traceback.print_exc()
            
            # If timelock not expired or other error, return what we can
            if hasattr(btd, 'decrypt_dict_debug') and callable(getattr(btd, 'decrypt_dict_debug')):
                try:
                    # Some implementations provide a debug method for testing
                    logger.warning("Attempting debug decryption (for testing only)")
                    decrypted = btd.decrypt_dict_debug(encrypted_submissions)
                    stats["decrypted_success"] = len(decrypted)
                    return decrypted, stats
                except Exception as e:
                    logger.error(f"Debug decryption also failed: {e}")
                    
            return {}, stats
    except ImportError as e:
        logger.error(f"Could not import QuicknetBittensorDrandTimelock: {e}")
        return {}, stats
    except Exception as e:
        logger.error(f"Unexpected error in decrypt_submissions: {e}")
        traceback.print_exc()
        return {}, stats


async def get_protein_challenge_updated(subtensor, netuid, block):
    """
    Get the protein challenge using the updated approach from the miner code.
    
    Args:
        subtensor: Subtensor instance
        netuid: Subnet ID
        block: Block number to get challenge for
        
    Returns:
        tuple: (target_proteins, antitarget_proteins)
    """
    try:
        # Get block hash
        block_hash = await subtensor.determine_block_hash(block)
        logger.info(f"Got block hash for block {block}: {block_hash}")
        
        # Use the updated function from miner code
        proteins = get_challenge_proteins_from_blockhash(
            block_hash=block_hash,
            num_targets=1,  # Default values, will be overridden by config
            num_antitargets=4
        )
        
        if not proteins:
            logger.error(f"Failed to get proteins for block {block}")
            return [], []
        
        logger.info(f"Retrieved protein challenge from block hash: {proteins}")
        return proteins["targets"], proteins["antitargets"]
        
    except Exception as e:
        logger.error(f"Error getting protein challenge: {e}")
        traceback.print_exc()
        return [], []
def run_model_on_molecules_batch_safely(psichic_wrapper, protein_sequence, molecules_data, batch_size=100):
    """
    Run the model on molecules in batches with robust error handling.
    
    Args:
        psichic_wrapper: PsichicWrapper instance
        protein_sequence: Protein sequence
        molecules_data: List of dictionaries with 'uid' and 'smiles' keys
        batch_size: Size of each batch for processing
        
    Returns:
        Dictionary mapping UIDs to scores
    """
    # Initialize model for protein sequence
    logger.info(f"Initializing model for protein sequence of length {len(protein_sequence)}")
    try:
        psichic_wrapper.run_challenge_start(protein_sequence)
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        try:
            # Try downloading model weights if missing
            os.system(f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt')} https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt")
            psichic_wrapper.run_challenge_start(protein_sequence)
            logger.info("Model initialized successfully after downloading weights")
        except Exception as e:
            logger.error(f"Error initializing model after download attempt: {e}")
            return {}
    
    # Process molecules in batches
    logger.info(f"Processing {len(molecules_data)} molecules in batches of {batch_size}")
    results = {}
    
    # Group molecules by batch
    for i in range(0, len(molecules_data), batch_size):
        batch = molecules_data[i:i+batch_size]
        batch_uids = [data['uid'] for data in batch]
        batch_smiles = [data['smiles'] for data in batch]
        
        try:
            # Run validation for the batch
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(molecules_data) + batch_size - 1)//batch_size} with {len(batch)} molecules")
            results_df = psichic_wrapper.run_validation(batch_smiles)
            
            if not results_df.empty and 'predicted_binding_affinity' in results_df.columns:
                # Map scores back to UIDs
                for j, uid in enumerate(batch_uids):
                    if j < len(results_df):
                        score = results_df.iloc[j]['predicted_binding_affinity']
                        if score is not None:
                            results[uid] = float(score)
            
            logger.info(f"Completed batch {i//batch_size + 1} with {len(results_df)} results")
            
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
            logger.info("Falling back to individual molecule processing for this batch")
            
            # Process problem batch molecule by molecule
            for j, data in enumerate(batch):
                uid = data['uid']
                smiles = data['smiles']
                
                try:
                    single_result = psichic_wrapper.run_validation([smiles])
                    if not single_result.empty and 'predicted_binding_affinity' in single_result.columns:
                        score = single_result.iloc[0]['predicted_binding_affinity']
                        if score is not None:
                            results[uid] = float(score)
                            logger.info(f"Successfully processed molecule {j+1}/{len(batch)} in fallback mode")
                except Exception as e2:
                    logger.error(f"Error processing UID {uid} in fallback mode: {e2}")
    
    logger.info(f"Completed all batches. Got scores for {len(results)}/{len(molecules_data)} molecules")
    return results


def filter_molecules_by_heavy_atoms(molecules_data, min_heavy_atoms):
    """
    Filter molecules that have at least min_heavy_atoms and valid SMILES.
    
    Args:
        molecules_data: List of dicts with 'uid', 'molecule_name', 'smiles' keys
        min_heavy_atoms: Minimum number of heavy atoms required
        
    Returns:
        List of filtered molecule dicts
    """
    logger.info(f"Filtering molecules for heavy atom count >= {min_heavy_atoms} and valid SMILES")
    start_time = time.time()
    
    valid_molecules = []
    invalid_count = 0
    
    for molecule in molecules_data:
        if not molecule.get('smiles'):
            invalid_count += 1
            continue
            
        smiles = molecule['smiles']
        
        # First check if SMILES is valid
        if not is_valid_smiles(smiles):
            invalid_count += 1
            continue
        
        # Then check heavy atom count
        try:
            atom_count = get_heavy_atom_count(smiles)
            if atom_count >= 26:
                valid_molecules.append(molecule)
            else:
                invalid_count += 1
        except Exception:
            # If get_heavy_atom_count fails, consider the SMILES invalid
            invalid_count += 1
    
    duration = time.time() - start_time
    logger.info(f"Filtered {len(valid_molecules)}/{len(molecules_data)} molecules in {duration:.2f} seconds")
    logger.info(f"Rejected {invalid_count} molecules with invalid SMILES or insufficient heavy atoms")
    
    return valid_molecules


def archive_molecules(results_df, molecule_data, decrypted_submissions, epoch_commitments, smiles_dict, archive_db_path):
    """
    Archive all molecules to the database, including those without SMILES strings.
    
    Args:
        results_df: DataFrame with evaluated molecules
        molecule_data: List of all molecule data collected
        decrypted_submissions: Dictionary of all decrypted submissions
        epoch_commitments: Dictionary of all commitments in the epoch
        smiles_dict: Dictionary of molecule names to SMILES strings
        archive_db_path: Path to the archive database
    """
    logger.info(f"Archiving all committed and extracted molecules to database: {archive_db_path}")
    
    try:
        # Initialize molecule archive database
        molecule_db = MoleculeDB(archive_db_path)
        molecule_db.init_db()
        
        # First, prepare all molecules from decrypted submissions
        all_molecules_to_archive = []
        
        # Add molecules from decrypted submissions (these may or may not have SMILES)
        for uid, molecule_name in decrypted_submissions.items():
            if molecule_name:  # Skip None/empty names
                # Find the corresponding commitment (for in-memory ranking)
                block = 0
                for hk, commit in epoch_commitments.items():
                    if commit['uid'] == uid:
                        block = commit['block']
                        break
                
                # Get SMILES if available, otherwise use None
                smiles = smiles_dict.get(molecule_name, None)
                
                all_molecules_to_archive.append({
                    'molecule_name': molecule_name,
                    'smiles': smiles,
                    'block': block
                })
        
        # Add any additional molecules from results_df that might not be in the above
        if not results_df.empty:
            for _, row in results_df.iterrows():
                if not any(m['molecule_name'] == row['molecule_name'] for m in all_molecules_to_archive):
                    all_molecules_to_archive.append({
                        'molecule_name': row['molecule_name'],
                        'smiles': row['smiles'],
                        'block': row['block']
                    })
        
        # Count molecules with and without SMILES
        with_smiles = len([m for m in all_molecules_to_archive if m['smiles'] is not None])
        without_smiles = len(all_molecules_to_archive) - with_smiles
        
        # Archive the molecules (only storing molecule, smiles, and block)
        added_count = molecule_db.add_molecules_batch(all_molecules_to_archive)
        
        # Clean up connections
        molecule_db.close()
        
        print(f"\nArchived {added_count} new molecules to database: {archive_db_path}")
        print(f" - {with_smiles} molecules with SMILES strings")
        print(f" - {without_smiles} molecules without SMILES strings")
        logger.info(f"Archive complete: {added_count} new molecules added to database")
        
    except Exception as e:
        logger.error(f"Error archiving molecules to database: {e}")
        traceback.print_exc()
        print(f"\nWarning: Failed to archive molecules to database. See log for details.")
# ----------------------------------------------------------------------------
# MAIN VALIDATOR FUNCTIONS
# ----------------------------------------------------------------------------

async def run_validator_check(
    network='finney', 
    netuid=68, 
    my_hotkey=None, 
    hide_molecules=False, 
    epoch=None, 
    archive_db_path="molecule_archive.db",
    min_heavy_atoms=26,
    target_weight=1.0,
    antitarget_weight=0.75,
    use_scraper=True,
    max_concurrent_requests=10
):
    """
    Individual evaluation validator check function with updated protein challenge logic
    and molecule scoring.
    
    Args:
        network: Bittensor network
        netuid: Subnet ID
        my_hotkey: Your hotkey to highlight in results (optional)
        hide_molecules: Whether to hide molecule names in output
        epoch: Specific epoch to check (None for current)
        archive_db_path: Path to the molecule archive database
        min_heavy_atoms: Minimum number of heavy atoms required
        target_weight: Weight for target protein affinity
        antitarget_weight: Weight for antitarget protein affinity
        use_scraper: Whether to use the SMILES scraper to fetch missing SMILES
        max_concurrent_requests: Maximum number of concurrent requests for SMILES scraper
    """
    print("Starting enhanced validator check...")
    logger.info(f"Starting validator check for network={network}, netuid={netuid}, epoch={epoch if epoch is not None else 'current'}")
    start_time = time.time()
    
    try:
        # Import PSICHIC here to allow for potential missing dependency
        print("Importing PSICHIC...")
        try:
            from PSICHIC.wrapper import PsichicWrapper
            psichic_wrapper = PsichicWrapper()
            logger.info("PSICHIC imported successfully")
            print("PSICHIC imported successfully.")
        except ImportError as e:
            logger.error(f"Error importing PSICHIC: {e}")
            print(f"Error: Could not import PSICHIC. Make sure it's installed correctly.")
            return False
        
        # Setup Bittensor
        print("Setting up Bittensor connection...")
        subtensor = await setup_bittensor(network)
        if not subtensor:
            logger.error("Failed to connect to Bittensor")
            print("Error: Failed to connect to Bittensor")
            return False
        
        # Get subnet info
        print("Getting subnet info...")
        subnet_info = await get_subnet_info(subtensor, netuid)
        if 'error' in subnet_info:
            logger.error(f"Error getting subnet info: {subnet_info['error']}")
            print(f"Error getting subnet info: {subnet_info['error']}")
            return False
        
        logger.info(f"Current block: {subnet_info['current_block']}")
        logger.info(f"Current epoch: {subnet_info['current_epoch']}")
        logger.info(f"Epoch length: {subnet_info['epoch_length']}")
        
        # Determine which epoch to check
        if epoch is None:
            target_epoch = subnet_info['current_epoch']
        else:
            target_epoch = epoch
            if target_epoch > subnet_info['current_epoch']:
                logger.error(f"Requested epoch {target_epoch} is in the future")
                print(f"Error: Requested epoch {target_epoch} is in the future")
                return False
        
        # Calculate epoch boundary block
        epoch_boundary_block = target_epoch * subnet_info['epoch_length']
        logger.info(f"Checking epoch: {target_epoch} (boundary block: {epoch_boundary_block})")
        print(f"Checking epoch: {target_epoch} (boundary block: {epoch_boundary_block})")
        
        # Get metagraph at the epoch boundary
        print("Getting metagraph...")
        logger.info(f"Getting metagraph at epoch boundary (block {epoch_boundary_block})...")
        try:
            metagraph = await subtensor.metagraph(netuid=netuid, block=epoch_boundary_block)
            logger.info(f"Metagraph loaded with {len(metagraph.hotkeys)} hotkeys")
            print(f"Metagraph loaded with {len(metagraph.hotkeys)} hotkeys")
        except Exception as e:
            logger.error(f"Error getting metagraph at block {epoch_boundary_block}: {e}")
            logger.warning("Trying with current metagraph instead...")
            print("Error getting metagraph. Trying with current metagraph instead...")
            metagraph = await subtensor.metagraph(netuid=netuid)
            logger.info(f"Current metagraph loaded with {len(metagraph.hotkeys)} hotkeys")
            print(f"Current metagraph loaded with {len(metagraph.hotkeys)} hotkeys")
        
        # Get protein challenge for the epoch using the new approach
        print("Getting protein challenge...")
        logger.info(f"Getting protein challenge for epoch {target_epoch}...")
        target_proteins, antitarget_proteins = await get_protein_challenge_updated(
            subtensor,
            netuid,
            epoch_boundary_block
        )
        
        if not target_proteins or not antitarget_proteins:
            logger.error(f"Failed to get protein challenge for epoch {target_epoch}")
            print(f"Error: Failed to get protein challenge for epoch {target_epoch}")
            return False
        
        logger.info(f"Target proteins: {target_proteins}")
        logger.info(f"Antitarget proteins: {antitarget_proteins}")
        print(f"Target proteins: {target_proteins}")
        print(f"Antitarget proteins: {antitarget_proteins}")
        
        # Log scoring parameters
        logger.info(f"Using scoring parameters - min_heavy_atoms: {min_heavy_atoms}, target_weight: {target_weight}, antitarget_weight: {antitarget_weight}")
        print(f"Using scoring parameters - min_heavy_atoms: {min_heavy_atoms}, target_weight: {target_weight}, antitarget_weight: {antitarget_weight}")
        
        # Get protein sequences
        print("Getting protein sequences...")
        target_sequences = []
        for target in target_proteins:
            seq = get_sequence_from_protein_code(target)
            if seq:
                target_sequences.append(seq)
                logger.info(f"Target {target} sequence length: {len(seq)}")
                print(f"Target {target} sequence length: {len(seq)}")
            else:
                logger.error(f"Failed to get sequence for target {target}")
        
        antitarget_sequences = []
        for antitarget in antitarget_proteins:
            seq = get_sequence_from_protein_code(antitarget)
            if seq:
                antitarget_sequences.append(seq)
                logger.info(f"Antitarget {antitarget} sequence length: {len(seq)}")
                print(f"Antitarget {antitarget} sequence length: {len(seq)}")
            else:
                logger.error(f"Failed to get sequence for antitarget {antitarget}")
        
        if not target_sequences or not antitarget_sequences:
            logger.error("Failed to get sequences for targets or antitargets")
            print("Error: Failed to get sequences for targets or antitargets")
            return False
        
        # Get current block hash (or epoch boundary block hash for past epochs)
        print("Getting block hash...")
        block_to_check = min(subnet_info['current_block'], (target_epoch + 1) * subnet_info['epoch_length'] - 1)
        logger.info(f"Getting block hash for block {block_to_check}...")
        block_hash = await subtensor.determine_block_hash(block_to_check)
        
        # Get all commitments
        print("Getting commitments...")
        logger.info("Getting commitments...")
        current_commitments = await get_commitments(
            subtensor, 
            metagraph, 
            block_hash, 
            netuid
        )
        
        if not current_commitments:
            logger.error("No commitments found")
            print("Error: No commitments found")
            return False
        
        # Filter commitments to those from the target epoch
        print("Filtering commitments for the target epoch...")
        epoch_commitments = {
            hotkey: commit for hotkey, commit in current_commitments.items()
            if commit['block'] >= epoch_boundary_block and commit['block'] < (target_epoch + 1) * subnet_info['epoch_length']
        }
        logger.info(f"Found {len(epoch_commitments)} commitments from epoch {target_epoch}")
        print(f"Found {len(epoch_commitments)} commitments from epoch {target_epoch}")
        
        # Decrypt submissions from GitHub URLs and get stats
        print("Decrypting submissions...")
        logger.info("Decrypting submissions...")
        decrypted_submissions, submission_stats = decrypt_submissions(epoch_commitments)
        logger.info(f"Decrypted {len(decrypted_submissions)} submissions")
        print(f"Decrypted {len(decrypted_submissions)} submissions")
        
        # STEP 1: Collect all molecule names and look up SMILES in batch
        print("Collecting molecule names and looking up SMILES...")
        logger.info("Collecting molecule names and looking up SMILES in batch...")
        
        # Create list to track molecule-to-miner mapping (retain uid and hotkey for ranking)
        molecule_data = []
        molecule_names = []
        
        for hotkey, commit in epoch_commitments.items():
            uid = commit['uid']
            if uid in decrypted_submissions:
                molecule_name = decrypted_submissions[uid]
                if molecule_name:  # Skip None/empty names
                    molecule_names.append(molecule_name)
                    molecule_data.append({
                        'uid': uid,
                        'hotkey': hotkey,
                        'molecule_name': molecule_name,
                        'url_path': commit['data'],
                        'block': commit['block'],
                        'stake': commit['stake']
                    })
        
        logger.info(f"Collected {len(molecule_names)} molecule names for batch lookup")
        print(f"Collected {len(molecule_names)} molecule names for batch lookup")
        
        # Look up SMILES in SAVI database first
        print("Looking up SMILES in SAVI database...")
        savi = SAVILookup("savi_lookup.db")
        savi_smiles_dict = savi.lookup_smiles_batch(molecule_names)
        savi_found_count = len(savi_smiles_dict)
        logger.info(f"Found {savi_found_count}/{len(molecule_names)} molecules in SAVI database ({savi_found_count/len(molecule_names)*100:.1f}% if any)")
        print(f"Found {savi_found_count}/{len(molecule_names)} molecules in SAVI database ({savi_found_count/len(molecule_names)*100:.1f}% if any)")
        
        # For molecules not found in SAVI, use NOVA dashboard scraper
        missing_molecules = [m for m in molecule_names if m not in savi_smiles_dict]
        nova_found_count = 0
        
        if missing_molecules and use_scraper:
            print(f"Using NOVA dashboard SMILES scraper for {len(missing_molecules)} molecules...")
            logger.info(f"Using NOVA dashboard SMILES scraper for {len(missing_molecules)} missing molecules...")
            
            # Fetch SMILES for missing molecules using the scraper
            nova_smiles_dict = fetch_smiles_batch(missing_molecules, max_concurrent_requests)
            
            # Count how many molecules have non-empty SMILES from scraper
            nova_found_count = sum(1 for smiles in nova_smiles_dict.values() if smiles)
            logger.info(f"Found additional {nova_found_count}/{len(missing_molecules)} molecules via NOVA dashboard scraper")
            print(f"Found additional {nova_found_count}/{len(missing_molecules)} molecules via NOVA dashboard scraper")
            
            # Combine SAVI and NOVA results
            smiles_dict = {**savi_smiles_dict, **nova_smiles_dict}
        else:
            # Just use SAVI results
            smiles_dict = savi_smiles_dict
        
        # Add SMILES to molecule data
        total_smiles_found = len([v for v in smiles_dict.values() if v])
        logger.info(f"Total molecules with SMILES strings: {total_smiles_found}/{len(molecule_names)} ({total_smiles_found/len(molecule_names)*100:.1f}%)")
        print(f"Total molecules with SMILES strings: {total_smiles_found}/{len(molecule_names)} ({total_smiles_found/len(molecule_names)*100:.1f}%)")
        
        # STEP 2: Prepare for model evaluation
        print("Preparing molecules for evaluation...")
        logger.info("Preparing molecules for evaluation...")
        
        # Create a list of molecules with both name and SMILES
        valid_molecules_raw = []
        for data in molecule_data:
            molecule_name = data['molecule_name']
            if molecule_name in smiles_dict and smiles_dict[molecule_name]:
                data['smiles'] = smiles_dict[molecule_name]
                valid_molecules_raw.append(data)
        
        # Filter molecules by heavy atom count
        valid_molecules = filter_molecules_by_heavy_atoms(valid_molecules_raw, min_heavy_atoms)
        
        logger.info(f"Found {len(valid_molecules)}/{len(valid_molecules_raw)} molecules with valid SMILES and sufficient heavy atoms")
        print(f"Found {len(valid_molecules)}/{len(valid_molecules_raw)} molecules with valid SMILES and sufficient heavy atoms")
        
        # Early exit if no valid molecules
        if not valid_molecules:
            logger.warning("No molecules with valid SMILES and sufficient heavy atoms found")
            print("No molecules with valid SMILES and sufficient heavy atoms found for evaluation")
            
            # Still archive all available molecules
            print("Archiving all molecules with or without SMILES...")
            empty_df = pd.DataFrame()
            archive_molecules(empty_df, molecule_data, decrypted_submissions, epoch_commitments, smiles_dict, archive_db_path)
            return True
# STEP 3: Run model on each molecule for all target proteins
        print("Running model for target proteins...")
        logger.info("Running model for target proteins...")
        
        all_target_scores = {}
        for i, target_sequence in enumerate(target_sequences):
            target = target_proteins[i]
            print(f"Evaluating against target protein {target}...")
            
            target_scores = run_model_on_molecules_batch_safely(
                psichic_wrapper, 
                target_sequence,
                valid_molecules
            )
            
            # Store scores for each molecule
            for uid, score in target_scores.items():
                if uid not in all_target_scores:
                    all_target_scores[uid] = []
                all_target_scores[uid].append(score)
        
        # STEP 4: Run model on each molecule for all antitarget proteins
        print("Running model for antitarget proteins...")
        logger.info("Running model for antitarget proteins...")
        
        all_antitarget_scores = {}
        for i, antitarget_sequence in enumerate(antitarget_sequences):
            antitarget = antitarget_proteins[i]
            print(f"Evaluating against antitarget protein {antitarget}...")
            
            antitarget_scores = run_model_on_molecules_batch_safely(
                psichic_wrapper, 
                antitarget_sequence,
                valid_molecules
            )
            
            # Store scores for each molecule
            for uid, score in antitarget_scores.items():
                if uid not in all_antitarget_scores:
                    all_antitarget_scores[uid] = []
                all_antitarget_scores[uid].append(score)
        
        # STEP 5: Combine results and calculate weighted difference scores
        print("Calculating weighted difference scores...")
        logger.info("Calculating weighted difference scores...")
        results = []
        
        for data in valid_molecules:
            uid = data['uid']
            if uid in all_target_scores and uid in all_antitarget_scores:
                # Calculate average scores across all targets and antitargets
                target_score_avg = sum(all_target_scores[uid]) / len(all_target_scores[uid])
                antitarget_score_avg = sum(all_antitarget_scores[uid]) / len(all_antitarget_scores[uid])
                
                # Calculate weighted difference score
                weighted_diff_score = (target_weight * target_score_avg) - (antitarget_weight * antitarget_score_avg)
                
                results.append({
                    'uid': uid,
                    'hotkey': data['hotkey'],
                    'molecule_name': data['molecule_name'],
                    'url_path': data['url_path'],
                    'smiles': data['smiles'],
                    'target_score': target_score_avg,
                    'antitarget_score': antitarget_score_avg,
                    'weighted_diff_score': weighted_diff_score,
                    'block': data['block'],
                    'stake': data['stake']
                })
                
                logger.info(f"Calculated scores for UID {uid}: target={target_score_avg:.4f}, antitarget={antitarget_score_avg:.4f}, weighted_diff={weighted_diff_score:.4f}")
        
        # Convert to DataFrame
        print("Converting results to DataFrame...")
        if results:
            results_df = pd.DataFrame(results)
            
            # Calculate submission order within block
            print("Calculating submission order within blocks...")
            block_groups = results_df.groupby('block')
            results_df['order_in_block'] = 0
            
            for block, group in block_groups:
                sorted_indices = group.sort_values('uid').index
                for i, idx in enumerate(sorted_indices):
                    results_df.loc[idx, 'order_in_block'] = i
            
            # Calculate final score based on weighted_diff_score and block order
            print("Calculating final scores...")
            block_penalty = 0.001
            results_df['final_score'] = results_df['weighted_diff_score'] - (results_df['order_in_block'] * block_penalty)
            
            results_df = results_df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
            
            logger.info(f"Evaluated {len(results_df)} molecules")
            print(f"Evaluated {len(results_df)} molecules")
            
            # Save to CSV
            print("Saving results to CSV...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"validator_results_epoch{target_epoch}_{timestamp}.csv"
            results_df.to_csv(csv_filename, index=False)
            logger.info(f"Saved results to {csv_filename}")
            print(f"Saved results to {csv_filename}")
            
            # Print submission summary
            print("\n" + "="*80)
            print("SUBMISSION SUMMARY")
            print("="*80)
            print(f"Epoch: {target_epoch}")
            print(f"Epoch boundary block: {epoch_boundary_block}")
            print(f"Total commitments in epoch: {len(epoch_commitments)}")
            print(f"Total GitHub URL submissions: {submission_stats['total_url_commitments']}")
            print(f"  - Successful fetches: {submission_stats['fetch_success']}")
            print(f"  - 404 errors: {submission_stats['fetch_404']}")
            print(f"  - Other fetch errors: {submission_stats['fetch_other_error']}")
            print(f"  - Valid hash format: {submission_stats['valid_hash_format']}")
            print(f"  - Successfully decrypted: {submission_stats['decrypted_success']}")
            print(f"Molecules found in SAVI database: {savi_found_count}/{len(molecule_names)}")
            if use_scraper:
                print(f"Additional molecules found via NOVA scraper: {nova_found_count}/{len(missing_molecules)}")
            print(f"Total molecules with SMILES: {total_smiles_found}/{len(molecule_names)}")
            print(f"Molecules with sufficient heavy atoms: {len(valid_molecules)}/{len(valid_molecules_raw)}")
            print(f"Molecules successfully evaluated: {len(results)}")
            print(f"Evaluation parameters: min_heavy_atoms={min_heavy_atoms}, target_weight={target_weight}, antitarget_weight={antitarget_weight}")
            print("="*80)
# Print ranking results
            print(f"\n=== Validator Check: Epoch {target_epoch} Ranking ===")
            print(f"Target Proteins: {', '.join(target_proteins)}")
            print(f"Antitarget Proteins: {', '.join(antitarget_proteins)}")
            print(f"Total Molecules Evaluated: {len(results_df)}")
            
            if hide_molecules:
                print("-" * 110)
                print(f"{'Rank':<5} {'UID':<5} {'Hotkey':<12} {'Weighted Score':<15} {'Target':<10} {'Anti':<10} {'Block':<8} {'Order':<5}")
                print("-" * 110)
            else:
                print("-" * 140)
                print(f"{'Rank':<5} {'UID':<5} {'Molecule Name':<45} {'Hotkey':<12} {'Weighted Score':<15} {'Target':<10} {'Anti':<10} {'Block':<8} {'Order':<5}")
                print("-" * 140)
            
            top_n = min(10, len(results_df))
            for i in range(top_n):
                row = results_df.iloc[i]
                is_my_hotkey = my_hotkey and row['hotkey'].startswith(my_hotkey)
                prefix = " *> " if is_my_hotkey else "   "
                
                display_hotkey = row['hotkey'][:6] + "..." + row['hotkey'][-3:] if len(row['hotkey']) > 12 else row['hotkey']
                
                if hide_molecules:
                    print(f"{prefix}{i+1:<3} {row['uid']:<5} {display_hotkey:<12} {row['weighted_diff_score']:<15.4f} "
                          f"{row['target_score']:<10.4f} {row['antitarget_score']:<10.4f} {row['block']:<8} {int(row['order_in_block']):<5}")
                else:
                    display_molecule = row['molecule_name']
                    if len(display_molecule) > 45:
                        display_molecule = display_molecule[:42] + "..."
                    
                    print(f"{prefix}{i+1:<3} {row['uid']:<5} {display_molecule:<45} {display_hotkey:<12} {row['weighted_diff_score']:<15.4f} "
                          f"{row['target_score']:<10.4f} {row['antitarget_score']:<10.4f} {row['block']:<8} {int(row['order_in_block']):<5}")
            
            if hide_molecules:
                print("-" * 110)
            else:
                print("-" * 140)
            
            if my_hotkey:
                my_rows = results_df[results_df['hotkey'].str.startswith(my_hotkey)]
                if not my_rows.empty:
                    my_rank = my_rows.index[0] + 1
                    my_row = my_rows.iloc[0]
                    print(f"\nYour Position:")
                    print(f"Rank: {my_rank}/{len(results_df)}")
                    print(f"UID: {my_row['uid']}")
                    print(f"Molecule: {my_row['molecule_name']}")
                    print(f"Weighted Diff Score: {my_row['weighted_diff_score']:.4f}")
                    print(f"Target Score: {my_row['target_score']:.4f}")
                    print(f"Antitarget Score: {my_row['antitarget_score']:.4f}")
                    print(f"Final Score (with block order): {my_row['final_score']:.4f}")
                    print(f"Block: {my_row['block']}")
                else:
                    print(f"\nYour hotkey {my_hotkey} not found in epoch {target_epoch} results")
            
            print(f"\nResults saved to: {csv_filename}")
            
            # Archive all molecules to the database (only storing columns that exist)
            print("Archiving all molecules to database (including those without SMILES)...")
            archive_molecules(results_df, molecule_data, decrypted_submissions, epoch_commitments, smiles_dict, archive_db_path)
            
        else:
            logger.warning("No molecules evaluated")
            print(f"\n=== Validator Check Results for Epoch {target_epoch} ===")
            print("No molecules were successfully evaluated")
            print(f"Total decrypted submissions: {len(decrypted_submissions)}")
            print("Archiving all molecules with or without SMILES...")
            empty_df = pd.DataFrame()
            archive_molecules(empty_df, molecule_data, decrypted_submissions, epoch_commitments, smiles_dict, archive_db_path)
        
        logger.info(f"Validator check completed in {time.time() - start_time:.2f} seconds")
        print(f"Validator check completed in {time.time() - start_time:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Error in validator check: {e}")
        print(f"ERROR in validator check: {e}")
        traceback.print_exc()
        return False
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate current or previous epoch")
    parser.add_argument("--network", type=str, default="finney",
                        help="Bittensor network (default: finney)")
    parser.add_argument("--netuid", type=int, default=68,
                        help="Subnet ID (default: 68)")
    parser.add_argument("--my-hotkey", type=str, default=None,
                       help="Your hotkey to highlight in results (default: None)")
    parser.add_argument("--hide-molecules", action="store_true",
                       help="Hide molecule names in output for cleaner display")
    epoch_group = parser.add_mutually_exclusive_group()
    epoch_group.add_argument("--epoch", type=int, default=None,
                      help="Specific epoch to check (default: current epoch)")
    epoch_group.add_argument("--previous", action="store_true",
                      help="Check the previous epoch instead of the current one")
    parser.add_argument("--archive-db", type=str, default="molecule_archive.db",
                       help="Path to molecule archive database (default: molecule_archive.db)")
    
    # New arguments for the updated validator
    parser.add_argument("--min-heavy-atoms", type=int, default=26,
                        help="Minimum number of heavy atoms for molecules (default: 24)")
    parser.add_argument("--target-weight", type=float, default=1.0,
                        help="Weight for target protein affinity scores (default: 1.0)")
    parser.add_argument("--antitarget-weight", type=float, default=0.75,
                        help="Weight for antitarget protein affinity scores (default: 1.0)")
    
    # SMILES scraper related arguments
    parser.add_argument("--no-scraper", action="store_true",
                       help="Disable using NOVA dashboard SMILES scraper for missing molecules")
    parser.add_argument("--max-concurrent", type=int, default=10,
                        help="Maximum number of concurrent requests for SMILES scraper (default: 10)")
    
    return parser.parse_args()


async def main_async():
    """Async main function."""
    print("\n" + "="*80)
    print("Enhanced Validator Dashboard Starting")
    print("Using updated protein challenge retrieval and weighted scoring")
    print("With integrated NOVA SMILES scraper for missing molecules")
    print("="*80 + "\n")
    
    args = parse_args()
    
    print("Connecting to Bittensor...")
    subtensor = await setup_bittensor(args.network)
    if not subtensor:
        print("Error: Failed to connect to Bittensor")
        return
        
    print("Getting subnet info...")
    subnet_info = await get_subnet_info(subtensor, args.netuid)
    if 'error' in subnet_info:
        print(f"Error getting subnet info: {subnet_info['error']}")
        return
    
    print(f"Current block: {subnet_info['current_block']}")
    print(f"Current epoch: {subnet_info['current_epoch']}")
    
    target_epoch = None
    if args.previous:
        target_epoch = subnet_info['current_epoch'] - 1
        if target_epoch < 0:
            print("Error: Cannot check epoch before genesis")
            return
    elif args.epoch is not None:
        target_epoch = args.epoch
        if target_epoch > subnet_info['current_epoch']:
            print(f"Error: Requested epoch {target_epoch} is in the future")
            return
        if target_epoch < 0:
            print("Error: Cannot check epoch before genesis")
            return
    
    print("\n" + "="*80)
    print("Validator Configuration")
    if args.previous:
        print(f"Checking previous epoch: {target_epoch}")
    elif target_epoch is not None:
        print(f"Checking epoch: {target_epoch}")
    else:
        print("Checking current epoch")
    print(f"{'Molecule names are hidden' if args.hide_molecules else 'Showing molecule names'}")
    print(f"Archiving molecules to: {args.archive_db}")
    print(f"Min heavy atoms: {args.min_heavy_atoms}")
    print(f"Target weight: {args.target_weight}")
    print(f"Antitarget weight: {args.antitarget_weight}")
    print(f"NOVA SMILES scraper: {'Disabled' if args.no_scraper else 'Enabled'}")
    print(f"Max concurrent requests: {args.max_concurrent}")
    print("="*80)
    
    print("Calling run_validator_check...")
    success = await run_validator_check(
        network=args.network,
        netuid=args.netuid,
        my_hotkey=args.my_hotkey,
        hide_molecules=args.hide_molecules,
        epoch=target_epoch,
        archive_db_path=args.archive_db,
        min_heavy_atoms=args.min_heavy_atoms,
        target_weight=args.target_weight,
        antitarget_weight=args.antitarget_weight,
        use_scraper=not args.no_scraper,
        max_concurrent_requests=args.max_concurrent
    )
    
    if success:
        print("\nValidator check completed successfully!")
    else:
        print("\nValidator check failed. See log for details.")


def main():
    """Main function."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nOperation canceled by user")
    except Exception as e:
        print(f"\nFatal error in main: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
