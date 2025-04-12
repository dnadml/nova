import os
import sys
import time
import asyncio
import logging
import argparse
import traceback
import yaml
from ast import literal_eval
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple, Any, Union
import numpy as np

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

# Import utilities
from my_utils import (
    get_smiles, 
    get_sequence_from_protein_code, 
    get_heavy_atom_count, 
    get_challenge_proteins_from_blockhash,
    upload_file_to_github,
    compute_maccs_entropy
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
# CONFIGURATION FUNCTIONS
# ----------------------------------------------------------------------------

def load_config(path: str = "config/config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to the YAML config file
        
    Returns:
        Dictionary with configuration settings
    """
    if not os.path.exists(path):
        logger.warning(f"Config file not found at {path}. Using default values.")
        return {
            'weekly_target': None,
            'num_antitargets': 4,
            'no_submission_blocks': 10,
            'antitarget_weight': 0.75,
            'min_heavy_atoms': 20,
            'min_rotatable_bonds': 1,
            'max_rotatable_bonds': 10,
            'num_molecules': 10,
            'entropy_weight': 1.0,
            'entropy_bonus_threshold': 0,
            'molecule_repetition_weight': 1.0,
            'molecule_repetition_threshold': 0
        }
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # Load configuration options
        weekly_target = config["protein_selection"]["weekly_target"]
        num_antitargets = config["protein_selection"]["num_antitargets"]
        no_submission_blocks = config["competition"]["no_submission_blocks"]
        
        validation_config = config["molecule_validation"]
        antitarget_weight = validation_config["antitarget_weight"]
        min_heavy_atoms = validation_config["min_heavy_atoms"]
        min_rotatable_bonds = validation_config["min_rotatable_bonds"]
        max_rotatable_bonds = validation_config["max_rotatable_bonds"]
        num_molecules = validation_config["num_molecules"]
        entropy_weight = validation_config["entropy_weight"]
        entropy_bonus_threshold = validation_config["entropy_bonus_threshold"]
        molecule_repetition_weight = validation_config["molecule_repetition_weight"]
        molecule_repetition_threshold = validation_config["molecule_repetition_threshold"]
        
        return {
            'weekly_target': weekly_target,
            'num_antitargets': num_antitargets,
            'no_submission_blocks': no_submission_blocks,
            'antitarget_weight': antitarget_weight,
            'min_heavy_atoms': min_heavy_atoms,
            'min_rotatable_bonds': min_rotatable_bonds,
            'max_rotatable_bonds': max_rotatable_bonds,
            'num_molecules': num_molecules,
            'entropy_weight': entropy_weight,
            'entropy_bonus_threshold': entropy_bonus_threshold,
            'molecule_repetition_weight': molecule_repetition_weight,
            'molecule_repetition_threshold': molecule_repetition_threshold
        }
    except Exception as e:
        logger.error(f"Error loading config from {path}: {e}")
        logger.warning("Using default values instead")
        return {
            'weekly_target': None,
            'num_antitargets': 4,
            'no_submission_blocks': 10,
            'antitarget_weight': 0.75,
            'min_heavy_atoms': 20,
            'min_rotatable_bonds': 1,
            'max_rotatable_bonds': 10,
            'num_molecules': 10,
            'entropy_weight': 1.0,
            'entropy_bonus_threshold': 0,
            'molecule_repetition_weight': 1.0,
            'molecule_repetition_threshold': 0
        }

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

def is_valid_smiles(smiles_str: str) -> bool:
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


def validate_molecule(molecule_name: str, smiles: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a molecule against the competition requirements.
    
    Args:
        molecule_name: The name of the molecule
        smiles: The SMILES string for the molecule
        config: Configuration dictionary with validation parameters
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "molecule_name": molecule_name,
        "smiles": smiles,
        "is_valid": False,
        "heavy_atoms": 0,
        "rotatable_bonds": 0,
        "error": None
    }
    
    if not smiles:
        result["error"] = "Missing SMILES string"
        return result
    
    if not is_valid_smiles(smiles):
        result["error"] = "Invalid SMILES format"
        return result
    
    try:
        # Check heavy atom count
        heavy_atoms = get_heavy_atom_count(smiles)
        result["heavy_atoms"] = heavy_atoms
        
        if heavy_atoms < config["min_heavy_atoms"]:
            result["error"] = f"Insufficient heavy atoms: {heavy_atoms} (min: {config['min_heavy_atoms']})"
            return result
        
        # Import RDKit here to avoid dependency issues if someone doesn't have it
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                result["error"] = "Could not parse with RDKit"
                return result
                
            # Check rotatable bonds
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            result["rotatable_bonds"] = rotatable_bonds
            
            if rotatable_bonds < config["min_rotatable_bonds"]:
                result["error"] = f"Too few rotatable bonds: {rotatable_bonds} (min: {config['min_rotatable_bonds']})"
                return result
                
            if rotatable_bonds > config["max_rotatable_bonds"]:
                result["error"] = f"Too many rotatable bonds: {rotatable_bonds} (max: {config['max_rotatable_bonds']})"
                return result
                
            # If we get here, the molecule is valid
            result["is_valid"] = True
            return result
            
        except ImportError:
            # If RDKit is not available, we'll skip the rotatable bond check
            logger.warning("RDKit not available, skipping rotatable bond validation")
            result["is_valid"] = True
            return result
            
    except Exception as e:
        result["error"] = f"Validation error: {str(e)}"
        return result

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


class SubmissionDB:
    """Database for storing miner submissions with multiple molecules."""
    
    def __init__(self, db_path="submissions_archive.db"):
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
        """Initialize the submissions tables."""
        self.connect()
        cursor = self.conn.cursor()
        
        # Create submissions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch INTEGER,
            uid INTEGER,
            hotkey TEXT,
            url_path TEXT,
            block INTEGER,
            stake REAL,
            entropy REAL,
            final_score REAL,
            submission_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        
        # Create submission_molecules table (for the many-to-many relationship)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS submission_molecules (
            submission_id INTEGER,
            molecule_name TEXT,
            smiles TEXT,
            target_score REAL,
            antitarget_score REAL,
            weighted_score REAL,
            primary_idx INTEGER,  -- Position in the submission (0-9)
            FOREIGN KEY (submission_id) REFERENCES submissions (id),
            PRIMARY KEY (submission_id, molecule_name)
        )""")
        
        self.conn.commit()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_submissions_epoch ON submissions (epoch)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_submissions_uid ON submissions (uid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_submissions_hotkey ON submissions (hotkey)")
        self.conn.commit()
        
        self.logger.info(f"Initialized submissions database at {self.db_path}")
    
    def submission_exists(self, epoch, uid):
        """Check if a submission already exists for this epoch and UID."""
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM submissions WHERE epoch = ? AND uid = ?", (epoch, uid))
        return cursor.fetchone() is not None
    
    def add_submission(self, submission_data, molecules_data):
        """
        Add a submission with its molecules to the database.
        
        Args:
            submission_data: Dict with submission metadata (epoch, uid, hotkey, etc.)
            molecules_data: List of dicts with molecule data
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        self.connect()
        
        # Skip if this submission already exists
        if self.submission_exists(submission_data['epoch'], submission_data['uid']):
            self.logger.info(f"Submission for epoch {submission_data['epoch']}, UID {submission_data['uid']} already exists")
            return False
        
        cursor = self.conn.cursor()
        try:
            # Insert submission record
            cursor.execute(
                """
                INSERT INTO submissions 
                (epoch, uid, hotkey, url_path, block, stake, entropy, final_score) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    submission_data['epoch'],
                    submission_data['uid'],
                    submission_data['hotkey'],
                    submission_data.get('url_path', ''),
                    submission_data['block'],
                    submission_data.get('stake', 0.0),
                    submission_data.get('entropy', None),
                    submission_data.get('final_score', None)
                )
            )
            
            submission_id = cursor.lastrowid
            
            # Insert molecule records
            for idx, mol_data in enumerate(molecules_data):
                cursor.execute(
                    """
                    INSERT INTO submission_molecules
                    (submission_id, molecule_name, smiles, target_score, antitarget_score, weighted_score, primary_idx)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        submission_id,
                        mol_data['molecule_name'],
                        mol_data.get('smiles', None),
                        mol_data.get('target_score', None),
                        mol_data.get('antitarget_score', None),
                        mol_data.get('weighted_score', None),
                        idx
                    )
                )
            
            self.conn.commit()
            self.logger.info(f"Added submission for epoch {submission_data['epoch']}, UID {submission_data['uid']} with {len(molecules_data)} molecules")
            return True
            
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error adding submission: {e}")
            traceback.print_exc()
            return False
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
        - decrypted_submissions is a dict mapping UIDs to lists of molecule names
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
            raw_decrypted = btd.decrypt_dict(encrypted_submissions)
            stats["decrypted_success"] = len(raw_decrypted)
            logger.info(f"Successfully decrypted {len(raw_decrypted)} submissions")
            
            # NEW: Parse the decryption results to get lists of molecules
            # Each decrypted item is now a comma-separated list of molecules
            decrypted_submissions = {}
            for uid, decryption in raw_decrypted.items():
                if decryption:
                    # Split the comma-separated list
                    molecules = decryption.split(',')
                    molecules = [m.strip() for m in molecules if m.strip()]
                    
                    if molecules:
                        decrypted_submissions[uid] = molecules
                        logger.info(f"UID {uid}: Decrypted {len(molecules)} molecules")
                    else:
                        logger.warning(f"UID {uid}: Decrypted submission contained no valid molecules")
            
            return decrypted_submissions, stats
            
        except Exception as e:
            logger.error(f"Error decrypting submissions: {e}")
            traceback.print_exc()
            
            # If timelock not expired or other error, return what we can
            if hasattr(btd, 'decrypt_dict_debug') and callable(getattr(btd, 'decrypt_dict_debug')):
                try:
                    # Some implementations provide a debug method for testing
                    logger.warning("Attempting debug decryption (for testing only)")
                    raw_decrypted = btd.decrypt_dict_debug(encrypted_submissions)
                    
                    # Process debug decryption results the same way
                    decrypted_submissions = {}
                    for uid, decryption in raw_decrypted.items():
                        if decryption:
                            molecules = decryption.split(',')
                            molecules = [m.strip() for m in molecules if m.strip()]
                            
                            if molecules:
                                decrypted_submissions[uid] = molecules
                                logger.info(f"UID {uid}: Debug decrypted {len(molecules)} molecules")
                    
                    stats["decrypted_success"] = len(decrypted_submissions)
                    return decrypted_submissions, stats
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


def calculate_molecule_name_counts(valid_molecules_by_uid):
    """
    Count occurrences of each molecule name across all valid submissions.
    
    Args:
        valid_molecules_by_uid: Dict mapping UIDs to their valid molecule lists
        
    Returns:
        Dict mapping molecule names to their occurrence count
    """
    name_counts = {}
    
    for uid, molecules_data in valid_molecules_by_uid.items():
        for mol_data in molecules_data:
            molecule_name = mol_data['molecule_name']
            name_counts[molecule_name] = name_counts.get(molecule_name, 0) + 1
    
    return name_counts


def run_model_on_molecules_batch_safely(psichic_wrapper, protein_sequence, molecules_data, batch_size=100):
    """
    Run the model on molecules in batches with robust error handling.
    
    Args:
        psichic_wrapper: PsichicWrapper instance
        protein_sequence: Protein sequence
        molecules_data: List of dictionaries with molecule info
        batch_size: Size of each batch for processing
        
    Returns:
        Dictionary mapping indices to scores
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
        batch_indices = list(range(i, min(i+batch_size, len(molecules_data))))
        batch_smiles = [data['smiles'] for data in batch]
        
        try:
            # Run validation for the batch
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(molecules_data) + batch_size - 1)//batch_size} with {len(batch)} molecules")
            results_df = psichic_wrapper.run_validation(batch_smiles)
            
            if not results_df.empty and 'predicted_binding_affinity' in results_df.columns:
                # Map scores back to indices
                for j, idx in enumerate(batch_indices):
                    if j < len(results_df):
                        score = results_df.iloc[j]['predicted_binding_affinity']
                        if score is not None:
                            results[idx] = float(score)
            
            logger.info(f"Completed batch {i//batch_size + 1} with {len(results_df)} results")
            
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
            logger.info("Falling back to individual molecule processing for this batch")
            
            # Process problem batch molecule by molecule
            for j, (idx, data) in enumerate(zip(batch_indices, batch)):
                smiles = data['smiles']
                
                try:
                    single_result = psichic_wrapper.run_validation([smiles])
                    if not single_result.empty and 'predicted_binding_affinity' in single_result.columns:
                        score = single_result.iloc[0]['predicted_binding_affinity']
                        if score is not None:
                            results[idx] = float(score)
                            logger.info(f"Successfully processed molecule {j+1}/{len(batch)} in fallback mode")
                except Exception as e2:
                    logger.error(f"Error processing index {idx} in fallback mode: {e2}")
    
    logger.info(f"Completed all batches. Got scores for {len(results)}/{len(molecules_data)} molecules")
    return results
# ----------------------------------------------------------------------------
# SCORING FUNCTIONS
# ----------------------------------------------------------------------------

def calculate_entropy_for_valid_molecules(valid_molecules_by_uid, config):
    """
    Calculate MACCS entropy for each UID's set of molecules.
    
    Args:
        valid_molecules_by_uid: Dict mapping UIDs to their valid molecules
        config: Configuration dictionary
        
    Returns:
        Dict mapping UIDs to their entropy scores
    """
    entropy_scores = {}
    
    for uid, molecules in valid_molecules_by_uid.items():
        if not molecules:
            entropy_scores[uid] = None
            continue
            
        try:
            # Extract SMILES strings
            smiles_list = [mol['smiles'] for mol in molecules if mol['smiles']]
            
            if len(smiles_list) < 2:  # Need at least 2 molecules for entropy calculation
                entropy_scores[uid] = 0.0
                continue
                
            # Calculate entropy
            entropy = compute_maccs_entropy(smiles_list)
            entropy_scores[uid] = entropy
            logger.info(f"UID {uid}: Calculated entropy = {entropy:.4f} (from {len(smiles_list)} molecules)")
            
        except Exception as e:
            logger.error(f"Error calculating entropy for UID {uid}: {e}")
            entropy_scores[uid] = None
    
    return entropy_scores


def score_target_and_antitarget_proteins(valid_molecules_by_uid, target_proteins, antitarget_proteins, psichic_wrapper):
    """
    Score all molecules against target and antitarget proteins.
    
    Args:
        valid_molecules_by_uid: Dict mapping UIDs to their valid molecules
        target_proteins: List of target protein codes
        antitarget_proteins: List of antitarget protein codes
        psichic_wrapper: PsichicWrapper instance
        
    Returns:
        Dict with target and antitarget scores for each UID and molecule
    """
    scoring_results = {}
    
    # Initialize scoring results structure
    for uid, molecules in valid_molecules_by_uid.items():
        scoring_results[uid] = {
            'target_scores': [[] for _ in range(len(target_proteins))],
            'antitarget_scores': [[] for _ in range(len(antitarget_proteins))],
        }
    
    # Score all molecules against target proteins
    for t_idx, target in enumerate(target_proteins):
        protein_sequence = get_sequence_from_protein_code(target)
        if not protein_sequence:
            logger.error(f"Failed to get sequence for target protein {target}")
            continue
            
        logger.info(f"Evaluating target protein {target} (sequence length: {len(protein_sequence)})")
        
        # Process each UID's molecules
        for uid, molecules in valid_molecules_by_uid.items():
            try:
                # Run model on this UID's molecules
                scores = run_model_on_molecules_batch_safely(
                    psichic_wrapper,
                    protein_sequence,
                    molecules
                )
                
                # Store scores in the appropriate position
                for mol_idx in range(len(molecules)):
                    score = scores.get(mol_idx, float('-inf'))
                    scoring_results[uid]['target_scores'][t_idx].append(score)
                    
                logger.info(f"UID {uid}: Completed target protein {target} scoring")
                
            except Exception as e:
                logger.error(f"Error scoring UID {uid} molecules for target {target}: {e}")
                # Fill with -inf for failures
                scoring_results[uid]['target_scores'][t_idx] = [float('-inf')] * len(molecules)
    
    # Score all molecules against antitarget proteins
    for a_idx, antitarget in enumerate(antitarget_proteins):
        protein_sequence = get_sequence_from_protein_code(antitarget)
        if not protein_sequence:
            logger.error(f"Failed to get sequence for antitarget protein {antitarget}")
            continue
            
        logger.info(f"Evaluating antitarget protein {antitarget} (sequence length: {len(protein_sequence)})")
        
        # Process each UID's molecules
        for uid, molecules in valid_molecules_by_uid.items():
            try:
                # Run model on this UID's molecules
                scores = run_model_on_molecules_batch_safely(
                    psichic_wrapper,
                    protein_sequence,
                    molecules
                )
                
                # Store scores in the appropriate position
                for mol_idx in range(len(molecules)):
                    score = scores.get(mol_idx, float('-inf'))
                    scoring_results[uid]['antitarget_scores'][a_idx].append(score)
                    
                logger.info(f"UID {uid}: Completed antitarget protein {antitarget} scoring")
                
            except Exception as e:
                logger.error(f"Error scoring UID {uid} molecules for antitarget {antitarget}: {e}")
                # Fill with -inf for failures
                scoring_results[uid]['antitarget_scores'][a_idx] = [float('-inf')] * len(molecules)
    
    return scoring_results


def calculate_final_scores(valid_molecules_by_uid, scoring_results, entropy_scores, molecule_name_counts, config):
    """
    Calculate final scores for each UID based on target/antitarget scores, entropy, and molecule repetition.
    
    Args:
        valid_molecules_by_uid: Dict mapping UIDs to their valid molecules
        scoring_results: Dict with target and antitarget scores for each UID
        entropy_scores: Dict mapping UIDs to their entropy scores
        molecule_name_counts: Dict mapping molecule names to occurrence counts
        config: Configuration dictionary
        
    Returns:
        Dict with final scores and related data for each UID
    """
    final_results = {}
    
    for uid, molecules in valid_molecules_by_uid.items():
        if not molecules:
            continue
            
        uid_targets = scoring_results[uid]['target_scores']
        uid_antitargets = scoring_results[uid]['antitarget_scores']
        
        # Initialize results structure for this UID
        final_results[uid] = {
            'molecules': [],
            'entropy': entropy_scores.get(uid, 0.0),
            'final_score': 0.0
        }
        
        # Process each molecule
        total_weighted_score = 0.0
        
        for mol_idx, molecule in enumerate(molecules):
            # Calculate average target score for this molecule
            target_scores = [targets[mol_idx] for targets in uid_targets]
            if not target_scores or all(s == float('-inf') for s in target_scores):
                # Skip this molecule if we have no valid target scores
                continue
                
            avg_target = sum(s for s in target_scores if s != float('-inf')) / len([s for s in target_scores if s != float('-inf')])
            
            # Calculate average antitarget score for this molecule
            antitarget_scores = [antitargets[mol_idx] for antitargets in uid_antitargets]
            if not antitarget_scores or all(s == float('-inf') for s in antitarget_scores):
                # Handle case with no valid antitarget scores
                avg_antitarget = 0.0
            else:
                avg_antitarget = sum(s for s in antitarget_scores if s != float('-inf')) / len([s for s in antitarget_scores if s != float('-inf')])
            
            # Calculate weighted difference score
            weighted_diff = avg_target - (config['antitarget_weight'] * avg_antitarget)
            
            # Apply molecule repetition adjustment
            molecule_name = molecule['molecule_name']
            repetition_count = molecule_name_counts.get(molecule_name, 1)
            
            if weighted_diff > config['molecule_repetition_threshold']:
                # Penalize repeated high-scoring molecules
                adjusted_score = weighted_diff / (config['molecule_repetition_weight'] * repetition_count)
            else:
                # Reward repeated low-scoring molecules
                adjusted_score = weighted_diff * config['molecule_repetition_weight'] * repetition_count
            
            # Store molecule-specific results
            molecule_result = {
                'molecule_name': molecule_name,
                'smiles': molecule['smiles'],
                'target_score': avg_target,
                'antitarget_score': avg_antitarget,
                'weighted_diff': weighted_diff,
                'repetition_count': repetition_count,
                'adjusted_score': adjusted_score
            }
            
            final_results[uid]['molecules'].append(molecule_result)
            total_weighted_score += adjusted_score
        
        # Calculate final score with entropy bonus
        entropy = entropy_scores.get(uid, 0.0) or 0.0
        if total_weighted_score > config['entropy_bonus_threshold'] and entropy > 0:
            final_score = total_weighted_score * (config['entropy_weight'] + entropy)
        else:
            final_score = total_weighted_score
            
        final_results[uid]['total_weighted_score'] = total_weighted_score
        final_results[uid]['final_score'] = final_score
        
        logger.info(f"UID {uid}: Final score = {final_score:.4f} (total weighted: {total_weighted_score:.4f}, entropy: {entropy:.4f})")
    
    return final_results


def filter_molecules_by_criteria(molecules_data, config):
    """
    Filter molecules that meet all competition criteria.
    
    Args:
        molecules_data: List of dicts with molecule information
        config: Configuration dictionary
        
    Returns:
        List of filtered molecule dicts with validation info
    """
    logger.info(f"Filtering molecules based on competition criteria...")
    start_time = time.time()
    
    valid_molecules = []
    failed_validations = []
    
    for molecule in molecules_data:
        molecule_name = molecule['molecule_name']
        smiles = molecule.get('smiles')
        
        if not smiles:
            failed_validations.append({
                'molecule_name': molecule_name,
                'error': "Missing SMILES"
            })
            continue
        
        # Validate the molecule
        validation = validate_molecule(molecule_name, smiles, config)
        
        if validation['is_valid']:
            # Add all validated properties to the molecule dict
            molecule.update({
                'is_valid': True,
                'heavy_atoms': validation['heavy_atoms'],
                'rotatable_bonds': validation['rotatable_bonds']
            })
            valid_molecules.append(molecule)
        else:
            failed_validations.append({
                'molecule_name': molecule_name,
                'error': validation['error']
            })
    
    duration = time.time() - start_time
    logger.info(f"Filtered {len(valid_molecules)}/{len(molecules_data)} molecules in {duration:.2f} seconds")
    logger.info(f"Rejected {len(failed_validations)} molecules due to validation failures")
    
    # Log reasons for rejection
    error_types = {}
    for failed in failed_validations:
        error = failed.get('error', 'Unknown error')
        error_types[error] = error_types.get(error, 0) + 1
    
    for error, count in error_types.items():
        logger.info(f"Rejection reason: {error} - {count} molecule(s)")
    
    return valid_molecules


def archive_molecules_individually(molecule_data, smiles_dict, archive_db_path):
    """
    Archive all individual molecules to the molecule database.
    
    Args:
        molecule_data: List of all molecule data collected
        smiles_dict: Dictionary of molecule names to SMILES strings
        archive_db_path: Path to the archive database
    """
    logger.info(f"Archiving all individual molecules to database: {archive_db_path}")
    
    try:
        # Initialize molecule archive database
        molecule_db = MoleculeDB(archive_db_path)
        molecule_db.init_db()
        
        # Prepare all molecules to archive
        all_molecules_to_archive = []
        
        for data in molecule_data:
            molecule_name = data['molecule_name']
            block = data.get('block', 0)
            
            # Get SMILES if available, otherwise use None
            smiles = smiles_dict.get(molecule_name, None)
            
            all_molecules_to_archive.append({
                'molecule_name': molecule_name,
                'smiles': smiles,
                'block': block
            })
        
        # Count molecules with and without SMILES
        with_smiles = len([m for m in all_molecules_to_archive if m['smiles'] is not None])
        without_smiles = len(all_molecules_to_archive) - with_smiles
        
        # Archive the molecules
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


def archive_submissions(target_epoch, final_results, valid_molecules_by_uid, decrypted_submissions, epoch_commitments, submissions_db_path):
    """
    Archive complete submissions (sets of molecules) to the submissions database.
    
    Args:
        target_epoch: The epoch being evaluated
        final_results: Dict with final scoring results
        valid_molecules_by_uid: Dict mapping UIDs to their valid molecules
        decrypted_submissions: Dict mapping UIDs to their decrypted molecule names
        epoch_commitments: Dict with all commitments from the epoch
        submissions_db_path: Path to the submissions archive database
    """
    logger.info(f"Archiving complete submissions to database: {submissions_db_path}")
    
    try:
        # Initialize submissions database
        submissions_db = SubmissionDB(submissions_db_path)
        submissions_db.init_db()
        
        added_count = 0
        
        # Process each UID with a submission
        for uid, molecules in decrypted_submissions.items():
            # Find the hotkey for this UID
            hotkey = None
            url_path = None
            block = 0
            stake = 0.0
            
            for hk, commit in epoch_commitments.items():
                if commit['uid'] == uid:
                    hotkey = hk
                    url_path = commit['data']
                    block = commit['block']
                    stake = commit['stake']
                    break
            
            if not hotkey:
                logger.warning(f"Could not find hotkey for UID {uid} in epoch commitments")
                continue
            
            # Get the final results for this UID if available
            entropy = None
            final_score = None
            
            if uid in final_results:
                entropy = final_results[uid].get('entropy')
                final_score = final_results[uid].get('final_score')
            
            # Prepare submission data
            submission_data = {
                'epoch': target_epoch,
                'uid': uid,
                'hotkey': hotkey,
                'url_path': url_path,
                'block': block,
                'stake': stake,
                'entropy': entropy,
                'final_score': final_score
            }
            
            # Prepare molecules data
            molecules_data = []
            
            # If we have final results for this UID, use the scored molecules
            if uid in final_results and 'molecules' in final_results[uid]:
                for mol_result in final_results[uid]['molecules']:
                    molecules_data.append({
                        'molecule_name': mol_result['molecule_name'],
                        'smiles': mol_result['smiles'],
                        'target_score': mol_result['target_score'],
                        'antitarget_score': mol_result['antitarget_score'],
                        'weighted_score': mol_result['weighted_diff']
                    })
            else:
                # Otherwise use the basic molecule data
                for i, mol_name in enumerate(molecules):
                    molecules_data.append({
                        'molecule_name': mol_name,
                        'smiles': None  # We don't have scores or SMILES in this case
                    })
            
            # Add submission to database
            success = submissions_db.add_submission(submission_data, molecules_data)
            if success:
                added_count += 1
        
        # Clean up connection
        submissions_db.close()
        
        print(f"\nArchived {added_count} submissions to database: {submissions_db_path}")
        logger.info(f"Archive complete: {added_count} submissions added to database")
        
    except Exception as e:
        logger.error(f"Error archiving submissions to database: {e}")
        traceback.print_exc()
        print(f"\nWarning: Failed to archive submissions to database. See log for details.")
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
    submissions_db_path="submissions_archive.db",
    config_path="config/config.yaml",
    use_scraper=True,
    max_concurrent_requests=10
):
    """
    Runs a validator check for the specified epoch.
    
    Args:
        network: Bittensor network
        netuid: Subnet ID
        my_hotkey: Your hotkey to highlight in results (optional)
        hide_molecules: Whether to hide molecule names in output
        epoch: Specific epoch to check (None for current)
        archive_db_path: Path to the molecule archive database
        submissions_db_path: Path to the submissions archive database
        config_path: Path to the config.yaml file
        use_scraper: Whether to use the SMILES scraper for missing molecules
        max_concurrent_requests: Maximum concurrent requests for SMILES scraper
    """
    print("Starting enhanced validator check...")
    logger.info(f"Starting validator check for network={network}, netuid={netuid}, epoch={epoch if epoch is not None else 'current'}")
    start_time = time.time()
    
    try:
        # Load configuration
        config = load_config(config_path)
        logger.info(f"Loaded configuration: {config}")
        
        # Import PSICHIC
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
        
        # Get protein challenge for the epoch
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
        epoch_commitments = {}
        for hotkey, commit in current_commitments.items():
            if commit['block'] >= epoch_boundary_block and commit['block'] < (target_epoch + 1) * subnet_info['epoch_length']:
                epoch_commitments[hotkey] = commit
                
        logger.info(f"Found {len(epoch_commitments)} commitments from epoch {target_epoch}")
        print(f"Found {len(epoch_commitments)} commitments from epoch {target_epoch}")
        
        # Decrypt submissions from GitHub URLs and get stats
        print("Decrypting submissions...")
        logger.info("Decrypting submissions...")
        decrypted_submissions, submission_stats = decrypt_submissions(epoch_commitments)
        logger.info(f"Decrypted {len(decrypted_submissions)} submissions")
        print(f"Decrypted {len(decrypted_submissions)} submissions")
        
        # Verify number of molecules per submission
        print("Verifying molecule counts in submissions...")
        valid_decrypted = {}
        for uid, molecules in decrypted_submissions.items():
            if len(molecules) != config['num_molecules']:
                logger.warning(f"UID {uid} submission has {len(molecules)} molecules (expected {config['num_molecules']})")
                print(f"Warning: UID {uid} has {len(molecules)} molecules (expected {config['num_molecules']})")
                # Still keep it for analysis
                valid_decrypted[uid] = molecules
            else:
                valid_decrypted[uid] = molecules
                
        logger.info(f"Found {len(valid_decrypted)} submissions with molecules")
        print(f"Found {len(valid_decrypted)} submissions with molecules")
        
        # STEP 1: Collect all molecule names and look up SMILES in batch
        print("Collecting molecule names and looking up SMILES...")
        logger.info("Collecting all molecule names for batch lookup...")
        
        # Create list to track all molecules
        all_molecule_data = []
        all_molecule_names = set()  # Use set to avoid duplicates
        
        # Map for UID -> molecules with their metadata
        uid_to_molecules = {}
        
        for uid, molecule_names in valid_decrypted.items():
            uid_molecules = []
            
            # Find the corresponding commitment
            hotkey = None
            block = 0
            url_path = None
            stake = 0.0
            
            for hk, commit in epoch_commitments.items():
                if commit['uid'] == uid:
                    hotkey = hk
                    block = commit['block']
                    url_path = commit['data']
                    stake = commit['stake']
                    break
                    
            if not hotkey:
                logger.warning(f"Could not find hotkey for UID {uid}")
                continue
                
            # Process each molecule in this submission
            for mol_name in molecule_names:
                if not mol_name:  # Skip empty names
                    continue
                    
                # Add to the set of all molecule names
                all_molecule_names.add(mol_name)
                
                # Create data structure for this molecule
                mol_data = {
                    'uid': uid,
                    'hotkey': hotkey,
                    'molecule_name': mol_name,
                    'block': block,
                    'url_path': url_path,
                    'stake': stake
                }
                
                # Add to the list of all molecules
                all_molecule_data.append(mol_data)
                
                # Add to this UID's molecule list
                uid_molecules.append(mol_data)
                
            # Store this UID's molecule list
            uid_to_molecules[uid] = uid_molecules
        
        logger.info(f"Collected {len(all_molecule_names)} unique molecule names from {len(valid_decrypted)} submissions")
        print(f"Collected {len(all_molecule_names)} unique molecule names from {len(valid_decrypted)} submissions")
        
        # Look up SMILES in SAVI database first
        print("Looking up SMILES in SAVI database...")
        savi = SAVILookup("savi_lookup.db")
        savi_smiles_dict = savi.lookup_smiles_batch(list(all_molecule_names))
        savi_found_count = len(savi_smiles_dict)
        savi_found_pct = (savi_found_count / len(all_molecule_names) * 100) if all_molecule_names else 0
        logger.info(f"Found {savi_found_count}/{len(all_molecule_names)} molecules in SAVI database ({savi_found_pct:.1f}%)")
        print(f"Found {savi_found_count}/{len(all_molecule_names)} molecules in SAVI database ({savi_found_pct:.1f}%)")
        
        # For molecules not found in SAVI, use NOVA dashboard scraper
        missing_molecules = [m for m in all_molecule_names if m not in savi_smiles_dict]
        nova_found_count = 0
        
        if missing_molecules and use_scraper:
            print(f"Using NOVA dashboard SMILES scraper for {len(missing_molecules)} molecules...")
            logger.info(f"Using NOVA dashboard SMILES scraper for {len(missing_molecules)} missing molecules...")
            
            # Fetch SMILES for missing molecules using the scraper
            nova_smiles_dict = fetch_smiles_batch(missing_molecules, max_concurrent_requests)
            
            # Count how many molecules have non-empty SMILES from scraper
            nova_found_count = sum(1 for smiles in nova_smiles_dict.values() if smiles)
            nova_found_pct = (nova_found_count / len(missing_molecules) * 100) if missing_molecules else 0
            logger.info(f"Found additional {nova_found_count}/{len(missing_molecules)} molecules via NOVA dashboard scraper ({nova_found_pct:.1f}%)")
            print(f"Found additional {nova_found_count}/{len(missing_molecules)} molecules via NOVA dashboard scraper ({nova_found_pct:.1f}%)")
            
            # Combine SAVI and NOVA results
            smiles_dict = {**savi_smiles_dict, **nova_smiles_dict}
        else:
            # Just use SAVI results
            smiles_dict = savi_smiles_dict
        
        # Add SMILES to molecule data
        total_smiles_found = len([v for v in smiles_dict.values() if v])
        total_found_pct = (total_smiles_found / len(all_molecule_names) * 100) if all_molecule_names else 0
        logger.info(f"Total molecules with SMILES strings: {total_smiles_found}/{len(all_molecule_names)} ({total_found_pct:.1f}%)")
        print(f"Total molecules with SMILES strings: {total_smiles_found}/{len(all_molecule_names)} ({total_found_pct:.1f}%)")
        
        # Add SMILES to the molecule data structures
        for mol_data in all_molecule_data:
            mol_name = mol_data['molecule_name']
            mol_data['smiles'] = smiles_dict.get(mol_name)
        
        # STEP 2: Filter molecules that meet competition criteria
        print("Filtering molecules by competition criteria...")
        valid_molecules_by_uid = {}
        
        for uid, molecules in uid_to_molecules.items():
            # Filter molecules with valid SMILES and competition criteria
            valid_mols = filter_molecules_by_criteria(molecules, config)
            if valid_mols:
                valid_molecules_by_uid[uid] = valid_mols
                logger.info(f"UID {uid}: {len(valid_mols)}/{len(molecules)} valid molecules")
                
        logger.info(f"Found {len(valid_molecules_by_uid)} UIDs with valid molecules")
        print(f"Found {len(valid_molecules_by_uid)} UIDs with valid molecules")
        
        # Count molecule repetition across all valid submissions
        molecule_name_counts = calculate_molecule_name_counts(valid_molecules_by_uid)
        logger.info(f"Molecule name counts: {len(molecule_name_counts)} unique molecules across all valid submissions")
        
        # STEP 3: Calculate entropy for each UID's molecules
        print("Calculating entropy for molecule sets...")
        entropy_scores = calculate_entropy_for_valid_molecules(valid_molecules_by_uid, config)
        
        valid_entropy_count = sum(1 for score in entropy_scores.values() if score is not None)
        logger.info(f"Calculated entropy for {valid_entropy_count}/{len(valid_molecules_by_uid)} UIDs")
        print(f"Calculated entropy for {valid_entropy_count}/{len(valid_molecules_by_uid)} UIDs")
        
        # STEP 4: Score molecules against target and antitarget proteins
        print("Scoring molecules against target and antitarget proteins...")
        scoring_results = score_target_and_antitarget_proteins(
            valid_molecules_by_uid,
            target_proteins,
            antitarget_proteins,
            psichic_wrapper
        )
        
        logger.info(f"Completed scoring for {len(scoring_results)} UIDs")
        print(f"Completed scoring for {len(scoring_results)} UIDs")
        
        # STEP 5: Calculate final scores with entropy and repetition adjustments
        print("Calculating final scores...")
        final_results = calculate_final_scores(
            valid_molecules_by_uid,
            scoring_results,
            entropy_scores,
            molecule_name_counts,
            config
        )
        
        logger.info(f"Calculated final scores for {len(final_results)} UIDs")
        print(f"Calculated final scores for {len(final_results)} UIDs")
        
        # STEP 6: Prepare results for display and output
        print("Preparing results for display...")
        
        # Convert to DataFrame for easier sorting and display
        results_data = []
        
        for uid, result in final_results.items():
            # Find the hotkey for this UID
            hotkey = None
            for mol in result['molecules']:
                if 'hotkey' in mol:
                    hotkey = mol['hotkey']
                    break
            
            if not hotkey:
                continue
                
            # Extract the block submitted (using the first molecule)
            block = 0
            if result['molecules']:
                if 'block' in result['molecules'][0]:
                    block = result['molecules'][0]['block']
            
            # Build the result row
            row = {
                'uid': uid,
                'hotkey': hotkey,
                'entropy': result['entropy'],
                'total_weighted_score': result['total_weighted_score'],
                'final_score': result['final_score'],
                'block': block,
                'num_molecules': len(result['molecules']),
                'molecules': [] if hide_molecules else [m['molecule_name'] for m in result['molecules']]
            }
            
            results_data.append(row)
        
        # Create the results DataFrame and sort by final score
        if results_data:
            results_df = pd.DataFrame(results_data)
            results_df = results_df.sort_values('final_score', ascending=False).reset_index(drop=True)
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"validator_results_epoch{target_epoch}_{timestamp}.csv"
            results_df.to_csv(csv_filename, index=False)
            logger.info(f"Saved results to {csv_filename}")
            print(f"Saved results to {csv_filename}")
            
            # Print results
            print("\n" + "="*80)
            print(f"VALIDATOR RESULTS FOR EPOCH {target_epoch}")
            print("="*80)
            
            if len(results_df) > 0:
                # Display the top results
                top_n = min(10, len(results_df))
                
                print(f"\nTop {top_n} submissions:")
                print("-" * 100)
                if hide_molecules:
                    print(f"{'Rank':<5} {'UID':<5} {'Hotkey':<12} {'Final Score':<12} {'Weighted Score':<15} {'Entropy':<10} {'Molecules':<10} {'Block':<8}")
                    print("-" * 100)
                    
                    for i in range(top_n):
                        row = results_df.iloc[i]
                        is_my_hotkey = my_hotkey and row['hotkey'].startswith(my_hotkey)
                        prefix = " *> " if is_my_hotkey else "    "
                        
                        display_hotkey = row['hotkey'][:6] + "..." + row['hotkey'][-3:] if len(row['hotkey']) > 12 else row['hotkey']
                        
                        print(f"{prefix}{i+1:<3} {row['uid']:<5} {display_hotkey:<12} {row['final_score']:<12.4f} "
                              f"{row['total_weighted_score']:<15.4f} {row['entropy']:<10.4f} {row['num_molecules']:<10} {row['block']:<8}")
                else:
                    print(f"{'Rank':<5} {'UID':<5} {'Hotkey':<12} {'Final Score':<12} {'Weighted Score':<15} {'Entropy':<10} {'Molecules':<10} {'Block':<8}")
                    print("-" * 100)
                    
                    for i in range(top_n):
                        row = results_df.iloc[i]
                        is_my_hotkey = my_hotkey and row['hotkey'].startswith(my_hotkey)
                        prefix = " *> " if is_my_hotkey else "    "
                        
                        display_hotkey = row['hotkey'][:6] + "..." + row['hotkey'][-3:] if len(row['hotkey']) > 12 else row['hotkey']
                        
                        # Truncate the molecules list for display
                        molecules_str = str(row['molecules'])
                        if len(molecules_str) > 40:
                            molecules_str = molecules_str[:37] + "..."
                        
                        print(f"{prefix}{i+1:<3} {row['uid']:<5} {display_hotkey:<12} {row['final_score']:<12.4f} "
                              f"{row['total_weighted_score']:<15.4f} {row['entropy']:<10.4f} {row['num_molecules']:<10} {row['block']:<8}")
                
                print("-" * 100)
                
                # Show my position if hotkey is provided
                if my_hotkey:
                    my_rows = results_df[results_df['hotkey'].str.startswith(my_hotkey)]
                    if not my_rows.empty:
                        my_rank = my_rows.index[0] + 1
                        my_row = my_rows.iloc[0]
                        print(f"\nYour Position:")
                        print(f"Rank: {my_rank}/{len(results_df)}")
                        print(f"UID: {my_row['uid']}")
                        print(f"Final Score: {my_row['final_score']:.4f}")
                        print(f"Weighted Score: {my_row['total_weighted_score']:.4f}")
                        print(f"Entropy: {my_row['entropy']:.4f}")
                        print(f"Number of Valid Molecules: {my_row['num_molecules']}")
                        print(f"Block: {my_row['block']}")
                    else:
                        print(f"\nYour hotkey {my_hotkey} not found in epoch {target_epoch} results")
            else:
                print("No submissions with valid scores found for this epoch.")
        else:
            print("\nNo valid submissions found for scoring.")
            
        # STEP 7: Archive results to databases
        print("\nArchiving molecules to databases...")
        
        # Archive individual molecules to molecule_archive.db
        archive_molecules_individually(all_molecule_data, smiles_dict, archive_db_path)
        
        # Archive complete submissions to submissions_archive.db
        archive_submissions(
            target_epoch,
            final_results,
            valid_molecules_by_uid,
            valid_decrypted,
            epoch_commitments,
            submissions_db_path
        )
        
        logger.info(f"Validator check completed in {time.time() - start_time:.2f} seconds")
        print(f"\nValidator check completed in {time.time() - start_time:.2f} seconds")
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
    
    # Database options
    parser.add_argument("--archive-db", type=str, default="molecule_archive.db",
                       help="Path to molecule archive database (default: molecule_archive.db)")
    parser.add_argument("--submissions-db", type=str, default="submissions_archive.db",
                       help="Path to submissions archive database (default: submissions_archive.db)")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file (default: config/config.yaml)")
    
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
    print("With support for multi-molecule submissions and weighted scoring")
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
    print(f"Archiving individual molecules to: {args.archive_db}")
    print(f"Archiving submissions to: {args.submissions_db}")
    print(f"Using config from: {args.config}")
    print(f"NOVA SMILES scraper: {'Disabled' if args.no_scraper else 'Enabled'}")
    print(f"Max concurrent requests: {args.max_concurrent}")
    print("="*80)
    
    print("Running validator check...")
    success = await run_validator_check(
        network=args.network,
        netuid=args.netuid,
        my_hotkey=args.my_hotkey,
        hide_molecules=args.hide_molecules,
        epoch=target_epoch,
        archive_db_path=args.archive_db,
        submissions_db_path=args.submissions_db,
        config_path=args.config,
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
