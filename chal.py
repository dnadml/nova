#!/usr/bin/env python3
"""
Optimized script to get challenge proteins from a blockhash and evaluate molecules against them in parallel.
Features:
- Single model loading
- Batch protein processing
- Molecule caching
- Parallel execution
- Molecule pre-filtering
- Top 5 molecules calculation
- Automatic submission of winning molecule
- Support for both CSV and SQLite DB molecule sources
"""
import os
import sys
import argparse
import asyncio
import subprocess
import concurrent.futures
import sqlite3
import pandas as pd
import numpy as np
import bittensor as bt
import tempfile
import base64
import hashlib
import datetime
import gc
import torch
from typing import Any, Tuple, List, Dict
from substrateinterface import SubstrateInterface
from bittensor.core.errors import MetadataError
from dotenv import load_dotenv

# Adjust the BASE_DIR to properly import dependencies
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import necessary functions
from my_utils import get_challenge_proteins_from_blockhash, get_sequence_from_protein_code, get_heavy_atom_count, upload_file_to_github
from config.config_loader import load_config
from btdr import QuicknetBittensorDrandTimelock

# Import enhanced wrapper
sys.path.append(os.path.join(BASE_DIR, "PSICHIC"))
from enhanced_wrapper import EnhancedPsichicWrapper

# ----------------------------------------------------------------------------
# DATABASE INTEGRATION
# ----------------------------------------------------------------------------

class MoleculeDB:
    """
    Database for querying molecules from a SQLite database.
    """
    def __init__(self, db_path="molecule_archive.db"):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Connect to the database."""
        if not self.conn:
            if not os.path.exists(self.db_path):
                bt.logging.error(f"Database file not found: {self.db_path}")
                return False
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            return True
        return True
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def get_all_molecules_with_smiles(self, limit=None, shard_index=0, shard_count=1, min_heavy_atoms=24):
        """
        Get all molecules that have SMILES strings, optionally filtered by shard.
        
        Args:
            limit (int, optional): Maximum number of molecules to retrieve
            shard_index (int): Index of this shard (0-based)
            shard_count (int): Total number of shards
            min_heavy_atoms (int): Minimum number of heavy atoms for filtering
            
        Returns:
            list: List of dicts with 'molecule' and 'smiles' keys
        """
        if not self.connect():
            return []
            
        try:
            cursor = self.conn.cursor()
            
            # When not sharding, use the original query
            if shard_count <= 1:
                query = 'SELECT molecule, smiles FROM molecules WHERE smiles IS NOT NULL'
                
                if limit is not None and isinstance(limit, int) and limit > 0:
                    query += f' LIMIT {limit}'
                    
            # When sharding, use the rowid modulo to distribute the rows
            else:
                query = f'''
                SELECT molecule, smiles FROM molecules 
                WHERE smiles IS NOT NULL AND rowid % {shard_count} = {shard_index}
                '''
                
                if limit is not None and isinstance(limit, int) and limit > 0:
                    query += f' LIMIT {limit}'
                
                bt.logging.info(f"Shard {shard_index+1}/{shard_count}: Processing molecules where rowid % {shard_count} = {shard_index}")
                
            cursor.execute(query)
            results = []
            for row in cursor.fetchall():
                molecule = row['molecule']
                smiles = row['smiles']
                
                # Basic SMILES validation
                if not is_valid_smiles(smiles):
                    continue
                
                # Heavy atom count check
                try:
                    if get_heavy_atom_count(smiles) >= min_heavy_atoms:
                        results.append({
                            'molecule': molecule,
                            'smiles': smiles
                        })
                except Exception:
                    continue
                
                # Apply limit if specified during post-filtering
                if limit is not None and len(results) >= limit:
                    break
            
            bt.logging.info(f"Retrieved {len(results)} valid molecules with SMILES strings from database after filtering")
            return results
        except Exception as e:
            bt.logging.error(f"Error getting molecules from database: {e}")
            return []

def is_valid_smiles(smiles_str):
    """
    Check if a SMILES string is valid by:
    1. Checking for balanced parentheses
    2. Checking for balanced square brackets
    3. Ensuring it's not empty or None
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
    
    return True

def load_github_path() -> str:
    """
    Constructs the path for GitHub operations from environment variables.
    
    Returns:
        str: The fully qualified GitHub path (owner/repo/branch/path).
    Raises:
        ValueError: If the final path exceeds 100 characters.
    """
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH')
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER')
    github_repo_path = os.environ.get('GITHUB_REPO_PATH')

    if github_repo_name is None or github_repo_branch is None or github_repo_owner is None:
        raise ValueError("Missing one or more GitHub environment variables (GITHUB_REPO_*)")

    if github_repo_path == "":
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}"
    else:
        github_path = f"{github_repo_owner}/{github_repo_name}/{github_repo_branch}/{github_repo_path}"

    if len(github_path) > 100:
        raise ValueError("GitHub path is too long. Please shorten it to 100 characters or less.")

    return github_path
async def submit_molecule(state, molecule_name, molecule_smiles):
    """
    Submit a molecule to the chain.
    
    Args:
        state (dict): State dictionary with references to blockchain objects
        molecule_name (str): Name of the molecule to submit
        molecule_smiles (str): SMILES string of the molecule
        
    Returns:
        bool: True if submission was successful, False otherwise
    """
    if not molecule_name:
        bt.logging.warning("No molecule provided for submission")
        return False
    
    bt.logging.info(f"=== SUBMISSION PROCESS STARTED ===")
    bt.logging.info(f"Submitting molecule: {molecule_name}")
    bt.logging.info(f"SMILES: {molecule_smiles}")
    
    bt.logging.info("Step 1: Encrypting response...")
    encrypted_response = state['bdt'].encrypt(state['miner_uid'], molecule_name)
    bt.logging.info("Encryption successful")
    
    bt.logging.info("Step 2: Preparing temporary file...")
    tmp_file = tempfile.NamedTemporaryFile(delete=True)
    with open(tmp_file.name, 'w+') as f:
        f.write(str(encrypted_response))
        f.flush()
        f.seek(0)
        content_str = f.read()
        encoded_content = base64.b64encode(content_str.encode()).decode()
        filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
        commit_content = f"{state['github_path']}/{filename}.txt"
        bt.logging.info(f"Prepared commit content: {commit_content}")
    
    bt.logging.info("Step 3: Setting chain commitment...")
    try:
        commitment_status = await state['subtensor'].set_commitment(
            wallet=state['wallet'],
            netuid=state['config'].netuid,
            data=commit_content
        )
        bt.logging.info(f"Chain commitment status: {commitment_status}")
    except MetadataError:
        bt.logging.info("Too soon to commit again. Will keep looking for better candidates.")
        return False
    except Exception as e:
        bt.logging.error(f"Error during chain commitment: {e}")
        return False
        
    if commitment_status:
        try:
            bt.logging.info("Step 4: Chain commitment was successful")
            bt.logging.info("Attempting GitHub upload...")
            github_status = upload_file_to_github(filename, encoded_content)
            if github_status:
                bt.logging.info(f"SUCCESS: File uploaded to GitHub at {commit_content}")
                state['last_submission_time'] = datetime.datetime.now()
                state['last_submitted_molecule'] = molecule_name
                bt.logging.info(f"=== SUBMISSION PROCESS COMPLETED SUCCESSFULLY ===")
                return True
            else:
                bt.logging.error(f"FAILED: Could not upload file to GitHub for {commit_content}")
                return False
        except Exception as e:
            bt.logging.error(f"FAILED: Error uploading file for {commit_content}: {e}")
            return False
    else:
        bt.logging.error("FAILED: Chain commitment was unsuccessful")
        return False

async def setup_bittensor_objects(config: argparse.Namespace) -> Tuple[Any, Any, Any, int, int]:
    """
    Initializes wallet, subtensor, and metagraph. Fetches the epoch length
    and calculates the miner UID.

    Args:
        config (argparse.Namespace): The miner configuration object.

    Returns:
        tuple: A 5-element tuple of
            (wallet, subtensor, metagraph, miner_uid, epoch_length).
    """
    bt.logging.info("Setting up Bittensor objects.")

    # Initialize wallet
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # Initialize subtensor (asynchronously)
    try:
        async with bt.async_subtensor(network=config.network) as subtensor:
            bt.logging.info(f"Connected to subtensor network: {config.network}")
            
            # Sync metagraph
            metagraph = await subtensor.metagraph(config.netuid)
            await metagraph.sync()
            bt.logging.info(f"Metagraph synced successfully.")

            bt.logging.info(f"Subtensor: {subtensor}")
            bt.logging.info(f"Metagraph synced: {metagraph}")

            # Get miner UID
            miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            bt.logging.info(f"Miner UID: {miner_uid}")

            # Query epoch length - THIS IS THE IMPORTANT PART WITH EXACT SAME CODE
            node = SubstrateInterface(url=config.network)
            epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value
            bt.logging.info(f"Epoch length query successful: {epoch_length} blocks")

        return wallet, subtensor, metagraph, miner_uid, epoch_length
    except Exception as e:
        bt.logging.error(f"Failed to setup Bittensor objects: {e}")
        bt.logging.error("Please check your network connection and the subtensor network status")
        raise

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments and merges with config defaults.

    Returns:
        argparse.Namespace: The combined configuration object.
    """
    parser = argparse.ArgumentParser(description="Get proteins and evaluate molecules against them")
    
    # Network arguments
    parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK'), help='Network to use')
    parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
    
    # Molecule source options
    parser.add_argument('--source', choices=['csv', 'db'], default='csv', help="Source of molecules: 'csv' or 'db'")
    parser.add_argument('--csv-file', type=str, default="molecule_smiles.csv", help="CSV file with 'molecule,smiles' columns (used when source=csv)")
    parser.add_argument('--molecule-db', type=str, default="molecule_archive.db", help="SQLite database file with molecules (used when source=db)")
    
    parser.add_argument('--output-dir', type=str, default="./results", help="Directory to store results")
    parser.add_argument('--limit', type=int, help="Limit the number of molecules to process")
    parser.add_argument('--batch-size', type=int, default=20, help="Number of molecules to process in each batch")
    parser.add_argument('--target-weight', type=float, default=1.0, help="Weight for target protein")
    parser.add_argument('--antitarget-weight', type=float, default=0.75, help="Weight for antitarget protein")
    parser.add_argument('--parallel', type=int, default=4, help="Number of proteins to evaluate in parallel")
    parser.add_argument('--no-cache', action="store_true", help="Disable molecule caching")
    parser.add_argument('--top-n', type=int, default=5, help="Number of top molecules to display")
    parser.add_argument('--min-heavy-atoms', type=int, default=26, help="Minimum number of heavy atoms for molecules")
    parser.add_argument('--no-commit', action="store_true", help="Disable automatic commitment of top molecule")
    
    # Add sharding parameters (from second script)
    parser.add_argument('--shard-index', type=int, default=0, help='Index of this shard (0-based)')
    parser.add_argument('--shard-count', type=int, default=1, help='Total number of shards')
    
    # Add continuous mode option (from second script)
    parser.add_argument('--mode', choices=['continuous', 'once'], default='continuous', 
                      help='Operation mode: continuous (keep running) or once (single run)')
    parser.add_argument('--check-interval', type=int, default=30,
                      help='Interval in seconds between checks in continuous mode')
    
    # Bittensor standard argument additions
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    
    # Parse combined config
    config = bt.config(parser)

    # Load all config parameters from YAML
    config_params = load_config(os.path.join(BASE_DIR, 'config/config.yaml'))
    config.num_targets = config_params['num_targets']
    config.num_antitargets = config_params['num_antitargets']
    
    return config

def load_smiles_from_csv(csv_path: str, min_heavy_atoms: int = 26, limit: int = None) -> Tuple[List[str], List[str]]:
    """
    Load molecule names and SMILES strings from a CSV file with pre-filtering.
    
    Args:
        csv_path: Path to CSV file with 'molecule,smiles' columns
        min_heavy_atoms: Minimum number of heavy atoms required
        limit: Optional limit on number of molecules to load
        
    Returns:
        Tuple of (molecule_names, smiles_list)
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        if 'molecule' not in df.columns or 'smiles' not in df.columns:
            print(f"Error: CSV file must have 'molecule' and 'smiles' columns")
            return [], []
        
        print(f"Total molecules in CSV: {len(df)}")
        
        # Filter valid molecules
        valid_molecules = []
        for _, row in df.iterrows():
            smiles = row['smiles']
            
            # Basic SMILES validation
            if not is_valid_smiles(smiles):
                continue
            
            # Heavy atom count check
            try:
                if get_heavy_atom_count(smiles) >= min_heavy_atoms:
                    valid_molecules.append({
                        'molecule': row['molecule'],
                        'smiles': smiles
                    })
            except Exception:
                continue
            
            # Apply limit if specified
            if limit and len(valid_molecules) >= limit:
                break
        
        valid_df = pd.DataFrame(valid_molecules)
        if valid_df.empty:
            print("No valid molecules found after filtering")
            return [], []
        
        # Extract molecule names and SMILES strings
        molecule_names = valid_df['molecule'].tolist()
        smiles_list = valid_df['smiles'].tolist()
        
        print(f"Loaded {len(molecule_names)} valid molecules from {csv_path} after filtering")
        return molecule_names, smiles_list
    
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return [], []
def load_molecules(config):
    """
    Load molecules from the specified source (CSV or DB).
    
    Args:
        config: Configuration object with source parameters
        
    Returns:
        Tuple of (molecule_names_list, smiles_list) or None if loading failed
    """
    if config.source == 'csv':
        bt.logging.info(f"Loading and filtering molecules from CSV: {config.csv_file}")
        molecule_names, smiles_list = load_smiles_from_csv(
            config.csv_file,
            config.min_heavy_atoms,
            config.limit
        )
        
        if not molecule_names or not smiles_list:
            bt.logging.error("Failed to load valid molecules from CSV file")
            return None
            
        bt.logging.info(f"Loaded {len(molecule_names)} valid molecules from {config.csv_file}")
        return molecule_names, smiles_list
        
    elif config.source == 'db':
        bt.logging.info(f"Loading and filtering molecules from DB: {config.molecule_db}")
        db = MoleculeDB(config.molecule_db)
        molecules_data = db.get_all_molecules_with_smiles(
            limit=config.limit,
            shard_index=config.shard_index,
            shard_count=config.shard_count,
            min_heavy_atoms=config.min_heavy_atoms
        )
        db.close()
        
        if not molecules_data:
            bt.logging.error("Failed to load valid molecules from database")
            return None
            
        # Convert the list of dicts to two parallel lists
        df = pd.DataFrame(molecules_data)
        molecule_names = df['molecule'].tolist()
        smiles_list = df['smiles'].tolist()
        
        bt.logging.info(f"Loaded {len(molecule_names)} valid molecules from database")
        return molecule_names, smiles_list
    
    else:
        bt.logging.error(f"Invalid source specified: {config.source}")
        return None

def batch_process_protein(
    protein_code: str,
    protein_sequence: str,
    molecule_names: List[str],
    smiles_list: List[str],
    model: EnhancedPsichicWrapper,
    preloaded_dict: dict,
    is_target: bool,
    target_weight: float = 1.0,
    antitarget_weight: float = 0.75,
    output_file: str = None
) -> pd.DataFrame:
    """
    Process a single protein against all molecules using the shared model.
    
    Args:
        protein_code: Code of the protein
        protein_sequence: Amino acid sequence of the protein
        molecule_names: List of molecule names
        smiles_list: List of SMILES strings
        model: Shared EnhancedPsichicWrapper instance
        preloaded_dict: Preloaded molecule dictionary
        is_target: Whether this is a target protein
        target_weight: Weight for target proteins
        antitarget_weight: Weight for antitarget proteins
        output_file: Path to save results (optional)
        
    Returns:
        DataFrame with evaluation results
    """
    protein_type = "target" if is_target else "antitarget"
    print(f"Processing {protein_type} protein {protein_code}")
    
    try:
        # Create a batch dataframe for this protein with all molecules
        batch_df = pd.DataFrame({
            'Protein': [protein_sequence] * len(smiles_list),
            'Ligand': smiles_list
        })
        
        # Process entire batch
        batch_results = model.run_preloaded_validation(batch_df, preloaded_dict)
        
        # Process results
        results = []
        for j, (name, smiles) in enumerate(zip(molecule_names, smiles_list)):
            score = float(batch_results['predicted_binding_affinity'].values[j])
            weighted_score = score if is_target else -antitarget_weight * score
            
            results.append({
                'molecule': name,
                'smiles': smiles,
                'raw_score': score,
                'weighted_score': weighted_score,
                'protein': protein_code,
                'is_target': is_target
            })
        
        # Create DataFrame and sort
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by='weighted_score', ascending=False).reset_index(drop=True)
            
            # Save to file if requested
            if output_file:
                df.to_csv(output_file, index=False)
                print(f"Results saved to {output_file}")
        
        return df
    
    except Exception as e:
        print(f"Error processing protein {protein_code}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

async def run_once(state):
    """
    Run the miner once and exit.
    
    Args:
        state (dict): State dictionary with references to blockchain objects
    """
    try:
        # Get current block and calculate epoch
        current_block = await state['subtensor'].get_current_block()
        current_epoch = current_block // state['epoch_length']
        epoch_boundary = current_epoch * state['epoch_length']
        
        bt.logging.info(f"Current block: {current_block}, Current epoch: {current_epoch}")
        bt.logging.info(f"Epoch boundary: {epoch_boundary}")
        
        # Get block hash
        block_hash = await state['subtensor'].determine_block_hash(epoch_boundary)
        bt.logging.info(f"Block hash for epoch boundary {epoch_boundary}: {block_hash}")
        
        # Get proteins
        proteins = get_challenge_proteins_from_blockhash(
            block_hash=block_hash, 
            num_targets=state['config'].num_targets, 
            num_antitargets=state['config'].num_antitargets
        )
        
        if not proteins:
            bt.logging.error("Failed to get proteins from block hash")
            return
        
        target_codes = proteins["targets"]
        antitarget_codes = proteins["antitargets"]
        
        bt.logging.info(f"Challenge targets ({len(target_codes)}): {target_codes}")
        bt.logging.info(f"Challenge antitargets ({len(antitarget_codes)}): {antitarget_codes}")
        
        # Create output directory
        os.makedirs(state['config'].output_dir, exist_ok=True)
        
        # Create molecule cache directory
        molecule_cache_dir = os.path.join(BASE_DIR, "molecule_cache")
        os.makedirs(molecule_cache_dir, exist_ok=True)
        
        # Load molecules from specified source
        molecules_data = load_molecules(state['config'])
        if not molecules_data:
            return
            
        molecule_names, smiles_list = molecules_data
        
        # Get all protein sequences in parallel
        protein_sequences = []
        protein_code_to_sequence = {}
        
        print("Fetching protein sequences...")
        for code in target_codes + antitarget_codes:
            sequence = get_sequence_from_protein_code(code)
            if sequence:
                protein_sequences.append(sequence)
                protein_code_to_sequence[code] = sequence
                print(f"Fetched sequence for {code}: {len(sequence)} amino acids")
            else:
                print(f"Failed to fetch sequence for {code}")
        
        # Initialize model only once
        print("Initializing PSICHIC model...")
        model = EnhancedPsichicWrapper()
        
        # Preprocess all molecules
        print(f"Preprocessing {len(smiles_list)} molecules...")
        preloaded_dict = model.initialize_smiles(smiles_list)
        
        # Preprocess all proteins in batch
        print(f"Preprocessing {len(protein_sequences)} proteins in batch...")
        model.initialize_proteins_batch(protein_sequences)
        
        # Process proteins and molecules
        all_proteins = []
        all_results = []
        
        # Prepare protein processing tasks
        for protein_code in target_codes + antitarget_codes:
            is_target = protein_code in target_codes
            sequence = protein_code_to_sequence.get(protein_code)
            
            if not sequence:
                print(f"Skipping protein {protein_code}: sequence not found")
                continue
                
            output_file = os.path.join(
                state['config'].output_dir, 
                f"{protein_code}_{'target' if is_target else 'antitarget'}.csv"
            )
            
            all_proteins.append({
                'protein_code': protein_code,
                'protein_sequence': sequence,
                'is_target': is_target,
                'output_file': output_file
            })
        
        # Process proteins in sequence (already using optimized batch processing for molecules)
        for protein in all_proteins:
            result_df = batch_process_protein(
                protein_code=protein['protein_code'],
                protein_sequence=protein['protein_sequence'],
                molecule_names=molecule_names,
                smiles_list=smiles_list,
                model=model,
                preloaded_dict=preloaded_dict,
                is_target=protein['is_target'],
                target_weight=state['config'].target_weight,
                antitarget_weight=state['config'].antitarget_weight,
                output_file=protein['output_file']
            )
            
            if not result_df.empty:
                all_results.append(result_df)
        
        # Calculate top molecules across all proteins
        if all_results:
            bt.logging.info("Calculating top molecules across all proteins...")
            
            # Get all unique molecules
            molecules_df = pd.DataFrame({
                'molecule': molecule_names,
                'smiles': smiles_list
            })
            
            # Create dictionaries to store scores for each molecule
            target_scores = {}
            antitarget_scores = {}
            
            # Process each protein's results
            for result_df in all_results:
                protein_code = result_df['protein'].iloc[0]
                is_target = result_df['is_target'].iloc[0]
                
                # Group by molecule and average scores
                for _, row in result_df.iterrows():
                    molecule = row['molecule']
                    raw_score = row['raw_score']
                    
                    if is_target:
                        if molecule not in target_scores:
                            target_scores[molecule] = []
                        target_scores[molecule].append(raw_score)
                    else:
                        if molecule not in antitarget_scores:
                            antitarget_scores[molecule] = []
                        antitarget_scores[molecule].append(raw_score)
            
            # Calculate final scores using the formula
            final_scores = []
            for molecule in molecules_df['molecule']:
                # Get average target score
                avg_target_score = np.mean(target_scores.get(molecule, [0])) if molecule in target_scores else 0
                
                # Get average antitarget score
                avg_antitarget_score = np.mean(antitarget_scores.get(molecule, [0])) if molecule in antitarget_scores else 0
                
                # Calculate weighted score: (target_weight * target_affinity) - (antitarget_weight * antitarget_affinity)
                weighted_score = (state['config'].target_weight * avg_target_score) - (state['config'].antitarget_weight * avg_antitarget_score)
                
                # Get SMILES for this molecule
                smiles = molecules_df.loc[molecules_df['molecule'] == molecule, 'smiles'].iloc[0]
                
                final_scores.append({
                    'molecule': molecule,
                    'smiles': smiles,
                    'target_affinity': avg_target_score,
                    'antitarget_affinity': avg_antitarget_score,
                    'weighted_score': weighted_score
                })
            
            # Create final DataFrame and sort by weighted score
            final_df = pd.DataFrame(final_scores)
            final_df = final_df.sort_values(by='weighted_score', ascending=False).reset_index(drop=True)
            
            # Save to CSV
            final_output = os.path.join(state['config'].output_dir, "final_rankings.csv")
            final_df.to_csv(final_output, index=False)
            bt.logging.info(f"Final rankings saved to {final_output}")
            
            # Display top molecules
            top_n = min(state['config'].top_n, len(final_df))
            bt.logging.info(f"\n=== TOP {top_n} MOLECULES ===")
            for i in range(top_n):
                molecule = final_df['molecule'].iloc[i]
                target_aff = final_df['target_affinity'].iloc[i]
                antitarget_aff = final_df['antitarget_affinity'].iloc[i]
                score = final_df['weighted_score'].iloc[i]
                bt.logging.info(f"#{i+1}: {molecule}")
                bt.logging.info(f"  Target affinity: {target_aff:.4f}")
                bt.logging.info(f"  Antitarget affinity: {antitarget_aff:.4f}")
                bt.logging.info(f"  Weighted score: {score:.4f}")
                bt.logging.info("-" * 50)
            
            # Also print a concise summary
            bt.logging.info("\nConcise summary:")
            for i in range(top_n):
                molecule = final_df['molecule'].iloc[i]
                score = final_df['weighted_score'].iloc[i]
                bt.logging.info(f"  #{i+1}: {molecule} - Score: {score:.4f}")
            
            # Submit the best molecule if auto-commit is enabled
            if not state['config'].no_commit and len(final_df) > 0:
                best_molecule = final_df['molecule'].iloc[0]
                best_smiles = final_df['smiles'].iloc[0]
                best_score = final_df['weighted_score'].iloc[0]
                
                bt.logging.info(f"\nPreparing to submit best molecule: {best_molecule} with score {best_score:.4f}")
                
                # Submit the molecule
                submission_success = await submit_molecule(
                    state, 
                    best_molecule, 
                    best_smiles
                )
                
                if submission_success:
                    bt.logging.info(f"Successfully submitted molecule {best_molecule}")
                    state['last_submitted_molecule'] = best_molecule
                else:
                    bt.logging.error(f"Failed to submit molecule {best_molecule}")
            elif state['config'].no_commit:
                bt.logging.info("Auto-commit disabled. Not submitting molecule.")
            
        bt.logging.info("\nAll evaluations complete!")
        bt.logging.info(f"Results saved to {state['config'].output_dir}/")
        
        # Clean up memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        bt.logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
async def run_continuous(state):
    """
    Run the miner continuously, monitoring for epoch changes.
    
    Args:
        state (dict): State dictionary with references to objects
    """
    bt.logging.info("Running in continuous monitoring mode")
    
    current_tracked_epoch = None
    
    while True:
        try:
            current_block = await state['subtensor'].get_current_block()
            current_epoch = current_block // state['epoch_length']
            blocks_to_next_epoch = state['epoch_length'] - (current_block % state['epoch_length'])
            bt.logging.info(f"Current block: {current_block}, Epoch: {current_epoch}")
            bt.logging.info(f"Blocks to next epoch: {blocks_to_next_epoch}")
            
            if current_tracked_epoch is None or current_epoch > current_tracked_epoch:
                bt.logging.info(f"🔄 New epoch detected: {current_epoch} (previous: {current_tracked_epoch or 'None'})")
                current_tracked_epoch = current_epoch
                
                # Clear any cached data
                if 'torch' in sys.modules and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                epoch_boundary = current_epoch * state['epoch_length']
                block_hash = await state['subtensor'].determine_block_hash(epoch_boundary)
                bt.logging.info(f"Block hash for epoch {current_epoch}: {block_hash}")
                
                proteins = get_challenge_proteins_from_blockhash(
                    block_hash=block_hash,
                    num_targets=state['config'].num_targets,
                    num_antitargets=state['config'].num_antitargets
                )
                
                if not proteins:
                    bt.logging.error("Failed to get proteins from block hash")
                    await asyncio.sleep(state['config'].check_interval)
                    continue
                
                target_codes = proteins["targets"]
                antitarget_codes = proteins["antitargets"]
                
                bt.logging.info(f"Challenge targets: {target_codes}")
                bt.logging.info(f"Challenge antitargets: {antitarget_codes}")
                
                # Reset for new challenge
                state['last_submitted_molecule'] = None
                
                # Run the optimization
                await run_once(state)
                
                # Clean up memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                bt.logging.info("🧹 GPU memory explicitly cleaned up after epoch processing")
            
            await asyncio.sleep(state['config'].check_interval)
            
        except Exception as e:
            bt.logging.error(f"Error in continuous loop: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(60)
        except KeyboardInterrupt:
            bt.logging.info("Keyboard interrupt received. Exiting.")
            break

async def main():
    """Main function with optimized protein and molecule processing"""
    try:
        # Parse arguments
        config = parse_arguments()
        
        # Setup logging
        bt.logging(config=config)
        bt.logging.set_debug(True)
        
        bt.logging.info(f"Using configuration with {config.num_targets} targets and {config.num_antitargets} antitargets")
        bt.logging.info(f"Source: {config.source} ({config.csv_file if config.source == 'csv' else config.molecule_db})")
        bt.logging.info(f"Target weight: {config.target_weight}, Antitarget weight: {config.antitarget_weight}")
        bt.logging.info(f"Minimum heavy atoms: {config.min_heavy_atoms}")
        bt.logging.info(f"Auto-commit: {not config.no_commit}")
        bt.logging.info(f"Mode: {config.mode}")
        
        if config.shard_count > 1:
            bt.logging.info(f"Sharding: {config.shard_index+1}/{config.shard_count}")
        
        # Initialize objects
        wallet, subtensor, metagraph, miner_uid, epoch_length = await setup_bittensor_objects(config)
        
        # Create state dictionary with all necessary objects
        state = {
            'config': config,
            'wallet': wallet,
            'subtensor': subtensor,
            'metagraph': metagraph,
            'miner_uid': miner_uid,
            'epoch_length': epoch_length,
            'github_path': load_github_path(),
            'bdt': QuicknetBittensorDrandTimelock(),
            'last_submission_time': None,
            'last_submitted_molecule': None,
        }
        
        # Run in the specified mode
        if config.mode == 'once':
            await run_once(state)
        else:
            await run_continuous(state)
            
    except Exception as e:
        bt.logging.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_dotenv()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting...")
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
        import traceback
        traceback.print_exc()
