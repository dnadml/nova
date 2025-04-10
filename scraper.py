#!/usr/bin/env python3
"""
NOVA Leaderboard Molecule Scraper

This script scrapes molecule names from the NOVA dashboard leaderboard for the most recent epochs,
fetches SMILES strings for each molecule using the smiles_scraper.py module, and saves them to a
molecules.txt file before loading them into the molecule archive database.

It automatically detects the current epoch from the blockchain and can run as a scheduled task
that executes at epoch start + 30 blocks.

Requirements:
- Python 3.6+
- requests
- beautifulsoup4
- selenium (with chromedriver)
- tqdm
- schedule (for scheduled execution)

Install dependencies:
pip install requests beautifulsoup4 selenium webdriver-manager tqdm schedule
"""

import os
import re
import time
import random
import argparse
import logging
import subprocess
import threading
import traceback
from typing import List, Set, Optional, Tuple, Dict
from datetime import datetime, timedelta
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import schedule

# Import SMILES scraper module
try:
    from smiles_scraper import (fetch_smiles_batch, load_existing_smiles_db, 
                               save_smiles_db, save_molecules_as_csv, clean_smiles)
    SMILES_SCRAPER_AVAILABLE = True
except ImportError:
    SMILES_SCRAPER_AVAILABLE = False
    print("WARNING: smiles_scraper.py module not found. SMILES fetching will be disabled.")
    print("Please ensure smiles_scraper.py is in the same directory as this script.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("nova_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://nova-dashboard-frontend.vercel.app/leaderboard?epoch_number="
DEFAULT_EPOCHS_TO_SCRAPE = 5
DEFAULT_OUTPUT_FILE = "molecules.txt"
DEFAULT_TOP_WINNERS_FILE = "top_molecules.txt"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
DELAY_BETWEEN_REQUESTS = 2  # seconds
def get_current_epoch_from_blockchain(args) -> Optional[int]:
    """
    Get the current epoch directly from the blockchain, then subtract 1 to get
    the latest completed epoch for which a leaderboard would exist.
    
    Args:
        args: Command line arguments containing network and netuid info
        
    Returns:
        The latest completed epoch number or None if retrieval fails
    """
    logger.info("Getting current epoch directly from the blockchain...")
    
    try:
        # Run a simple python command to get current epoch info
        check_cmd = [
            "python3", "-c",
            f"""
import asyncio
import bittensor as bt

async def check():
    try:
        subtensor = bt.async_subtensor(network='{args.network}')
        await subtensor.initialize()
        
        # Get epoch length
        epoch_length = (await subtensor.substrate.query(
            module="SubtensorModule",
            storage_function="Tempo",
            params=[{args.netuid}]
        )).value
        
        # Get current block
        current_block = await subtensor.get_current_block()
        current_epoch = current_block // epoch_length
        
        # The latest completed epoch is the current_epoch - 1
        latest_completed_epoch = current_epoch - 1
        
        print(f"{{latest_completed_epoch}}")
    except Exception as e:
        print(f"ERROR: {{e}}")

asyncio.run(check())
            """
        ]
        
        result = subprocess.check_output(check_cmd).decode('utf-8').strip()
        
        if result.startswith("ERROR:"):
            logger.error(f"Error checking epoch: {result}")
            return None
        
        latest_completed_epoch = int(result)
        logger.info(f"Latest completed epoch from blockchain: {latest_completed_epoch}")
        return latest_completed_epoch
        
    except Exception as e:
        logger.error(f"Error getting epoch from blockchain: {e}", exc_info=True)
        return None

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

def extract_molecules_from_html(html_content: str) -> List[str]:
    """
    Extract molecule IDs from HTML content.
    
    Args:
        html_content: HTML content as string
        
    Returns:
        List of molecule IDs
    """
    # Define potential regex patterns for molecule IDs
    patterns = [
        r'[A-F0-9]{16}_[A-F0-9]{16}_\d{4}_[A-Z]{2}',  # Standard format
        r'[A-F0-9]+_[A-F0-9]+_\d+_[A-Z]+',            # More flexible format
    ]
    
    molecules = []
    for pattern in patterns:
        matches = re.findall(pattern, html_content)
        if matches:
            molecules.extend(matches)
            
    # Remove duplicates while preserving order
    seen = set()
    return [x for x in molecules if not (x in seen or seen.add(x))]
def extract_molecules_with_ranks_from_soup(soup: BeautifulSoup) -> List[Tuple[int, float, str]]:
    """
    Extract molecule IDs with their ranks and scores from BeautifulSoup object.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        List of tuples (rank, score, molecule_name)
    """
    ranked_molecules = []
    
    # Method 1: Look for table rows with molecule data and extract ranks
    if soup.find('table'):
        rows = soup.find_all('tr')
        current_rank = None
        current_score = None
        
        for row in rows:
            # Look for rank row (with score)
            rank_cell = row.find('td', string=lambda t: t and t.strip().startswith('#'))
            if rank_cell:
                cells = row.find_all('td')
                if len(cells) >= 4:  # Has enough cells for rank, score
                    try:
                        current_rank = int(rank_cell.text.strip('#').strip())
                        score_cell = cells[3]  # Typically the score is in the 4th column
                        current_score = float(score_cell.text.strip())
                    except (ValueError, IndexError):
                        continue
            
            # Look for molecule row
            cells = row.find_all('td')
            if len(cells) == 1:  # Molecule rows typically have just one cell
                text = cells[0].get_text().strip()
                if re.match(r'^[A-F0-9_]+_[A-F0-9_]+_\d+_[A-Z]+$', text) and current_rank is not None:
                    ranked_molecules.append((current_rank, current_score, text))
    
    # If method 1 fails, just extract molecules without rank information
    if not ranked_molecules:
        molecules = extract_molecules_from_html(str(soup))
        ranked_molecules = [(i+1, 0.0, mol) for i, mol in enumerate(molecules)]
    
    return ranked_molecules

def scrape_leaderboard(epoch_number: int) -> List[Tuple[int, float, str]]:
    """
    Scrape molecules with their ranks from a specific epoch leaderboard.
    
    Args:
        epoch_number: Epoch number to scrape
        
    Returns:
        List of tuples (rank, score, molecule_name)
    """
    logger.info(f"Scraping epoch {epoch_number}...")
    
    url = f"{BASE_URL}{epoch_number}"
    soup = get_soup_from_url(url)
    
    if not soup:
        logger.error(f"Failed to retrieve data for epoch {epoch_number}")
        return []
    
    ranked_molecules = extract_molecules_with_ranks_from_soup(soup)
    logger.info(f"Found {len(ranked_molecules)} ranked molecules in epoch {epoch_number}")
    
    return ranked_molecules

def run_molecule_loader(molecules_file: str) -> None:
    """
    Run the molecule_loader.py script on the generated molecules file.
    
    Args:
        molecules_file: Path to the molecules file
    """
    logger.info(f"Running molecule_loader.py on {molecules_file}...")
    
    try:
        result = subprocess.run(
            ["python", "molecule_loader.py", "--molecules-file", molecules_file],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("Molecule loader completed successfully")
        
        # Log stdout if there's any output
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                logger.info(f"Molecule loader: {line}")
        
        # Log stderr as warnings
        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                logger.warning(f"Molecule loader: {line}")
                
    except subprocess.CalledProcessError as e:
        logger.error(f"Molecule loader failed with exit code {e.returncode}")
        
        if e.stdout:
            for line in e.stdout.strip().split('\n'):
                logger.info(f"Molecule loader: {line}")
        
        if e.stderr:
            for line in e.stderr.strip().split('\n'):
                logger.error(f"Molecule loader error: {line}")
                
    except FileNotFoundError:
        logger.error("Error: molecule_loader.py script not found in the current directory")
        
    except Exception as e:
        logger.error(f"Unexpected error running molecule_loader.py: {e}", exc_info=True)

def extract_from_paste_file(file_path: str = "paste.txt") -> Set[str]:
    """
    Extract molecules from a paste file.
    
    Args:
        file_path: Path to the paste file
        
    Returns:
        Set of molecule IDs
    """
    if not os.path.exists(file_path):
        logger.warning(f"Paste file not found: {file_path}")
        return set()
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        molecules = extract_molecules_from_html(content)
        logger.info(f"Extracted {len(molecules)} molecules from {file_path}")
        return set(molecules)
        
    except Exception as e:
        logger.error(f"Failed to extract molecules from {file_path}: {e}", exc_info=True)
        return set()
def try_selenium_backup(epoch: int) -> List[Tuple[int, float, str]]:
    """
    Backup method using Selenium if requests fails.
    This is only used when the primary method fails.
    
    Args:
        epoch: Epoch number to scrape
        
    Returns:
        List of tuples (rank, score, molecule_name)
    """
    try:
        # Only import these if needed to avoid unnecessary dependencies
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        logger.info(f"Trying selenium backup for epoch {epoch}...")
        
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument(f"user-agent={USER_AGENT}")
        
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        try:
            driver.get(f"{BASE_URL}{epoch}")
            
            # Wait for the page to load (wait for table to appear)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            
            # Get the page source and parse it with BeautifulSoup
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            ranked_molecules = extract_molecules_with_ranks_from_soup(soup)
            
            logger.info(f"Selenium found {len(ranked_molecules)} molecules in epoch {epoch}")
            return ranked_molecules
            
        finally:
            driver.quit()
            
    except Exception as e:
        logger.error(f"Selenium backup failed for epoch {epoch}: {e}", exc_info=True)
        return []

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Scrape NOVA leaderboard molecules")
    parser.add_argument(
        "--current-epoch", 
        type=int, 
        default=None,
        help="Current epoch number (default: auto-detect from blockchain)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=DEFAULT_EPOCHS_TO_SCRAPE,
        help=f"Number of epochs to scrape (default: {DEFAULT_EPOCHS_TO_SCRAPE})"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output file name (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--top", 
        type=int, 
        default=0,
        help="Number of top molecules to extract (0 means don't extract top molecules)"
    )
    parser.add_argument(
        "--top-output", 
        type=str, 
        default=DEFAULT_TOP_WINNERS_FILE,
        help=f"Output file for top molecules (default: {DEFAULT_TOP_WINNERS_FILE})"
    )
    parser.add_argument(
        "--no-run-loader", 
        action="store_true",
        help="Don't run molecule_loader.py after scraping"
    )
    parser.add_argument(
        "--top-only", 
        action="store_true",
        help="Only save top molecules, skip saving all molecules"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    parser.add_argument(
        "--schedule", 
        action="store_true",
        help="Run in scheduled mode at epoch start + 30 blocks"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=5,
        help="Check interval in minutes when in scheduled mode (default: 5)"
    )
    parser.add_argument(
        "--netuid", 
        type=int, 
        default=68,
        help="Subnet ID for epoch monitoring (default: 68)"
    )
    parser.add_argument(
        "--network", 
        type=str, 
        default="finney",
        help="Bittensor network (default: finney)"
    )
    parser.add_argument(
        "--blocks-after-start", 
        type=int, 
        default=30,
        help="Execute scraper when this many blocks after epoch start (default: 30)"
    )
    parser.add_argument(
        "--check-interval", 
        type=int, 
        default=30,
        help="Check interval in seconds for block checking (default: 2)"
    )
    parser.add_argument(
        "--fetch-smiles", 
        action="store_true",
        help="Fetch SMILES strings for each molecule (default: True)"
    )
    parser.add_argument(
        "--no-fetch-smiles", 
        action="store_true",
        help="Don't fetch SMILES strings for each molecule"
    )
    parser.add_argument(
        "--smiles-db", 
        type=str, 
        default="smiles_db.txt",
        help="File to store molecule-to-SMILES mappings (default: smiles_db.txt)"
    )
    parser.add_argument(
        "--max-concurrent", 
        type=int, 
        default=5,
        help="Maximum number of concurrent SMILES requests (default: 5)"
    )
    parser.add_argument(
        "--csv-output", 
        type=str, 
        default="",
        help="Save molecule data to a CSV file (specify filename)"
    )
    return parser.parse_args()
# Functions for epoch scheduling
def check_blocks_until_epoch_end(args):
    """
    Check how many blocks remain until the next epoch end and how many blocks
    since the epoch started.
    
    Returns (blocks_until_next_epoch, blocks_since_epoch_start, current_block, current_epoch)
    """
    logger.info("Checking epoch block information...")
    
    try:
        # Run a simple python command to get epoch info
        check_cmd = [
            "python3", "-c",
            f"""
import asyncio
import bittensor as bt

async def check():
    try:
        subtensor = bt.async_subtensor(network='{args.network}')
        await subtensor.initialize()
        
        # Get epoch length
        epoch_length = (await subtensor.substrate.query(
            module="SubtensorModule",
            storage_function="Tempo",
            params=[{args.netuid}]
        )).value
        
        # Get current block
        current_block = await subtensor.get_current_block()
        current_epoch = current_block // epoch_length
        blocks_until_next_epoch = epoch_length - (current_block % epoch_length)
        blocks_since_epoch_start = (current_block % epoch_length)
        
        print(f"{{blocks_until_next_epoch}},{{blocks_since_epoch_start}},{{current_block}},{{current_epoch}}")
    except Exception as e:
        print(f"ERROR: {{e}}")

asyncio.run(check())
            """
        ]
        
        result = subprocess.check_output(check_cmd).decode('utf-8').strip()
        
        if result.startswith("ERROR:"):
            logger.error(f"Error checking blocks: {result}")
            return None, None, None, None
        
        parts = result.split(',')
        blocks_until_next_epoch = int(parts[0])
        blocks_since_epoch_start = int(parts[1])
        current_block = int(parts[2])
        current_epoch = int(parts[3])
        
        logger.info(f"Current block: {current_block}, Current epoch: {current_epoch}")
        logger.info(f"Blocks since epoch start: {blocks_since_epoch_start}")
        logger.info(f"Blocks until next epoch: {blocks_until_next_epoch}")
        
        return blocks_until_next_epoch, blocks_since_epoch_start, current_block, current_epoch
        
    except Exception as e:
        logger.error(f"Error checking epoch block information: {e}")
        traceback.print_exc()
        return None, None, None, None

def scheduled_check(args):
    """
    Scheduled job to check if we're at epoch start + N blocks and run the scraper if so.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"\n\n=== Scheduled check at {timestamp} ===")
    print(f"\n\n=== Scheduled check at {timestamp} ===")
    
    try:
        # Check epoch block information
        blocks_until_next_epoch, blocks_since_epoch_start, current_block, current_epoch = check_blocks_until_epoch_end(args)
        
        if blocks_since_epoch_start is None:
            logger.error("Failed to check epoch block information")
            return
        
        # Static global to track last processed epoch
        if not hasattr(scheduled_check, "last_processed_epoch"):
            scheduled_check.last_processed_epoch = None
        
        # Check if we're at the target block after epoch start and haven't processed this epoch yet
        if blocks_since_epoch_start == args.blocks_after_start and scheduled_check.last_processed_epoch != current_epoch:
            logger.info(f"🔔 Exactly {args.blocks_after_start} blocks after epoch start, running scraper!")
            print(f"🔔 Exactly {args.blocks_after_start} blocks after epoch start, running scraper!")
            
            # The current epoch is already ongoing, so we want to scrape the previous epoch
            completed_epoch = current_epoch - 1
            
            # Run the scraper with the completed epoch
            args.current_epoch = completed_epoch
            run_scraper(args)
            
            # Update last processed epoch
            scheduled_check.last_processed_epoch = current_epoch
            
            logger.info(f"Completed scraping for epoch {completed_epoch}")
            print(f"Completed scraping for epoch {completed_epoch}")
        elif blocks_since_epoch_start == args.blocks_after_start:
            logger.info(f"Already processed scraping for epoch {current_epoch}")
            print(f"Already processed scraping for epoch {current_epoch}")
        else:
            logger.info(f"Not at target block ({blocks_since_epoch_start} != {args.blocks_after_start}), waiting...")
            print(f"Not at target block ({blocks_since_epoch_start} != {args.blocks_after_start}), waiting...")
    
    except Exception as e:
        logger.error(f"Error in scheduled check: {e}")
        print(f"Error in scheduled check: {e}")
        traceback.print_exc()

def run_scheduler(args):
    """Run the scheduler with the given arguments."""
    logger.info("Starting Nova Leaderboard Scraper Scheduler")
    print("Starting Nova Leaderboard Scraper Scheduler")
    
    logger.info(f"Arguments: {args}")
    
    # Schedule the check to run at the specified interval
    schedule.every(args.interval).minutes.do(scheduled_check, args)
    
    # Log next scheduled run
    next_run = datetime.now() + timedelta(minutes=args.interval)
    logger.info(f"Scheduled check every {args.interval} minutes")
    logger.info(f"Will run scraper at exactly {args.blocks_after_start} blocks after epoch start")
    logger.info(f"Next check at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"Scheduled check every {args.interval} minutes")
    print(f"Will run scraper at exactly {args.blocks_after_start} blocks after epoch start")
    print(f"Next check at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run once immediately
    logger.info("Running initial check now...")
    print("Running initial check now...")
    scheduled_check(args)
    
    # Main loop - check at shorter intervals than scheduled to catch exactly the right block
    try:
        # Track the last check time
        last_check_time = time.time()
        
        while True:
            schedule.run_pending()
            
            # Also do a check with higher frequency than the scheduled interval
            # This ensures we don't miss the exact block we want
            if time.time() - last_check_time >= args.check_interval:
                scheduled_check(args)
                last_check_time = time.time()
            
            time.sleep(1)  # Sleep briefly between iterations
            
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        print("\nScheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        print(f"\nScheduler error: {e}")
        traceback.print_exc()
def run_scraper(args):
    """Run the scraper with the given arguments."""
    start_time = datetime.now()
    logger.info(f"Starting scraper at {start_time}")
    
    # If current_epoch is not provided, try to auto-detect or use the command line argument
    if args.current_epoch is None:
        current_epoch = get_current_epoch_from_blockchain(args)
        if current_epoch is None:
            logger.error("Failed to get current epoch from blockchain. Please provide it using --current-epoch")
            return False
        args.current_epoch = current_epoch
    
    logger.info(f"Using current epoch: {args.current_epoch}")
    logger.info(f"Will scrape epochs {args.current_epoch} to {args.current_epoch - args.epochs + 1}")
    
    all_molecules = set()
    molecule_stats = defaultdict(list)  # {molecule: [(rank, score, epoch), ...]}
    
    start_epoch = args.current_epoch
    end_epoch = args.current_epoch - args.epochs + 1
    
    # Create progress bar
    progress_bar = tqdm(range(start_epoch, end_epoch - 1, -1), desc="Scraping epochs")
    
    for epoch in progress_bar:
        try:
            progress_bar.set_description(f"Scraping epoch {epoch}")
            ranked_molecules = scrape_leaderboard(epoch)
            
            # Add molecules to the set and collect rank/score information
            for rank, score, molecule in ranked_molecules:
                all_molecules.add(molecule)
                molecule_stats[molecule].append((rank, score, epoch))
            
            # Add a delay to avoid overwhelming the server
            if epoch > end_epoch:
                time.sleep(DELAY_BETWEEN_REQUESTS)
            
        except Exception as e:
            logger.error(f"Failed to process epoch {epoch}: {e}", exc_info=True)
    
    # Process SMILES if requested and available - default is True unless --no-fetch-smiles is specified
    molecule_to_smiles = {}
    should_fetch_smiles = args.fetch_smiles and not args.no_fetch_smiles
    
    if should_fetch_smiles:
        if SMILES_SCRAPER_AVAILABLE:
            # Load existing SMILES database
            molecule_to_smiles = load_existing_smiles_db(args.smiles_db)
            
            # Determine which molecules need SMILES lookup
            molecules_needing_smiles = [m for m in all_molecules if m not in molecule_to_smiles or not molecule_to_smiles[m]]
            
            if molecules_needing_smiles:
                logger.info(f"Need to fetch SMILES for {len(molecules_needing_smiles)} new molecules")
                
                # Save molecules to a temporary file for smiles_scraper
                temp_molecules_file = "temp_molecules_for_smiles.txt"
                with open(temp_molecules_file, 'w') as f:
                    for molecule in molecules_needing_smiles:
                        f.write(f"{molecule}\n")
                
                # Fetch SMILES for molecules that don't have them yet
                new_smiles = fetch_smiles_batch(molecules_needing_smiles, args.max_concurrent)
                molecule_to_smiles.update(new_smiles)
                
                # Save updated SMILES database
                save_smiles_db(molecule_to_smiles, args.smiles_db)
                
                # Clean up temporary file
                try:
                    os.remove(temp_molecules_file)
                except:
                    pass
            else:
                logger.info("All molecules already have SMILES strings in the database")
        else:
            logger.warning("SMILES fetching requested but smiles_scraper.py module not available")
    
    # Save as CSV if requested
    if args.csv_output and SMILES_SCRAPER_AVAILABLE:
        save_molecules_as_csv(molecule_to_smiles, args.csv_output, list(all_molecules))

# Write all molecules to the output file if not in top-only mode
    if not args.top_only:
        with open(args.output, 'w') as f:
            for molecule in sorted(all_molecules):
                smiles = molecule_to_smiles.get(molecule, "")
                f.write(f"{molecule}\t{smiles}\n")
        
        logger.info(f"Scraping complete! Found {len(all_molecules)} unique molecules.")
        logger.info(f"Results saved to {args.output}")
        
        # Run molecule_loader.py on the all molecules file if not disabled
        # Run molecule_loader.py on the CSV file if not disabled
    if not args.no_run_loader and args.csv_output:
        run_molecule_loader(args.csv_output)
    
    # Process and save top molecules if requested
    if args.top > 0:
        # Calculate stats for each molecule
        top_molecules_data = []
        for molecule, appearances in molecule_stats.items():
            frequency = len(appearances)
            wins = sum(1 for a in appearances if a[0] == 1)  # Count number of #1 rankings
            avg_rank = sum(a[0] for a in appearances) / frequency
            best_rank = min(a[0] for a in appearances)
            avg_score = sum(a[1] for a in appearances) / frequency
            epochs_seen = sorted([a[2] for a in appearances], reverse=True)
            win_epochs = sorted([a[2] for a in appearances if a[0] == 1], reverse=True)
            smiles = molecule_to_smiles.get(molecule, "")
            
            top_molecules_data.append({
                'molecule': molecule,
                'smiles': smiles,
                'wins': wins,
                'frequency': frequency,
                'win_ratio': wins / frequency if frequency > 0 else 0,
                'avg_rank': avg_rank,
                'best_rank': best_rank,
                'avg_score': avg_score,
                'epochs_seen': epochs_seen,
                'win_epochs': win_epochs
            })
# Sort by number of wins (desc), then win ratio (desc), then avg_score (desc)
        top_molecules_data.sort(key=lambda x: (-x['wins'], -x['win_ratio'], -x['avg_score']))
        
        # Get top N molecules
        top_n = top_molecules_data[:args.top]
        
        # Save to file with detailed information
        with open(args.top_output, 'w') as f:
            f.write(f"# Top {len(top_n)} NOVA Leaderboard Winners\n")
            f.write(f"# Molecules ranked by number of #1 placements\n")
            f.write(f"# Scraped from epochs {start_epoch} to {end_epoch}\n")
            f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for idx, mol_data in enumerate(top_n, 1):
                f.write(f"#{idx}: {mol_data['molecule']}\n")
                if mol_data['smiles']:
                    f.write(f"  SMILES: {mol_data['smiles']}\n")
                f.write(f"  Wins: {mol_data['wins']} (ranked #1 in {mol_data['wins']} epochs)\n")
                f.write(f"  Frequency: {mol_data['frequency']}/{args.epochs} epochs\n")
                f.write(f"  Win Ratio: {mol_data['win_ratio']:.1%} of appearances\n")
                f.write(f"  Best Rank: #{mol_data['best_rank']}\n")
                f.write(f"  Avg Rank: #{mol_data['avg_rank']:.1f}\n")
                f.write(f"  Avg Score: {mol_data['avg_score']:.3f}\n")
                
                if mol_data['win_epochs']:
                    f.write(f"  Winning Epochs: {', '.join(map(str, mol_data['win_epochs'][:5]))}")
                    if len(mol_data['win_epochs']) > 5:
                        f.write(f" + {len(mol_data['win_epochs']) - 5} more")
                    f.write("\n")
                    
                f.write(f"  Most Recent Epochs: {', '.join(map(str, mol_data['epochs_seen'][:5]))}")
                if len(mol_data['epochs_seen']) > 5:
                    f.write(f" + {len(mol_data['epochs_seen']) - 5} more")
                f.write("\n\n")
            
            # Add raw list at the end for easy copy-paste
            f.write("# Raw molecule list (for easy copy-paste):\n")
            for mol_data in top_n:
                f.write(f"{mol_data['molecule']}\t{mol_data['smiles']}\n")
        
        logger.info(f"Saved top {len(top_n)} molecules to {args.top_output}")
        
        # If top-only mode is enabled, run molecule_loader on the top molecules
        if args.top_only and not args.no_run_loader:
            # Create a file with just the molecule names for the loader
            loader_file = f"top_{args.top}_molecules_for_loader.txt"
            with open(loader_file, 'w') as f:
                for mol_data in top_n:
                    f.write(f"{mol_data['molecule']}\t{mol_data['smiles']}\n")
            
            run_molecule_loader(loader_file)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Total time: {duration:.2f} seconds")
    return True

def main():
    """Main function to scrape molecules from multiple epochs."""
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    # Check if SMILES scraper is available when requested
    if args.fetch_smiles and not args.no_fetch_smiles and not SMILES_SCRAPER_AVAILABLE:
        logger.warning("WARNING: SMILES fetching requested but smiles_scraper.py module not found.")
        logger.warning("SMILES fetching will be disabled. Please ensure smiles_scraper.py is in the same directory.")
    
    # If running in scheduled mode, start the scheduler
    if args.schedule:
        run_scheduler(args)
    else:
        # Run the scraper once
        # If current epoch was not manually specified, get it from blockchain
        if args.current_epoch is None:
            current_epoch = get_current_epoch_from_blockchain(args)
            if current_epoch is None:
                logger.error("Failed to get current epoch from blockchain and no --current-epoch was specified")
                logger.error("Please run the script again with --current-epoch EPOCH_NUMBER")
                return
            args.current_epoch = current_epoch
            
        # Validate the epoch number
        if args.current_epoch <= 0:
            logger.error(f"Invalid epoch number: {args.current_epoch}")
            logger.error("Please run the script again with --current-epoch EPOCH_NUMBER")
            return
            
        run_scraper(args)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user.")
    except Exception as e:
        logger.critical(f"Fatal error in scraping process: {e}", exc_info=True)
