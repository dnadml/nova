#!/usr/bin/env python3

import os
import sys
import time
import logging
import argparse
import subprocess
import traceback
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper_scheduler.log")
    ]
)

logger = logging.getLogger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Schedule scraper to run before epoch end and update molecules list")
    
    parser.add_argument("--blocks-before-end", type=int, default=20,
                      help="Run scraper when this many blocks remain until epoch end (default: 20)")
    parser.add_argument("--check-interval", type=float, default=3.0,
                      help="Check interval in seconds (default: 3.0)")
    parser.add_argument("--scraper-script", type=str, default="./scraper.py",
                      help="Path to scraper script")
    parser.add_argument("--temp-csv", type=str, default="./molecules.csv",
                      help="Path to temporary CSV output from scraper")
    parser.add_argument("--output-csv", type=str, default="./quick_mols.csv",
                      help="Path to final CSV that will be updated with new molecules")
    parser.add_argument("--netuid", type=int, default=68,
                      help="Subnet ID (default: 68)")
    parser.add_argument("--network", type=str, default="finney",
                      help="Bittensor network (default: finney)")
    parser.add_argument("--fetch-epochs", type=int, default=1,
                      help="Number of epochs to fetch (default: 1)")
    
    return parser.parse_args()

def check_blocks_until_epoch_end(netuid, network):
    """
    Check how many blocks remain until the next epoch end.
    Returns (blocks_until_next_epoch, current_block, current_epoch, epoch_length)
    """
    logger.info("Checking blocks until epoch end...")
    
    try:
        # Run a simple python command to get epoch info
        check_cmd = [
            "python3", "-c",
            f"""
import asyncio
import bittensor as bt

async def check():
    try:
        subtensor = bt.async_subtensor(network='{network}')
        await subtensor.initialize()
        
        # Get epoch length
        epoch_length = (await subtensor.substrate.query(
            module="SubtensorModule",
            storage_function="Tempo",
            params=[{netuid}]
        )).value
        
        # Get current block
        current_block = await subtensor.get_current_block()
        current_epoch = current_block // epoch_length
        blocks_until_next_epoch = epoch_length - (current_block % epoch_length)
        
        print(f"{{blocks_until_next_epoch}},{{current_block}},{{current_epoch}},{{epoch_length}}")
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
        current_block = int(parts[1])
        current_epoch = int(parts[2])
        epoch_length = int(parts[3])
        
        logger.info(f"Current block: {current_block}, Current epoch: {current_epoch}, Epoch length: {epoch_length}")
        logger.info(f"Blocks until next epoch: {blocks_until_next_epoch}")
        
        return blocks_until_next_epoch, current_block, current_epoch, epoch_length
        
    except Exception as e:
        logger.error(f"Error checking blocks until epoch end: {e}")
        traceback.print_exc()
        return None, None, None, None

def run_scraper(args):
    """Run the scraper script"""
    logger.info("🔍 Running scraper script")
    
    scraper_cmd = [
        "python3", args.scraper_script,
        "--fetch-smiles",
        "--csv-output", args.temp_csv,
        "--epochs", str(args.fetch_epochs),
        "--network", args.network,
        "--netuid", str(args.netuid)
    ]
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            scraper_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            logger.info(line.strip())
            print(line.strip())
            
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            logger.info("✅ Successfully ran scraper script")
            return True
        else:
            logger.warning(f"❌ Scraper script failed, return code: {process.returncode}")
            return False
    
    except Exception as e:
        logger.error(f"Error executing scraper command: {str(e)}")
        traceback.print_exc()
        return False

def update_molecules_file(temp_csv, output_csv):
    """Update the output CSV with new molecules without creating duplicates"""
    logger.info(f"Updating {output_csv} with new molecules from {temp_csv}")
    
    try:
        # Check if temp CSV exists
        if not os.path.exists(temp_csv):
            logger.error(f"Temporary CSV file not found: {temp_csv}")
            return False
            
        # Read the temp CSV
        new_molecules = pd.read_csv(temp_csv)
        logger.info(f"Found {len(new_molecules)} molecules in the temporary CSV")
        
        # Check if output CSV exists, create it if it doesn't
        if not os.path.exists(output_csv):
            logger.info(f"Output CSV doesn't exist, creating: {output_csv}")
            new_molecules.to_csv(output_csv, index=False)
            logger.info(f"Added {len(new_molecules)} new molecules to {output_csv}")
            return True
            
        # Read the existing output CSV
        existing_molecules = pd.read_csv(output_csv)
        logger.info(f"Found {len(existing_molecules)} existing molecules in {output_csv}")
        
        # Determine the key columns to check for duplicates
        # Assuming 'molecule' column is unique identifier - adjust as needed
        id_column = 'molecule' if 'molecule' in new_molecules.columns else new_molecules.columns[0]
        
        # Find new molecules (not in the existing CSV)
        existing_ids = set(existing_molecules[id_column])
        new_entries = new_molecules[~new_molecules[id_column].isin(existing_ids)]
        
        # If there are new entries, append them to the output CSV
        if len(new_entries) > 0:
            logger.info(f"Found {len(new_entries)} new molecules to add")
            
            # Combine existing and new
            updated_molecules = pd.concat([existing_molecules, new_entries], ignore_index=True)
            
            # Save to output CSV
            updated_molecules.to_csv(output_csv, index=False)
            logger.info(f"Updated {output_csv} with {len(new_entries)} new molecules")
            
            return True
        else:
            logger.info("No new molecules found to add")
            return True
            
    except Exception as e:
        logger.error(f"Error updating molecules file: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main function."""
    args = parse_args()
    
    # Keep track of processed epochs
    last_processed_epoch = None
    
    logger.info(f"Starting Scraper Scheduler")
    logger.info(f"Will run scraper when exactly {args.blocks_before_end} blocks before epoch end")
    logger.info(f"Checking blocks every {args.check_interval} seconds")
    
    try:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\nChecking blocks at {current_time}...")
            
            # Check blocks until epoch end
            blocks_until_next_epoch, current_block, current_epoch, epoch_length = check_blocks_until_epoch_end(
                args.netuid, args.network
            )
            
            if blocks_until_next_epoch is None:
                logger.error("Failed to check blocks, will retry...")
                time.sleep(args.check_interval)
                continue
            
            # Check if it's time to run the scraper
            if blocks_until_next_epoch == args.blocks_before_end and last_processed_epoch != current_epoch:
                logger.info(f"🔔 Exactly {args.blocks_before_end} blocks before epoch end, running scraper!")
                print(f"🔔 Exactly {args.blocks_before_end} blocks before epoch end, running scraper!")
                
                # Run scraper
                success = run_scraper(args)
                
                if success:
                    # Update molecules file
                    update_success = update_molecules_file(args.temp_csv, args.output_csv)
                    if update_success:
                        logger.info("✅ Successfully updated molecules file")
                    else:
                        logger.error("❌ Failed to update molecules file")
                
                # Mark this epoch as processed
                last_processed_epoch = current_epoch
                
                logger.info(f"Completed scraper process for epoch {current_epoch}")
                
            else:
                if last_processed_epoch == current_epoch:
                    print(f"Already processed epoch {current_epoch}")
                else:
                    print(f"Not time to run scraper yet. {blocks_until_next_epoch} blocks until epoch end.")
            
            # Sleep before next check
            time.sleep(args.check_interval)
            
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        print("\nScheduler stopped by user")

if __name__ == "__main__":
    main()
