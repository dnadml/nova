#!/usr/bin/env python3

import os
import sys
import time
import logging
import argparse
import subprocess
import traceback
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
    parser = argparse.ArgumentParser(description="Scheduler to run scraper at specific block before epoch end")
    
    parser.add_argument("--blocks-before-end", type=int, default=310,
                      help="Run script when this many blocks remain until epoch end (default: 310)")
    parser.add_argument("--check-interval", type=float, default=3.0,
                      help="Check interval in seconds (default: 3.0)")
    parser.add_argument("--epochs", type=int, default=3,
                      help="Number of epochs to scrape (default: 3)")
    parser.add_argument("--csv-output", type=str, default="molecules.csv",
                      help="CSV output file (default: molecules.csv)")
    parser.add_argument("--netuid", type=int, default=68,
                      help="Subnet ID (default: 68)")
    parser.add_argument("--network", type=str, default="finney",
                      help="Bittensor network (default: finney)")
    
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
    logger.info("🚀 Running scraper")
    
    run_cmd = [
        "python", "scraper.py",
        "--fetch-smiles",
        "--csv-output", args.csv_output,
        "--epochs", str(args.epochs)
    ]
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            run_cmd, 
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
            logger.info("✅ Successfully completed scraper execution")
            return True
        else:
            logger.warning(f"❌ Scraper execution failed, return code: {process.returncode}")
            return False
    
    except Exception as e:
        logger.error(f"Error executing scraper: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main function."""
    args = parse_args()
    
    # Keep track of processed epochs
    last_processed_epoch = None
    
    logger.info(f"Starting Scraper Scheduler")
    logger.info(f"Will run scraper when exactly {args.blocks_before_end} blocks remain until epoch end")
    logger.info(f"Checking blocks every {args.check_interval} seconds")
    logger.info(f"Command to run: python scraper.py --fetch-smiles --csv-output {args.csv_output} --epochs {args.epochs}")
    
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
            
            # Check if it's time to run the scraper (exactly at the target block)
            if blocks_until_next_epoch <= args.blocks_before_end and blocks_until_next_epoch > (args.blocks_before_end - 3) and last_processed_epoch != current_epoch:
                logger.info(f"🔔 Exactly {args.blocks_before_end} blocks before epoch end, running scraper!")
                print(f"🔔 Exactly {args.blocks_before_end} blocks before epoch end, running scraper!")
                
                # Run the scraper
                success = run_scraper(args)
                
                # Mark this epoch as processed
                last_processed_epoch = current_epoch
                
                logger.info(f"Completed scraper execution for epoch {current_epoch}")
                
            else:
                if last_processed_epoch == current_epoch:
                    print(f"Already processed epoch {current_epoch}")
                else:
                    print(f"Not time to run yet. {blocks_until_next_epoch} blocks until epoch end (target: {args.blocks_before_end}).")
            
            # Sleep before next check
            time.sleep(args.check_interval)
            
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        print("\nScheduler stopped by user")

if __name__ == "__main__":
    main()
