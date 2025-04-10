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
        logging.FileHandler("missing_scheduler.log")
    ]
)

logger = logging.getLogger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Scheduler to run missing.py at specific block in epoch")
    
    parser.add_argument("--blocks-after-start", type=int, default=60,
                      help="Run script when this many blocks after epoch start (default: 60)")
    parser.add_argument("--check-interval", type=float, default=3.0,
                      help="Check interval in seconds (default: 3.0)")
    parser.add_argument("--run-script", type=str, default="./update.py",
                      help="Path to run script (default: ./missing.py)")
    parser.add_argument("--db-path", type=str, default="molecule_archive.db",
                      help="Path to molecule archive database (default: molecule_archive.db)")
    parser.add_argument("--max-concurrent", type=int, default=5,
                      help="Maximum concurrent SMILES requests (default: 5)")
    parser.add_argument("--netuid", type=int, default=68,
                      help="Subnet ID (default: 68)")
    parser.add_argument("--network", type=str, default="finney",
                      help="Bittensor network (default: finney)")
    
    return parser.parse_args()

def check_blocks_since_epoch_start(netuid, network):
    """
    Check how many blocks have passed since the epoch start.
    Returns (blocks_since_epoch_start, current_block, current_epoch, epoch_length)
    """
    logger.info("Checking blocks since epoch start...")
    
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
        blocks_since_epoch_start = current_block % epoch_length
        
        print(f"{{blocks_since_epoch_start}},{{current_block}},{{current_epoch}},{{epoch_length}}")
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
        blocks_since_epoch_start = int(parts[0])
        current_block = int(parts[1])
        current_epoch = int(parts[2])
        epoch_length = int(parts[3])
        
        logger.info(f"Current block: {current_block}, Current epoch: {current_epoch}, Epoch length: {epoch_length}")
        logger.info(f"Blocks since epoch start: {blocks_since_epoch_start}")
        
        return blocks_since_epoch_start, current_block, current_epoch, epoch_length
        
    except Exception as e:
        logger.error(f"Error checking blocks since epoch start: {e}")
        traceback.print_exc()
        return None, None, None, None

def run_missing_script(args):
    """Run the missing.py script"""
    logger.info("🚀 Running missing.py script")
    
    run_cmd = [
        "python", args.run_script,
        args.db_path,
        str(args.max_concurrent)
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
            logger.info("✅ Successfully completed missing.py execution")
            return True
        else:
            logger.warning(f"❌ Script execution failed, return code: {process.returncode}")
            return False
    
    except Exception as e:
        logger.error(f"Error executing script: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main function."""
    args = parse_args()
    
    # Keep track of processed epochs
    last_processed_epoch = None
    
    logger.info(f"Starting Missing SMILES Scheduler")
    logger.info(f"Will run missing.py when exactly {args.blocks_after_start} blocks after epoch start")
    logger.info(f"Checking blocks every {args.check_interval} seconds")
    logger.info(f"Script to run: python {args.run_script} {args.db_path} {args.max_concurrent}")
    
    try:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\nChecking blocks at {current_time}...")
            
            # Check blocks since epoch start
            blocks_since_epoch_start, current_block, current_epoch, epoch_length = check_blocks_since_epoch_start(
                args.netuid, args.network
            )
            
            if blocks_since_epoch_start is None:
                logger.error("Failed to check blocks, will retry...")
                time.sleep(args.check_interval)
                continue
            
            # Check if it's time to run the script (exactly at the target block)
            if blocks_since_epoch_start >= args.blocks_after_start and blocks_since_epoch_start < (args.blocks_after_start + 3) and last_processed_epoch != current_epoch:
                logger.info(f"🔔 Exactly {args.blocks_after_start} blocks after epoch start, running missing.py!")
                print(f"🔔 Exactly {args.blocks_after_start} blocks after epoch start, running missing.py!")
                
                # Run the script
                success = run_missing_script(args)
                
                # Mark this epoch as processed
                last_processed_epoch = current_epoch
                
                logger.info(f"Completed script execution for epoch {current_epoch}")
                
            else:
                if last_processed_epoch == current_epoch:
                    print(f"Already processed epoch {current_epoch}")
                else:
                    print(f"Not time to run yet. {blocks_since_epoch_start} blocks since epoch start (target: {args.blocks_after_start}).")
            
            # Sleep before next check
            time.sleep(args.check_interval)
            
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        print("\nScheduler stopped by user")

if __name__ == "__main__":
    main()
