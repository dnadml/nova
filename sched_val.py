#!/usr/bin/env python3

import os
import sys
import time
import logging
import argparse
import subprocess
import traceback
from datetime import datetime, timedelta
import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("extraction_scheduler.log")
    ]
)
logger = logging.getLogger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Schedule collect_val.py to run before epoch end"
    )
    parser.add_argument("--interval", type=int, default=1,
                        help="Check interval in minutes (default: 1)")
    parser.add_argument("--blocks-before-end", type=int, default=20,
                        help="Run extraction when this many blocks remain until epoch end (default: 20)")
    parser.add_argument("--extraction-path", type=str, default="./val.py",
                        help="Path to validate.py (default: ./val.py)")
    parser.add_argument("--netuid", type=int, default=68,
                        help="Subnet ID (default: 68)")
    parser.add_argument("--network", type=str, default="finney",
                        help="Bittensor network (default: finney)")
    parser.add_argument("--archive-db", type=str, default="molecule_archive.db",
                        help="Path to molecule archive database (default: molecule_archive.db)")
    return parser.parse_args()

def check_blocks_until_epoch_end(netuid, network):
    """
    Check how many blocks remain until the next epoch end.
    Returns (blocks_until_next_epoch, current_block, current_epoch)
    """
    logger.info("Checking blocks until epoch end...")
    try:
        # Run an inline Python command to retrieve epoch information
        check_cmd = [
            "python3", "-c",
            f"""
import asyncio
import bittensor as bt

async def check():
    try:
        subtensor = bt.async_subtensor(network='{network}')
        await subtensor.initialize()
        epoch_length = (await subtensor.substrate.query(
            module="SubtensorModule",
            storage_function="Tempo",
            params=[{netuid}]
        )).value
        current_block = await subtensor.get_current_block()
        current_epoch = current_block // epoch_length
        blocks_until_next_epoch = epoch_length - (current_block % epoch_length)
        print(f"{{blocks_until_next_epoch}},{{current_block}},{{current_epoch}}")
    except Exception as e:
        print(f"ERROR: {{e}}")

asyncio.run(check())
            """
        ]
        result = subprocess.check_output(check_cmd).decode('utf-8').strip()
        if result.startswith("ERROR:"):
            logger.error(f"Error checking blocks: {result}")
            return None, None, None
        parts = result.split(',')
        blocks_until_next_epoch = int(parts[0])
        current_block = int(parts[1])
        current_epoch = int(parts[2])
        logger.info(f"Current block: {current_block}, Current epoch: {current_epoch}")
        logger.info(f"Blocks until next epoch: {blocks_until_next_epoch}")
        return blocks_until_next_epoch, current_block, current_epoch
    except Exception as e:
        logger.error(f"Error checking blocks until epoch end: {e}")
        traceback.print_exc()
        return None, None, None

def run_extraction(extraction_path, args, current_epoch=None):
    """Run the collect_val.py extraction script."""
    try:
        cmd = ["python3", extraction_path]
        # Add network and netuid arguments
        cmd.extend(["--network", args.network])
        cmd.extend(["--netuid", str(args.netuid)])
        cmd.extend(["--archive-db", args.archive_db])
        # If current_epoch is provided, pass it to the extraction script
        if current_epoch is not None:
            cmd.extend(["--epoch", str(current_epoch)])
        logger.info(f"Running extraction: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        for line in process.stdout:
            print(line, end='')
            logger.info(line.strip())
        process.wait()
        if process.returncode == 0:
            logger.info("Molecule extraction completed successfully")
            return True
        else:
            logger.error(f"Molecule extraction failed with return code: {process.returncode}")
            return False
    except Exception as e:
        logger.error(f"Error running molecule extraction: {e}")
        traceback.print_exc()
        return False

# Flag to track if extraction has already run for this epoch
last_extraction_epoch = None

def scheduled_check(args):
    """
    Scheduled job to check if we're near epoch end and run extraction if needed.
    """
    global last_extraction_epoch
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"\n\n=== Scheduled check at {timestamp} ===")
    print(f"\n\n=== Scheduled check at {timestamp} ===")
    
    blocks_until_next_epoch, current_block, current_epoch = check_blocks_until_epoch_end(
        args.netuid, args.network
    )
    
    if blocks_until_next_epoch is None:
        logger.error("Failed to check blocks until epoch end")
        return
        
    # Check if we already ran extraction for this epoch
    if last_extraction_epoch == current_epoch:
        logger.info(f"Already ran extraction for epoch {current_epoch}, skipping")
        print(f"Already ran extraction for epoch {current_epoch}, skipping")
        return
        
    if blocks_until_next_epoch <= args.blocks_before_end:
        logger.info(f"Within {args.blocks_before_end} blocks of epoch end, running extraction...")
        print(f"Within {args.blocks_before_end} blocks of epoch end, running extraction...")
        success = run_extraction(args.extraction_path, args, current_epoch)
        if success:
            last_extraction_epoch = current_epoch
            logger.info(f"Marked extraction as completed for epoch {current_epoch}")
    else:
        logger.info(f"Not within threshold ({blocks_until_next_epoch} > {args.blocks_before_end}), skipping extraction")
        print(f"Not within threshold ({blocks_until_next_epoch} > {args.blocks_before_end}), skipping extraction")

def main():
    logger.info("Starting molecule extraction scheduler")
    print("Starting molecule extraction scheduler")
    args = parse_args()
    logger.info(f"Arguments: {args}")
    
    if not os.path.exists(args.extraction_path):
        logger.error(f"Extraction script not found at: {args.extraction_path}")
        print(f"ERROR: Extraction script not found at: {args.extraction_path}")
        sys.exit(1)
        
    # Convert interval from minutes to seconds for internal use
    interval_seconds = args.interval * 60
    
    # Schedule the check to run every X minutes (not seconds)
    schedule.every(interval_seconds).seconds.do(scheduled_check, args)
    
    next_run = datetime.now() + timedelta(seconds=interval_seconds)
    logger.info(f"Scheduled check every {args.interval} minutes")
    logger.info(f"Will run extraction when within {args.blocks_before_end} blocks of epoch end")
    logger.info(f"Next check at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scheduled check every {args.interval} minutes")
    print(f"Will run extraction when within {args.blocks_before_end} blocks of epoch end")
    print(f"Next check at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # You can optionally run an initial check now or wait for the first scheduled run
    scheduled_check(args)
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        print("\nScheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        print(f"\nScheduler error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
