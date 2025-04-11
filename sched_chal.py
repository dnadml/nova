#!/usr/bin/env python3

import os
import sys
import time
import logging
import argparse
import subprocess
import traceback
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("run_scheduler.log")
    ]
)

logger = logging.getLogger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pure timer-based scheduler with minimal blockchain queries")
    
    parser.add_argument("--blocks-before-end", type=int, default=1,
                      help="Run script when this many blocks remain until epoch end (default: 1)")
    parser.add_argument("--check-interval", type=float, default=3.0,
                      help="Resync with blockchain every N epochs (default: 3.0)")
    parser.add_argument("--run-script", type=str, default="./run.py",
                      help="Path to run script")
    parser.add_argument("--wallet-name", type=str, default="code3",
                      help="Wallet name (default: code3)")
    parser.add_argument("--wallet-hotkey", type=str, default="m1",
                      help="Wallet hotkey (default: m1)")
    parser.add_argument("--antitarget-weight", type=float, default=1.0,
                      help="Antitarget weight (default: 1.0)")
    parser.add_argument("--netuid", type=int, default=68,
                      help="Subnet ID (default: 68)")
    parser.add_argument("--network", type=str, default="finney",
                      help="Bittensor network (default: finney)")
    parser.add_argument("--block-time", type=float, default=12.0,
                      help="Average block time in seconds (default: 12.0)")
    
    return parser.parse_args()

def get_epoch_info(netuid, network):
    """
    Get detailed epoch information from the blockchain.
    Returns (blocks_until_next_epoch, current_block, current_epoch, epoch_length, timestamp)
    """
    logger.info("Getting epoch information from blockchain...")
    
    try:
        # Run a single python command to get all the epoch info we need
        check_cmd = [
            "python3", "-c",
            f"""
import asyncio
import bittensor as bt
import time

async def check():
    try:
        start_time = time.time()
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
        
        # Time taken for the query
        query_time = time.time() - start_time
        timestamp = time.time()
        
        print(f"{{blocks_until_next_epoch}},{{current_block}},{{current_epoch}},{{epoch_length}},{{timestamp}},{{query_time}}")
    except Exception as e:
        print(f"ERROR: {{e}}")

asyncio.run(check())
            """
        ]
        
        result = subprocess.check_output(check_cmd).decode('utf-8').strip()
        
        if result.startswith("ERROR:"):
            logger.error(f"Error checking blocks: {result}")
            return None, None, None, None, None, None
        
        parts = result.split(',')
        blocks_until_next_epoch = int(parts[0])
        current_block = int(parts[1])
        current_epoch = int(parts[2])
        epoch_length = int(parts[3])
        timestamp = float(parts[4])
        query_time = float(parts[5])
        
        logger.info(f"Current block: {current_block}, Current epoch: {current_epoch}, Epoch length: {epoch_length}")
        logger.info(f"Blocks until next epoch: {blocks_until_next_epoch}")
        logger.info(f"Query time: {query_time:.2f}s")
        
        return blocks_until_next_epoch, current_block, current_epoch, epoch_length, timestamp, query_time
        
    except Exception as e:
        logger.error(f"Error getting epoch info: {e}")
        traceback.print_exc()
        return None, None, None, None, None, None

def run_command(args):
    """Run the specified command"""
    logger.info("🚀 Running command")
    
    run_cmd = [
        "python", args.run_script,
        f"--wallet.name", args.wallet_name,
        f"--wallet.hotkey", args.wallet_hotkey,
        f"--antitarget-weight={args.antitarget_weight}",
        "--source", "db",
        "--molecule-db", "molecule_archive.db",
        "--mode", "once"  # Ensure it runs once
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
            logger.info("✅ Successfully completed command execution")
            return True
        else:
            logger.warning(f"❌ Command execution failed, return code: {process.returncode}")
            return False
    
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main function."""
    args = parse_args()
    
    # Keep track of processed epochs
    last_processed_epoch = None
    last_sync_epoch = None
    epochs_since_sync = 0
    
    logger.info(f"Starting Pure Timer-Based Run Scheduler")
    logger.info(f"Will run script when exactly {args.blocks_before_end} blocks remain until epoch end")
    logger.info(f"Will resync with blockchain every {args.check_interval} epochs (configurable with --check-interval)")
    logger.info(f"Using block time of {args.block_time} seconds")
    logger.info(f"Command to run: python {args.run_script} --wallet.name {args.wallet_name} --wallet.hotkey {args.wallet_hotkey}")
    
    # Initial sync with blockchain to establish baseline
    logger.info("Performing initial blockchain synchronization...")
    
    while True:
        # Get epoch information
        epoch_info = get_epoch_info(args.netuid, args.network)
        
        if epoch_info[0] is not None:
            blocks_until_next_epoch, current_block, current_epoch, epoch_length, sync_timestamp, _ = epoch_info
            
            # Calculate the number of seconds in an epoch
            epoch_seconds = epoch_length * args.block_time
            logger.info(f"Epoch length: {epoch_length} blocks = {epoch_seconds:.1f} seconds")
            
            # Break out of the loop if we got valid data
            break
        else:
            logger.error("Failed to get initial epoch info, retrying in 10 seconds...")
            time.sleep(10)
    
    # Remember when we did the initial sync
    last_sync_epoch = current_epoch
    last_sync_time = sync_timestamp
    
    try:
        while True:
            current_time = time.time()
            seconds_since_sync = current_time - last_sync_time
            blocks_since_sync = seconds_since_sync / args.block_time
            
            # Calculate current block based on time elapsed since last sync
            estimated_current_block = current_block + blocks_since_sync
            
            # Calculate current epoch
            estimated_current_epoch = int(estimated_current_block // epoch_length)
            
            # Calculate blocks until next epoch
            blocks_in_current_epoch = estimated_current_block % epoch_length
            estimated_blocks_until_next_epoch = epoch_length - blocks_in_current_epoch
            
            # Check if epoch changed since last sync
            epochs_since_sync = estimated_current_epoch - last_sync_epoch
            
            # Format time for readability
            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"\nStatus at {current_time_str}:")
            logger.info(f"Time since last sync: {seconds_since_sync:.1f}s ({blocks_since_sync:.1f} blocks)")
            logger.info(f"Estimated current block: {estimated_current_block:.1f}")
            logger.info(f"Estimated current epoch: {estimated_current_epoch}")
            logger.info(f"Estimated blocks until next epoch: {estimated_blocks_until_next_epoch:.1f}")
            
            # Check if we should resync with the blockchain
            if epochs_since_sync >= float(args.check_interval):
                logger.info(f"Reached {epochs_since_sync} epochs since last sync, re-synchronizing with blockchain (check interval: {args.check_interval} epochs)...")
                
                # Resync with blockchain
                epoch_info = get_epoch_info(args.netuid, args.network)
                
                if epoch_info[0] is not None:
                    blocks_until_next_epoch, current_block, current_epoch, epoch_length, sync_timestamp, _ = epoch_info
                    
                    # Update our tracking variables
                    last_sync_epoch = current_epoch
                    last_sync_time = sync_timestamp
                    epochs_since_sync = 0
                    
                    # Recalculate epoch_seconds in case epoch_length changed
                    epoch_seconds = epoch_length * args.block_time
                    logger.info(f"Updated epoch length: {epoch_length} blocks = {epoch_seconds:.1f} seconds")
                    
                    # Recalculate current estimates
                    estimated_current_block = current_block
                    estimated_current_epoch = current_epoch
                    estimated_blocks_until_next_epoch = blocks_until_next_epoch
                    
                    logger.info(f"Resync complete. Current block: {current_block}, Current epoch: {current_epoch}")
                else:
                    logger.error("Failed to resync with blockchain, will continue with timer estimates")
                    # Sleep briefly and continue with our estimates
                    time.sleep(10)
                    continue
            
            # Calculate if we need to execute now
            target_blocks_remaining = float(args.blocks_before_end)
            execution_threshold = 0.5  # How close we need to be to our target block
            
            # Check if we're close to our target execution point and we haven't processed this epoch yet
            if (abs(estimated_blocks_until_next_epoch - target_blocks_remaining) < execution_threshold and 
                last_processed_epoch != estimated_current_epoch):
                
                logger.info(f"🔔 Target execution point reached! {estimated_blocks_until_next_epoch:.1f} blocks until epoch end")
                
                # Run the command
                success = run_command(args)
                
                # Mark this epoch as processed
                last_processed_epoch = estimated_current_epoch
                
                logger.info(f"Completed command execution for epoch {estimated_current_epoch}")
                
                # Calculate time until next execution point
                # This will be the remainder of this epoch plus (epoch_length - target_blocks) blocks into the next epoch
                blocks_until_next_execution = estimated_blocks_until_next_epoch + (epoch_length - target_blocks_remaining)
                seconds_until_next_execution = blocks_until_next_execution * args.block_time
                
                # Sleep until just before next execution to save resources
                # Wake up 10 seconds before the expected execution time
                next_check_time = min(60, seconds_until_next_execution - 10)
                if next_check_time > 0:
                    logger.info(f"Sleeping for {next_check_time:.1f}s until near next execution point")
                    time.sleep(next_check_time)
            else:
                # Calculate time until target execution point
                if estimated_blocks_until_next_epoch > target_blocks_remaining:
                    # Target is in this epoch
                    blocks_until_execution = estimated_blocks_until_next_epoch - target_blocks_remaining
                else:
                    # Target is in next epoch
                    blocks_until_execution = estimated_blocks_until_next_epoch + (epoch_length - target_blocks_remaining)
                
                seconds_until_execution = blocks_until_execution * args.block_time
                
                logger.info(f"Time until next execution point: {seconds_until_execution:.1f}s ({blocks_until_execution:.1f} blocks)")
                
                # Sleep time calculation
                # If we're close to execution, check more frequently
                if seconds_until_execution < 60:  # Within a minute of execution
                    sleep_time = 5  # Check every 5 seconds
                elif seconds_until_execution < 300:  # Within 5 minutes
                    sleep_time = 15  # Check every 15 seconds
                else:
                    # Sleep for longer, but wake up 30 seconds before the expected execution time
                    sleep_time = max(15, min(60, seconds_until_execution - 30))
                
                logger.info(f"Sleeping for {sleep_time:.1f}s before next check")
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        print("\nScheduler stopped by user")

if __name__ == "__main__":
    main()
