#!/usr/bin/env python3
import os
import sys
import time
import asyncio
import argparse
import traceback
from dotenv import load_dotenv
import bittensor as bt
from bittensor.core.errors import MetadataError

class ParallelBatchCommitter:
    def __init__(self):
        load_dotenv()
        
        parser = argparse.ArgumentParser(description="Submit pre-encrypted molecule commitments to the blockchain in parallel")
        
        # Add network and netuid arguments
        parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK', 'finney'), help='Network to use')
        parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
        
        # Add input file argument
        parser.add_argument('--input', type=str, required=True, help='Text file containing commitment data')
        
        # Add flags to control behavior
        parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no actual blockchain transactions)')
        parser.add_argument('--max-concurrent', type=int, default=10, help='Maximum number of concurrent submissions (default: 10)')
        
        # Add wallet filter arguments
        parser.add_argument('--wallet', type=str, help='Filter submissions to specific wallet (optional)')
        parser.add_argument('--hotkey', type=str, help='Filter submissions to specific hotkey (optional)')
        
        self.args = parser.parse_args()
        self.setup_logging()
        
        # Track success/failure counts
        self.successful = 0
        self.failed = 0
        self.total = 0
        
        # Setup semaphore for limiting concurrent connections
        self.semaphore = asyncio.Semaphore(self.args.max_concurrent)
    
    def setup_logging(self):
        # Just use print statements for simplicity
        print(f"Running parallel batch committer for subnet: {self.args.netuid} on network: {self.args.network}")
        print(f"Input file: {self.args.input}")
        print(f"Max concurrent submissions: {self.args.max_concurrent}")
    
    def parse_commitment_file(self):
        """Parse the commitment file and return a list of entries"""
        entries = []
        
        try:
            with open(self.args.input, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split(',')
                    if len(parts) >= 5:
                        entry = {
                            'wallet': parts[0],
                            'hotkey': parts[1],
                            'molecule': parts[2],
                            'proteins': parts[3],
                            'commitment': parts[4]
                        }
                        
                        # Apply filters if specified
                        if self.args.wallet and entry['wallet'] != self.args.wallet:
                            continue
                        if self.args.hotkey and entry['hotkey'] != self.args.hotkey:
                            continue
                            
                        entries.append(entry)
            
            print(f"Loaded {len(entries)} entries from commitment file")
            return entries
            
        except Exception as e:
            print(f"Error reading commitment file: {e}")
            return []
            
    async def submit_commitment(self, entry):
        """Submit a commitment to the blockchain"""
        wallet_name = entry['wallet']
        hotkey_name = entry['hotkey']
        commitment = entry['commitment']
        
        # Use semaphore to limit concurrent connections
        async with self.semaphore:
            print(f"Submitting commitment for {wallet_name}/{hotkey_name}")
            
            if self.args.dry_run:
                print(f"[DRY RUN] Would submit commitment for {wallet_name}/{hotkey_name}")
                self.successful += 1
                return True
                
            try:
                # Initialize wallet directly
                wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
                
                # Initialize subtensor
                async with bt.async_subtensor(network=self.args.network) as subtensor:
                    # Set commitment
                    commitment_status = await subtensor.set_commitment(
                        wallet=wallet,
                        netuid=self.args.netuid,
                        data=commitment
                    )
                    
                    if commitment_status:
                        print(f"✅ Successfully set commitment for {wallet_name}/{hotkey_name}")
                        self.successful += 1
                        return True
                    else:
                        print(f"❌ Failed to set commitment for {wallet_name}/{hotkey_name}")
                        self.failed += 1
                        return False
                        
            except MetadataError as e:
                print(f"❌ Metadata error for {wallet_name}/{hotkey_name}: too soon to commit again")
                self.failed += 1
                return False
            except Exception as e:
                print(f"❌ Error for {wallet_name}/{hotkey_name}: {str(e)}")
                self.failed += 1
                return False
    
    async def run(self):
        """Main execution function"""
        # Parse the commitment file
        entries = self.parse_commitment_file()
        if not entries:
            print("No valid entries found in commitment file")
            return False
            
        # Submit commitments in parallel
        self.total = len(entries)
        
        print(f"Starting parallel commitment submission for {self.total} entries")
        print(f"{'Dry run mode enabled - no actual transactions' if self.args.dry_run else ''}")
        
        start_time = time.time()
        
        # Create tasks for all submissions to run in parallel
        tasks = [self.submit_commitment(entry) for entry in entries]
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        
        print(f"\nParallel commitment submission complete: {self.successful}/{self.total} submitted successfully")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        
        return self.successful == self.total

async def main():
    committer = ParallelBatchCommitter()
    await committer.run()

if __name__ == "__main__":
    # Increase connection limit for parallel requests
    asyncio.run(main())
