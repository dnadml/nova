import os
import sys
import time
import asyncio
import argparse
from datetime import datetime, timedelta
import bittensor as bt

class EpochTracker:
    def __init__(
        self,
        wallet_name: str = "default",
        wallet_hotkey: str = "default",
        network: str = 'finney', 
        netuid: int = 68,
        block_time: int = 12,  # Seconds per block
    ):
        self.wallet_name = wallet_name
        self.wallet_hotkey = wallet_hotkey
        self.network = network
        self.netuid = netuid
        self.block_time = block_time  # Seconds
        self.subtensor = None
        self.connected = False
        self.epoch_length = None
        
    async def connect(self):
        """Connect to the network"""
        try:
            # Initialize wallet
            parser = argparse.ArgumentParser()
            bt.wallet.add_args(parser)
            config = bt.config(parser)
            
            # Set wallet name and hotkey
            config.wallet.name = self.wallet_name
            config.wallet.hotkey = self.wallet_hotkey
            
            # Connect to subtensor
            self.subtensor = await bt.async_subtensor(network=self.network).__aenter__()
            
            # Get epoch information
            tempo_result = await self.subtensor.substrate.query(
                module="SubtensorModule",
                storage_function="Tempo",
                params=[self.netuid]
            )
            self.epoch_length = tempo_result.value
            self.connected = True
            
            print(f"✓ Connected to {self.network}")
            print(f"✓ Subnet {self.netuid} epoch length: {self.epoch_length} blocks")
            print(f"✓ Estimated block time: {self.block_time} seconds")
            print("\nStarting tracker...\n")
            return True
            
        except Exception as e:
            print(f"Error connecting to network: {e}")
            return False

    async def get_current_data(self):
        """Get current block and epoch data"""
        try:
            current_block = await self.subtensor.get_current_block()
            current_epoch = current_block // self.epoch_length
            blocks_in_epoch = current_block % self.epoch_length
            blocks_until_next_epoch = self.epoch_length - blocks_in_epoch
            
            # Calculate time
            seconds_until_next_block = self.block_time
            minutes_until_next_epoch = blocks_until_next_epoch * self.block_time / 60
            
            return {
                'current_block': current_block,
                'current_epoch': current_epoch,
                'blocks_in_epoch': blocks_in_epoch,
                'blocks_until_next_epoch': blocks_until_next_epoch,
                'seconds_until_next_block': seconds_until_next_block,
                'minutes_until_next_epoch': minutes_until_next_epoch
            }
        except Exception as e:
            print(f"Error getting current data: {e}")
            return None

    def display(self, data):
        """Display the current information in a nice format"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Terminal colors
        GREEN = '\033[92m'
        BLUE = '\033[94m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"{BOLD}════════════════════════════════════════════{RESET}")
        print(f"{BOLD}{BLUE}          BITTENSOR EPOCH TRACKER           {RESET}")
        print(f"{BOLD}════════════════════════════════════════════{RESET}")
        print(f"  Network: {YELLOW}{self.network}{RESET}  |  Subnet: {YELLOW}{self.netuid}{RESET}")
        print(f"  Time: {now}")
        print(f"{BOLD}────────────────────────────────────────────{RESET}")
        print(f"  {BOLD}Current Block:{RESET} {GREEN}{data['current_block']}{RESET}")
        print(f"  {BOLD}Current Epoch:{RESET} {GREEN}{data['current_epoch']}{RESET}")
        print(f"{BOLD}────────────────────────────────────────────{RESET}")
        print(f"  {BOLD}Blocks in this Epoch:{RESET} {data['blocks_in_epoch']} / {self.epoch_length}")
        print(f"  {BOLD}Blocks until next Epoch:{RESET} {YELLOW}{data['blocks_until_next_epoch']}{RESET}")
        print(f"  {BOLD}Time until next Epoch:{RESET} ~{YELLOW}{data['minutes_until_next_epoch']:.1f}{RESET} minutes")
        print(f"{BOLD}────────────────────────────────────────────{RESET}")
        print(f"  {BOLD}Next block ETA:{RESET}")
        
        # Countdown timer
        return data['seconds_until_next_block']
    
    async def countdown(self, seconds):
        """Display a countdown timer to the next block"""
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            remaining = max(0, seconds - elapsed)
            
            if remaining <= 0:
                print(f"\r  Next block arriving now...                    ", end="")
                break
                
            minutes, secs = divmod(int(remaining), 60)
            timer = f"{minutes:02d}:{secs:02d}"
            print(f"\r  Next block in: {timer}                     ", end="")
            await asyncio.sleep(0.5)
            
        print()
        return True

    async def run(self):
        """Main loop for the tracker"""
        if not self.connected and not await self.connect():
            print("Failed to connect. Exiting.")
            return

        try:
            while True:
                try:
                    data = await self.get_current_data()
                    if not data:
                        print("Error getting data. Retrying in 5 seconds...")
                        await asyncio.sleep(5)
                        continue
                        
                    seconds = self.display(data)
                    await self.countdown(seconds)
                    
                except KeyboardInterrupt:
                    print("\nTracker stopped by user.")
                    break
                except Exception as e:
                    print(f"Error in tracker loop: {e}")
                    await asyncio.sleep(5)
        finally:
            # Ensure we close the subtensor connection
            if self.subtensor:
                await self.subtensor.__aexit__(None, None, None)

async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Bittensor Epoch Tracker")
    parser.add_argument("--wallet", type=str, default="default", help="Wallet name")
    parser.add_argument("--hotkey", type=str, default="default", help="Wallet hotkey")
    parser.add_argument("--network", type=str, default="finney", help="Network to connect to")
    parser.add_argument("--netuid", type=int, default=68, help="Subnet UID")
    parser.add_argument("--blocktime", type=int, default=12, help="Seconds per block")
    
    args = parser.parse_args()
    
    # Create and run the tracker
    tracker = EpochTracker(
        wallet_name=args.wallet,
        wallet_hotkey=args.hotkey,
        network=args.network,
        netuid=args.netuid,
        block_time=args.blocktime
    )
    
    await tracker.run()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
