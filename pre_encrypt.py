#!/usr/bin/env python3
import os
import sys
import json
import base64
import hashlib
import argparse
import tempfile
import asyncio
from dotenv import load_dotenv

import bittensor as bt
from btdr import QuicknetBittensorDrandTimelock

from my_utils import upload_file_to_github

class MoleculePreEncryptor:
    def __init__(self):
        load_dotenv()
        
        # Load configuration from environment variables
        self.github_repo_name = os.environ.get('GITHUB_REPO_NAME', '')
        self.github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH', 'main')
        self.github_repo_owner = os.environ.get('GITHUB_REPO_OWNER', '')
        self.github_repo_path = os.environ.get('GITHUB_REPO_PATH', '')

        if self.github_repo_path == "":
            self.github_path = f"{self.github_repo_owner}/{self.github_repo_name}/{self.github_repo_branch}"
        else:
            self.github_path = f"{self.github_repo_owner}/{self.github_repo_name}/{self.github_repo_branch}/{self.github_repo_path}"

        if len(self.github_path) > 100:
            raise ValueError("Github path is too long. Please shorten it to 100 characters or less.")

        self.args = self.parse_args()
        self.bdt = QuicknetBittensorDrandTimelock()
        
        # Create output directory
        os.makedirs("pre_encrypted", exist_ok=True)
        # Create results file
        self.results_file = os.path.join("pre_encrypted", "commitment_data.txt")
        self.init_results_file()

    def init_results_file(self):
        with open(self.results_file, 'w') as f:
            f.write("# Commitment Data for Blockchain\n")
            f.write("# wallet,hotkey,molecule,proteins,commitment_data\n")

    def parse_args(self):
        # Set up the configuration parser
        parser = argparse.ArgumentParser()
        # Adds override arguments for network
        parser.add_argument('--network', default=os.getenv('SUBTENSOR_NETWORK'), help='Network to use')
        # Adds override arguments for netuid
        parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
        # Add input file argument
        parser.add_argument('--input', type=str, required=True, help='JSON file containing molecules to encrypt')
        # Add delay argument
        parser.add_argument('--delay', type=float, default=3.0, help='Delay between GitHub uploads in seconds')
        # Parse args
        return parser.parse_args()

    async def setup_bittensor_objects(self, wallet_name, hotkey_name):
        """Set up Bittensor objects with specific wallet/hotkey combination"""
        try:
            # Initialize wallet with specified names
            wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
            print(f"Wallet: {wallet}")

            # Initialize subtensor
            async with bt.async_subtensor(network=self.args.network) as subtensor:
                print(f"Subtensor: {subtensor}")

                # Initialize and sync metagraph
                metagraph = await subtensor.metagraph(self.args.netuid)
                await metagraph.sync()
                print(f"Metagraph synced")
                
                try:
                    # Get miner uid
                    miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
                    print(f"Miner UID: {miner_uid}")
                    return miner_uid, wallet, subtensor
                except ValueError:
                    print(f"ERROR: Hotkey {wallet.hotkey.ss58_address} not found in metagraph")
                    return None, None, None
        except Exception as e:
            print(f"ERROR setting up bittensor objects: {e}")
            return None, None, None

    async def pre_encrypt_molecule(self, wallet_name, hotkey_name, molecule_name, proteins):
        """Pre-encrypt a molecule and upload to GitHub"""
        print(f"Pre-encrypting molecule: {molecule_name} for {wallet_name}/{hotkey_name}")
        
        # Set up Bittensor objects for this wallet/hotkey
        miner_uid, wallet, subtensor = await self.setup_bittensor_objects(wallet_name, hotkey_name)
        if miner_uid is None:
            return None, None
        
        # Encrypt the molecule
        encrypted_response = self.bdt.encrypt(miner_uid, molecule_name)
        print(f"Encrypted molecule for UID {miner_uid}")
        
        # Create tmp file with encrypted response
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            # Write the content
            tmp_file.write(str(encrypted_response).encode())
            tmp_file.flush()
            
            # Read the content back
            tmp_file.seek(0)
            content_str = tmp_file.read().decode()
            
            # Base64-encode it
            encoded_content = base64.b64encode(content_str.encode()).decode()
            
            # Generate filename using SHA-256 hash
            filename = hashlib.sha256(content_str.encode()).hexdigest()[:20]
            print(f"Generated filename hash: {filename}")
            
            # Format chain commitment content
            commit_content = f"{self.github_path}/{filename}.txt"
            print(f"Commitment data: {commit_content}")
            
            # Save raw encrypted data for future use
            encrypted_data_path = os.path.join("pre_encrypted", f"{wallet_name}_{hotkey_name}_{molecule_name}.json")
            with open(encrypted_data_path, 'w') as f:
                json.dump({
                    "wallet": wallet_name,
                    "hotkey": hotkey_name,
                    "molecule": molecule_name,
                    "proteins": proteins,
                    "uid": miner_uid,
                    "encrypted_data": content_str,
                    "filename": filename,
                    "commitment_data": commit_content
                }, f, indent=2)
            
        # Upload to GitHub - we'll do this once for each molecule
        upload_status = upload_file_to_github(filename, encoded_content)
        
        if upload_status:
            print(f"Successfully uploaded {filename}.txt to GitHub")
            
            # Append to results file
            with open(self.results_file, 'a') as f:
                f.write(f"{wallet_name},{hotkey_name},{molecule_name},{proteins},{commit_content}\n")
                
            return filename, commit_content
        else:
            print(f"Failed to upload {filename}.txt to GitHub")
            return None, None

async def process_molecules(encryptor, input_file):
    """Process all molecules from the input file"""
    try:
        with open(input_file, 'r') as f:
            molecules = json.load(f)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False
        
    total = len(molecules)
    successful = 0
    
    for idx, molecule_data in enumerate(molecules):
        wallet = molecule_data["wallet"]
        hotkey = molecule_data["hotkey"]
        molecule = molecule_data["molecule"]
        proteins = molecule_data.get("proteins", "")
        
        print(f"Processing {idx+1}/{total}: {molecule} for {wallet}/{hotkey}")
        
        filename, commit_data = await encryptor.pre_encrypt_molecule(
            wallet,
            hotkey,
            molecule,
            proteins
        )
        
        if filename and commit_data:
            successful += 1
            print(f"✅ Successfully pre-encrypted and uploaded {molecule}")
        else:
            print(f"❌ Failed to pre-encrypt {molecule}")
            
        # Add a delay to avoid GitHub rate limits
        if idx < total - 1:
            print(f"Waiting {encryptor.args.delay} seconds before next operation...")
            await asyncio.sleep(encryptor.args.delay)
    
    print(f"\nPre-encryption complete: {successful}/{total} molecules processed successfully")
    print(f"Commitment data saved to: {encryptor.results_file}")
    return True

async def main():
    encryptor = MoleculePreEncryptor()
    await process_molecules(encryptor, encryptor.args.input)

if __name__ == "__main__":
    asyncio.run(main())
