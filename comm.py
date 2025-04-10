#!/usr/bin/env python3

import os
import sys
import json
import asyncio
import argparse
from typing import cast, Dict, Any
from datetime import datetime

# Bittensor imports
import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata


async def get_subnet_info(subtensor, netuid: int, protein_validator_uid: int = 5):
    """
    Get subnet information including epoch length and current protein challenge.
    
    Args:
        subtensor: Bittensor subtensor client
        netuid: The subnet ID
        protein_validator_uid: UID of validator setting protein
        
    Returns:
        Dictionary with subnet information
    """
    # Get epoch length
    try:
        epoch_length = (await subtensor.substrate.query(
            module="SubtensorModule",
            storage_function="Tempo",
            params=[netuid]
        )).value
        
        # Get current block
        current_block = await subtensor.get_current_block()
        current_epoch = current_block // epoch_length
        
        # Get metagraph
        metagraph = await subtensor.metagraph(netuid)
        
        # Get protein validator's commitment
        commitments = await get_all_commitments(subtensor, netuid)
        protein_validator_hotkey = None
        for hotkey, data in commitments.items():
            if data['uid'] == protein_validator_uid:
                protein_validator_hotkey = hotkey
                break
                
        if protein_validator_hotkey and protein_validator_hotkey in commitments:
            protein_code = commitments[protein_validator_hotkey]['data']
        else:
            protein_code = "Unknown"
            
        return {
            "netuid": netuid,
            "current_block": current_block,
            "epoch_length": epoch_length,
            "current_epoch": current_epoch,
            "blocks_until_next_epoch": epoch_length - (current_block % epoch_length),
            "protein_code": protein_code,
            "protein_validator_uid": protein_validator_uid
        }
    except Exception as e:
        print(f"Error getting subnet info: {e}")
        return {
            "netuid": netuid,
            "error": str(e)
        }


async def get_all_commitments(
    subtensor,
    netuid: int = 68, 
    block: int = None, 
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Query all commitments for all UIDs in a subnet.
    
    Args:
        subtensor: The subtensor client object
        netuid: The subnet ID (default: 68 for NOVA)
        block: Specific block to query (default: latest)
        verbose: Print detailed logs
    
    Returns:
        Dictionary of all commitments
    """
    if verbose:
        print(f"Querying subnet {netuid} commitments...")
    
    # Get metagraph
    metagraph = await subtensor.metagraph(netuid)
    num_hotkeys = len(metagraph.hotkeys)
    
    if verbose:
        print(f"Found {num_hotkeys} hotkeys in subnet {netuid}")
    
    # Get current block if not specified
    if block is None:
        block = await subtensor.get_current_block()
        if verbose:
            print(f"Using current block: {block}")
    else:
        if verbose:
            print(f"Using specified block: {block}")
        
    block_hash = await subtensor.determine_block_hash(block)
    
    # Query commitments for all hotkeys
    if verbose:
        print("Gathering commitments for all UIDs...")
        
    commits_tasks = []
    for hotkey in metagraph.hotkeys:
        task = subtensor.substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey],
            block_hash=block_hash,
        )
        commits_tasks.append(task)
        
    # Gather all results
    all_commits = await asyncio.gather(*commits_tasks)
    
    # Process results
    commitments = {}
    for uid, hotkey in enumerate(metagraph.hotkeys):
        commit = cast(dict, all_commits[uid])
        if commit:
            try:
                metadata = decode_metadata(commit)
                # Try to decode JSON if it looks like JSON
                if isinstance(metadata, str) and (metadata.startswith('{') or metadata.startswith('[')):
                    try:
                        parsed_data = json.loads(metadata)
                        metadata = parsed_data
                    except:
                        # Keep as string if not valid JSON
                        pass
                
                # Add to commitments dict
                commitments[hotkey] = {
                    'uid': uid,
                    'hotkey': hotkey,
                    'block': commit['block'],
                    'data': metadata,
                    'stake': float(metagraph.S[uid]),
                    'time': datetime.now().isoformat()
                }
            except Exception as e:
                if verbose:
                    print(f"Error decoding metadata for UID {uid}: {e}")
                commitments[hotkey] = {
                    'uid': uid,
                    'hotkey': hotkey,
                    'block': commit['block'],
                    'data': "ERROR_DECODING",
                    'error': str(e),
                    'time': datetime.now().isoformat()
                }
                
    return commitments


async def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Query all commitments for a Bittensor subnet")
    parser.add_argument('--netuid', type=int, default=68, help="Subnet UID (default: 68)")
    parser.add_argument('--network', type=str, default='finney', help="Network (default: finney)")
    parser.add_argument('--block', type=int, default=None, help="Block number (default: latest)")
    parser.add_argument('--output', type=str, default='commitments.json', help="Output file (default: commitments.json)")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--filter-empty', action='store_true', help="Filter out empty commitments")
    parser.add_argument('--protein-validator-uid', type=int, default=5, 
                        help="UID of validator that sets protein challenge (default: 5)")
    parser.add_argument('--sort-by', type=str, default='uid', choices=['uid', 'block', 'stake'],
                        help="Sort output by field (default: uid)")
    args = parser.parse_args()
    
    # Connect to subtensor
    print(f"Connecting to {args.network} network...")
    async with bt.async_subtensor(network=args.network) as subtensor:
        
        # Get subnet info
        subnet_info = await get_subnet_info(
            subtensor, 
            args.netuid, 
            args.protein_validator_uid
        )
        
        # Print subnet info
        print("\nSubnet Information:")
        print(f"Network: {args.network}")
        print(f"Subnet: {args.netuid}")
        print(f"Current Block: {subnet_info['current_block']}")
        print(f"Epoch Length: {subnet_info['epoch_length']}")
        print(f"Current Epoch: {subnet_info['current_epoch']}")
        print(f"Blocks Until Next Epoch: {subnet_info['blocks_until_next_epoch']}")
        print(f"Current Protein Challenge: {subnet_info['protein_code']}")
        
        # Query commitments
        commitments = await get_all_commitments(
            subtensor,
            netuid=args.netuid,
            block=args.block,
            verbose=args.verbose
        )
        
        # Filter empty commitments if requested
        if args.filter_empty:
            before_count = len(commitments)
            commitments = {k: v for k, v in commitments.items() if v.get('data')}
            after_count = len(commitments)
            print(f"Filtered out {before_count - after_count} empty commitments")
        
        # Count non-empty commitments
        non_empty = sum(1 for v in commitments.values() if v.get('data'))
        print(f"Found {non_empty} non-empty commitments out of {len(commitments)} total")
        
        # Sort commitments
        if args.sort_by == 'uid':
            sorted_commits = sorted(commitments.items(), key=lambda x: x[1]['uid'])
        elif args.sort_by == 'block':
            sorted_commits = sorted(commitments.items(), key=lambda x: x[1]['block'], reverse=True)
        elif args.sort_by == 'stake':
            sorted_commits = sorted(commitments.items(), key=lambda x: x[1]['stake'], reverse=True)
        
        # Format for display (top 10)
        formatted = []
        for hotkey, data in sorted_commits[:10]:
            if data.get('data'):
                formatted.append({
                    'uid': data['uid'],
                    'hotkey': hotkey[:10] + '...',
                    'block': data['block'],
                    'stake': data['stake'],
                    'data': str(data['data'])[:50] + ('...' if len(str(data['data'])) > 50 else '')
                })
        
        if formatted:
            print("\nSample of commitments (first 10):")
            print(f"{'UID':<5} {'Hotkey':<14} {'Block':<10} {'Stake':<10} Data")
            print("-" * 80)
            for item in formatted:
                print(f"{item['uid']:<5} {item['hotkey']:<14} {item['block']:<10} {item['stake']:<10.2f} {item['data']}")
        
        # Save to file
        with open(args.output, 'w') as f:
            json.dump({
                'metadata': {
                    'netuid': args.netuid,
                    'network': args.network,
                    'block': args.block if args.block else subnet_info['current_block'],
                    'timestamp': datetime.now().isoformat(),
                    'total_commitments': len(commitments),
                    'non_empty_commitments': non_empty,
                    'current_epoch': subnet_info['current_epoch'],
                    'epoch_length': subnet_info['epoch_length'],
                    'protein_code': subnet_info['protein_code']
                },
                'commitments': commitments
            }, f, indent=2)
        
        print(f"\nSaved complete results to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
