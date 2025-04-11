#!/bin/bash
# run_nova_challenge.sh - Scheduler script for Nova challenge
# This script will check the current epoch and only run the main challenge script
# when we're exactly 2 blocks into a new epoch.

# Configuration
WORKSPACE="/workspace/nova"
MAIN_SCRIPT="${WORKSPACE}/chal.py"
MOLECULE_DB="${WORKSPACE}/molecule_archive.db"
WALLET_NAME="code5"
WALLET_HOTKEY="m2"
LOG_FILE="${WORKSPACE}/nova_scheduler.log"
TARGET_BLOCKS_SINCE_EPOCH=2  # Run when we're exactly this many blocks into a new epoch

# Function to log messages
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Make sure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"
log "Starting Nova challenge scheduler check"

# Get current blockchain state
BLOCKCHAIN_INFO=$(python3 -c "
import asyncio
import bittensor as bt

async def check():
    try:
        subtensor = bt.async_subtensor(network='finney')
        await subtensor.initialize()
        
        # Get epoch length
        epoch_length = (await subtensor.substrate.query(
            module=\"SubtensorModule\",
            storage_function=\"Tempo\",
            params=[68]
        )).value
        
        # Get current block
        current_block = await subtensor.get_current_block()
        current_epoch = current_block // epoch_length
        blocks_since_epoch_start = current_block % epoch_length
        
        print(f\"{blocks_since_epoch_start},{current_block},{current_epoch},{epoch_length}\")
    except Exception as e:
        print(f\"ERROR: {e}\")

asyncio.run(check())
")

# Check for errors
if [[ $BLOCKCHAIN_INFO == ERROR* ]]; then
  log "Error getting blockchain info: $BLOCKCHAIN_INFO"
  exit 1
fi

# Parse blockchain info
IFS=',' read -r BLOCKS_SINCE_START CURRENT_BLOCK CURRENT_EPOCH EPOCH_LENGTH <<< "$BLOCKCHAIN_INFO"

log "Current block: $CURRENT_BLOCK, Epoch: $CURRENT_EPOCH"
log "Blocks since epoch start: $BLOCKS_SINCE_START"

# Get the last run epoch from state file
STATE_FILE="${WORKSPACE}/nova_scheduler_state.txt"
LAST_RUN_EPOCH=0
if [ -f "$STATE_FILE" ]; then
  LAST_RUN_EPOCH=$(cat "$STATE_FILE")
fi

# Check if we should run the main script
SHOULD_RUN=false
RUN_REASON=""

# Case 1: We're exactly 2 blocks into a new epoch
if [ "$BLOCKS_SINCE_START" -eq "$TARGET_BLOCKS_SINCE_EPOCH" ]; then
  SHOULD_RUN=true
  RUN_REASON="we're exactly $TARGET_BLOCKS_SINCE_EPOCH blocks into a new epoch"
fi

# Case 2: We're in a new epoch that we haven't run for yet and we've passed the 2-block mark
if [ "$CURRENT_EPOCH" -gt "$LAST_RUN_EPOCH" ] && [ "$BLOCKS_SINCE_START" -ge "$TARGET_BLOCKS_SINCE_EPOCH" ]; then
  SHOULD_RUN=true
  RUN_REASON="new epoch detected (previous run was for epoch $LAST_RUN_EPOCH) and we've passed the $TARGET_BLOCKS_SINCE_EPOCH-block mark"
fi

# Run the main script if conditions are met
if [ "$SHOULD_RUN" = true ]; then
  log "Running main challenge script: $RUN_REASON"
  
  # Run the main script with "once" mode
  cd "$WORKSPACE"
  python3 "$MAIN_SCRIPT" \
    --source db \
    --molecule-db "$MOLECULE_DB" \
    --wallet.name "$WALLET_NAME" \
    --wallet.hotkey "$WALLET_HOTKEY" \
    --mode once \
    >> "$LOG_FILE" 2>&1
  
  # Save the current epoch to state file
  echo "$CURRENT_EPOCH" > "$STATE_FILE"
  
  log "Finished running challenge script for epoch $CURRENT_EPOCH"
else
  log "Skipping run: not at target block position (waiting for block $TARGET_BLOCKS_SINCE_EPOCH of epoch)"
fi

log "Scheduler check completed"

