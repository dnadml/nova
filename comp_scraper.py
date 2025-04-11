#!/usr/bin/env python3
"""
Nova Dashboard API Scraper

This script fetches data from the Nova Dashboard API for a range of epochs,
stores it in an SQLite database, and exports molecule data to a CSV file.
"""

import os
import time
import sqlite3
import csv
import requests
import json
import argparse
from tqdm import tqdm

def create_database():
    """Create an SQLite database to store epoch and molecule data."""
    conn = sqlite3.connect('nova_molecules.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS epochs (
        epoch_number INTEGER PRIMARY KEY,
        date_collected TEXT,
        json_data TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS molecules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        epoch_number INTEGER,
        rank INTEGER,
        molecule_id TEXT,
        smiles TEXT,
        final_score REAL,
        FOREIGN KEY (epoch_number) REFERENCES epochs(epoch_number),
        UNIQUE(molecule_id, epoch_number)
    )
    ''')
    
    conn.commit()
    return conn

def fetch_epoch_data(epoch_number, headers):
    """Fetch leaderboard data for a specific epoch from the API."""
    url = f"https://dashboard-backend-production-0103.up.railway.app/api/leaderboard?epoch_number={epoch_number}"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for epoch {epoch_number}: {e}")
        return None

def store_epoch_data(conn, epoch_number, epoch_data):
    """Store epoch data in the database."""
    cursor = conn.cursor()
    
    # Store the epoch data
    cursor.execute(
        "INSERT OR REPLACE INTO epochs (epoch_number, date_collected, json_data) VALUES (?, datetime('now'), ?)",
        (epoch_number, json.dumps(epoch_data))
    )
    
    # Store each molecule
    for rank, submission in enumerate(epoch_data.get('leaderboard', []), 1):
        if 'molecule' in submission and 'smiles' in submission:
            cursor.execute(
                """
                INSERT OR REPLACE INTO molecules 
                (epoch_number, rank, molecule_id, smiles, final_score) 
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    epoch_number,
                    rank,
                    submission.get('molecule', ''),
                    submission.get('smiles', ''),
                    submission.get('final_score', 0.0)
                )
            )
    
    conn.commit()

def export_to_csv(conn, csv_filename):
    """Export all molecules to a CSV file."""
    cursor = conn.cursor()
    
    # Query for all molecules
    cursor.execute("""
    SELECT molecule_id, smiles 
    FROM molecules
    ORDER BY molecule_id
    """)
    
    # Write to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['molecule', 'smiles'])  # Header
        
        # Use a set to track unique molecule_id/smiles pairs
        unique_pairs = set()
        
        for row in cursor.fetchall():
            molecule_id, smiles = row
            
            # Only write if this pair is unique
            pair = (molecule_id, smiles)
            if pair not in unique_pairs:
                unique_pairs.add(pair)
                csv_writer.writerow([molecule_id, smiles])
    
    print(f"Exported {len(unique_pairs)} unique molecules to {csv_filename}")

def main():
    parser = argparse.ArgumentParser(description='Fetch Nova Dashboard data for a range of epochs')
    parser.add_argument('--start', type=int, default=14559, help='Starting epoch number')
    parser.add_argument('--end', type=int, default=14789, help='Ending epoch number')
    parser.add_argument('--output', default='molecules.csv', help='Output CSV filename')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between API requests in seconds')
    args = parser.parse_args()
    
    # Set up database
    conn = create_database()
    
    # Set up request headers
    headers = {
        "Origin": "https://dashboard-frontend-dusky.vercel.app",
        "Referer": "https://dashboard-frontend-dusky.vercel.app/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15"
    }
    
    # Define epoch range
    start_epoch = args.start
    end_epoch = args.end
    
    # Fetch data for each epoch
    print(f"Fetching data for epochs {start_epoch} to {end_epoch}...")
    
    # Using tqdm for a progress bar
    for epoch_number in tqdm(range(start_epoch, end_epoch + 1)):
        epoch_data = fetch_epoch_data(epoch_number, headers)
        
        if epoch_data:
            store_epoch_data(conn, epoch_number, epoch_data)
            
            # Add a small delay to avoid overwhelming the API
            time.sleep(args.delay)
    
    # Export to CSV
    export_to_csv(conn, args.output)
    
    # Close database connection
    conn.close()
    print("Done!")

if __name__ == "__main__":
    main()
