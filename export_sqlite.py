import sqlite3
import csv
import sys

def export_sqlite_to_csv(db_path, output_csv, table_name=None):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # If no table name is provided, get the first table
    if table_name is None:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if not tables:
            print("No tables found in the database.")
            conn.close()
            return
        table_name = tables[0][0]
        print(f"Using table: {table_name}")
    
    # Get all data from the table
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    
    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(columns)  # Write header
        csv_writer.writerows(rows)    # Write data
    
    print(f"Data exported successfully to {output_csv}")
    conn.close()

if __name__ == "__main__":
    # You can change these values or pass them as command line arguments
    db_path = "molecule_archive.db"
    output_csv = "molecule_archive_export.csv"
    
    # If you know the specific table name, you can specify it here
    table_name = None  # This will use the first table if None
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    if len(sys.argv) > 3:
        table_name = sys.argv[3]
    
    export_sqlite_to_csv(db_path, output_csv, table_name)
