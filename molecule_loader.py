import os
import csv
import sqlite3
import argparse
import logging

def setup_logging(log_file=None, log_level=logging.INFO):
    """Configure logging for the script."""
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

class MoleculeDB:
    """Database for storing molecules and their SMILES strings."""
    
    def __init__(self, db_path="molecule_archive.db"):
        self.db_path = db_path
        self.conn = None
        self.logger = logging.getLogger()
    
    def connect(self):
        """Connect to the SQLite database."""
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def init_db(self):
        """Initialize the molecules table."""
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS molecules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                molecule TEXT UNIQUE,
                smiles TEXT,
                first_seen_block INTEGER,
                first_seen_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_molecules_molecule ON molecules (molecule)')
        self.conn.commit()
        self.logger.info(f"Initialized molecule database at {self.db_path}")
    
    def molecule_exists(self, molecule):
        """Check if a molecule already exists in the database."""
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM molecules WHERE molecule = ?", (molecule,))
        return cursor.fetchone() is not None
    
    def add_molecule(self, molecule, smiles, block=0):
        """Add a single molecule entry to the database."""
        self.connect()
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO molecules (molecule, smiles, first_seen_block) VALUES (?, ?, ?)",
                (molecule, smiles, block)
            )
            self.conn.commit()
            self.logger.info(f"Added molecule '{molecule}' with SMILES: {smiles}")
        except sqlite3.IntegrityError:
            self.logger.info(f"Molecule '{molecule}' already exists, skipping")
    
    def add_molecules(self, molecules, block=0, savi_lookup=None, file_smiles_dict=None):
        """
        For each molecule in the list, check if it exists. If not, lookup its SMILES (if available)
        via the SAVI lookup or from the file_smiles_dict and add it to the database.
        """
        self.connect()
        count = 0
        smiles_dict = {}
        
        # First use file_smiles_dict if available
        if file_smiles_dict:
            smiles_dict = file_smiles_dict
        
        # If a SAVI lookup instance is provided, get SMILES strings for molecules not in file_smiles_dict
        if savi_lookup:
            molecules_to_lookup = [m for m in molecules if m not in smiles_dict or not smiles_dict[m]]
            if molecules_to_lookup:
                savi_results = savi_lookup.lookup_smiles(molecules_to_lookup)
                # Update smiles_dict with SAVI results only for molecules without SMILES
                for mol, smiles in savi_results.items():
                    if mol not in smiles_dict or not smiles_dict[mol]:
                        smiles_dict[mol] = smiles
        
        for molecule in molecules:
            if not self.molecule_exists(molecule):
                smiles = smiles_dict.get(molecule, None)
                self.add_molecule(molecule, smiles, block)
                count += 1
            else:
                self.logger.info(f"Molecule '{molecule}' already exists, skipping")
        
        self.logger.info(f"Added {count} new molecule(s) to the archive database")
        return count

class SAVILookup:
    """
    Provides a simple lookup for molecule names to SMILES strings using an existing SAVI lookup database.
    The lookup database should have a table with columns 'name' and 'smiles'.
    """
    
    def __init__(self, db_path="savi_lookup.db"):
        self.db_path = db_path
        self.conn = None
        self.logger = logging.getLogger()
    
    def connect(self):
        """Connect to the SAVI lookup database."""
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
    
    def close(self):
        """Close the SAVI lookup database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def lookup_smiles(self, molecule_names):
        """
        Look up SMILES strings for the given list of molecule names.
        
        Returns:
            A dictionary mapping molecule names to their SMILES strings.
        """
        if not os.path.exists(self.db_path):
            self.logger.error(f"SAVI lookup database not found: {self.db_path}")
            return {}
        self.connect()
        cursor = self.conn.cursor()
        results = {}
        placeholders = ','.join(['?'] * len(molecule_names))
        query = f"SELECT name, smiles FROM molecules WHERE name IN ({placeholders})"
        try:
            cursor.execute(query, molecule_names)
            rows = cursor.fetchall()
            for row in rows:
                results[row['name']] = row['smiles']
            self.logger.info(f"Found SMILES for {len(results)} out of {len(molecule_names)} molecule(s)")
            return results
        except Exception as e:
            self.logger.error(f"Error during SMILES lookup: {e}")
            return {}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load molecules into archive DB with SMILES strings"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--molecules", type=str, nargs='+',
                       help="List of molecule names (separated by space)")
    group.add_argument("--molecules-file", type=str,
                       help="Path to a text file containing molecules (CSV or tab-separated)")
    parser.add_argument("--molecule-db", type=str, default="molecule_archive.db",
                        help="Path to molecule archive DB (default: molecule_archive.db)")
    parser.add_argument("--savi-db", type=str, default="savi_lookup.db",
                        help="Path to SAVI lookup DB (default: savi_lookup.db)")
    parser.add_argument("--block", type=int, default=0,
                        help="Block number for first_seen_block (default: 0)")
    parser.add_argument("--log-file", type=str, default="molecule_loader.log",
                        help="Path to log file (default: molecule_loader.log)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    return parser.parse_args()

def load_molecule_list(args):
    """
    Load molecule list and SMILES dictionary from arguments or file.
    Supports CSV, tab-separated, or simple list formats.
    
    Returns:
        Tuple of (molecules list, SMILES dictionary)
    """
    molecules = []
    smiles_dict = {}
    
    if args.molecules:
        # Direct command line input - no SMILES
        return args.molecules, {}
    
    elif args.molecules_file:
        if not os.path.exists(args.molecules_file):
            logging.error(f"Molecules file not found: {args.molecules_file}")
            return [], {}
        
        # Determine the file format based on extension and content
        is_csv = args.molecules_file.lower().endswith('.csv')
        
        with open(args.molecules_file, 'r') as f:
            # Check first line for format if not already determined
            if not is_csv:
                first_line = f.readline().strip()
                # Check if it's a CSV file with header
                is_csv = ',' in first_line and ('molecule' in first_line.lower() or 'smiles' in first_line.lower())
                # Reset file pointer
                f.seek(0)
            
            # Process CSV file
            if is_csv:
                try:
                    csv_reader = csv.reader(f)
                    header = next(csv_reader)  # Read header
                    
                    # Identify column indices
                    mol_idx = header.index('molecule') if 'molecule' in header else 0
                    smiles_idx = header.index('smiles') if 'smiles' in header else 1
                    
                    # Read data rows
                    for row in csv_reader:
                        if not row or len(row) <= max(mol_idx, smiles_idx):
                            continue  # Skip empty or incomplete rows
                        
                        molecule = row[mol_idx].strip()
                        if molecule:
                            molecules.append(molecule)
                            if len(row) > smiles_idx and row[smiles_idx].strip():
                                smiles_dict[molecule] = row[smiles_idx].strip()
                except Exception as e:
                    logging.error(f"Error parsing CSV file: {e}")
                    return [], {}
            
            # Process tab-separated or simple list
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if line has a tab separator
                    if '\t' in line:
                        parts = line.split('\t', 1)
                        molecule = parts[0].strip()
                        molecules.append(molecule)
                        if len(parts) > 1 and parts[1].strip():
                            smiles_dict[molecule] = parts[1].strip()
                    else:
                        molecules.append(line)
    
    logging.info(f"Loaded {len(molecules)} molecules and {len(smiles_dict)} SMILES strings")
    return molecules, smiles_dict

def main():
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(args.log_file, log_level)
    logging.info("Starting molecule loader")

    # Load molecule list and SMILES dictionary
    molecule_list, file_smiles_dict = load_molecule_list(args)
    if not molecule_list:
        logging.error("No molecules provided!")
        return

    # Initialize molecule archive database
    molecule_db = MoleculeDB(args.molecule_db)
    molecule_db.init_db()
    
    # Initialize SAVI lookup if the database exists
    savi_lookup = None
    if os.path.exists(args.savi_db):
        savi_lookup = SAVILookup(args.savi_db)
        logging.info("SAVI lookup enabled")
    else:
        logging.warning(f"SAVI lookup DB not found: {args.savi_db}")
    
    # Process the list of molecules
    molecule_db.add_molecules(molecule_list, block=args.block, savi_lookup=savi_lookup, file_smiles_dict=file_smiles_dict)
    
    # Clean up connections
    molecule_db.close()
    if savi_lookup:
        savi_lookup.close()
    
    logging.info("Finished loading molecules")

if __name__ == "__main__":
    main()
