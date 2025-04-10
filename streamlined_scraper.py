#!/usr/bin/env python3
"""
Nova Dashboard Epoch Scraper - Simplified Version
Extracts molecule names and SMILES strings with minimal overhead
"""

import time
import logging
import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Import webdriver_manager for automatic ChromeDriver management
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nova_ui_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NovaDashboardScraper:
    def __init__(self, config):
        self.url = config.get('url', 'https://dashboard-frontend-dusky.vercel.app')
        self.output_file = config.get('output_file', 'molecules.csv')
        self.start_epoch = config.get('start_epoch', 14748)
        self.end_epoch = config.get('end_epoch', 14745)
        self.timeout = config.get('timeout', 30)
        self.navigation_delay = config.get('navigation_delay', 1.0)  # Reduced from original
        self.action_delay = config.get('action_delay', 0.5)         # Reduced from original
        self.retries = config.get('retries', 3)
        self.config = config  # Store the full config for later use
        
        # Setup Chrome driver
        options = Options()
        if config.get('headless', True):
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        self.driver.set_page_load_timeout(self.timeout)
        
        # Create CSV file with headers
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['molecule', 'smiles'])
        
    def run(self):
        """Run the scraper through all specified epochs"""
        try:
            # Navigate to the dashboard
            logger.info(f"Navigating to {self.url}")
            print(f"Navigating to {self.url}...")
            self.driver.get(self.url)
            time.sleep(self.navigation_delay)
            
            # Process each epoch
            for epoch_number in range(self.start_epoch, self.end_epoch - 1, -1):
                logger.info(f"Processing epoch {epoch_number}...")
                print(f"\nEpoch {epoch_number}:")
                
                # Try to navigate to this epoch
                success = self._navigate_to_epoch(epoch_number)
                if not success:
                    logger.warning(f"Could not navigate to epoch {epoch_number}, skipping")
                    print(f"Could not navigate to epoch {epoch_number}, skipping")
                    continue
                
                # Wait for the page to load completely
                time.sleep(self.navigation_delay)
                
                # Extract molecules from this epoch
                molecules = self._extract_molecules(epoch_number)
                
                # Write to the CSV file
                self._save_to_csv(molecules)
                
                logger.info(f"Completed processing epoch {epoch_number}: {len(molecules)} molecules extracted")
                print(f"Completed epoch {epoch_number}: {len(molecules)} molecules extracted")
            
            logger.info("Scraping completed successfully")
        except Exception as e:
            logger.error(f"Error during scraping: {e}", exc_info=True)
        finally:
            self.driver.quit()
    
    def _navigate_to_epoch(self, epoch_number):
        """
        Try multiple strategies to navigate to a specific epoch
        
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        # Backup the current page in case we need to restore
        current_url = self.driver.current_url
        
        for attempt in range(self.retries):
            try:
                # Based on the screenshots, we can now target the exact UI elements
                print(f"Attempt {attempt+1}/{self.retries} to navigate to epoch {epoch_number}")
                
                # Strategy 1: Look for the dropdown with "Epoch XXXXX" text in the UI
                try:
                    # Look for the epoch dropdown/button in the "Competition Tracker" section
                    epoch_dropdown_elements = self.driver.find_elements(
                        By.XPATH, 
                        "//div[contains(text(), 'Epoch')] | //span[contains(text(), 'Epoch')] | //button[contains(text(), 'Epoch')]"
                    )
                    
                    if epoch_dropdown_elements:
                        logger.info(f"Found {len(epoch_dropdown_elements)} epoch dropdown elements")
                        
                        for i, dropdown in enumerate(epoch_dropdown_elements):
                            try:
                                # First, check if this element already shows the correct epoch
                                if str(epoch_number) in dropdown.text:
                                    logger.info(f"Already on epoch {epoch_number}")
                                    return True
                                
                                # Scroll to make the dropdown visible
                                self.driver.execute_script("arguments[0].scrollIntoView(true);", dropdown)
                                time.sleep(0.5)
                                
                                # Click to open the dropdown
                                logger.info(f"Clicking epoch dropdown {i+1}: {dropdown.text}")
                                self.driver.execute_script("arguments[0].click();", dropdown)
                                time.sleep(self.action_delay)
                                
                                # Now look for the specific epoch in the dropdown list
                                # Based on the screenshot, each epoch is in its own div/row
                                epoch_options = self.driver.find_elements(
                                    By.XPATH, 
                                    f"//div[contains(text(), 'Epoch {epoch_number}')] | //span[contains(text(), 'Epoch {epoch_number}')]"
                                )
                                
                                if epoch_options:
                                    logger.info(f"Found {len(epoch_options)} options for epoch {epoch_number}")
                                    
                                    for option in epoch_options:
                                        try:
                                            # Scroll to make the option visible
                                            self.driver.execute_script("arguments[0].scrollIntoView(true);", option)
                                            time.sleep(0.5)
                                            
                                            # Click the option
                                            logger.info(f"Clicking epoch option: {option.text}")
                                            self.driver.execute_script("arguments[0].click();", option)
                                            time.sleep(self.action_delay * 2)  # Extra time for page to update
                                            
                                            # Check if navigation was successful
                                            if self._verify_epoch_navigation(epoch_number):
                                                return True
                                        except Exception as e:
                                            logger.warning(f"Error clicking epoch option: {e}")
                                            continue
                            except Exception as e:
                                logger.warning(f"Error with epoch dropdown {i+1}: {e}")
                                continue
                except Exception as e:
                    logger.warning(f"Error finding epoch dropdown: {e}")
                
                # Strategy 2: Look at the left panel sidebar with list of epochs
                try:
                    # Look for elements in the sidebar that contain the epoch text
                    sidebar_epoch_elements = self.driver.find_elements(
                        By.XPATH, 
                        f"//div[contains(text(), 'Epoch {epoch_number}')] | //span[contains(text(), 'Epoch {epoch_number}')]"
                    )
                    
                    if sidebar_epoch_elements:
                        logger.info(f"Found {len(sidebar_epoch_elements)} sidebar elements for epoch {epoch_number}")
                        
                        for i, element in enumerate(sidebar_epoch_elements):
                            try:
                                # Scroll to make the element visible
                                self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                                time.sleep(0.5)
                                
                                # Click the element
                                logger.info(f"Clicking sidebar epoch element {i+1}: {element.text}")
                                self.driver.execute_script("arguments[0].click();", element)
                                time.sleep(self.action_delay * 2)  # Extra time for page to update
                                
                                # Check if navigation was successful
                                if self._verify_epoch_navigation(epoch_number):
                                    return True
                            except Exception as e:
                                logger.warning(f"Error clicking sidebar epoch element {i+1}: {e}")
                                continue
                except Exception as e:
                    logger.warning(f"Error finding sidebar epoch elements: {e}")
                
                # Strategy 3: Check if the page title/header already shows the correct epoch
                try:
                    # Look for the page header that should contain the epoch number
                    header_elements = self.driver.find_elements(
                        By.XPATH, 
                        "//h1 | //h2 | //h3 | //div[contains(@class, 'header')] | //div[contains(@class, 'title')]"
                    )
                    
                    for header in header_elements:
                        if f"Epoch {epoch_number}" in header.text or f"Leaderboard - Epoch {epoch_number}" in header.text:
                            logger.info(f"Already on epoch {epoch_number} as indicated by header: {header.text}")
                            return True
                except Exception as e:
                    logger.warning(f"Error checking page headers: {e}")
                
                # If all strategies failed, refresh and try again
                if attempt < self.retries - 1:
                    logger.warning(f"Navigation attempt {attempt+1} failed, refreshing page")
                    self.driver.refresh()
                    time.sleep(self.navigation_delay * 2)  # Extra time for page to load
            
            except Exception as e:
                logger.error(f"Error during navigation attempt {attempt+1}: {e}")
                if attempt < self.retries - 1:
                    # Try to restore the page
                    try:
                        self.driver.get(current_url)
                        time.sleep(self.navigation_delay)
                    except:
                        pass
        
        logger.error(f"Failed to navigate to epoch {epoch_number} after {self.retries} attempts")
        return False
    
    def _verify_epoch_navigation(self, epoch_number):
        """Verify that we've successfully navigated to the specified epoch"""
        try:
            # Check for the epoch number in the page text
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            
            # Check for exact epoch number match with the word "Epoch" nearby
            if f"Epoch {epoch_number}" in page_text or f"epoch {epoch_number}" in page_text.lower():
                logger.info(f"Successfully navigated to epoch {epoch_number}")
                return True
                
            # Check if the epoch number appears anywhere on the page
            if str(epoch_number) in page_text:
                # Look for any elements with the epoch number
                epoch_elements = self.driver.find_elements(By.XPATH, f"//*[contains(text(), '{epoch_number}')]")
                for element in epoch_elements:
                    # Check if this element or its parent contains the word "Epoch"
                    element_text = element.text.lower()
                    if 'epoch' in element_text:
                        logger.info(f"Found epoch indicator with text: {element.text}")
                        return True
            
            logger.warning(f"Could not verify navigation to epoch {epoch_number}")
            return False
        except Exception as e:
            logger.error(f"Error verifying epoch navigation: {e}")
            return False
    
    def _extract_molecules(self, epoch_number):
        """Extract molecule data from the current epoch page"""
        molecules = []
        
        try:
            # Wait for the page to load completely
            time.sleep(self.navigation_delay)
            
            # Find the leaderboard
            leaderboard_headers = self.driver.find_elements(
                By.XPATH, 
                "//div[contains(text(), 'Leaderboard')] | //h2[contains(text(), 'Leaderboard')] | //h3[contains(text(), 'Leaderboard')]"
            )
            
            if leaderboard_headers:
                logger.info("Found Leaderboard page")
                
                # Look for the table with columns
                try:
                    # First find the headers to confirm we have the right structure
                    header_elements = self.driver.find_elements(
                        By.XPATH, 
                        "//th[contains(text(), 'Rank')] | //th[contains(text(), 'SMILES')] | //th[contains(text(), 'Actions')] | " +
                        "//div[contains(text(), 'Rank')] | //div[contains(text(), 'SMILES')] | //div[contains(text(), 'Actions')]"
                    )
                    
                    if header_elements:
                        logger.info(f"Found table headers: {[h.text for h in header_elements]}")
                        
                        # Look for the "Inspect" links/buttons
                        inspect_links = self.driver.find_elements(
                            By.XPATH, 
                            "//a[contains(text(), 'Inspect')] | //button[contains(text(), 'Inspect')] | " +
                            "//div[contains(text(), 'Inspect')] | //span[contains(text(), 'Inspect')]"
                        )
                        
                        if inspect_links:
                            logger.info(f"Found {len(inspect_links)} 'Inspect' links")
                            print(f"Processing {len(inspect_links)} submissions...")
                            
                            # Process each link
                            for i, link in enumerate(inspect_links):
                                try:
                                    print(f"Submission {i+1}/{len(inspect_links)}...")
                                    
                                    # Get rank from row if possible
                                    rank_value = i + 1  # Default fallback
                                    
                                    try:
                                        row = link
                                        for _ in range(5):  # Go up max 5 levels to find the row
                                            try:
                                                parent = row.find_element(By.XPATH, "./..")
                                                if "tr" in parent.tag_name or "row" in parent.get_attribute("class") or "":
                                                    row = parent
                                                    break
                                                row = parent
                                            except:
                                                break
                                        
                                        # Extract rank from the first cell
                                        row_cells = row.find_elements(By.XPATH, ".//td | .//div[contains(@class, 'cell')]")
                                        if row_cells and len(row_cells) > 0:
                                            rank_text = row_cells[0].text.strip()
                                            if rank_text.isdigit():
                                                rank_value = int(rank_text)
                                    except:
                                        pass
                                    
                                    # Scroll to make the link visible
                                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link)
                                    time.sleep(self.action_delay / 2)
                                    
                                    # Store window handles to detect new windows
                                    original_window = self.driver.current_window_handle
                                    original_windows = self.driver.window_handles
                                    
                                    # Click the inspect link
                                    logger.info(f"Clicking 'Inspect' link {i+1}")
                                    self.driver.execute_script("arguments[0].click();", link)
                                    
                                    # Wait for popup or new window
                                    try:
                                        # Check if a new window opened
                                        WebDriverWait(self.driver, 3).until(
                                            lambda d: len(d.window_handles) > len(original_windows)
                                        )
                                        
                                        # Switch to the new window
                                        for window_handle in self.driver.window_handles:
                                            if window_handle != original_window:
                                                self.driver.switch_to.window(window_handle)
                                                logger.info("Switched to new window")
                                                break
                                    except TimeoutException:
                                        # No new window, must be a popup
                                        logger.info("No new window detected, looking for popup")
                                        time.sleep(self.action_delay)
                                    
                                    # Extract data
                                    molecule_data = self._extract_from_popup(rank_value)
                                    
                                    # If we have a new window, close it and switch back
                                    if len(self.driver.window_handles) > len(original_windows):
                                        self.driver.close()
                                        self.driver.switch_to.window(original_window)
                                    else:
                                        # Close popup
                                        self._close_popup()
                                    
                                    if molecule_data:
                                        molecules.append(molecule_data)
                                        
                                    time.sleep(self.action_delay / 2)
                                    
                                except Exception as e:
                                    logger.error(f"Error processing link {i+1}: {e}")
                                    # Try to restore window state
                                    try:
                                        if len(self.driver.window_handles) > len(original_windows):
                                            self.driver.close()
                                            self.driver.switch_to.window(original_window)
                                        else:
                                            self._close_popup()
                                    except:
                                        pass
                        else:
                            logger.warning("No 'Inspect' links found")
                    else:
                        logger.warning("Could not find table headers")
                except Exception as e:
                    logger.error(f"Error processing table: {e}")
            
            logger.info(f"Extracted {len(molecules)} molecules")
            
        except Exception as e:
            logger.error(f"Error extracting molecules: {e}")
        
        return molecules
    
    def _extract_from_popup(self, rank):
        """Extract molecule name and SMILES string from popup"""
        try:
            # Extract molecule ID
            molecule_id = None
            molecule_elements = self.driver.find_elements(By.XPATH, 
                "//div[contains(text(), 'Molecule')] | //span[contains(text(), 'Molecule')]")
            
            if molecule_elements:
                for element in molecule_elements:
                    parent = element
                    for _ in range(3):  # Go up a few levels
                        try:
                            parent = parent.find_element(By.XPATH, "./..")
                        except:
                            break
                    
                    candidates = parent.find_elements(By.XPATH, ".//div | .//span | .//pre | .//code | .//p")
                    for elem in candidates:
                        # Skip the label element
                        if "Molecule" in elem.text:
                            continue
                        
                        text = elem.text.strip()
                        # IDs often contain underscores
                        if text and ('_' in text or text.startswith('F')):
                            molecule_id = text
                            break
                    
                    if molecule_id:
                        break
            
            # Extract SMILES string
            smiles_string = None
            smiles_elements = self.driver.find_elements(By.XPATH, 
                "//div[contains(text(), 'SMILES')] | //span[contains(text(), 'SMILES')]")
            
            if smiles_elements:
                for element in smiles_elements:
                    parent = element
                    for _ in range(3):  # Go up a few levels
                        try:
                            parent = parent.find_element(By.XPATH, "./..")
                        except:
                            break
                    
                    candidates = parent.find_elements(By.XPATH, ".//div | .//span | .//pre | .//code | .//p")
                    for elem in candidates:
                        # Skip the label element
                        if "SMILES" in elem.text:
                            continue
                        
                        text = elem.text.strip()
                        # SMILES have patterns like C=, C(, etc.
                        if text and len(text) > 10 and ('C' in text) and any(pattern in text for pattern in ['=', '(', '[', '#']):
                            smiles_string = text
                            break
                    
                    if smiles_string:
                        break
            
            # If we still don't have a SMILES string, try one more general approach
            if not smiles_string:
                all_elements = self.driver.find_elements(By.XPATH, ".//div | .//span | .//pre | .//code | .//p")
                for elem in all_elements:
                    text = elem.text.strip()
                    # Skip texts that are too short/long or have spaces
                    if not text or len(text) < 10 or len(text) > 500 or ' ' in text:
                        continue
                    
                    # SMILES pattern check
                    if ('C' in text) and any(pattern in text for pattern in ['=', '(', '[', '#']):
                        smiles_string = text
                        break
            
            # Return data if we have SMILES
            if smiles_string:
                return (molecule_id or f"unknown_{rank}", smiles_string)
            return None
            
        except Exception as e:
            logger.error(f"Error extracting from popup: {e}")
            return None
    
    def _close_popup(self):
        """Close the popup - simplified version"""
        try:
            # First try Escape key (fastest method)
            webdriver.ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(self.action_delay / 2)
            
            # Try Close button
            close_buttons = self.driver.find_elements(By.XPATH, 
                "//button[contains(text(), 'Close')] | //button[@aria-label='Close']")
            
            if close_buttons:
                self.driver.execute_script("arguments[0].click();", close_buttons[0])
                time.sleep(self.action_delay / 2)
        except Exception as e:
            logger.error(f"Error closing popup: {e}")
    
    def _save_to_csv(self, molecules):
        """Save extracted molecules to CSV file"""
        try:
            with open(self.output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for molecule_id, smiles in molecules:
                    writer.writerow([molecule_id, smiles])
            
            logger.info(f"Saved {len(molecules)} molecules to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description='Nova Dashboard Scraper')
    
    parser.add_argument('--url', default='https://dashboard-frontend-dusky.vercel.app',
                      help='Dashboard URL')
    parser.add_argument('--output-file', default='molecules.csv',
                      help='Output CSV file path')
    parser.add_argument('--start-epoch', type=int, default=14748,
                      help='Start from this epoch number')
    parser.add_argument('--end-epoch', type=int, default=14747,
                      help='End at this epoch number')
    parser.add_argument('--navigation-delay', type=float, default=1.0,
                      help='Delay after navigation in seconds')
    parser.add_argument('--action-delay', type=float, default=0.5,
                      help='Delay between actions in seconds')
    parser.add_argument('--retries', type=int, default=3,
                      help='Number of retries for failed operations')
    parser.add_argument('--no-headless', action='store_true',
                      help='Disable headless mode (show browser window)')
    
    args = parser.parse_args()
    
    config = {
        'url': args.url,
        'output_file': args.output_file,
        'start_epoch': args.start_epoch,
        'end_epoch': args.end_epoch,
        'navigation_delay': args.navigation_delay,
        'action_delay': args.action_delay,
        'retries': args.retries,
        'headless': not args.no_headless
    }
    
    scraper = NovaDashboardScraper(config)
    scraper.run()

if __name__ == "__main__":
    main()
