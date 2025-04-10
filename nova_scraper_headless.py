#!/usr/bin/env python3
"""
Nova Dashboard Headless Scraper

This script is designed to run on RunPod or other headless environments.
It automates the process of extracting molecule data from the Nova Dashboard
without requiring a web interface.
"""

import os
import json
import time
import logging
import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, 
    ElementNotInteractableException, StaleElementReferenceException
)

# Import ChromeDriverManager for automatic ChromeDriver handling
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nova_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NovaDashboardScraper:
    """
    A web scraper for the Nova Dashboard that extracts molecule data.
    Designed for headless environments like RunPod.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the scraper with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.url = config.get('url', 'https://dashboard-frontend-dusky.vercel.app')
        self.output_dir = Path(config.get('output_dir', './nova_data'))
        self.start_epoch = config.get('start_epoch', 14748)
        self.end_epoch = config.get('end_epoch', 14745)
        self.timeout = config.get('timeout', 30)
        self.navigation_delay = config.get('navigation_delay', 2.0)
        self.action_delay = config.get('action_delay', 1.0)
        self.retries = config.get('retries', 3)
        self.take_screenshots = config.get('take_screenshots', True)
        self.verbose = config.get('verbose', True)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a screenshots directory if needed
        if self.take_screenshots:
            self.screenshots_dir = self.output_dir / 'screenshots'
            self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the database to store molecule data
        self.db_path = self.output_dir / 'molecules_database.json'
        self.db = self._load_database()
        
        # Progress tracking
        self.molecules_processed = 0
        self.start_time = datetime.now()
        
        # Setup webdriver - Always use headless on RunPod
        self.driver = self._setup_webdriver(True)
    
    def _setup_webdriver(self, headless: bool = True) -> webdriver.Chrome:
        """
        Set up and configure the Chrome WebDriver using webdriver-manager.
        
        Args:
            headless: Whether to run in headless mode
            
        Returns:
            A configured Chrome WebDriver instance
        """
        chrome_options = Options()
        
        # Always use headless mode on RunPod
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Additional options for stability
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-infobars")
        
        try:
            # Create a Chrome browser instance using webdriver-manager
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            
            driver.set_page_load_timeout(self.timeout)
            return driver
        except Exception as e:
            logger.error(f"Failed to initialize webdriver: {e}")
            raise
    
    def _load_database(self) -> Dict[str, Any]:
        """
        Load existing database or create a new one.
        
        Returns:
            Dictionary containing the molecule database
        """
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    db = json.load(f)
                logger.info(f"Loaded existing database with {len(db)} epochs")
                return db
            except Exception as e:
                logger.error(f"Error loading database: {e}")
                return {}
        else:
            logger.info("No existing database found, creating new one")
            return {}
    
    def _save_database(self) -> None:
        """Save the current state of the database to disk."""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.db, f, indent=2)
            logger.info(f"Saved database to {self.db_path}")
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def run(self) -> None:
        """
        Run the scraper to extract molecule data from the Nova Dashboard.
        """
        try:
            # Navigate to the dashboard
            logger.info(f"Navigating to {self.url}")
            print(f"Navigating to {self.url}...")
            self.driver.get(self.url)
            time.sleep(self.navigation_delay)
            
            if self.verbose:
                self._print_progress_header()
            
            # Process each epoch in descending order
            for epoch_number in range(self.start_epoch, self.end_epoch - 1, -1):
                logger.info(f"Processing epoch {epoch_number}...")
                print(f"\nEpoch {epoch_number}:")
                
                # Initialize this epoch in the database if it doesn't exist
                if str(epoch_number) not in self.db:
                    self.db[str(epoch_number)] = {
                        "epochNumber": epoch_number,
                        "processingDate": datetime.now().isoformat(),
                        "molecules": []
                    }
                
                # Select the epoch with retry logic
                for attempt in range(self.retries):
                    try:
                        self._select_epoch(epoch_number)
                        break
                    except Exception as e:
                        if attempt < self.retries - 1:
                            logger.warning(f"Failed to select epoch {epoch_number} (attempt {attempt+1}/{self.retries}): {e}")
                            time.sleep(self.action_delay * 2)
                        else:
                            logger.error(f"Failed to select epoch {epoch_number} after {self.retries} attempts: {e}")
                            raise
                
                # Wait for the page to stabilize after epoch selection
                time.sleep(self.navigation_delay)
                
                # Scroll to the leaderboard section
                self._scroll_to_leaderboard()
                
                # Take a screenshot of the leaderboard for debugging
                if self.take_screenshots:
                    screenshot_path = self.screenshots_dir / f"epoch_{epoch_number}_leaderboard.png"
                    self.driver.save_screenshot(str(screenshot_path))
                
                # Extract data from the leaderboard table
                molecules = self._extract_molecule_data(epoch_number)
                
                # Update the database with the new molecules
                self.db[str(epoch_number)]["molecules"] = molecules
                self.db[str(epoch_number)]["lastUpdated"] = datetime.now().isoformat()
                
                # Save the database after processing each epoch
                self._save_database()
                
                # Generate a CSV export for this epoch
                self._export_epoch_to_csv(epoch_number, molecules)
                
                logger.info(f"Completed processing epoch {epoch_number}: {len(molecules)} molecules extracted")
                print(f"Completed epoch {epoch_number}: {len(molecules)} molecules extracted")
            
            logger.info("Scraping completed successfully.")
            self._print_summary()
        
        except Exception as e:
            logger.error(f"Error during scraping: {e}", exc_info=True)
        finally:
            # Always close the browser and save the database
            self.driver.quit()
            self._save_database()
    
    def _print_progress_header(self):
        """Print a header for progress tracking."""
        print("\n" + "="*80)
        print(" NOVA DASHBOARD MOLECULE SCRAPER ".center(80, "="))
        print("="*80)
        print(f"URL: {self.url}")
        print(f"Epochs: {self.start_epoch} to {self.end_epoch}")
        print(f"Output Directory: {self.output_dir}")
        print("="*80 + "\n")
    
    def _print_summary(self):
        """Print a summary of the scraping results."""
        elapsed_time = datetime.now() - self.start_time
        hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n" + "="*80)
        print(" SCRAPING COMPLETED ".center(80, "="))
        print("="*80)
        print(f"Total molecules processed: {self.molecules_processed}")
        print(f"Total time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Output files saved to: {self.output_dir}")
        print("="*80 + "\n")
    
    def _select_epoch(self, epoch_number: int) -> None:
        """
        Select an epoch from the dropdown on the dashboard.
        
        Args:
            epoch_number: The epoch number to select
        """
        # Take a screenshot before selecting epoch
        if self.take_screenshots:
            screenshot_path = self.screenshots_dir / f"before_epoch_select_{epoch_number}.png"
            self.driver.save_screenshot(str(screenshot_path))
        
        # Try various potential selectors for the epoch dropdown
        dropdown_selectors = [
            'select[name="epoch"]',
            'select:contains("Epoch")',
            'button:contains("Epoch")',
            'div[role="combobox"]',
            '.epoch-selector',
            '.dropdown',
            'div:contains("Epoch"):has(select)'
        ]
        
        dropdown_found = False
        
        # Try to find the dropdown with different CSS selectors
        for selector in dropdown_selectors:
            try:
                # First try using CSS selector
                dropdown = self.driver.find_element(By.CSS_SELECTOR, selector)
                dropdown.click()
                dropdown_found = True
                logger.info(f"Found and clicked epoch dropdown using selector: {selector}")
                time.sleep(self.action_delay)
                break
            except (NoSuchElementException, ElementNotInteractableException):
                # CSS selector didn't work, continue trying other selectors
                continue
        
        # If standard selectors fail, look for any element containing "Epoch"
        if not dropdown_found:
            logger.warning("Could not find epoch dropdown with standard selectors, trying alternative approach")
            
            # Find all elements and look for one containing "Epoch" in its text
            elements = self.driver.find_elements(By.XPATH, "//*")
            for element in elements:
                try:
                    if "Epoch" in element.text and (
                        element.tag_name in ["button", "select"] or 
                        element.get_attribute("role") or
                        "select" in element.get_attribute("class") or ''
                    ):
                        element.click()
                        dropdown_found = True
                        logger.info("Found and clicked epoch element via text search")
                        time.sleep(self.action_delay)
                        break
                except StaleElementReferenceException:
                    continue
        
        # If still not found, try JavaScript approach
        if not dropdown_found:
            logger.warning("Could not find epoch dropdown with DOM methods, trying JavaScript approach")
            try:
                # Use JavaScript to find and click any epoch-related element
                result = self.driver.execute_script("""
                    // Look for elements containing 'Epoch' text or attributes
                    const elements = Array.from(document.querySelectorAll('*'));
                    for (const el of elements) {
                        if (el.textContent.includes('Epoch') || 
                            (el.getAttribute('aria-label') && el.getAttribute('aria-label').includes('Epoch')) ||
                            (el.id && el.id.includes('epoch')) ||
                            (el.className && el.className.includes('epoch'))) {
                            
                            // Try to click it
                            el.click();
                            return true;
                        }
                    }
                    return false;
                """)
                
                if result:
                    dropdown_found = True
                    logger.info("Found and clicked epoch element via JavaScript")
                    time.sleep(self.action_delay)
            except Exception as e:
                logger.error(f"JavaScript approach failed: {e}")
        
        if not dropdown_found:
            # Take a screenshot for debugging
            if self.take_screenshots:
                screenshot_path = self.screenshots_dir / f"dropdown_not_found_{epoch_number}.png"
                self.driver.save_screenshot(str(screenshot_path))
            
            raise ValueError("Could not find epoch selector with any method")
        
        # Now try to select the specific epoch
        try:
            # First try to find an option with this exact epoch
            epoch_options = [
                f"option[value='{epoch_number}']",
                f"li[data-value='{epoch_number}']",
                f"div[role='option'][data-value='{epoch_number}']"
            ]
            
            option_found = False
            for selector in epoch_options:
                try:
                    option = self.driver.find_element(By.CSS_SELECTOR, selector)
                    option.click()
                    option_found = True
                    logger.info(f"Selected epoch {epoch_number} by clicking on option")
                    time.sleep(self.action_delay)
                    break
                except NoSuchElementException:
                    continue
            
            # If no exact match found, try using a text search
            if not option_found:
                elements = self.driver.find_elements(By.XPATH, f"//*[contains(text(), '{epoch_number}')]")
                for element in elements:
                    try:
                        if str(epoch_number) in element.text:
                            element.click()
                            option_found = True
                            logger.info(f"Selected epoch {epoch_number} via text search")
                            time.sleep(self.action_delay)
                            break
                    except StaleElementReferenceException:
                        continue
            
            # If still not found, try typing the epoch number
            if not option_found:
                # Try to find an input field
                try:
                    input_field = self.driver.find_element(By.CSS_SELECTOR, "input")
                    input_field.clear()
                    input_field.send_keys(str(epoch_number))
                    input_field.send_keys(Keys.ENTER)
                    logger.info(f"Typed epoch {epoch_number} and pressed Enter")
                    time.sleep(self.action_delay)
                except NoSuchElementException:
                    # If no input field, try sending keys to the active element
                    active_element = self.driver.switch_to.active_element
                    active_element.send_keys(str(epoch_number))
                    active_element.send_keys(Keys.ENTER)
                    logger.info(f"Sent epoch {epoch_number} to active element and pressed Enter")
                    time.sleep(self.action_delay)
        
        except Exception as e:
            logger.error(f"Error selecting specific epoch {epoch_number}: {e}")
            raise
    
    def _scroll_to_leaderboard(self) -> None:
        """Scroll to the leaderboard section of the page."""
        try:
            # Execute JavaScript to scroll to the leaderboard
            self.driver.execute_script("""
                // Look for the leaderboard heading or table
                const selectors = [
                    'h2:contains("Leaderboard")', 
                    'div:contains("Leaderboard")',
                    'table', 
                    'div.table-container'
                ];
                
                for (const selector of selectors) {
                    const element = document.querySelector(selector);
                    if (element) {
                        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        return true;
                    }
                }
                
                // If no specific element found, scroll down the page
                window.scrollTo(0, document.body.scrollHeight * 0.7);
                return false;
            """)
            
            time.sleep(self.action_delay)
            logger.info("Scrolled to leaderboard section")
        except Exception as e:
            logger.error(f"Error scrolling to leaderboard: {e}")
    
    def _extract_molecule_data(self, epoch_number: int) -> List[Dict[str, Any]]:
        """
        Extract molecule data from the leaderboard table.
        
        Args:
            epoch_number: The current epoch number
            
        Returns:
            List of dictionaries containing molecule data
        """
        # Get existing molecules for this epoch
        existing_molecules = self.db.get(str(epoch_number), {}).get("molecules", [])
        
        # Create a map of existing molecules for quick lookup
        existing_map = {molecule.get("rank"): molecule for molecule in existing_molecules if molecule.get("rank")}
        
        # Result list will contain both existing and new molecules
        molecules = list(existing_molecules)
        
        try:
            # Count how many "Inspect" buttons are in the table
            inspect_buttons = self.driver.find_elements(By.XPATH, 
                "//button[contains(text(), 'Inspect')] | //a[contains(text(), 'Inspect')]")
            
            inspect_button_count = len(inspect_buttons)
            logger.info(f"Found {inspect_button_count} submission entries to inspect for epoch {epoch_number}")
            
            if inspect_button_count == 0:
                logger.warning(f"No inspect buttons found for epoch {epoch_number}. The page might not have loaded correctly.")
                
                # Take a screenshot of the current state for debugging
                if self.take_screenshots:
                    screenshot_path = self.screenshots_dir / f"no_buttons_epoch_{epoch_number}.png"
                    self.driver.save_screenshot(str(screenshot_path))
                    logger.info(f"Saved diagnostic screenshot to {screenshot_path}")
                
                # Try an alternative approach to find buttons - look for any elements that might be "Inspect" buttons
                try:
                    logger.info("Trying alternative approach to find Inspect buttons")
                    
                    # Try to find the table first
                    tables = self.driver.find_elements(By.TAG_NAME, "table")
                    if tables:
                        # Look for buttons within the table
                        buttons = tables[0].find_elements(By.TAG_NAME, "button")
                        logger.info(f"Found {len(buttons)} buttons in the table")
                        
                        # Filter buttons that might be "Inspect" buttons
                        inspect_buttons = [b for b in buttons if "inspect" in b.get_attribute("class").lower() or 
                                          "view" in b.get_attribute("class").lower() or 
                                          "detail" in b.get_attribute("class").lower()]
                        
                        if inspect_buttons:
                            logger.info(f"Found {len(inspect_buttons)} potential inspect buttons")
                            inspect_button_count = len(inspect_buttons)
                        else:
                            # If still no luck, just use all buttons in the table
                            logger.info("Using all buttons in the table as potential inspect buttons")
                            inspect_buttons = buttons
                            inspect_button_count = len(buttons)
                except Exception as e:
                    logger.error(f"Alternative button finding approach failed: {e}")
            
            # If still no buttons, return existing molecules
            if inspect_button_count == 0:
                return molecules
            
            # Progress tracking
            progress_increment = max(1, inspect_button_count // 10)  # Show progress every ~10% 
            
            # Process each submission by clicking "Inspect"
            for i in range(inspect_button_count):
                rank = i + 1
                
                # Print progress
                if i % progress_increment == 0 or i == inspect_button_count - 1:
                    progress = f"[{i+1}/{inspect_button_count}]"
                    elapsed = datetime.now() - self.start_time
                    print(f"{progress} Processing submission {rank}... (Elapsed: {elapsed.seconds}s)")
                
                # Skip if we already have data for this rank and it's complete
                if rank in existing_map and existing_map[rank].get("moleculeId") and existing_map[rank].get("smilesString"):
                    logger.info(f"Skipping rank {rank} in epoch {epoch_number}, already processed")
                    continue
                
                logger.info(f"Processing submission rank {rank}/{inspect_button_count} for epoch {epoch_number}")
                
                try:
                    # Get an updated list of inspect buttons (they might have changed)
                    inspect_buttons = self.driver.find_elements(By.XPATH, 
                        "//button[contains(text(), 'Inspect')] | //a[contains(text(), 'Inspect')]")
                    
                    # Click the current inspect button if it exists
                    if i < len(inspect_buttons):
                        inspect_buttons[i].click()
                        time.sleep(self.action_delay)
                        
                        # Extract molecule data from the popup
                        molecule_data = self._extract_molecule_from_popup(epoch_number, rank)
                        
                        if molecule_data:
                            self.molecules_processed += 1
                            
                            # Add or update molecule in our result list
                            if rank in existing_map:
                                # Update existing entry
                                for idx, molecule in enumerate(molecules):
                                    if molecule.get("rank") == rank:
                                        molecules[idx] = {**molecule, **molecule_data}
                                        break
                            else:
                                # Add new entry
                                molecules.append(molecule_data)
                            
                            logger.info(f"Successfully extracted data for molecule rank {rank}")
                        
                        # Close the popup
                        popup_closed = self._close_popup()
                        if not popup_closed:
                            logger.warning(f"Could not close popup normally for rank {rank}, trying page refresh")
                            self.driver.refresh()
                            time.sleep(self.navigation_delay)
                            self._select_epoch(epoch_number)
                            time.sleep(self.navigation_delay)
                            self._scroll_to_leaderboard()
                    
                    time.sleep(self.action_delay)
                
                except Exception as e:
                    logger.error(f"Error processing submission rank {rank}: {e}")
                    
                    # Take a screenshot of the error state
                    if self.take_screenshots:
                        error_screenshot_path = self.screenshots_dir / f"error_epoch_{epoch_number}_rank_{rank}.png"
                        self.driver.save_screenshot(str(error_screenshot_path))
                    
                    # Try to recover by refreshing the page
                    self.driver.refresh()
                    time.sleep(self.navigation_delay)
                    self._select_epoch(epoch_number)
                    time.sleep(self.navigation_delay)
                    self._scroll_to_leaderboard()
        
        except Exception as e:
            logger.error(f"Error extracting molecule data for epoch {epoch_number}: {e}")
        
        # Sort molecules by rank to ensure consistent order
        molecules.sort(key=lambda x: x.get("rank", 0))
        
        return molecules
    
    def _extract_molecule_from_popup(self, epoch_number: int, rank: int) -> Optional[Dict[str, Any]]:
        """
        Extract molecule details from the popup.
        
        Args:
            epoch_number: The current epoch number
            rank: The rank of the current molecule
            
        Returns:
            Dictionary containing molecule data or None if extraction failed
        """
        try:
            # Wait for popup content to be visible
            WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 
                    'div[role="dialog"], .modal, .modal-content, .popup'))
            )
            
            # Take a screenshot of the molecule visualization if enabled
            if self.take_screenshots:
                screenshot_dir = self.screenshots_dir / f"epoch_{epoch_number}"
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                
                screenshot_path = screenshot_dir / f"molecule_{rank}.png"
                self.driver.save_screenshot(str(screenshot_path))
                logger.info(f"Saved molecule screenshot to {screenshot_path}")
            
            # Helper function to find text by label using JavaScript
            def find_value_by_label(label_text):
                js_code = f"""
                    // Helper function to find text by label
                    const findValueByLabel = (labelText) => {{
                        // Find elements that might contain labels
                        const possibleLabels = Array.from(document.querySelectorAll('label, th, div, strong, span, p'))
                            .filter(el => el.textContent.includes(labelText));
                        
                        if (possibleLabels.length === 0) return null;
                        
                        // For each potential label, try to find the associated value
                        for (const label of possibleLabels) {{
                            // Check if the next sibling has the value
                            if (label.nextElementSibling) {{
                                return label.nextElementSibling.textContent.trim();
                            }}
                            
                            // Check if the parent has the value in another child
                            const parent = label.parentElement;
                            if (parent) {{
                                const children = Array.from(parent.children);
                                const labelIndex = children.indexOf(label);
                                if (labelIndex >= 0 && labelIndex + 1 < children.length) {{
                                    return children[labelIndex + 1].textContent.trim();
                                }}
                                
                                // Check for value in sibling elements
                                const siblings = Array.from(parent.children).filter(el => el !== label);
                                for (const sibling of siblings) {{
                                    if (!sibling.textContent.includes(labelText)) {{
                                        const value = sibling.textContent.trim();
                                        if (value) return value;
                                    }}
                                }}
                            }}
                            
                            // Check if value is in a nearby element with a specific class
                            const valueContainers = document.querySelectorAll('.value, .data, .info, .details');
                            for (const container of valueContainers) {{
                                if (container.previousElementSibling === label) {{
                                    return container.textContent.trim();
                                }}
                            }}
                        }}
                        
                        return null;
                    }};
                    
                    return findValueByLabel("{label_text}");
                """
                return self.driver.execute_script(js_code)
            
            # Extract various pieces of data
            molecule_id = find_value_by_label("Molecule ID") or find_value_by_label("ID")
            
            # If regular lookup failed, try pattern matching for Molecule ID
            if not molecule_id:
                molecule_id_js = """
                    const texts = Array.from(document.querySelectorAll('div, span, p'))
                        .map(el => el.textContent.trim())
                        .filter(text => /^[A-F0-9]{16}_[A-F0-9]{16}/.test(text) || 
                                        /^[A-F0-9]{8}.*_\\d+/.test(text));
                    
                    return texts.length > 0 ? texts[0] : null;
                """
                molecule_id = molecule_id or self.driver.execute_script(molecule_id_js)
            
            # Look for SMILES string
            smiles_string = find_value_by_label("SMILES")
            
            # If regular lookup failed, try pattern matching for SMILES
            if not smiles_string:
                smiles_js = """
                    const texts = Array.from(document.querySelectorAll('div, span, p'))
                        .map(el => el.textContent.trim())
                        .filter(text => {
                            // SMILES strings typically contain these characters
                            return /^[A-Za-z0-9\\(\\)=\\[\\]\\.#@\\-\\\\\\/\\+]+$/.test(text) && 
                                text.length > 10 &&
                                (text.includes('=') || text.includes('C') || text.includes('N'));
                        });
                    
                    return texts.length > 0 ? texts[0] : null;
                """
                smiles_string = smiles_string or self.driver.execute_script(smiles_js)
            
            # Extract score if available
            score_text = find_value_by_label("Score")
            score = None
            if score_text:
                # Try to extract a numeric value from the score text
                import re
                score_match = re.search(r'[-+]?[0-9]*\.?[0-9]+', score_text)
                if score_match:
                    score = float(score_match.group(0))
            
            # Extract submitter info
            submitter = (find_value_by_label("Submitted by") or 
                        find_value_by_label("UID") or 
                        find_value_by_label("User"))
            
            # Extract coldkey/hotkey if available
            coldkey = find_value_by_label("Coldkey")
            hotkey = find_value_by_label("Hotkey")
            
            # Construct the molecule data dictionary
            return {
                "moleculeId": molecule_id,
                "smilesString": smiles_string,
                "score": score,
                "submitter": submitter,
                "coldkey": coldkey,
                "hotkey": hotkey,
                "rank": rank,
                "epochNumber": epoch_number,
                "extractionTimestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error extracting data from popup: {e}")
            return None
    
    def _close_popup(self) -> bool:
        """
        Attempt to close the popup window.
        
        Returns:
            True if popup was closed successfully, False otherwise
        """
        try:
            # Try multiple possible selectors for close buttons
            close_selectors = [
                'button:contains("Close")', 
                'button.close', 
                'button[aria-label="Close"]',
                '.modal-close',
                '.close-button',
                '.modal-header button',
                'svg[aria-label="Close"]'
            ]
            
            for selector in close_selectors:
                try:
                    close_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    close_button.click()
                    time.sleep(self.action_delay)
                    logger.info(f"Closed popup using selector: {selector}")
                    return True
                except NoSuchElementException:
                    continue
            
            # If no button found, try pressing Escape key
            self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ESCAPE)
            time.sleep(self.action_delay)
            
            # Check if popup is still visible
            try:
                self.driver.find_element(By.CSS_SELECTOR, 'div[role="dialog"], .modal, .modal-content, .popup')
                # Popup is still visible, try clicking outside
                self.driver.execute_script("""
                    const modal = document.querySelector('div[role="dialog"], .modal, .modal-content, .popup');
                    if (modal && modal.parentElement) {
                        // Click on the modal backdrop/overlay
                        modal.parentElement.click();
                        return true;
                    }
                    return false;
                """)
                time.sleep(self.action_delay)
                logger.info('Attempted to close popup by clicking outside')
            except NoSuchElementException:
                # Popup is gone
                logger.info('Closed popup using Escape key')
                return True
            
            # Final check if popup is gone
            try:
                self.driver.find_element(By.CSS_SELECTOR, 'div[role="dialog"], .modal, .modal-content, .popup')
                return False  # Popup still visible
            except NoSuchElementException:
                return True  # Popup is gone
        
        except Exception as e:
            logger.error(f"Error closing popup: {e}")
            return False
    
    def _export_epoch_to_csv(self, epoch_number: int, molecules: List[Dict[str, Any]]) -> None:
        """
        Export epoch data to CSV format.
        
        Args:
            epoch_number: The epoch number
            molecules: List of molecule data dictionaries
        """
        try:
            csv_path = self.output_dir / f"epoch_{epoch_number}_molecules.csv"
            
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = [
                    'Rank', 
                    'Molecule ID', 
                    'SMILES', 
                    'Score',
                    'Submitter',
                    'Coldkey',
                    'Hotkey',
                    'Extraction Timestamp'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for molecule in molecules:
                    writer.writerow({
                        'Rank': molecule.get('rank', ''),
                        'Molecule ID': molecule.get('moleculeId', ''),
                        'SMILES': molecule.get('smilesString', ''),
                        'Score': molecule.get('score', ''),
                        'Submitter': molecule.get('submitter', ''),
                        'Coldkey': molecule.get('coldkey', ''),
                        'Hotkey': molecule.get('hotkey', ''),
                        'Extraction Timestamp': molecule.get('extractionTimestamp', '')
                    })
            
            logger.info(f"Exported {len(molecules)} molecules to CSV: {csv_path}")
            
            # Also create a simple SMILES-only file for convenience
            smiles_path = self.output_dir / f"epoch_{epoch_number}_smiles.txt"
            with open(smiles_path, 'w') as f:
                for molecule in molecules:
                    if molecule.get('smilesString'):
                        f.write(f"{molecule.get('smilesString')}\n")
            
            logger.info(f"Exported SMILES strings to: {smiles_path}")
        
        except Exception as e:
            logger.error(f"Error exporting CSV for epoch {epoch_number}: {e}")


def main():
    """Main entry point for the scraper."""
    parser = argparse.ArgumentParser(description='Nova Dashboard Molecule Scraper (Headless)')
    
    parser.add_argument('--url', default='https://dashboard-frontend-dusky.vercel.app',
                        help='URL of the Nova Dashboard')
    parser.add_argument('--output-dir', default='./nova_data',
                        help='Directory to save output data')
    parser.add_argument('--start-epoch', type=int, default=14748,
                        help='Start from this epoch number')
    parser.add_argument('--end-epoch', type=int, default=14745,
                        help='End at this epoch number')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Page load timeout in seconds')
    parser.add_argument('--navigation-delay', type=float, default=2.0,
                        help='Delay after page navigation in seconds')
    parser.add_argument('--action-delay', type=float, default=1.0,
                        help='Delay between actions in seconds')
    parser.add_argument('--retries', type=int, default=3,
                        help='Number of retries for failed operations')
    parser.add_argument('--no-screenshots', action='store_true',
                        help='Disable taking screenshots of molecules')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--config', type=str,
                        help='Path to JSON configuration file')
    
    args = parser.parse_args()
    
    # Load configuration from file if specified
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration from {args.config}: {e}")
            config = {}
    else:
        config = {}
    
    # Command line arguments override configuration file
    config.update({
        'url': args.url,
        'output_dir': args.output_dir,
        'start_epoch': args.start_epoch,
        'end_epoch': args.end_epoch,
        'timeout': args.timeout,
        'navigation_delay': args.navigation_delay,
        'action_delay': args.action_delay,
        'retries': args.retries,
        'take_screenshots': not args.no_screenshots,
        'verbose': args.verbose or not args.quiet
    })
    
    # Display configuration for debugging
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create and run the scraper
    scraper = NovaDashboardScraper(config)
    scraper.run()


if __name__ == "__main__":
    main()
