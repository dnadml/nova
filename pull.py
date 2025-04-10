#!/usr/bin/env python3
"""
Nova Dashboard Epoch Scraper (Optimized Version)

This version navigates through different epochs on the Nova Dashboard
and extracts molecule IDs and SMILES strings. Screenshots and extra logging
are only enabled in debug mode.
"""

import os
import time
import logging
import argparse
import csv
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
        self.output_dir = Path(config.get('output_dir', './nova_data'))
        self.start_epoch = config.get('start_epoch', 14748)
        self.end_epoch = config.get('end_epoch', 14745)
        self.timeout = config.get('timeout', 30)
        self.navigation_delay = config.get('navigation_delay', 0.3)  # reduced default for faster run
        self.action_delay = config.get('action_delay', 0.1)          # reduced default for faster run
        self.retries = config.get('retries', 3)
        self.verbose = config.get('verbose', True)
        self.debug = config.get('debug', False)
        self.max_submissions = config.get('max_submissions', 0)
        self.config = config  # full config
        
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'screenshots').mkdir(exist_ok=True)
        
        # Start the browser
        self.driver = self._setup_webdriver()
        
    def _capture_screenshot(self, filename):
        """Save screenshot only if debug mode is active"""
        if self.debug:
            self.driver.save_screenshot(filename)
        
    def _setup_webdriver(self):
        """Set up Chrome WebDriver with reduced wait times for speed"""
        options = Options()
        if self.config.get('headless', True):
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        driver.set_page_load_timeout(self.timeout)
        # Set an implicit wait (use a small value to speed up element lookup)
        driver.implicitly_wait(self.action_delay)
        return driver
        
    def run(self):
        """Run the scraper through all specified epochs"""
        try:
            logger.info(f"Navigating to {self.url}")
            print(f"Navigating to {self.url}...")
            self.driver.get(self.url)
            time.sleep(self.navigation_delay)
            
            if self.verbose:
                self._print_progress_header()
            
            # Process each epoch from start to end (decrementing)
            for epoch_number in range(self.start_epoch, self.end_epoch - 1, -1):
                logger.info(f"Processing epoch {epoch_number}...")
                print(f"\nEpoch {epoch_number}:")
                
                if not self._navigate_to_epoch(epoch_number):
                    logger.warning(f"Could not navigate to epoch {epoch_number}, skipping")
                    continue
                
                # Allow brief wait after navigation
                time.sleep(self.navigation_delay)
                
                molecules = self._extract_molecules(epoch_number)
                self._export_to_csv(molecules)
                
                logger.info(f"Completed processing epoch {epoch_number}: {len(molecules)} molecules extracted")
                print(f"Completed epoch {epoch_number}: {len(molecules)} molecules extracted")
            
            logger.info("Scraping completed successfully")
        except Exception as e:
            logger.error(f"Error during scraping: {e}", exc_info=True)
        finally:
            self.driver.quit()
        # (Optional) Save any database if needed here
    
    def _navigate_to_epoch(self, epoch_number):
        """
        Try multiple strategies to navigate to a specific epoch.
        Returns True if navigation was successful.
        """
        current_url = self.driver.current_url
        self._capture_screenshot(f"{self.output_dir}/screenshots/before_epoch_{epoch_number}.png")
        
        for attempt in range(self.retries):
            try:
                print(f"Attempt {attempt+1}/{self.retries} to navigate to epoch {epoch_number}")
                # Strategy 1: Look for dropdown elements mentioning "Epoch"
                dropdown_elements = self.driver.find_elements(
                    By.XPATH, 
                    "//div[contains(text(), 'Epoch')] | //span[contains(text(), 'Epoch')] | //button[contains(text(), 'Epoch')]"
                )
                if dropdown_elements:
                    logger.info(f"Found {len(dropdown_elements)} epoch dropdown elements")
                    for i, dropdown in enumerate(dropdown_elements):
                        try:
                            if str(epoch_number) in dropdown.text:
                                logger.info(f"Already on epoch {epoch_number}")
                                return True
                            
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", dropdown)
                            time.sleep(0.2)
                            logger.info(f"Clicking epoch dropdown {i+1}: {dropdown.text}")
                            self.driver.execute_script("arguments[0].click();", dropdown)
                            time.sleep(self.action_delay)
                            
                            # Look for the matching epoch in the dropdown list
                            epoch_options = self.driver.find_elements(
                                By.XPATH, 
                                f"//div[contains(text(), 'Epoch {epoch_number}')] | //span[contains(text(), 'Epoch {epoch_number}')]"
                            )
                            if epoch_options:
                                logger.info(f"Found {len(epoch_options)} options for epoch {epoch_number}")
                                for option in epoch_options:
                                    try:
                                        self.driver.execute_script("arguments[0].scrollIntoView(true);", option)
                                        time.sleep(0.2)
                                        logger.info(f"Clicking epoch option: {option.text}")
                                        self.driver.execute_script("arguments[0].click();", option)
                                        time.sleep(self.action_delay * 2)
                                        if self._verify_epoch_navigation(epoch_number):
                                            return True
                                    except Exception as e:
                                        logger.warning(f"Error clicking epoch option: {e}")
                                        continue
                        except Exception as e:
                            logger.warning(f"Error with epoch dropdown {i+1}: {e}")
                            continue
                
                # Strategy 2: Look in the sidebar for the epoch
                sidebar_elements = self.driver.find_elements(
                    By.XPATH, 
                    f"//div[contains(text(), 'Epoch {epoch_number}')] | //span[contains(text(), 'Epoch {epoch_number}')]"
                )
                if sidebar_elements:
                    logger.info(f"Found {len(sidebar_elements)} sidebar elements for epoch {epoch_number}")
                    for i, element in enumerate(sidebar_elements):
                        try:
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                            time.sleep(0.2)
                            logger.info(f"Clicking sidebar element {i+1}: {element.text}")
                            self.driver.execute_script("arguments[0].click();", element)
                            time.sleep(self.action_delay * 2)
                            if self._verify_epoch_navigation(epoch_number):
                                return True
                        except Exception as e:
                            logger.warning(f"Error clicking sidebar element {i+1}: {e}")
                            continue
                
                # Strategy 3: Check if the header already indicates the correct epoch
                header_elements = self.driver.find_elements(
                    By.XPATH, 
                    "//h1 | //h2 | //h3 | //div[contains(@class, 'header')] | //div[contains(@class, 'title')]"
                )
                for header in header_elements:
                    if f"Epoch {epoch_number}" in header.text or f"Leaderboard - Epoch {epoch_number}" in header.text:
                        logger.info(f"Already on epoch {epoch_number} as indicated by header: {header.text}")
                        return True
                
                if attempt < self.retries - 1:
                    logger.warning(f"Navigation attempt {attempt+1} failed, refreshing page")
                    self.driver.refresh()
                    time.sleep(self.navigation_delay * 2)
            
            except Exception as e:
                logger.error(f"Error during navigation attempt {attempt+1}: {e}")
                if attempt < self.retries - 1:
                    try:
                        self.driver.get(current_url)
                        time.sleep(self.navigation_delay)
                    except:
                        pass
        
        logger.error(f"Failed to navigate to epoch {epoch_number} after {self.retries} attempts")
        return False
    
    def _verify_epoch_navigation(self, epoch_number):
        """Check if the current page shows the specified epoch."""
        try:
            self._capture_screenshot(f"{self.output_dir}/screenshots/verify_epoch_{epoch_number}.png")
            page_text = self.driver.find_element(By.TAG_NAME, "body").text
            if f"Epoch {epoch_number}" in page_text or f"epoch {epoch_number}" in page_text.lower():
                logger.info(f"Successfully navigated to epoch {epoch_number}")
                return True
            
            epoch_elements = self.driver.find_elements(By.XPATH, f"//*[contains(text(), '{epoch_number}')]")
            for element in epoch_elements:
                if 'epoch' in element.text.lower():
                    logger.info(f"Found epoch indicator with text: {element.text}")
                    return True
            
            logger.warning(f"Could not verify navigation to epoch {epoch_number}")
            return False
        except Exception as e:
            logger.error(f"Error verifying epoch navigation: {e}")
            return False
    
    def _extract_molecules(self, epoch_number):
        """Extract molecule data from the current epoch page."""
        molecules = []
        try:
            self._capture_screenshot(f"{self.output_dir}/screenshots/epoch_{epoch_number}_page.png")
            time.sleep(self.navigation_delay)
            
            leaderboard_headers = self.driver.find_elements(
                By.XPATH, 
                "//div[contains(text(), 'Leaderboard')] | //h2[contains(text(), 'Leaderboard')] | //h3[contains(text(), 'Leaderboard')]"
            )
            if leaderboard_headers:
                logger.info("Found Leaderboard page")
                header_elements = self.driver.find_elements(
                    By.XPATH, 
                    "//th[contains(text(), 'Rank')] | //th[contains(text(), 'SMILES')] | //th[contains(text(), 'Actions')] | " +
                    "//div[contains(text(), 'Rank')] | //div[contains(text(), 'SMILES')] | //div[contains(text(), 'Actions')]"
                )
                if header_elements:
                    logger.info(f"Found table headers: {[h.text for h in header_elements]}")
                    inspect_links = self.driver.find_elements(
                        By.XPATH, 
                        "//a[contains(text(), 'Inspect')] | //button[contains(text(), 'Inspect')] | " +
                        "//div[contains(text(), 'Inspect')] | //span[contains(text(), 'Inspect')]"
                    )
                    if inspect_links:
                        logger.info(f"Found {len(inspect_links)} 'Inspect' links")
                        print(f"Processing {len(inspect_links)} submissions...")
                        for i, link in enumerate(inspect_links):
                            # If a max submission limit is set, stop after reaching it
                            if self.max_submissions and i >= self.max_submissions:
                                break
                            try:
                                print(f"Submission {i+1}/{len(inspect_links)}...")
                                # Attempt to retrieve row data (if available)
                                row = link
                                for _ in range(5):  # try to move up the DOM tree
                                    parent = row.find_element(By.XPATH, "./..")
                                    if "tr" in parent.tag_name.lower() or "row" in parent.get_attribute("class").lower():
                                        row = parent
                                        break
                                    row = parent
                                
                                row_cells = row.find_elements(By.XPATH, ".//td | .//div[contains(@class, 'cell')]")
                                rank_value = None
                                if row_cells:
                                    rank_text = row_cells[0].text.strip()
                                    if rank_text.isdigit():
                                        rank_value = int(rank_text)
                            except Exception:
                                rank_value = i + 1
                            
                            self._capture_screenshot(f"{self.output_dir}/screenshots/epoch_{epoch_number}_before_click_{i+1}.png")
                            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link)
                            time.sleep(0.2)
                            
                            original_window = self.driver.current_window_handle
                            original_windows = self.driver.window_handles
                            logger.info(f"Clicking 'Inspect' link {i+1}")
                            self.driver.execute_script("arguments[0].click();", link)
                            
                            # Wait for a new window/tab if it opens
                            try:
                                WebDriverWait(self.driver, 2).until(
                                    lambda d: len(d.window_handles) > len(original_windows)
                                )
                                for window_handle in self.driver.window_handles:
                                    if window_handle != original_window:
                                        self.driver.switch_to.window(window_handle)
                                        logger.info("Switched to new window for molecule details")
                                        break
                                self._capture_screenshot(f"{self.output_dir}/screenshots/epoch_{epoch_number}_new_window_{i+1}.png")
                            except TimeoutException:
                                logger.info("No new window detected, assuming popup on same page")
                                time.sleep(self.action_delay)
                            
                            self._capture_screenshot(f"{self.output_dir}/screenshots/epoch_{epoch_number}_after_click_{i+1}.png")
                            
                            # Use iframe support to extract the molecule data
                            molecule_data = self._extract_from_popup_with_iframe_support(epoch_number, rank_value if rank_value else i + 1)
                            
                            # If a new window was opened, close it and switch back.
                            if len(self.driver.window_handles) > len(original_windows):
                                self.driver.close()
                                self.driver.switch_to.window(original_window)
                                logger.info("Closed new window and switched back")
                            else:
                                self._close_popup()
                            
                            if molecule_data:
                                molecules.append(molecule_data)
                            
                            time.sleep(0.2)
                    else:
                        logger.warning("No 'Inspect' links found in the table")
                else:
                    logger.warning("Table headers not found in the Leaderboard")
        except Exception as e:
            logger.error(f"Error extracting molecules from epoch {epoch_number}: {e}")
        logger.info(f"Extracted {len(molecules)} molecules from epoch {epoch_number}")
        return molecules
    
    def _extract_from_popup_with_iframe_support(self, epoch_number, rank):
        """Extract data from popup, supporting iframes if present."""
        try:
            data = self._extract_from_popup(epoch_number, rank)
            if data:
                return data
            logger.info("No data found on main page, checking iframes")
            if self._check_and_switch_to_iframe():
                logger.info("Switched to iframe, attempting data extraction")
                data = self._extract_from_popup(epoch_number, rank)
                self.driver.switch_to.default_content()
                return data
            logger.warning(f"No data found for epoch {epoch_number}, rank {rank}")
            return None
        except Exception as e:
            logger.error(f"Error in iframe-supported extraction: {e}")
            try:
                self.driver.switch_to.default_content()
            except:
                pass
            return None
    
    def _check_and_switch_to_iframe(self):
        """Check available iframes and switch if one contains expected data."""
        try:
            iframes = self.driver.find_elements(By.TAG_NAME, "iframe")
            if not iframes:
                logger.info("No iframes found on page")
                return False
            logger.info(f"Found {len(iframes)} iframes, examining each")
            for i, iframe in enumerate(iframes):
                try:
                    logger.info(f"Switching to iframe {i+1}")
                    self.driver.switch_to.frame(iframe)
                    self._capture_screenshot(f"{self.output_dir}/screenshots/iframe_{i+1}.png")
                    indicators = self.driver.find_elements(By.XPATH,
                        "//div[contains(text(), 'SMILES')] | //div[contains(text(), 'Molecule')] | " +
                        "//div[contains(text(), 'Score')] | //div[contains(text(), 'Submission')]"
                    )
                    if indicators:
                        logger.info(f"Found relevant data in iframe {i+1}")
                        return True
                    self.driver.switch_to.default_content()
                except Exception as e:
                    logger.warning(f"Error examining iframe {i+1}: {e}")
                    self.driver.switch_to.default_content()
            logger.info("No relevant data found in iframes")
            return False
        except Exception as e:
            logger.error(f"Error checking iframes: {e}")
            try:
                self.driver.switch_to.default_content()
            except:
                pass
            return False
    
    def _extract_from_popup(self, epoch_number, rank):
        """Quick extraction of molecule ID and SMILES from the popup."""
        try:
            smiles_string = None
            molecule_id = None
            
            # Find the SMILES element using a targeted XPath
            smiles_elements = self.driver.find_elements(
                By.XPATH, 
                "//div[contains(text(), 'SMILES')]/following-sibling::*[1] | //div[contains(text(), 'SMILES')]/..//div[not(contains(text(), 'SMILES'))]"
            )
            if smiles_elements:
                for element in smiles_elements:
                    text = element.text.strip()
                    if text and 'C' in text and any(ch in text for ch in "= (#"):
                        smiles_string = text
                        logger.debug(f"Found SMILES: {text[:20]}...")
                        break
            
            # Find the Molecule ID element using another targeted XPath
            molecule_elements = self.driver.find_elements(
                By.XPATH, 
                "//div[contains(text(), 'Molecule')]/following-sibling::*[1] | //div[contains(text(), 'Molecule')]/..//div[not(contains(text(), 'Molecule'))]"
            )
            if molecule_elements:
                for element in molecule_elements:
                    text = element.text.strip()
                    if text and ('_' in text or text.startswith('F')):
                        molecule_id = text
                        logger.debug(f"Found Molecule ID: {text}")
                        break
            
            if smiles_string:
                # Return a dictionary for consistency with the CSV export.
                return {'moleculeId': molecule_id or f"unknown_{rank}", 'smilesString': smiles_string}
            return None
        except Exception as e:
            logger.error(f"Error extracting data: {e}")
            return None
    
    def _close_popup(self):
        """Try multiple methods to close the popup or modal."""
        try:
            close_buttons = self.driver.find_elements(
                By.XPATH, 
                "//button[contains(text(), 'Close')] | //button[@aria-label='Close'] | " +
                "//button[contains(@class, 'close')] | //div[contains(@role, 'button')][contains(text(), 'Close')] | " +
                "//span[contains(text(), 'Close')] | //button[contains(@class, 'dialog-close')]"
            )
            if close_buttons:
                logger.info(f"Found {len(close_buttons)} close buttons")
                for button in close_buttons:
                    try:
                        logger.debug("Clicking a close button")
                        self.driver.execute_script("arguments[0].click();", button)
                        time.sleep(self.action_delay)
                        if not self._is_popup_visible():
                            logger.info("Popup closed via button")
                            return True
                    except Exception as e:
                        logger.warning(f"Error clicking close button: {e}")
            x_buttons = self.driver.find_elements(
                By.XPATH,
                "//button[contains(@class, 'close')] | //button[contains(@aria-label, 'close')] | " +
                "//div[contains(@class, 'close')] | //span[contains(@class, 'close')] | " +
                "//button[contains(@class, 'modal-close')] | //button[contains(@class, 'dialog-close')]"
            )
            if x_buttons:
                logger.info(f"Found {len(x_buttons)} X buttons")
                for button in x_buttons:
                    try:
                        self.driver.execute_script("arguments[0].click();", button)
                        time.sleep(self.action_delay)
                        if not self._is_popup_visible():
                            logger.info("Popup closed via X button")
                            return True
                    except Exception as e:
                        logger.warning(f"Error clicking X button: {e}")
            logger.info("Attempting to close popup with Escape key")
            webdriver.ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(self.action_delay)
            if not self._is_popup_visible():
                logger.info("Popup closed with Escape key")
                return True
            logger.info("Clicking outside popup area")
            try:
                self.driver.execute_script("document.elementFromPoint(10,10).click();")
                time.sleep(self.action_delay)
                if not self._is_popup_visible():
                    logger.info("Popup closed by clicking corner")
                    return True
            except Exception as e:
                logger.warning(f"Error clicking outside popup: {e}")
            for _ in range(2):
                webdriver.ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
                time.sleep(0.5)
            logger.warning("Popup closure could not be verified, continuing")
            return True
        except Exception as e:
            logger.error(f"Error closing popup: {e}")
            return False
    
    def _is_popup_visible(self):
        """Check if a popup or modal is still visible."""
        try:
            popup_elements = self.driver.find_elements(
                By.XPATH, 
                "//div[@role='dialog'] | //div[contains(@class, 'modal')] | //div[contains(@class, 'popup')] | " +
                "//div[contains(@class, 'dialog')] | //div[contains(@class, 'overlay')] | " +
                "//div[@aria-modal='true']"
            )
            section_headers = self.driver.find_elements(
                By.XPATH,
                "//div[contains(text(), 'Submission Details')] | " +
                "//div[contains(text(), 'SMILES')][ancestor::div[@role='dialog' or contains(@class, 'modal')]] | " +
                "//div[contains(text(), 'Molecule')][ancestor::div[@role='dialog' or contains(@class, 'modal')]]"
            )
            return len(popup_elements) > 0 or len(section_headers) > 0
        except Exception as e:
            logger.warning(f"Error checking popup visibility: {e}")
            return False

    def _print_progress_header(self):
        """Print a header showing the scraping configuration."""
        print("\n" + "="*80)
        print(" NOVA DASHBOARD MOLECULE SCRAPER ".center(80, "="))
        print("="*80)
        print(f"URL: {self.url}")
        print(f"Epochs: {self.start_epoch} to {self.end_epoch}")
        print(f"Output Directory: {self.output_dir}")
        print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Nova Dashboard Epoch Scraper')
    
    parser.add_argument('--url', default='https://dashboard-frontend-dusky.vercel.app',
                        help='Dashboard URL')
    parser.add_argument('--output-dir', default='./nova_data',
                        help='Directory to save output data')
    parser.add_argument('--start-epoch', type=int, default=14748,
                        help='Start from this epoch number')
    parser.add_argument('--end-epoch', type=int, default=14747,
                        help='End at this epoch number')
    parser.add_argument('--navigation-delay', type=float, default=0.3,
                        help='Delay after navigation in seconds')
    parser.add_argument('--action-delay', type=float, default=0.1,
                        help='Delay between actions in seconds')
    parser.add_argument('--retries', type=int, default=3,
                        help='Number of retries for failed operations')
    parser.add_argument('--no-headless', action='store_true',
                        help='Disable headless mode (show browser window)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--max-submissions', type=int, default=0,
                        help='Maximum number of submissions to process (0 = all)')
    
    args = parser.parse_args()
    
    config = {
        'url': args.url,
        'output_dir': args.output_dir,
        'start_epoch': args.start_epoch,
        'end_epoch': args.end_epoch,
        'navigation_delay': args.navigation_delay,
        'action_delay': args.action_delay,
        'retries': args.retries,
        'headless': not args.no_headless,
        'debug': args.debug,
        'max_submissions': args.max_submissions
    }
    
    scraper = NovaDashboardScraper(config)
    scraper.run()

if __name__ == "__main__":
    main()
