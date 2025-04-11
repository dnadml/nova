#!/usr/bin/env python3
"""
Nova Dashboard Epoch Scraper

This script navigates through different epochs on the Nova Dashboard
and extracts SMILES strings by clicking the 'Inspect' buttons.
"""

import os
import time
import logging
import argparse
import json
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
        self.output_dir = Path(config.get('output_dir', './nova_data'))
        self.start_epoch = config.get('start_epoch', 14748)
        self.end_epoch = config.get('end_epoch', 14745)
        self.timeout = config.get('timeout', 30)
        self.navigation_delay = config.get('navigation_delay', 0.3)
        self.action_delay = config.get('action_delay', 0.1)
        self.retries = config.get('retries', 3)
        self.verbose = config.get('verbose', True)
        self.debug = config.get('debug', False)
        self.config = config  # Store the full config for later use
        
        # Configure debug logging if enabled
        if self.debug:
            logger.setLevel(logging.DEBUG)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        #(self.output_dir / 'screenshots').mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.output_dir / 'molecules_database.json'
#        self.db = self._load_database()
        
        # Start the browser
        self.driver = self._setup_webdriver()
        
    
    def _setup_webdriver(self):
        """Set up Chrome WebDriver"""
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
        return driver
        
    def run(self):
        """Run the scraper through all specified epochs"""
        try:
            # Navigate to the dashboard
            logger.info(f"Navigating to {self.url}")
            print(f"Navigating to {self.url}...")
            self.driver.get(self.url)
            time.sleep(self.navigation_delay)
            
            if self.verbose:
                self._print_progress_header()
            
            # Take a screenshot of the initial page
      #      self.driver.save_screenshot(f"{self.output_dir}/screenshots/initial_page.png")
            
            # Process each epoch
            for epoch_number in range(self.start_epoch, self.end_epoch - 1, -1):
                logger.info(f"Processing epoch {epoch_number}...")
                print(f"\nEpoch {epoch_number}:")
                
                # Initialize this epoch in the database if it doesn't exist
       #         if str(epoch_number) not in self.db:
       #             self.db[str(epoch_number)] = {
       #                 "epochNumber": epoch_number,
       #                 "processingDate": datetime.now().isoformat(),
       #                 "molecules": []
        #            }
                
                # Try to navigate to this epoch
                success = self._navigate_to_epoch(epoch_number)
                if not success:
                    logger.warning(f"Could not navigate to epoch {epoch_number}, skipping")
                    continue
                
                # Wait for the page to load completely
                time.sleep(self.navigation_delay)
                
                # Extract molecules from this epoch
                molecules = self._extract_molecules(epoch_number)
                self._export_to_csv(molecules)
                # Update database and save
         #       self.db[str(epoch_number)]["molecules"] = molecules
         #       self.db[str(epoch_number)]["lastUpdated"] = datetime.now().isoformat()
         #       self._save_database()
                
                # Export to CSV and SMILES file
          #      self._export_epoch(epoch_number, molecules)
                
                logger.info(f"Completed processing epoch {epoch_number}: {len(molecules)} molecules extracted")
                print(f"Completed epoch {epoch_number}: {len(molecules)} molecules extracted")
            
            logger.info("Scraping completed successfully")
        except Exception as e:
            logger.error(f"Error during scraping: {e}", exc_info=True)
        finally:
            self.driver.quit()
            self._save_database()
    
    
    def _scroll_dropdown_to_find_epoch(self, epoch_number):
        """
        Specialized method to handle long dropdowns where scrolling is needed
        to find the specific epoch.
        
        Returns:
            bool: True if successfully selected the epoch, False otherwise
        """
        try:
            # First, find and click the dropdown to open it
            dropdown_selectors = self.driver.find_elements(By.XPATH, 
                "//button[contains(text(), 'Epoch')] | "
                "//div[contains(@role, 'button') and contains(text(), 'Epoch')] | "
                "//div[contains(@class, 'dropdown') and contains(text(), 'Epoch')]"
            )
            
            if not dropdown_selectors:
                logger.warning("No epoch dropdown selectors found")
                return False
            
            # Click to open the dropdown
            logger.info(f"Clicking dropdown: {dropdown_selectors[0].text}")
            self.driver.execute_script("arguments[0].click();", dropdown_selectors[0])
            time.sleep(1.5)  # Wait for dropdown to open
            
            # Take a screenshot of the opened dropdown
            self.driver.save_screenshot(f"{self.output_dir}/screenshots/dropdown_open_{epoch_number}.png")
            
            # Find the dropdown menu/list container
            dropdown_containers = self.driver.find_elements(By.XPATH,
                "//div[contains(@class, 'dropdown-menu')] | "
                "//div[contains(@class, 'menu')] | "
                "//div[contains(@class, 'list')] | "
                "//ul[contains(@class, 'dropdown-menu')] | "
                "//ul[contains(@class, 'menu')]"
            )
            
            # If no container found, try finding any newly appeared elements after clicking
            if not dropdown_containers:
                # Get all elements that might be part of the dropdown
                all_elements = self.driver.find_elements(By.XPATH, "//div | //ul | //ol")
                # Filter to those that contain "Epoch" text anywhere in them
                dropdown_containers = [elem for elem in all_elements 
                                      if "Epoch" in elem.text and len(elem.text.split('\n')) > 3]
            
            if not dropdown_containers:
                logger.warning("Could not find dropdown container")
                return False
            
            # Use the first dropdown container
            dropdown_container = dropdown_containers[0]
            logger.info(f"Found dropdown container with {len(dropdown_container.text.split())} items")
            
            # Check if our target epoch is already visible in the container
            visible_text = dropdown_container.text
            if f"Epoch {epoch_number}" in visible_text or f"epoch {epoch_number}" in visible_text.lower():
                logger.info(f"Target epoch {epoch_number} is visible in dropdown")
                
                # Find the specific option for our epoch
                epoch_options = dropdown_container.find_elements(By.XPATH,
                    f".//div[contains(text(), 'Epoch {epoch_number}')] | "
                    f".//div[text()='{epoch_number}'] | "
                    f".//li[contains(text(), 'Epoch {epoch_number}')] | "
                    f".//li[text()='{epoch_number}']"
                )
                
                if epoch_options:
                    logger.info(f"Found epoch option: {epoch_options[0].text}")
                    self.driver.execute_script("arguments[0].click();", epoch_options[0])
                    time.sleep(1.5)
                    return self._verify_epoch_navigation(epoch_number)
            
            # Target epoch not visible - need to scroll to find it
            logger.info(f"Target epoch {epoch_number} not immediately visible, scrolling to find it")
            
            # First, determine which direction to scroll based on current epoch vs target
            current_epoch_text = dropdown_selectors[0].text
            current_epoch_match = re.search(r'Epoch\s+(\d+)', current_epoch_text)
            
            if current_epoch_match:
                current_epoch = int(current_epoch_match.group(1))
                logger.info(f"Current epoch is {current_epoch}, target is {epoch_number}")
                
                # Determine scroll direction (up or down)
                if epoch_number < current_epoch:
                    logger.info(f"Need to scroll DOWN to find older epoch {epoch_number}")
                    scroll_direction = "down"
                else:
                    logger.info(f"Need to scroll UP to find newer epoch {epoch_number}")
                    scroll_direction = "up"
            else:
                # If we can't determine the current epoch, default to scrolling down
                logger.info(f"Could not determine current epoch, defaulting to scroll down")
                scroll_direction = "down"
            
            # Scroll progressively in the dropdown to find our epoch
            max_scroll_attempts = 20
            for attempt in range(max_scroll_attempts):
                logger.info(f"Scroll attempt {attempt+1}/{max_scroll_attempts}")
                
                # Scroll in the appropriate direction
                if scroll_direction == "down":
                    self.driver.execute_script(
                        "arguments[0].scrollTop = arguments[0].scrollTop + 200;", 
                        dropdown_container
                    )
                else:
                    self.driver.execute_script(
                        "arguments[0].scrollTop = arguments[0].scrollTop - 200;", 
                        dropdown_container
                    )
                
                time.sleep(0.5)  # Wait for scroll to complete
                
                # Take a screenshot after each scroll to see what's visible
                if attempt % 5 == 0:  # Every 5 attempts to avoid too many screenshots
                    self.driver.save_screenshot(
                        f"{self.output_dir}/screenshots/dropdown_scroll_{epoch_number}_attempt_{attempt+1}.png"
                    )
                
                # Check if our target epoch is now visible
                updated_visible_text = dropdown_container.text
                if f"Epoch {epoch_number}" in updated_visible_text or f"epoch {epoch_number}" in updated_visible_text.lower():
                    logger.info(f"Found target epoch {epoch_number} after scrolling!")
                    
                    # Find the specific option for our epoch
                    epoch_options = dropdown_container.find_elements(By.XPATH,
                        f".//div[contains(text(), 'Epoch {epoch_number}')] | "
                        f".//div[text()='{epoch_number}'] | "
                        f".//li[contains(text(), 'Epoch {epoch_number}')] | "
                        f".//li[text()='{epoch_number}']"
                    )
                    
                    if epoch_options:
                        logger.info(f"Found epoch option: {epoch_options[0].text}")
                        self.driver.execute_script("arguments[0].click();", epoch_options[0])
                        time.sleep(1.5)
                        return self._verify_epoch_navigation(epoch_number)
                    else:
                        logger.warning(f"Text indicates epoch {epoch_number} is visible, but couldn't find clickable element")
                
                # Try an alternative approach - look for any elements within the container that have our epoch
                all_items = dropdown_container.find_elements(By.XPATH, ".//*")
                for item in all_items:
                    try:
                        item_text = item.text.strip()
                        if f"Epoch {epoch_number}" in item_text or item_text == str(epoch_number):
                            logger.info(f"Found item with text: {item_text}")
                            self.driver.execute_script("arguments[0].click();", item)
                            time.sleep(1.5)
                            if self._verify_epoch_navigation(epoch_number):
                                return True
                    except:
                        continue
            
            logger.warning(f"Could not find epoch {epoch_number} after {max_scroll_attempts} scroll attempts")
            
            # Close the dropdown by clicking elsewhere
            try:
                body = self.driver.find_element(By.TAG_NAME, "body")
                self.driver.execute_script("arguments[0].click();", body)
            except:
                pass
            
            return False
            
        except Exception as e:
            logger.error(f"Error in dropdown scrolling: {e}")
            
            # Try to close the dropdown to clean up
            try:
                body = self.driver.find_element(By.TAG_NAME, "body")
                self.driver.execute_script("arguments[0].click();", body)
            except:
                pass
            
            return False
    def _navigate_to_epoch(self, epoch_number):
        """
        Navigate to a specific epoch using a more robust approach with dropdown scrolling
        
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        # Backup the current page in case we need to restore
        current_url = self.driver.current_url
        
        logger.info(f"Attempting to navigate to epoch {epoch_number}")
        
        for attempt in range(self.retries):
            try:
                print(f"Attempt {attempt+1}/{self.retries} to navigate to epoch {epoch_number}")
                
                # First, check if we're already on the right epoch
                if self._verify_epoch_navigation(epoch_number):
                    logger.info(f"Already on epoch {epoch_number}")
                    return True
                
                # STRATEGY 1: Use the specialized dropdown scrolling method
                logger.info("Trying dropdown scrolling method...")
                if self._scroll_dropdown_to_find_epoch(epoch_number):
                    return True
                
                # STRATEGY 2: Look for epoch numbers in the sidebar
                logger.info("Trying sidebar navigation method...")
                try:
                    sidebar_epochs = self.driver.find_elements(By.XPATH, 
                        f"//div[contains(@class, 'sidebar') or contains(@class, 'nav')]//div[contains(text(), '{epoch_number}')]")
                    
                    if sidebar_epochs:
                        logger.info(f"Found {len(sidebar_epochs)} sidebar elements for epoch {epoch_number}")
                        
                        for sidebar_epoch in sidebar_epochs:
                            try:
                                # Scroll to element
                                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", sidebar_epoch)
                                time.sleep(0.5)
                                
                                # Click the element
                                logger.info(f"Clicking sidebar epoch: {sidebar_epoch.text}")
                                self.driver.execute_script("arguments[0].click();", sidebar_epoch)
                                time.sleep(2)
                                
                                # Verify navigation
                                if self._verify_epoch_navigation(epoch_number):
                                    return True
                            except Exception as e:
                                logger.warning(f"Error clicking sidebar epoch: {e}")
                except Exception as e:
                    logger.warning(f"Error finding sidebar epochs: {e}")
                
                # STRATEGY 3: Direct URL manipulation if possible
                logger.info("Trying URL manipulation method...")
                try:
                    # Check if the current URL has a pattern we can modify
                    current_url = self.driver.current_url
                    if 'epoch=' in current_url or '/epoch/' in current_url:
                        # Try to construct a URL with the target epoch
                        if 'epoch=' in current_url:
                            new_url = re.sub(r'epoch=\d+', f'epoch={epoch_number}', current_url)
                        else:
                            new_url = re.sub(r'/epoch/\d+', f'/epoch/{epoch_number}', current_url)
                        
                        logger.info(f"Attempting direct URL navigation to: {new_url}")
                        self.driver.get(new_url)
                        time.sleep(2)
                        
                        # Verify navigation
                        if self._verify_epoch_navigation(epoch_number):
                            return True
                except Exception as e:
                    logger.warning(f"Error with direct URL navigation: {e}")
                
                # If all strategies failed on this attempt, refresh the page before trying again
                if attempt < self.retries - 1:
                    logger.warning(f"Navigation attempt {attempt+1} failed, refreshing page")
                    self.driver.refresh()
                    time.sleep(3)  # Wait longer for refresh
            
            except Exception as e:
                logger.error(f"Error during navigation attempt {attempt+1}: {e}")
                
                # Try to restore the page before next attempt
                if attempt < self.retries - 1:
                    try:
                        self.driver.get(current_url)
                        time.sleep(2)
                    except:
                        pass
        
        logger.error(f"Failed to navigate to epoch {epoch_number} after {self.retries} attempts")
        return False
    def _verify_epoch_navigation(self, epoch_number):
        """Verify that we've successfully navigated to the specified epoch"""
        try:
            # Check for epoch indicators in multiple ways
            
            # 1. Look for elements that specifically show the current epoch
            epoch_indicators = self.driver.find_elements(By.XPATH, 
                f"//div[contains(@class, 'title') or contains(@class, 'header')]//span[contains(text(), 'Epoch {epoch_number}')] | "
                f"//h1[contains(text(), 'Epoch {epoch_number}')] | "
                f"//h2[contains(text(), 'Epoch {epoch_number}')] | "
                f"//h3[contains(text(), 'Epoch {epoch_number}')] | "
                f"//div[contains(text(), 'Epoch') and contains(text(), '{epoch_number}')]"
            )
            
            if epoch_indicators:
                logger.info(f"Found epoch indicator: {epoch_indicators[0].text}")
                return True
            
            # 2. Look for the epoch dropdown to see if it shows the current epoch
            dropdown_elements = self.driver.find_elements(By.XPATH,
                "//button[contains(text(), 'Epoch')] | "
                "//div[contains(@role, 'button') and contains(text(), 'Epoch')]"
            )
            
            for dropdown in dropdown_elements:
                if f"Epoch {epoch_number}" in dropdown.text or f"epoch {epoch_number}" in dropdown.text.lower():
                    logger.info(f"Dropdown shows correct epoch: {dropdown.text}")
                    return True
            
            # 3. Look for the leaderboard title with epoch number
            leaderboard_headers = self.driver.find_elements(By.XPATH,
                f"//div[contains(text(), 'Leaderboard') and contains(text(), '{epoch_number}')] | "
                f"//h2[contains(text(), 'Leaderboard') and contains(text(), '{epoch_number}')] | "
                f"//h3[contains(text(), 'Leaderboard') and contains(text(), '{epoch_number}')]"
            )
            
            if leaderboard_headers:
                logger.info(f"Found leaderboard header for epoch {epoch_number}: {leaderboard_headers[0].text}")
                return True
            
            # 4. Check for molecules or submissions sections that mention the epoch
            submission_sections = self.driver.find_elements(By.XPATH,
                f"//div[contains(text(), 'Submissions') and contains(text(), '{epoch_number}')] | "
                f"//div[contains(text(), 'Molecules') and contains(text(), '{epoch_number}')]"
            )
            
            if submission_sections:
                logger.info(f"Found submission section for epoch {epoch_number}: {submission_sections[0].text}")
                return True
            
            # 5. Check URL if it contains the epoch number
            if f"epoch={epoch_number}" in self.driver.current_url or f"/epoch/{epoch_number}" in self.driver.current_url:
                logger.info(f"URL contains epoch {epoch_number}: {self.driver.current_url}")
                return True
            
            # If all checks fail, log and return False
            logger.warning(f"Could not verify navigation to epoch {epoch_number}")
            return False
        
        except Exception as e:
            logger.error(f"Error verifying epoch navigation: {e}")
            return False
    
    def _extract_molecules(self, epoch_number):
        """Extract molecule data from the current epoch page"""
        molecules = []
        
        try:
            # Take a screenshot of the current epoch page
            self.driver.save_screenshot(f"{self.output_dir}/screenshots/epoch_{epoch_number}_page.png")
            
            # Wait for the page to load completely
            time.sleep(self.navigation_delay)
            
            # From the screenshots, we can see there's a table with Inspect links in the rightmost column
            # First check if we're looking at the leaderboard page with the table
            leaderboard_headers = self.driver.find_elements(
                By.XPATH, 
                "//div[contains(text(), 'Leaderboard')] | //h2[contains(text(), 'Leaderboard')] | //h3[contains(text(), 'Leaderboard')]"
            )
            
            if leaderboard_headers:
                logger.info("Found Leaderboard page")
                
                # Look for the table with columns including "Actions" (where Inspect links are)
                try:
                    # First find the headers of the table to confirm we have the right structure
                    header_elements = self.driver.find_elements(
                        By.XPATH, 
                        "//th[contains(text(), 'Rank')] | //th[contains(text(), 'SMILES')] | //th[contains(text(), 'Actions')] | " +
                        "//div[contains(text(), 'Rank')] | //div[contains(text(), 'SMILES')] | //div[contains(text(), 'Actions')]"
                    )
                    
                    if header_elements:
                        logger.info(f"Found table headers: {[h.text for h in header_elements]}")
                        
                        # Look specifically for the "Inspect" links/buttons which are blue and in the Actions column
                        inspect_links = self.driver.find_elements(
                            By.XPATH, 
                            "//a[contains(text(), 'Inspect')] | //button[contains(text(), 'Inspect')] | " +
                            "//div[contains(text(), 'Inspect')] | //span[contains(text(), 'Inspect')]"
                        )
                        
                        if inspect_links:
                            logger.info(f"Found {len(inspect_links)} 'Inspect' links")
                            print(f"Processing {len(inspect_links)} submissions...")
                            
                            # Process each Inspect link
                            for i, link in enumerate(inspect_links):
                                try:
                                    print(f"Submission {i+1}/{len(inspect_links)}...")
                                    
                                    # Get the row this link belongs to for rank information
                                    try:
                                        # Find the parent row that contains this link
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
                                        
                                        # Extract row data like rank, SMILES from the cells in the same row
                                        row_cells = row.find_elements(By.XPATH, ".//td | .//div[contains(@class, 'cell')]")
                                        
                                        # Try to extract rank from the first cell
                                        rank_value = None
                                        if row_cells and len(row_cells) > 0:
                                            try:
                                                rank_text = row_cells[0].text.strip()
                                                if rank_text.isdigit():
                                                    rank_value = int(rank_text)
                                            except:
                                                pass
                                    except:
                                        rank_value = i + 1  # Fallback to index if we can't extract rank
                                    
                                    # Take a screenshot before clicking
                                    self.driver.save_screenshot(f"{self.output_dir}/screenshots/epoch_{epoch_number}_before_click_{i+1}.png")
                                    
                                    # Scroll to make the link visible
                                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link)
                                    time.sleep(0.5)
                                    
                                    # Store current window handles
                                    original_window = self.driver.current_window_handle
                                    original_windows = self.driver.window_handles
                                    
                                    # Click the inspect link using JavaScript to avoid intercept issues
                                    logger.info(f"Clicking 'Inspect' link {i+1}")
                                    self.driver.execute_script("arguments[0].click();", link)
                                    
                                    # Wait for the popup or new window/tab
                                    try:
                                        # Check if a new window/tab was opened
                                        WebDriverWait(self.driver, 5).until(
                                            lambda d: len(d.window_handles) > len(original_windows)
                                        )
                                        
                                        # Switch to the new window
                                        for window_handle in self.driver.window_handles:
                                            if window_handle != original_window:
                                                self.driver.switch_to.window(window_handle)
                                                logger.info("Switched to new window for molecule details")
                                                break
                                        
                                        # Take a screenshot of the new window
                                        self.driver.save_screenshot(f"{self.output_dir}/screenshots/epoch_{epoch_number}_new_window_{i+1}.png")
                                    except TimeoutException:
                                        # No new window opened, so it's probably a popup on the same page
                                        logger.info("No new window detected, looking for popup on same page")
                                        
                                        # Wait for any popup to be visible
                                        time.sleep(self.action_delay)
                                    
                                    # Take another screenshot after clicking/waiting to see what appeared
                                    self.driver.save_screenshot(f"{self.output_dir}/screenshots/epoch_{epoch_number}_after_click_{i+1}.png")
                                    
                                    # Extract data from the popup or new window
                                    rank_to_use = rank_value if rank_value is not None else i + 1
                                    molecule_data = self._extract_from_popup_with_iframe_support(epoch_number, rank_to_use)
                                    
                                    # If we switched to a new window, close it and switch back
                                    if len(self.driver.window_handles) > len(original_windows):
                                        self.driver.close()  # Close current window
                                        self.driver.switch_to.window(original_window)  # Switch back to original
                                        logger.info("Closed new window and switched back to main window")
                                    else:
                                        # Close the popup if it's on the same page
                                        self._close_popup()
                                    
                                    if molecule_data:
                                        molecules.append(molecule_data)
                                        
                                    time.sleep(0.5)  # Short delay after closing
                                    
                                except Exception as e:
                                    logger.error(f"Error processing inspect link {i+1}: {e}")
                                    # Try to restore the window state
                                    try:
                                        if len(self.driver.window_handles) > len(original_windows):
                                            self.driver.close()
                                            self.driver.switch_to.window(original_window)
                                        else:
                                            self._close_popup()
                                    except:
                                        pass
                                    
                                    # Break after first few for testing
                                    if i >= 4 and len(molecules) == 0:
                                        logger.warning("Breaking after 5 attempts with no successful extractions")
                                        break
                        else:
                            logger.warning("No 'Inspect' links found in the table")
                    else:
                        logger.warning("Could not find table headers in the Leaderboard")
                except Exception as e:
                    logger.error(f"Error processing table structure: {e}")
            
            logger.info(f"Extracted {len(molecules)} molecules from epoch {epoch_number}")
            
        except Exception as e:
            logger.error(f"Error extracting molecules from epoch {epoch_number}: {e}")
        
        return molecules
    
    def _extract_from_popup_with_iframe_support(self, epoch_number, rank):
        """Extract data from popup with iframe support"""
        try:
            # Try to extract from the main page first
            data = self._extract_from_popup(epoch_number, rank)
            if data:
                return data
            
            # If no data found, check for iframes
            logger.info("No data found in main page, checking iframes")
            if self._check_and_switch_to_iframe():
                # We're now inside an iframe that might have our data
                logger.info("Switched to iframe, extracting data")
                data = self._extract_from_popup(epoch_number, rank)
                
                # Switch back to main content
                self.driver.switch_to.default_content()
                
                return data
            
            # If we got here, no data was found in main page or iframes
            logger.warning(f"No data found for epoch {epoch_number}, rank {rank} in main page or iframes")
            return None
            
        except Exception as e:
            logger.error(f"Error in iframe-supported extraction: {e}")
            # Make sure we're back to the main content
            try:
                self.driver.switch_to.default_content()
            except:
                pass
            return None
    
    def _check_and_switch_to_iframe(self):
        """Check for iframes and switch to them if needed"""
        try:
            # Store the main window handle
            main_context = self.driver.current_window_handle
            
            # Find all iframes
            iframes = self.driver.find_elements(By.TAG_NAME, "iframe")
            if not iframes:
                logger.info("No iframes found on page")
                return False
                
            logger.info(f"Found {len(iframes)} iframes, checking each one")
            
            # Try each iframe
            for i, iframe in enumerate(iframes):
                try:
                    # Switch to this iframe
                    logger.info(f"Switching to iframe {i+1}/{len(iframes)}")
                    self.driver.switch_to.frame(iframe)
                    
                    # Take a screenshot to see what's in the iframe
                    iframe_screenshot_path = f"{self.output_dir}/screenshots/iframe_{i+1}.png"
                    self.driver.save_screenshot(iframe_screenshot_path)
                    logger.info(f"Saved iframe screenshot to {iframe_screenshot_path}")
                    
                    # Look for indicators that this iframe contains the data we want
                    indicators = self.driver.find_elements(By.XPATH,
                        "//div[contains(text(), 'SMILES')] | //div[contains(text(), 'Molecule')] | " +
                        "//div[contains(text(), 'Score')] | //div[contains(text(), 'Submission')]")
                    
                    if indicators:
                        logger.info(f"Found {len(indicators)} data indicators in iframe {i+1}")
                        return True  # Stay in this iframe
                    
                    # Switch back to the main context to try the next iframe
                    self.driver.switch_to.default_content()
                except Exception as e:
                    logger.warning(f"Error examining iframe {i+1}: {e}")
                    # Make sure we're back to the main content
                    self.driver.switch_to.default_content()
            
            logger.info("No relevant data found in any iframe")
            return False
            
        except Exception as e:
            logger.error(f"Error checking iframes: {e}")
            # Make sure we're back to the main content
            try:
                self.driver.switch_to.default_content()
            except:
                pass
            return False
    def _extract_from_popup(self, epoch_number, rank):
        """Extract molecule data from the popup or details page"""
        try:
            # Take a screenshot of the current state
            self.driver.save_screenshot(f"{self.output_dir}/screenshots/epoch_{epoch_number}_popup_{rank}.png")
            
            # Look for different types of modal/popup containers
            # Based on the screenshot, it looks like the popup might be using a different structure
            popup_elements = self.driver.find_elements(By.XPATH, 
                "//div[@role='dialog'] | //div[contains(@class, 'modal')] | //div[contains(@class, 'popup')] | " +
                "//div[contains(@class, 'dialog')] | //div[contains(@class, 'overlay')] | " +
                "//div[contains(@aria-label, 'details')] | //div[contains(@aria-label, 'submission')]")
            
            if popup_elements:
                logger.info(f"Found {len(popup_elements)} popup/dialog elements")
                
                # Use the first popup container as our search context
                container = popup_elements[0]
            else:
                # If we don't find a specific popup container, look for key sections that would be 
                # in the submission details (based on screenshot: Molecule ID, SMILES, Score)
                section_headers = self.driver.find_elements(By.XPATH,
                    "//div[contains(text(), 'Submission Details')] | //div[contains(text(), 'Molecule')] | " +
                    "//div[contains(text(), 'SMILES')] | //div[contains(text(), 'Score')]")
                
                if section_headers:
                    logger.info(f"Found {len(section_headers)} section headers in details view")
                    # Use the entire page as our container
                    container = self.driver.find_element(By.TAG_NAME, "body")
                else:
                    logger.warning("No popup elements or section headers found, using body as fallback")
                    container = self.driver.find_element(By.TAG_NAME, "body")
            
            # Write the HTML source to a file for debugging
            with open(f"{self.output_dir}/screenshots/epoch_{epoch_number}_popup_{rank}_source.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            
            # Extract SMILES string - with more specific targeting based on screenshot
            smiles_string = None
            
            # Look for elements with the "SMILES" label
            smiles_sections = container.find_elements(By.XPATH, 
                ".//div[contains(text(), 'SMILES')] | .//span[contains(text(), 'SMILES')]")
            
            if smiles_sections:
                logger.info(f"Found {len(smiles_sections)} SMILES sections")
                
                # For each SMILES section, look for the actual SMILES string
                for section in smiles_sections:
                    try:
                        # Get the parent container that holds both the label and value
                        parent = section
                        for _ in range(3):  # Go up max 3 levels
                            try:
                                parent = parent.find_element(By.XPATH, "./..")
                            except:
                                break
                        
                        # Look for elements that might contain the SMILES text
                        # (in the screenshot, it's in a separate element from the label)
                        candidates = parent.find_elements(By.XPATH, 
                            ".//div | .//span | .//pre | .//code | .//p")
                        
                        for elem in candidates:
                            # Skip the label element itself
                            if "SMILES" in elem.text:
                                continue
                                
                            text = elem.text.strip()
                            # Check for valid SMILES patterns - very specific based on the screenshot
                            # SMILES strings have specific chemical notation patterns
                            if text and len(text) > 8:
                                # More stringent check for SMILES-like strings
                                if (('C' in text) and 
                                    (any(pattern in text for pattern in ['=C', '#C', 'CC', 'CN', 'CO', 'C(', 'C[', 'C1', 'C2']))):
                                    smiles_string = text
                                    logger.info(f"Found SMILES string: {text[:30]}...")
                                    break
                        
                        if smiles_string:
                            break
                    except Exception as e:
                        logger.warning(f"Error examining SMILES section: {e}")
            
            # If we still don't have a SMILES string, try a different approach
            if not smiles_string:
                # Look for text that might be a SMILES string by pattern anywhere in the container
                try:
                    # Get all leaf text nodes that might be SMILES strings
                    all_elements = container.find_elements(By.XPATH, 
                        ".//div | .//span | .//pre | .//code | .//p")
                    
                    for elem in all_elements:
                        text = elem.text.strip()
                        
                        # Skip very short texts or very long texts (page content)
                        if not text or len(text) < 10 or len(text) > 300:
                            continue
                        
                        # More stringent check for SMILES-like strings
                        if (('C' in text) and 
                            (any(pattern in text for pattern in ['=C', '#C', 'CC', 'CN', 'CO', 'C(', 'C[', 'C1', 'C2']))):
                            # Additional check: SMILES shouldn't contain spaces or most punctuation
                            if not any(c in text for c in [' ', ',', ';', ':', '"', "'"]):
                                smiles_string = text
                                logger.info(f"Found SMILES-like pattern: {text[:30]}...")
                                break
                except Exception as e:
                    logger.warning(f"Error in pattern-based SMILES search: {e}")
            
            # Extract Molecule ID - based on screenshot
            molecule_id = None
            
            # Look for elements with "Molecule" label
            molecule_sections = container.find_elements(By.XPATH, 
                ".//div[contains(text(), 'Molecule')] | .//span[contains(text(), 'Molecule')]")
            
            if molecule_sections:
                logger.info(f"Found {len(molecule_sections)} Molecule sections")
                
                for section in molecule_sections:
                    try:
                        # Get parent container
                        parent = section
                        for _ in range(3):  # Go up max 3 levels
                            try:
                                parent = parent.find_element(By.XPATH, "./..")
                            except:
                                break
                        
                        # Look for elements that might have the ID
                        candidates = parent.find_elements(By.XPATH, 
                            ".//div | .//span | .//pre | .//code | .//p")
                        
                        for elem in candidates:
                            # Skip the label element itself
                            if "Molecule" in elem.text:
                                continue
                                
                            text = elem.text.strip()
                            # Molecule IDs typically are alphanumeric with underscores
                            if text and (('_' in text) or text.startswith('F') or text.isalnum()):
                                molecule_id = text
                                logger.info(f"Found Molecule ID: {text}")
                                break
                        
                        if molecule_id:
                            break
                    except Exception as e:
                        logger.warning(f"Error examining Molecule section: {e}")
            
            # Extract Score value
            score = None
            
            # Look for "Score" or "Final Score" labels
            score_sections = container.find_elements(By.XPATH, 
                ".//div[contains(text(), 'Score')] | .//span[contains(text(), 'Score')] | " +
                ".//div[contains(text(), 'Final Score')] | .//span[contains(text(), 'Final Score')]")
            
            if score_sections:
                logger.info(f"Found {len(score_sections)} Score sections")
                
                for section in score_sections:
                    try:
                        # First check if the score is in the same element as the label
                        section_text = section.text.strip()
                        
                        # Look for decimal numbers in the text
                        import re
                        score_matches = re.findall(r"0\.\d+", section_text)
                        if score_matches:
                            score = float(score_matches[0])
                            logger.info(f"Found score in section text: {score}")
                            break
                        
                        # If not found, check nearby elements
                        parent = section
                        for _ in range(3):  # Go up max 3 levels
                            try:
                                parent = parent.find_element(By.XPATH, "./..")
                            except:
                                break
                        
                        # Look for elements that might have the score
                        candidates = parent.find_elements(By.XPATH, 
                            ".//div | .//span | .//pre | .//code | .//p")
                        
                        for elem in candidates:
                            # Skip the label element itself
                            if "Score" in elem.text:
                                continue
                                
                            text = elem.text.strip()
                            # Try to extract score number (typically decimal like 0.1234)
                            try:
                                # If the text is just a number
                                if text and text.replace('.', '', 1).isdigit():
                                    score = float(text)
                                    logger.info(f"Found score: {score}")
                                    break
                                    
                                # Otherwise look for decimal pattern
                                score_matches = re.findall(r"0\.\d+", text)
                                if score_matches:
                                    score = float(score_matches[0])
                                    logger.info(f"Found score in text: {score}")
                                    break
                            except:
                                pass
                        
                        if score is not None:
                            break
                    except Exception as e:
                        logger.warning(f"Error examining Score section: {e}")
            
            # Create molecule data dictionary
            if smiles_string:
                return {
                    "rank": rank,
                    "epochNumber": epoch_number,
                    "moleculeId": molecule_id,
                    "smilesString": smiles_string,
                    "score": score,
                    "extractionTimestamp": datetime.now().isoformat()
                }
            else:
                logger.warning(f"No SMILES string found in popup/details view for epoch {epoch_number}, rank {rank}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting data from popup: {e}")
            return None
    def _close_popup(self):
        """Close the popup by various methods, with multiple fallbacks"""
        try:
            # First look for a Close button (visible in the screenshot)
            close_buttons = self.driver.find_elements(By.XPATH, 
                "//button[contains(text(), 'Close')] | //button[@aria-label='Close'] | " +
                "//button[contains(@class, 'close')] | //div[contains(@role, 'button')][contains(text(), 'Close')] | " +
                "//span[contains(text(), 'Close')] | //button[contains(@class, 'dialog-close')]")
            
            if close_buttons:
                logger.info(f"Found {len(close_buttons)} close buttons")
                for button in close_buttons:
                    try:
                        logger.info(f"Clicking close button: {button.get_attribute('outerHTML')[:100]}")
                        # Use JavaScript to click to avoid potential intercept issues
                        self.driver.execute_script("arguments[0].click();", button)
                        time.sleep(self.action_delay)
                        
                        # Check if popup closed successfully
                        if not self._is_popup_visible():
                            logger.info("Popup closed successfully via button")
                            return True
                    except Exception as e:
                        logger.warning(f"Error clicking close button: {e}")
            
            # Try clicking the X button which might not have "Close" text
            x_buttons = self.driver.find_elements(By.XPATH,
                "//button[contains(@class, 'close')] | //button[contains(@aria-label, 'close')] | " +
                "//div[contains(@class, 'close')] | //span[contains(@class, 'close')] | " +
                "//button[contains(@class, 'modal-close')] | //button[contains(@class, 'dialog-close')]")
            
            if x_buttons:
                logger.info(f"Found {len(x_buttons)} X buttons")
                for button in x_buttons:
                    try:
                        logger.info("Clicking X button")
                        self.driver.execute_script("arguments[0].click();", button)
                        time.sleep(self.action_delay)
                        
                        # Check if popup closed successfully
                        if not self._is_popup_visible():
                            logger.info("Popup closed successfully via X button")
                            return True
                    except Exception as e:
                        logger.warning(f"Error clicking X button: {e}")
            
            # If no close button was found or clicking it failed, try pressing Escape
            logger.info("Trying to close popup with Escape key")
            webdriver.ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(self.action_delay)
            
            # Check if popup is still visible
            if not self._is_popup_visible():
                logger.info("Popup closed successfully with Escape key")
                return True
            
            # Try clicking away from the modal (often closes modals)
            logger.info("Trying to close popup by clicking outside")
            try:
                # First try clicking at the very corner of the page
                self.driver.execute_script("document.elementFromPoint(10, 10).click();")
                time.sleep(self.action_delay)
                
                if not self._is_popup_visible():
                    logger.info("Popup closed successfully by clicking corner")
                    return True
                
                # Try clicking somewhere else (e.g., near the top of the page but not too close to the corner)
                self.driver.execute_script("document.elementFromPoint(document.body.clientWidth / 2, 10).click();")
                time.sleep(self.action_delay)
                
                if not self._is_popup_visible():
                    logger.info("Popup closed successfully by clicking top center")
                    return True
            except Exception as e:
                logger.warning(f"Error clicking outside popup: {e}")
            
            # Last resort: try sending Escape multiple times
            for _ in range(3):
                webdriver.ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
                time.sleep(0.5)
            
            logger.warning("Could not verify popup closure, continuing anyway")
            return True
        except Exception as e:
            logger.error(f"Error closing popup: {e}")
            return False
    # 1. In the _extract_from_popup method, add this simple CSV export
    def _export_to_csv(self, molecules, file_path="molecules.csv"):
        """Export molecules to a simple CSV file"""
    # Create file with headers if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['molecule', 'smiles'])
    
    # Append data
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for molecule in molecules:
                if molecule.get('moleculeId') and molecule.get('smilesString'):
                    writer.writerow([molecule.get('moleculeId'), molecule.get('smilesString')])

    def _is_popup_visible(self):
        """Check if a popup/modal is still visible on the page"""
        try:
            # Look for any popup/modal elements
            popup_elements = self.driver.find_elements(By.XPATH, 
                "//div[@role='dialog'] | //div[contains(@class, 'modal')] | //div[contains(@class, 'popup')] | " +
                "//div[contains(@class, 'dialog')] | //div[contains(@class, 'overlay')] | " +
                "//div[@aria-modal='true']")
            
            # Also check for specific section headers that would be in the popup
            section_headers = self.driver.find_elements(By.XPATH,
                "//div[contains(text(), 'Submission Details')] | " +
                "//div[contains(text(), 'SMILES')][ancestor::div[@role='dialog' or contains(@class, 'modal')]] | " +
                "//div[contains(text(), 'Molecule')][ancestor::div[@role='dialog' or contains(@class, 'modal')]]")
            
            return len(popup_elements) > 0 or len(section_headers) > 0
        except Exception as e:
            logger.warning(f"Error checking if popup is visible: {e}")
            return False
    
    def _debug_html_element(self, element, description="Element"):
        """Print debugging information about an HTML element"""
        try:
            element_html = element.get_attribute('outerHTML')
            element_text = element.text
            element_tag = element.tag_name
            element_class = element.get_attribute('class') or ""
            element_id = element.get_attribute('id') or ""
            
            debug_info = f"{description}:\n"
            debug_info += f"  Tag: {element_tag}\n"
            debug_info += f"  Class: {element_class}\n"
            debug_info += f"  ID: {element_id}\n"
            debug_info += f"  Text: {element_text[:100]}\n"
            debug_info += f"  HTML: {element_html[:200]}\n"
            
            logger.debug(debug_info)
            return debug_info
        except Exception as e:
            logger.warning(f"Error debugging element: {e}")
            return "Error debugging element"
    
    
    def _print_progress_header(self):
        """Print a header with configuration info"""
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
    parser.add_argument('--navigation-delay', type=float, default=3.0,
                      help='Delay after navigation in seconds')
    parser.add_argument('--action-delay', type=float, default=1.5,
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
