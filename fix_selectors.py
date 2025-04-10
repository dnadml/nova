import re

# Read the original file
with open('nova_scraper_headless.py', 'r') as file:
    content = file.read()

# Fix the dropdown selectors - removing invalid :contains() selectors
dropdown_selectors_pattern = r"dropdown_selectors = \[(.*?)\]"
dropdown_selectors_replacement = """dropdown_selectors = [
            'select[name="epoch"]',
            'select.epoch-selector',
            'button.epoch-button',
            'div[role="combobox"]',
            '.epoch-selector',
            '.dropdown',
            'div.epoch-container select'
        ]"""

# Use re.DOTALL to make the . pattern match newlines
content = re.sub(dropdown_selectors_pattern, dropdown_selectors_replacement, content, flags=re.DOTALL)

# Also replace the XPATH approach to be more specific
xpath_pattern = r'elements = self\.driver\.find_elements\(By\.XPATH, "/\*"\)'
xpath_replacement = 'elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), \'Epoch\')]")'
content = re.sub(xpath_pattern, xpath_replacement, content)

# Fix the epoch options selectors
epoch_options_pattern = r'epoch_options = \[(.*?)\]'
epoch_options_replacement = """epoch_options = [
                f"option[value='{epoch_number}']",
                f"li[data-value='{epoch_number}']",
                f"div[role='option'][data-value='{epoch_number}']"
            ]"""
content = re.sub(epoch_options_pattern, epoch_options_replacement, content, flags=re.DOTALL)

# Add a more robust approach to finding epoch dropdown
close_popup_pattern = r"def _close_popup\(self\) -> bool:"
close_popup_replacement = """def _find_epoch_dropdown_by_text(self) -> bool:
        """
        Find the epoch dropdown by looking for text containing 'Epoch'
        
        Returns:
            True if found and clicked, False otherwise
        """
        try:
            # Look for any elements containing the text "Epoch"
            elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Epoch')]")
            logger.info(f"Found {len(elements)} elements containing 'Epoch' text")
            
            for element in elements:
                try:
                    tag_name = element.tag_name
                    element_text = element.text[:30] + "..." if len(element.text) > 30 else element.text
                    logger.info(f"Trying to click element: {tag_name}, text: {element_text}")
                    
                    # Try to click this element
                    element.click()
                    time.sleep(self.action_delay)
                    logger.info(f"Successfully clicked on element with text containing 'Epoch'")
                    return True
                except Exception as e:
                    logger.info(f"Could not click element: {e}")
                    continue
            
            return False
        except Exception as e:
            logger.error(f"Error in _find_epoch_dropdown_by_text: {e}")
            return False
        
    def _close_popup(self) -> bool:"""
content = re.sub(close_popup_pattern, close_popup_replacement, content)

# Update the dropdown finding section in _select_epoch
find_dropdown_pattern = r"# Try to find the dropdown with different CSS selectors(.*?)# If standard selectors fail, look for any element containing \"Epoch\""
find_dropdown_replacement = """# Try to find the dropdown with different CSS selectors
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
                
        # If standard selectors fail, try using our custom method
        if not dropdown_found:
            logger.warning("Could not find epoch dropdown with standard selectors, trying text search approach")
            dropdown_found = self._find_epoch_dropdown_by_text()
                
        # If standard selectors fail, look for any element containing "Epoch\""""

content = re.sub(find_dropdown_pattern, find_dropdown_replacement, content, flags=re.DOTALL)

# Write the updated content back to the file
with open('nova_scraper_headless.py', 'w') as file:
    file.write(content)

print("Fixed selectors in the script!")
