import re

# Read the original file
with open('nova_scraper_headless.py', 'r') as file:
    content = file.read()

# Add imports at the top if they don't exist
if 'from webdriver_manager.chrome import ChromeDriverManager' not in content:
    imports_to_add = 'from selenium.webdriver.chrome.service import Service\nfrom webdriver_manager.chrome import ChromeDriverManager\n'
    # Find the import section
    import_section_end = content.find('# Configure logging')
    if import_section_end == -1:
        import_section_end = content.find('logging.basicConfig')
    
    # Insert the imports
    content = content[:import_section_end] + imports_to_add + content[import_section_end:]

# Update the _setup_webdriver method
setup_webdriver_pattern = r'def _setup_webdriver\(self, headless: bool = True\) -> webdriver\.Chrome:(.*?)return driver'
setup_webdriver_replacement = '''def _setup_webdriver(self, headless: bool = True) -> webdriver.Chrome:
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
            return driver'''

# Use re.DOTALL to make the . pattern match newlines
content = re.sub(setup_webdriver_pattern, setup_webdriver_replacement, content, flags=re.DOTALL)

# Write the updated content back to the file
with open('nova_scraper_headless.py', 'w') as file:
    file.write(content)

print("Script updated successfully to use webdriver-manager!")
