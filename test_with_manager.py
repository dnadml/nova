from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

print("Setting up Chrome options...")
options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
print("Chrome options set up")

print("Initializing Chrome WebDriver with webdriver-manager...")
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)
print("Chrome WebDriver initialized successfully")

print("Navigating to the Nova Dashboard...")
driver.get("https://dashboard-frontend-dusky.vercel.app")
print("Navigation successful, page title:", driver.title)

print("Taking screenshot...")
driver.save_screenshot("test_screenshot.png")
print("Screenshot saved")

print("Quitting browser...")
driver.quit()
print("Browser quit successfully")
