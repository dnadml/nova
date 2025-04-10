#!/bin/bash

# Nova Dashboard Scraper Setup for RunPod
# This script will install all required dependencies for running in a headless environment

echo "Setting up Nova Dashboard Molecule Scraper for RunPod..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Installing..."
    apt-get update
    apt-get install -y python3 python3-pip
fi

# Verify Python version
PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
echo "Using Python $PYTHON_VERSION"

# Create project directory structure
mkdir -p ./nova_data
mkdir -p ./nova_data/screenshots

# Install required Python packages
echo "Installing required Python packages..."
pip3 install selenium webdriver-manager

# Install Chrome and Chrome WebDriver
echo "Installing Chrome and ChromeDriver..."
apt-get update
apt-get install -y wget unzip
apt-get install -y google-chrome-stable

# Create a simple configuration file
if [ ! -f "config.json" ]; then
    echo "Creating configuration file..."
    cat > config.json << EOF
{
    "url": "https://dashboard-frontend-dusky.vercel.app",
    "output_dir": "./nova_data",
    "start_epoch": 14748,
    "end_epoch": 14745,
    "timeout": 30,
    "navigation_delay": 2.0,
    "action_delay": 1.0,
    "retries": 3,
    "take_screenshots": true,
    "verbose": true
}
EOF
fi

# Create a wrapper script to run the scraper
cat > run_scraper.sh << EOF
#!/bin/bash

echo "Running Nova Dashboard Molecule Scraper..."
python3 nova_scraper_headless.py "\$@"
EOF

chmod +x run_scraper.sh

echo "Setup complete!"
echo "To run the scraper: ./run_scraper.sh"
echo "To customize settings: edit config.json"
echo "To specify command-line options: ./run_scraper.sh --start-epoch 14748 --end-epoch 14745"
