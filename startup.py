#!/usr/bin/env python3
"""
CQI Dashboard Startup Script
Launches both the Flask API and web server for the dashboard
"""

import os
import sys
import time
import threading
import webbrowser
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler


def check_dependencies():
    """Check and install required packages"""
    required = {
        'flask': 'flask',
        'flask_cors': 'flask-cors',
        'snowflake.connector': 'snowflake-connector-python',
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"📦 Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("✅ Dependencies installed successfully!")
        return False
    return True


def start_flask_api():
    """Start the Flask API server"""
    try:
        # Set environment variables
        os.environ['FLASK_APP'] = 'app.py'
        
        # Import and run the Flask app
        import app
        print("🔧 Starting Flask API on http://localhost:5000")
        print("📊 Snowflake connection will be established...")
        app.app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
    except ImportError:
        print("❌ Error: app.py not found in current directory")
        print("Make sure the Flask API file (app.py) is in the same folder")
    except Exception as e:
        print(f"❌ Error starting Flask API: {e}")


def start_web_server():
    """Start the web server for the dashboard"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Only log errors
            if args[1] != '200':
                super().log_message(format, *args)
    
    print("🌐 Starting web server on http://localhost:8000")
    httpd = HTTPServer(('localhost', 8000), QuietHTTPRequestHandler)
    httpd.serve_forever()


def test_api_health(max_retries=10):
    """Test if the API is responding"""
    import urllib.request
    import json
    
    for i in range(max_retries):
        try:
            response = urllib.request.urlopen('http://localhost:5000/api/health')
            data = json.loads(response.read().decode())
            if data.get('status') == 'healthy':
                return True
        except:
            time.sleep(1)
    return False


def print_banner():
    """Print startup banner"""
    print("=" * 60)
    print("🚀 CQI OFFENDERS DASHBOARD")
    print("=" * 60)
    print("Interactive dashboard for network performance analysis")
    print("Displays top offenders by Contribution (IDXCONTR) or Failures")
    print("-" * 60)


def main():
    """Main startup function"""
    print_banner()
    
    # Check required files
    required_files = ['app.py', 'index.html']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("\nRequired files:")
        print("  - app.py: Flask API backend")
        print("  - index.html: Dashboard frontend")
        print("  - private_key.txt: Snowflake authentication (optional)")
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Check for Snowflake key
    if not os.path.exists('private_key.txt'):
        print("⚠️  Warning: private_key.txt not found")
        print("   Dashboard will work with sample data only")
        print("   Add private_key.txt for Snowflake connection")
        time.sleep(3)
    
    # Check and install dependencies
    print("\n📋 Checking dependencies...")
    deps_installed = check_dependencies()
    
    if not deps_installed:
        print("\n⚠️  Dependencies were installed. Please run this script again.")
        input("\nPress Enter to exit...")
        sys.exit(0)
    
    print("✅ All dependencies are ready")
    
    # Start Flask API in a separate thread
    print("\n🔄 Starting services...")
    api_thread = threading.Thread(target=start_flask_api, daemon=True)
    api_thread.start()
    
    # Give Flask time to start
    print("⏳ Waiting for Flask API to initialize...")
    time.sleep(3)
    
    # Test API health
    if test_api_health():
        print("✅ Flask API is running and healthy")
    else:
        print("⚠️  Flask API may not be fully ready yet")
        print("   If dashboard shows connection error, refresh in a few seconds")
    
    # Start web server in a separate thread
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    
    # Give web server time to start
    time.sleep(1)
    
    # Open browser
    # dashboard_url = "http://localhost:8000/index.html"
    # print(f"\n🌐 Opening dashboard in browser: {dashboard_url}")
    # webbrowser.open(dashboard_url)
    
    # Print success message
    print("\n" + "=" * 60)
    print("✨ CQI DASHBOARD IS RUNNING!")
    print("=" * 60)
    print("\n📊 Dashboard URL: http://localhost:8000/index.html")
    print("🔧 API Endpoint: http://localhost:5000/api")
    print("\n🎯 Key Features:")
    print("  • Initial load shows top offenders by Contribution (IDXCONTR)")
    print("  • Switch to 'Extra Failures' for different ranking")
    print("  • Fresh data pull from Snowflake on criteria change")
    print("  • Export results to CSV")
    print("  • Click USID for detailed metrics view")
    print("\n📝 Troubleshooting:")
    print("  • If 'API not running' error: Refresh page after 5 seconds")
    print("  • Check ports 5000 and 8000 are not in use")
    print("  • Ensure private_key.txt exists for real Snowflake data")
    print("\n⛔ Press Ctrl+C to stop all services")
    print("=" * 60)
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        print("Dashboard closed successfully. Goodbye! 👋")
        sys.exit(0)


if __name__ == "__main__":
    main()