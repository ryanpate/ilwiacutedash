# CQI Offenders Dashboard

An interactive web-based dashboard for displaying CQI (Customer Quality Index) offenders based on data from Snowflake tables. The dashboard provides real-time analysis of network performance metrics with ranking by Contribution (IDXCONTR) or Extra Failures.

## 🎯 Key Features

- **Dual Ranking Criteria**: 
  - Initial load shows top offenders by **Contribution (IDXCONTR)** - worst performers first
  - Switch to **Extra Failures** ranking for different perspective
  - **Fresh data pull from Snowflake** when ranking criteria changes

- **Interactive Filters**:
  - Submarket
  - CQE Cluster
  - Date Range (Period Start/End)
  - Metric Name
  - USID search

- **Data Visualization**:
  - Table view with sortable columns
  - Chart view for top 20 offenders
  - Color-coded severity levels (Critical, High, Medium, Low)
  - Statistics cards showing key metrics

- **Export Functionality**:
  - Export filtered data to CSV
  - Includes ranking based on selected criteria

## 📋 Prerequisites

- Python 3.7 or higher
- Snowflake account with access to `PRD_MOBILITY.PRD_MOBILITYSCORECARD_VIEWS.CQI2025_CQX_CONTRIBUTION`
- Private key file for Snowflake authentication

## 🚀 Quick Start

### Option 1: Using the Startup Script (Recommended)

1. Place all files in the same directory:
   - `index.html` - Dashboard frontend
   - `app.py` - Flask backend API
   - `startup.py` - Launch script
   - `private_key.txt` - Snowflake authentication key

2. Run the startup script:
   ```bash
   python startup.py
   ```

3. The dashboard will automatically open in your browser at `http://localhost:8080`

### Option 2: Manual Setup

1. Install dependencies:
   ```bash
   pip install flask flask-cors snowflake-connector-python pandas numpy
   ```

2. Start the Flask API:
   ```bash
   python app.py
   ```

3. In a new terminal, start the web server:
   ```bash
   python -m http.server 8080
   ```

4. Open browser to `http://localhost:8080/index.html`

## 📁 File Structure

```
cqi-dashboard/
├── index.html          # Enhanced dashboard frontend
├── app.py             # Flask backend with Snowflake connection
├── startup.py         # Automated startup script
├── private_key.txt    # Snowflake authentication (not included)
└── README.md          # This file
```

## 🔧 Configuration

### Snowflake Connection

Edit the `SNOWFLAKE_CONFIG` in `app.py`:

```python
SNOWFLAKE_CONFIG = {
    'account': 'your-account.region.privatelink',
    'user': 'your-username',
    'private_key_file': 'private_key.txt',
    'private_key_file_pwd': 'your-key-password',
    'warehouse': 'YOUR_WAREHOUSE',
    'database': 'YOUR_DATABASE',
    'schema': 'YOUR_SCHEMA'
}
```

### Metric Mapping

The dashboard uses friendly names for metrics. Modify `METRIC_MAPPING` in `app.py` to customize:

```python
METRIC_MAPPING = {
    'VOICE_CDR_RET_25': 'V-CDR',
    'LTE_IQI_NS_ESO_25': 'NS/ESO',
    # Add more mappings as needed
}
```

## 📊 Understanding the Rankings

### Contribution (IDXCONTR) Ranking
- **Default initial ranking**
- Shows performance impact on overall CQI
- **Negative values = worse performance** (shown in red)
- **Positive values = better performance** (shown in green)
- Sorted from most negative (worst) to most positive (best)

### Extra Failures Ranking
- Shows absolute failure counts
- Sorted from highest to lowest
- All values are positive (negative values set to 0)
- **Does not directly correlate with Contribution ranking**

### Why Fresh Data Pulls?

When switching between ranking criteria, the dashboard fetches fresh data from Snowflake because:
- Top offenders by Contribution may differ from top offenders by Failures
- Ensures accurate ranking based on selected criteria
- Provides real-time data for better decision making

## 🎨 Dashboard Interface

### Statistics Cards
- **Total Entries**: Number of records in current view
- **Total Contribution/Failures**: Sum based on selected ranking
- **Critical Offenders**: Entries contributing ≥10% of total
- **Active USIDs**: Unique site identifiers

### Severity Levels
- **CRITICAL**: ≥10% of total (red)
- **HIGH**: 5-10% of total (orange) 
- **MEDIUM**: 1-5% of total (yellow)
- **LOW**: <1% of total (green)

## 🐛 Troubleshooting

### "Flask API not running" error
- Wait 5 seconds and refresh the page
- Check if port 5000 is already in use
- Verify Flask is installed: `pip install flask`

### No data displayed
- Check date range filters (default: last 2 days)
- Verify Snowflake connection in `app.py`
- Ensure `private_key.txt` exists and is valid
- Check Snowflake table has data for selected period

### Connection to Snowflake fails
- Verify credentials in `SNOWFLAKE_CONFIG`
- Check network access to Snowflake
- Ensure private key file permissions are correct
- Test connection using Snowflake CLI or SnowSQL

## 📈 Performance Tips

- Use date filters to limit data range
- Filter by specific metrics for faster queries
- The dashboard limits results to top 1000 offenders
- Chart view shows only top 20 for clarity

## 🔐 Security Notes

- Never commit `private_key.txt` to version control
- Use environment variables for sensitive configuration
- Implement authentication for production deployment
- Consider using Snowflake OAuth for better security

## 📝 API Endpoints

The Flask backend provides these endpoints:

- `GET /api/health` - Health check
- `GET /api/filters` - Get filter options
- `GET /api/data` - Get CQI data with filters
- `GET /api/summary` - Get summary statistics
- `GET /api/test` - Test Snowflake connection

## 🚀 Production Deployment

For production use:

1. Use a production WSGI server (Gunicorn, uWSGI)
2. Deploy behind a reverse proxy (Nginx, Apache)
3. Implement authentication and authorization
4. Use HTTPS for secure communication
5. Set up monitoring and logging
6. Configure connection pooling for Snowflake

## 📄 License

This dashboard is for internal use only. Do not distribute without authorization.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Snowflake connection logs
3. Verify data availability in source tables
4. Contact your data engineering team

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Maintained By**: Data Analytics Team