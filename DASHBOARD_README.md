# Mobility Scorecard Dashboard

A modern, responsive dashboard for tracking network performance metrics and identifying top offenders by total impact score.

## Features

✨ **Modern UI with Dark Mode**
- Clean, professional interface with gradient accents
- Toggle between light and dark themes
- Fully responsive design for all screen sizes

📊 **Top Offenders Ranking**
- Markets ranked by TOTAL_IMPACT (combination of all metric impacts)
- Color-coded impact badges (Critical, High, Medium, Low)
- Visual impact breakdown showing contribution from each metric
- Gold/Silver/Bronze medals for top 3 offenders

🔍 **Advanced Filtering**
- Filter by date range (default: last 14 days)
- Filter by specific market
- Filter by specific metric
- Adjustable results limit (10, 20, 50, 100)

📈 **Visualizations**
- Impact distribution bar chart (top 10 markets)
- Metric breakdown doughnut chart
- Interactive charts that update with theme changes

💾 **Data Export**
- Export filtered results to CSV
- Includes all impact scores and metadata

## Impact Calculations

The dashboard uses exponential formulas to calculate impact scores for each metric:

### NS/ESO Impact
```
20 - (100 * EXP(metric_value * -16.961727679) * 0.2)
```

### Quality Impact
```
10 - (100 * EXP(metric_value * -5.645451413) * 0.1)
```

### UL Throughput Impact
```
5 - ((100 - 100 * EXP(metric_value * -0.002014335)) * 0.05)
```

### VoLTE CDR Accessibility Impact
```
10 - (100 * EXP(metric_value * -14.947954404) * 0.1)
```

### Voice Drop Impact
```
20 - (100 * EXP(metric_value * -44.673178216) * 0.2)
```

### VoLTE RAN Accessibility Impact
```
10 - (100 * EXP(metric_value * -41.884613262) * 0.1)
```

### Total Impact
```
TOTAL_IMPACT = NS_ESO_IMP + QUALITY_IMP + UL_TPUT_IMP + VCDR_ACC_IMP + VOICE_DROP_IMP + VRAN_ACC_IMP
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r dashboard_requirements.txt
   ```

2. **Generate sample data:**
   ```bash
   python generate_sample_data.py
   ```
   This creates `sample_data.json` with 90 days of sample metrics data.

3. **Start the Flask API:**
   ```bash
   python dashboard_app.py
   ```
   The API will start on `http://localhost:5001`

4. **Open the dashboard:**
   Open your browser and navigate to:
   ```
   http://localhost:5001
   ```

## API Endpoints

### Health Check
```
GET /api/health
```
Returns server status and number of records loaded.

### Get Filters
```
GET /api/filters
```
Returns available markets, metrics, and date ranges for filtering.

### Get Top Offenders
```
GET /api/top-offenders?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD&market=MarketName&metric=MetricName&limit=20
```
Returns markets ranked by TOTAL_IMPACT with detailed breakdown.

**Parameters:**
- `startDate` (optional): Start date for filtering (YYYY-MM-DD)
- `endDate` (optional): End date for filtering (YYYY-MM-DD)
- `market` (optional): Filter by specific market
- `metric` (optional): Filter by specific metric
- `limit` (optional): Number of results to return (default: 20)

### Get Market Detail
```
GET /api/market-detail?market=MarketName&startDate=YYYY-MM-DD&endDate=YYYY-MM-DD
```
Returns daily metrics and impacts for a specific market.

### Get Metric Trends
```
GET /api/metric-trends?metric=MetricName&startDate=YYYY-MM-DD&endDate=YYYY-MM-DD
```
Returns trend data for a specific metric across all markets.

### Get Summary Stats
```
GET /api/summary-stats?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD
```
Returns summary statistics including total markets, total impact, and worst performers.

## Metrics

The dashboard tracks the following metrics:

| Metric Code | Display Name | Description |
|------------|--------------|-------------|
| ALLRAT_DACC_25 | Data Accessibility | Data session accessibility rate |
| ALLRAT_DDR_25 | Data Drop Rate | Data session drop rate |
| ALLRAT_DL_TPUT_25 | DL Throughput | Downlink throughput (Mbps) |
| ALLRAT_UL_TPUT_25 | UL Throughput | Uplink throughput (Mbps) |
| LTE_IQI_NS_ESO_25 | NS/ESO | Network switching/equivalent system outage |
| LTE_IQI_QUALITY_25 | Quality | Network quality indicator |
| VOICE_CDR_RET_25 | Voice CDR Retention | Voice call drop rate retention |
| VOLTE_CDR_MOMT_ACC_25 | VoLTE CDR Accessibility | VoLTE call detail record accessibility |
| VOLTE_RAN_ACBACC_25_ALL | VoLTE RAN Accessibility | VoLTE radio access network accessibility |
| LTE_PS_DVOL_25 | LTE Data Volume | LTE packet switched data volume |
| NR_DATAVOL_F1U_25 | NR Data Volume | 5G NR data volume |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Browser (dashboard.html)             │
│  - Dark mode toggle                                  │
│  - Filters & controls                                │
│  - Data table with impact rankings                   │
│  - Charts (Chart.js)                                 │
└────────────────────┬────────────────────────────────┘
                     │ HTTP/JSON
                     ▼
┌─────────────────────────────────────────────────────┐
│              Flask API (dashboard_app.py)            │
│  - REST endpoints                                    │
│  - Impact calculations                               │
│  - Data filtering & aggregation                      │
└────────────────────┬────────────────────────────────┘
                     │ Read
                     ▼
┌─────────────────────────────────────────────────────┐
│          Sample Data (sample_data.json)              │
│  - 90 days of metrics data                           │
│  - 20 markets × 11 metrics × 91 days = 20,020 records│
└─────────────────────────────────────────────────────┘
```

## Sample Data

The `generate_sample_data.py` script creates realistic sample data with:
- **20 markets** across major US metropolitan areas
- **11 metrics** covering data, voice, and quality indicators
- **90 days** of daily data
- **20,020 total records**
- Realistic metric values based on typical network performance ranges
- Occasional outliers to simulate network issues

## Customization

### Changing Sample Data
Edit `generate_sample_data.py` to:
- Add/remove markets in the `MARKETS` list
- Adjust metric ranges in the `METRIC_RANGES` dictionary
- Change the number of days in the `generate_sample_data(days=90)` call

### Modifying Impact Calculations
Edit the calculation functions in `dashboard_app.py`:
- `calculate_ns_eso_impact()`
- `calculate_quality_impact()`
- `calculate_ul_tput_impact()`
- etc.

### Styling
Edit the CSS variables in `dashboard.html`:
- Light theme: `:root { ... }`
- Dark theme: `[data-theme="dark"] { ... }`

## Production Deployment

For production use:

1. **Use a production WSGI server** instead of Flask's development server:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5001 dashboard_app:app
   ```

2. **Replace sample data** with real database connection:
   - Modify `load_sample_data()` to query your database
   - Update endpoints to use live data

3. **Add authentication** if needed for security

4. **Configure CORS** properly for your domain

5. **Enable HTTPS** for secure communication

## Troubleshooting

### Flask won't start
- Ensure Flask is installed: `pip install flask flask-cors`
- Check port 5001 is not already in use
- Verify `sample_data.json` exists

### No data showing
- Run `python generate_sample_data.py` to create sample data
- Check browser console for API errors
- Verify API is running: `curl http://localhost:5001/api/health`

### Charts not displaying
- Ensure Chart.js CDN is accessible
- Check browser console for JavaScript errors
- Verify theme is set correctly

## Files

```
ilwiacutedash/
├── dashboard_app.py              # Flask API server
├── dashboard.html                # Main dashboard UI
├── generate_sample_data.py       # Sample data generator
├── dashboard_requirements.txt    # Python dependencies
├── sample_data.json             # Generated sample data
└── DASHBOARD_README.md          # This file
```

## Future Enhancements

Potential improvements for the dashboard:

- [ ] Market detail page with drill-down capabilities
- [ ] Trend analysis over time with line charts
- [ ] Real-time data updates via WebSocket
- [ ] User authentication and role-based access
- [ ] Custom alert thresholds and notifications
- [ ] Metric correlation analysis
- [ ] Export to PDF reports
- [ ] Historical data comparison
- [ ] Predictive analytics using ML models

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review API endpoint documentation
3. Inspect browser console for errors
4. Check Flask server logs for API issues

## License

This dashboard is provided as-is for internal use.
