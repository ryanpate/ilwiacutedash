"""
Mobility Scorecard Dashboard Flask API
Displays metrics with impact calculations and Top Offenders ranking
"""
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import math
from datetime import datetime, timedelta
from collections import defaultdict
import os

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load sample data
SAMPLE_DATA_FILE = 'sample_data.json'
sample_data = []

def load_sample_data():
    """Load sample data from JSON file"""
    global sample_data
    if os.path.exists(SAMPLE_DATA_FILE):
        with open(SAMPLE_DATA_FILE, 'r') as f:
            sample_data = json.load(f)
        print(f"✅ Loaded {len(sample_data)} sample records")
    else:
        print(f"⚠️  Sample data file not found: {SAMPLE_DATA_FILE}")
        print("   Run 'python generate_sample_data.py' to create sample data")


# Metric display names
METRIC_DISPLAY_NAMES = {
    'ALLRAT_DACC_25': 'Data Accessibility',
    'ALLRAT_DDR_25': 'Data Drop Rate',
    'ALLRAT_DL_TPUT_25': 'DL Throughput',
    'ALLRAT_UL_TPUT_25': 'UL Throughput',
    'LTE_IQI_NS_ESO_25': 'NS/ESO',
    'LTE_IQI_QUALITY_25': 'Quality',
    'VOICE_CDR_RET_25': 'Voice CDR Retention',
    'VOLTE_CDR_MOMT_ACC_25': 'VoLTE CDR Accessibility',
    'VOLTE_RAN_ACBACC_25_ALL': 'VoLTE RAN Accessibility',
    'LTE_PS_DVOL_25': 'LTE Data Volume',
    'NR_DATAVOL_F1U_25': 'NR Data Volume'
}


def calculate_ns_eso_impact(metric_value):
    """Calculate NS/ESO impact"""
    try:
        return 20 - (100 * math.exp(metric_value * -16.961727679) * 0.2)
    except (OverflowError, ValueError):
        return 0


def calculate_quality_impact(metric_value):
    """Calculate Quality impact"""
    try:
        return 10 - (100 * math.exp(metric_value * -5.645451413) * 0.1)
    except (OverflowError, ValueError):
        return 0


def calculate_ul_tput_impact(metric_value):
    """Calculate UL Throughput impact"""
    try:
        return 5 - ((100 - 100 * math.exp(metric_value * -0.002014335)) * 0.05)
    except (OverflowError, ValueError):
        return 0


def calculate_vcdr_acc_impact(metric_value):
    """Calculate VoLTE CDR Accessibility impact"""
    try:
        return 10 - (100 * math.exp(metric_value * -14.947954404) * 0.1)
    except (OverflowError, ValueError):
        return 0


def calculate_voice_drop_impact(metric_value):
    """Calculate Voice Drop impact"""
    try:
        return 20 - (100 * math.exp(metric_value * -44.673178216) * 0.2)
    except (OverflowError, ValueError):
        return 0


def calculate_vran_acc_impact(metric_value):
    """Calculate VoLTE RAN Accessibility impact"""
    try:
        return 10 - (100 * math.exp(metric_value * -41.884613262) * 0.1)
    except (OverflowError, ValueError):
        return 0


def calculate_impact(metric_name, metric_value):
    """Calculate impact score based on metric name and value"""
    if metric_name == 'LTE_IQI_NS_ESO_25':
        return calculate_ns_eso_impact(metric_value)
    elif metric_name == 'LTE_IQI_QUALITY_25':
        return calculate_quality_impact(metric_value)
    elif metric_name == 'ALLRAT_UL_TPUT_25':
        return calculate_ul_tput_impact(metric_value)
    elif metric_name == 'VOLTE_CDR_MOMT_ACC_25':
        return calculate_vcdr_acc_impact(metric_value)
    elif metric_name == 'VOICE_CDR_RET_25':
        return calculate_voice_drop_impact(metric_value)
    elif metric_name == 'VOLTE_RAN_ACBACC_25_ALL':
        return calculate_vran_acc_impact(metric_value)
    else:
        return 0


def calculate_total_impact(impacts):
    """Calculate total impact from individual metric impacts"""
    return sum(impacts.values())


@app.route('/')
def index():
    """Serve the main dashboard page"""
    return send_from_directory('.', 'dashboard.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'records_loaded': len(sample_data),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/filters', methods=['GET'])
def get_filters():
    """Get available filter options"""
    markets = sorted(list(set(record['edmarket'] for record in sample_data)))
    metrics = sorted(list(set(record['metricname'] for record in sample_data)))
    metric_display_names = [METRIC_DISPLAY_NAMES.get(m, m) for m in metrics]

    # Get date range
    dates = [datetime.fromisoformat(record['periodend']) for record in sample_data]
    min_date = min(dates).strftime('%Y-%m-%d') if dates else None
    max_date = max(dates).strftime('%Y-%m-%d') if dates else None

    return jsonify({
        'markets': markets,
        'metrics': metrics,
        'metricDisplayNames': metric_display_names,
        'metricMapping': METRIC_DISPLAY_NAMES,
        'dateRange': {
            'min': min_date,
            'max': max_date
        }
    })


@app.route('/api/top-offenders', methods=['GET'])
def get_top_offenders():
    """
    Get top offenders ranked by TOTAL_IMPACT
    Supports filtering by date range, market, and metric
    """
    # Get filter parameters
    start_date_str = request.args.get('startDate', '')
    end_date_str = request.args.get('endDate', '')
    market_filter = request.args.get('market', '')
    metric_filter = request.args.get('metric', '')
    limit = int(request.args.get('limit', 20))

    # Parse dates
    start_date = datetime.fromisoformat(start_date_str).date() if start_date_str else None
    end_date = datetime.fromisoformat(end_date_str).date() if end_date_str else None

    # Filter data
    filtered_data = sample_data

    if start_date:
        filtered_data = [r for r in filtered_data
                        if datetime.fromisoformat(r['periodend']).date() >= start_date]

    if end_date:
        filtered_data = [r for r in filtered_data
                        if datetime.fromisoformat(r['periodend']).date() <= end_date]

    if market_filter:
        filtered_data = [r for r in filtered_data if r['edmarket'] == market_filter]

    if metric_filter:
        filtered_data = [r for r in filtered_data if r['metricname'] == metric_filter]

    # Aggregate by market
    market_impacts = defaultdict(lambda: {
        'market': '',
        'impacts': defaultdict(float),
        'metric_values': defaultdict(list),
        'record_count': 0
    })

    for record in filtered_data:
        market = record['edmarket']
        metric = record['metricname']
        metric_value = record['metric_value']
        impact = calculate_impact(metric, metric_value)

        market_impacts[market]['market'] = market
        market_impacts[market]['impacts'][metric] += impact
        market_impacts[market]['metric_values'][metric].append(metric_value)
        market_impacts[market]['record_count'] += 1

    # Calculate total impact for each market
    results = []
    for market, data in market_impacts.items():
        total_impact = calculate_total_impact(data['impacts'])

        # Calculate average metric values
        avg_metrics = {
            metric: sum(values) / len(values) if values else 0
            for metric, values in data['metric_values'].items()
        }

        result = {
            'market': market,
            'total_impact': round(total_impact, 2),
            'ns_eso_impact': round(data['impacts'].get('LTE_IQI_NS_ESO_25', 0), 2),
            'quality_impact': round(data['impacts'].get('LTE_IQI_QUALITY_25', 0), 2),
            'ul_tput_impact': round(data['impacts'].get('ALLRAT_UL_TPUT_25', 0), 2),
            'vcdr_acc_impact': round(data['impacts'].get('VOLTE_CDR_MOMT_ACC_25', 0), 2),
            'voice_drop_impact': round(data['impacts'].get('VOICE_CDR_RET_25', 0), 2),
            'vran_acc_impact': round(data['impacts'].get('VOLTE_RAN_ACBACC_25_ALL', 0), 2),
            'avg_metrics': avg_metrics,
            'record_count': data['record_count']
        }

        results.append(result)

    # Sort by total impact (descending)
    results.sort(key=lambda x: x['total_impact'], reverse=True)

    # Limit results
    results = results[:limit]

    return jsonify(results)


@app.route('/api/market-detail', methods=['GET'])
def get_market_detail():
    """Get detailed metrics for a specific market over time"""
    market = request.args.get('market', '')
    start_date_str = request.args.get('startDate', '')
    end_date_str = request.args.get('endDate', '')

    if not market:
        return jsonify({'error': 'Market parameter is required'}), 400

    # Parse dates
    start_date = datetime.fromisoformat(start_date_str).date() if start_date_str else None
    end_date = datetime.fromisoformat(end_date_str).date() if end_date_str else None

    # Filter data
    filtered_data = [r for r in sample_data if r['edmarket'] == market]

    if start_date:
        filtered_data = [r for r in filtered_data
                        if datetime.fromisoformat(r['periodend']).date() >= start_date]

    if end_date:
        filtered_data = [r for r in filtered_data
                        if datetime.fromisoformat(r['periodend']).date() <= end_date]

    # Group by date
    daily_data = defaultdict(lambda: {
        'date': '',
        'metrics': {},
        'impacts': {},
        'total_impact': 0
    })

    for record in filtered_data:
        date = record['periodend']
        metric = record['metricname']
        metric_value = record['metric_value']
        impact = calculate_impact(metric, metric_value)

        if not daily_data[date]['date']:
            daily_data[date]['date'] = date

        daily_data[date]['metrics'][metric] = metric_value
        daily_data[date]['impacts'][metric] = impact
        daily_data[date]['total_impact'] += impact

    # Convert to list and sort by date
    results = list(daily_data.values())
    results.sort(key=lambda x: x['date'])

    # Add metric display names
    for result in results:
        result['metric_display'] = {
            metric: METRIC_DISPLAY_NAMES.get(metric, metric)
            for metric in result['metrics'].keys()
        }

    return jsonify(results)


@app.route('/api/metric-trends', methods=['GET'])
def get_metric_trends():
    """Get trend data for specific metrics across all markets"""
    metric_name = request.args.get('metric', '')
    start_date_str = request.args.get('startDate', '')
    end_date_str = request.args.get('endDate', '')

    if not metric_name:
        return jsonify({'error': 'Metric parameter is required'}), 400

    # Parse dates
    start_date = datetime.fromisoformat(start_date_str).date() if start_date_str else None
    end_date = datetime.fromisoformat(end_date_str).date() if end_date_str else None

    # Filter data
    filtered_data = [r for r in sample_data if r['metricname'] == metric_name]

    if start_date:
        filtered_data = [r for r in filtered_data
                        if datetime.fromisoformat(r['periodend']).date() >= start_date]

    if end_date:
        filtered_data = [r for r in filtered_data
                        if datetime.fromisoformat(r['periodend']).date() <= end_date]

    # Group by date
    daily_trend = defaultdict(lambda: {
        'date': '',
        'values': [],
        'impacts': []
    })

    for record in filtered_data:
        date = record['periodend']
        metric_value = record['metric_value']
        impact = calculate_impact(metric_name, metric_value)

        if not daily_trend[date]['date']:
            daily_trend[date]['date'] = date

        daily_trend[date]['values'].append(metric_value)
        daily_trend[date]['impacts'].append(impact)

    # Calculate averages
    results = []
    for date, data in daily_trend.items():
        avg_value = sum(data['values']) / len(data['values']) if data['values'] else 0
        avg_impact = sum(data['impacts']) / len(data['impacts']) if data['impacts'] else 0

        results.append({
            'date': date,
            'avg_value': round(avg_value, 4),
            'avg_impact': round(avg_impact, 4),
            'sample_count': len(data['values'])
        })

    # Sort by date
    results.sort(key=lambda x: x['date'])

    return jsonify({
        'metric': metric_name,
        'metric_display': METRIC_DISPLAY_NAMES.get(metric_name, metric_name),
        'trend_data': results
    })


@app.route('/api/summary-stats', methods=['GET'])
def get_summary_stats():
    """Get summary statistics for the dashboard"""
    start_date_str = request.args.get('startDate', '')
    end_date_str = request.args.get('endDate', '')

    # Parse dates
    start_date = datetime.fromisoformat(start_date_str).date() if start_date_str else None
    end_date = datetime.fromisoformat(end_date_str).date() if end_date_str else None

    # Filter data
    filtered_data = sample_data

    if start_date:
        filtered_data = [r for r in filtered_data
                        if datetime.fromisoformat(r['periodend']).date() >= start_date]

    if end_date:
        filtered_data = [r for r in filtered_data
                        if datetime.fromisoformat(r['periodend']).date() <= end_date]

    # Calculate statistics
    total_records = len(filtered_data)
    markets = set(record['edmarket'] for record in filtered_data)
    total_markets = len(markets)

    # Calculate total impacts
    total_impact = 0
    metric_impacts = defaultdict(float)

    for record in filtered_data:
        metric = record['metricname']
        metric_value = record['metric_value']
        impact = calculate_impact(metric, metric_value)

        total_impact += impact
        metric_impacts[metric] += impact

    # Find worst performing markets
    market_impacts = defaultdict(float)
    for record in filtered_data:
        market = record['edmarket']
        metric = record['metricname']
        metric_value = record['metric_value']
        impact = calculate_impact(metric, metric_value)
        market_impacts[market] += impact

    worst_markets = sorted(market_impacts.items(), key=lambda x: x[1], reverse=True)[:5]

    return jsonify({
        'total_records': total_records,
        'total_markets': total_markets,
        'total_impact': round(total_impact, 2),
        'avg_impact_per_record': round(total_impact / total_records, 4) if total_records > 0 else 0,
        'metric_impacts': {
            metric: round(impact, 2)
            for metric, impact in metric_impacts.items()
        },
        'worst_markets': [
            {'market': market, 'impact': round(impact, 2)}
            for market, impact in worst_markets
        ]
    })


if __name__ == '__main__':
    print("🚀 Starting Mobility Scorecard Dashboard API...")
    print("=" * 60)

    # Load sample data
    load_sample_data()

    print("\n📊 Available endpoints:")
    print("  GET  /                    - Dashboard UI")
    print("  GET  /api/health          - Health check")
    print("  GET  /api/filters         - Get available filters")
    print("  GET  /api/top-offenders   - Get top offenders by TOTAL_IMPACT")
    print("  GET  /api/market-detail   - Get detailed market metrics")
    print("  GET  /api/metric-trends   - Get metric trend data")
    print("  GET  /api/summary-stats   - Get summary statistics")

    print("\n📡 Server starting on: http://localhost:5001")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5001)
