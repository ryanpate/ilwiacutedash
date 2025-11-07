"""
Generate sample mobility scorecard data for testing
Creates realistic data for the dashboard without needing Snowflake connection
"""
import random
import math
from datetime import datetime, timedelta
import json


# Markets that will be used for sample data
MARKETS = [
    'New York Metro', 'Los Angeles', 'Chicago', 'Dallas-Fort Worth', 'Houston',
    'Philadelphia', 'Atlanta', 'Miami', 'Boston', 'San Francisco',
    'Phoenix', 'Seattle', 'Detroit', 'Minneapolis', 'San Diego',
    'Tampa', 'Denver', 'St. Louis', 'Baltimore', 'Charlotte'
]

# Metrics from the SQL query
METRICS = [
    'ALLRAT_DACC_25',
    'ALLRAT_DDR_25',
    'ALLRAT_DL_TPUT_25',
    'ALLRAT_UL_TPUT_25',
    'LTE_IQI_NS_ESO_25',
    'LTE_IQI_QUALITY_25',
    'VOICE_CDR_RET_25',
    'VOLTE_CDR_MOMT_ACC_25',
    'VOLTE_RAN_ACBACC_25_ALL',
    'LTE_PS_DVOL_25',
    'NR_DATAVOL_F1U_25'
]

# Metric display names
METRIC_NAMES = {
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

# Realistic metric value ranges (these create realistic impact scores)
METRIC_RANGES = {
    'ALLRAT_DACC_25': (0.90, 0.99),  # Percentage
    'ALLRAT_DDR_25': (0.90, 0.99),
    'ALLRAT_DL_TPUT_25': (10, 150),  # Mbps
    'ALLRAT_UL_TPUT_25': (5, 50),    # Mbps
    'LTE_IQI_NS_ESO_25': (0.001, 0.05),  # Small values
    'LTE_IQI_QUALITY_25': (0.02, 0.15),
    'VOICE_CDR_RET_25': (0.001, 0.02),
    'VOLTE_CDR_MOMT_ACC_25': (0.001, 0.05),
    'VOLTE_RAN_ACBACC_25_ALL': (0.001, 0.02),
    'LTE_PS_DVOL_25': (10000, 50000),  # Volume in MB
    'NR_DATAVOL_F1U_25': (5000, 30000)
}


def calculate_ns_eso_impact(metric_value):
    """Calculate NS/ESO impact: 20-(100*EXP(metric_value*-16.961727679)*0.2)"""
    try:
        return 20 - (100 * math.exp(metric_value * -16.961727679) * 0.2)
    except (OverflowError, ValueError):
        return 0


def calculate_quality_impact(metric_value):
    """Calculate Quality impact: 10-(100*EXP(metric_value*-5.645451413)*0.1)"""
    try:
        return 10 - (100 * math.exp(metric_value * -5.645451413) * 0.1)
    except (OverflowError, ValueError):
        return 0


def calculate_ul_tput_impact(metric_value):
    """Calculate UL Throughput impact: 5-((100-100*EXP(metric_value*-0.002014335))*0.05)"""
    try:
        return 5 - ((100 - 100 * math.exp(metric_value * -0.002014335)) * 0.05)
    except (OverflowError, ValueError):
        return 0


def calculate_vcdr_acc_impact(metric_value):
    """Calculate VoLTE CDR Accessibility impact: 10-(100*EXP(metric_value*-14.947954404)*0.1)"""
    try:
        return 10 - (100 * math.exp(metric_value * -14.947954404) * 0.1)
    except (OverflowError, ValueError):
        return 0


def calculate_voice_drop_impact(metric_value):
    """Calculate Voice Drop impact: 20-(100*EXP(metric_value*-44.673178216)*0.2)"""
    try:
        return 20 - (100 * math.exp(metric_value * -44.673178216) * 0.2)
    except (OverflowError, ValueError):
        return 0


def calculate_vran_acc_impact(metric_value):
    """Calculate VoLTE RAN Accessibility impact: 10-(100*EXP(metric_value*-41.884613262)*0.1)"""
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
        # For other metrics, use a simple formula for demo purposes
        return 0


def generate_metric_value(metric_name):
    """Generate a realistic metric value based on metric type"""
    min_val, max_val = METRIC_RANGES.get(metric_name, (0, 100))

    # Add some randomness but favor certain ranges for realism
    base_value = random.uniform(min_val, max_val)

    # Add occasional outliers (10% of the time)
    if random.random() < 0.1:
        # Make it worse to create higher impact
        if metric_name in ['LTE_IQI_NS_ESO_25', 'LTE_IQI_QUALITY_25',
                           'VOICE_CDR_RET_25', 'VOLTE_CDR_MOMT_ACC_25',
                           'VOLTE_RAN_ACBACC_25_ALL']:
            base_value = min(base_value * 1.5, max_val * 1.2)

    return base_value


def generate_sample_data(days=90):
    """Generate sample data for the specified number of days"""
    data = []
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    current_date = start_date
    while current_date <= end_date:
        for market in MARKETS:
            for metric in METRICS:
                # Generate metric values
                metricnum = random.randint(1000, 10000)
                metricden = random.randint(1001, 10001)

                # Calculate metric value based on type
                if 'CQI' in metric or 'NS_ESO' in metric or 'QUALITY' in metric or 'CDR' in metric or 'RAN' in metric:
                    metric_value = (metricnum * 100.0 / metricden) if metricden > 0 else 0
                else:
                    metric_value = (metricnum * 1.0 / metricden) if metricden > 0 else 0

                # Override with more realistic values
                metric_value = generate_metric_value(metric)

                # Calculate impact
                impact = calculate_impact(metric, metric_value)

                record = {
                    'periodend': current_date.isoformat(),
                    'edmarket': market,
                    'metricname': metric,
                    'metricnum': metricnum,
                    'metricden': metricden,
                    'metric_value': round(metric_value, 4),
                    'impact': round(impact, 4)
                }

                data.append(record)

        current_date += timedelta(days=1)

    return data


def save_sample_data(filename='sample_data.json'):
    """Generate and save sample data to a JSON file"""
    print(f"Generating sample data...")
    data = generate_sample_data(days=90)
    print(f"Generated {len(data)} records")

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Sample data saved to {filename}")

    # Print some statistics
    print("\nData Statistics:")
    print(f"Total records: {len(data)}")
    print(f"Markets: {len(MARKETS)}")
    print(f"Metrics: {len(METRICS)}")
    print(f"Date range: {data[0]['periodend']} to {data[-1]['periodend']}")

    # Calculate and print total impacts by metric
    print("\nAverage Impact by Metric:")
    for metric in METRICS:
        metric_data = [d for d in data if d['metricname'] == metric]
        avg_impact = sum(d['impact'] for d in metric_data) / len(metric_data) if metric_data else 0
        print(f"  {METRIC_NAMES.get(metric, metric)}: {avg_impact:.4f}")


if __name__ == '__main__':
    save_sample_data('sample_data.json')
