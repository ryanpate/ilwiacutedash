"""
CQI Dashboard Flask API - With FOCUSLEV-based Contribution Support
Uses FOCUSLEV to determine correct IDXCONTR and EXTRAFAILURES values
FOCUSLEV: 0=National, 1=Regional, 2=Market, 3=Submarket
"""

from dotenv import load_dotenv
import json
import logging
from functools import lru_cache
import os
from datetime import datetime, timedelta
import csv
import numpy as np
import pandas as pd
import snowflake.connector as sc
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress Snowflake connector INFO logs
snowflake_logger = logging.getLogger('snowflake.connector')
snowflake_logger.setLevel(logging.WARNING)

# Suppress werkzeug INFO logs
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.ERROR)

# Snowflake connection parameters from environment variables
SNOWFLAKE_CONFIG = {
    'account': os.getenv('SNOWFLAKE_ACCOUNT', 'nsasprd.east-us-2.privatelink'),
    'user': os.getenv('SNOWFLAKE_USER'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'USR_REPORTING_WH'),
    'database': os.getenv('SNOWFLAKE_DATABASE', 'PRD_MOBILITY'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA', 'PRD_MOBILITYSCORECARD_VIEWS')
}

# Path to the mapping CSV file
MAPPING_CSV_PATH = 'submkt_cqecluster_mapping.csv'

# Directory for district CSV files
DISTRICT_CSV_DIR = '.'  # Current directory, can be changed to a specific folder

# Handle authentication method
if os.getenv('SNOWFLAKE_PRIVATE_KEY_PATH'):
    SNOWFLAKE_CONFIG['private_key_file'] = os.getenv(
        'SNOWFLAKE_PRIVATE_KEY_PATH')
    if os.getenv('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE'):
        SNOWFLAKE_CONFIG['private_key_file_pwd'] = os.getenv(
            'SNOWFLAKE_PRIVATE_KEY_PASSPHRASE')
elif os.getenv('SNOWFLAKE_PASSWORD'):
    SNOWFLAKE_CONFIG['password'] = os.getenv('SNOWFLAKE_PASSWORD')
else:
    logger.warning(
        "No authentication method configured. Set either SNOWFLAKE_PRIVATE_KEY_PATH or SNOWFLAKE_PASSWORD")


def validate_config():
    """Validate that required configuration is present"""
    required = ['account', 'user', 'warehouse', 'database', 'schema']
    missing = [key for key in required if not SNOWFLAKE_CONFIG.get(key)]

    if missing:
        logger.error(
            f"Missing required Snowflake configuration: {', '.join(missing)}")
        logger.error("Please set the following environment variables:")
        for key in missing:
            logger.error(f"  SNOWFLAKE_{key.upper()}")
        return False

    has_auth = (
        'private_key_file' in SNOWFLAKE_CONFIG or 'password' in SNOWFLAKE_CONFIG)
    if not has_auth:
        logger.error("No authentication method configured.")
        logger.error(
            "Set either SNOWFLAKE_PRIVATE_KEY_PATH or SNOWFLAKE_PASSWORD")
        return False

    return True


def load_submarket_cluster_mapping():
    """Load the submarket-cluster mapping from CSV file"""
    mapping = {}

    if not os.path.exists(MAPPING_CSV_PATH):
        logger.warning(f"Mapping CSV file not found: {MAPPING_CSV_PATH}")
        logger.warning("Creating sample mapping file...")

        sample_data = [
            ['SUBMKT', 'CQECLUSTER'],
            ['NYC', 'CQE_NYC_MANHATTAN'],
            ['NYC', 'CQE_NYC_BROOKLYN'],
            ['LA', 'CQE_LA_DOWNTOWN'],
            ['LA', 'CQE_LA_HOLLYWOOD'],
            ['Chicago', 'CQE_CHI_NORTH'],
            ['Chicago', 'CQE_CHI_SOUTH'],
        ]

        with open(MAPPING_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(sample_data)

        logger.info(f"Sample mapping file created: {MAPPING_CSV_PATH}")

    try:
        with open(MAPPING_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                submarket = row.get('SUBMKT', '').strip()
                cluster = row.get('CQECLUSTER', '').strip()

                if submarket and cluster:
                    if submarket not in mapping:
                        mapping[submarket] = []
                    mapping[submarket].append(cluster)

        logger.info(
            f"Loaded mapping for {len(mapping)} submarkets from {MAPPING_CSV_PATH}")

    except Exception as e:
        logger.error(f"Error reading mapping CSV: {str(e)}")
        return {}

    return mapping


def load_district_mapping(submarket):
    """Load district mapping for a specific submarket from CSV file"""
    if not submarket:
        return {}

    # Create filename from submarket name
    district_file = os.path.join(DISTRICT_CSV_DIR, f"{submarket}.csv")

    if not os.path.exists(district_file):
        logger.info(
            f"District file not found for submarket '{submarket}': {district_file}")
        return {}

    district_mapping = {}

    try:
        with open(district_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    # First column is USID (as integer), second is district
                    try:
                        # Convert to string for consistency
                        usid = str(row[0]).strip()
                        district = str(row[1]).strip()
                        district_mapping[usid] = district
                    except (ValueError, IndexError) as e:
                        logger.warning(
                            f"Skipping invalid row in {district_file}: {row}")
                        continue

        logger.info(
            f"Loaded {len(district_mapping)} district mappings for submarket '{submarket}'")

    except Exception as e:
        logger.error(f"Error reading district file {district_file}: {str(e)}")
        return {}

    return district_mapping


def clean_numeric_value(value):
    """Clean numeric values, handling NaN, Infinity, None, and negative values"""
    if value is None:
        return 0
    if pd.isna(value):
        return 0
    if isinstance(value, (float, np.floating)):
        if np.isinf(value):
            return 0
        if np.isnan(value):
            return 0
        if value < 0:
            return 0
        if value == int(value):
            return int(value)
        return float(value)
    if isinstance(value, (int, np.integer)):
        return max(0, int(value))
    try:
        num_val = float(value)
        if np.isnan(num_val) or np.isinf(num_val):
            return 0
        if num_val < 0:
            return 0
        if num_val == int(num_val):
            return int(num_val)
        return num_val
    except (ValueError, TypeError):
        return 0


def clean_contribution_value(value):
    """Clean contribution values (can be negative)"""
    if value is None or pd.isna(value):
        return 0

    if isinstance(value, (float, np.floating)):
        if np.isinf(value) or np.isnan(value):
            return 0
        return float(value)

    if isinstance(value, (int, np.integer)):
        return float(value)

    try:
        num_val = float(value)
        if np.isnan(num_val) or np.isinf(num_val):
            return 0
        return num_val
    except (ValueError, TypeError):
        return 0


def get_snowflake_connection():
    """Create and return a Snowflake connection"""
    if not validate_config():
        raise ValueError(
            "Invalid Snowflake configuration. Check environment variables.")

    try:
        conn = sc.connect(**SNOWFLAKE_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {str(e)}")
        raise


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    config_valid = validate_config()
    csv_exists = os.path.exists(MAPPING_CSV_PATH)

    return jsonify({
        'status': 'healthy' if config_valid else 'unhealthy',
        'config_valid': config_valid,
        'mapping_csv_exists': csv_exists,
        'mapping_csv_path': MAPPING_CSV_PATH,
        'district_csv_dir': DISTRICT_CSV_DIR,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/test', methods=['GET'])
def test_connection():
    """Test endpoint to verify Snowflake connection and data"""
    try:
        conn = get_snowflake_connection()
        cur = conn.cursor()

        cur.execute(
            "SELECT CURRENT_USER(), CURRENT_DATABASE(), CURRENT_SCHEMA()")
        context = cur.fetchone()

        cur.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'PRD_MOBILITYSCORECARD_VIEWS' 
            AND TABLE_NAME = 'CQI2025_CQX_CONTRIBUTION'
        """)
        table_exists = cur.fetchone()[0] > 0

        row_count = 0
        recent_count = 0
        date_range = None

        if table_exists:
            cur.execute("SELECT COUNT(*) FROM CQI2025_CQX_CONTRIBUTION")
            row_count = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) 
                FROM CQI2025_CQX_CONTRIBUTION
                WHERE PERIODSTART >= DATEADD(day, -7, CURRENT_TIMESTAMP())
            """)
            recent_count = cur.fetchone()[0]

            cur.execute("""
                SELECT 
                    MIN(PERIODSTART) as earliest,
                    MAX(PERIODSTART) as latest
                FROM CQI2025_CQX_CONTRIBUTION
            """)
            date_range = cur.fetchone()

        cur.close()
        conn.close()

        mapping = load_submarket_cluster_mapping()
        mapping_info = {
            'total_submarkets': len(mapping),
            'total_mappings': sum(len(clusters) for clusters in mapping.values()),
            'sample_mappings': dict(list(mapping.items())[:3]) if mapping else {}
        }

        # Check for district CSV files
        district_files = [f for f in os.listdir(
            DISTRICT_CSV_DIR) if f.endswith('.csv') and f != MAPPING_CSV_PATH]

        response_data = {
            'connection': 'success',
            'user': context[0],
            'database': context[1],
            'schema': context[2],
            'table_exists': table_exists,
            'total_rows': row_count,
            'recent_rows_7days': recent_count,
            'csv_mapping': mapping_info,
            'district_files_found': district_files[:5]
        }

        if date_range:
            response_data['date_range'] = {
                'earliest': date_range[0].strftime('%Y-%m-%d %H:%M:%S') if date_range[0] else None,
                'latest': date_range[1].strftime('%Y-%m-%d %H:%M:%S') if date_range[1] else None
            }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Test connection failed: {str(e)}")
        return jsonify({
            'connection': 'failed',
            'error': str(e),
            'error_type': type(e).__name__
        }), 500


@app.route('/api/filters', methods=['GET'])
def get_filter_options():
    """Get available filter options from the database with CSV-based submarket-cluster mapping"""
    try:
        conn = get_snowflake_connection()
        cur = conn.cursor()

        filters = {}

        metric_mapping = {
            'VOICE_CDR_RET_25': 'V-CDR',
            'LTE_IQI_NS_ESO_25': 'NS/ESO',
            'LTE_IQI_RSRP_25': 'Quality RSRP',
            'LTE_IQI_QUALITY_25': 'Quality RSRQ',
            'VOLTE_RAN_ACBACC_25_ALL': 'V-ACC',
            'VOLTE_CDR_MOMT_ACC_25': 'V-ACC-E2E',
            'ALLRAT_DACC_25': 'D-ACC',
            'ALLRAT_DL_TPUT_25': 'DLTPUT',
            'ALLRAT_UL_TPUT_25': 'ULTPUT',
            'ALLRAT_DDR_25': 'D-RET',
            'VOLTE_WIFI_CDR_25': 'WIFI-RET'
        }

        cur.execute("""
            SELECT DISTINCT SUBMKT 
            FROM CQI2025_CQX_CONTRIBUTION 
            WHERE SUBMKT IS NOT NULL 
            ORDER BY SUBMKT
        """)
        db_submarkets = [row[0] for row in cur.fetchall()]

        cur.execute("""
            SELECT DISTINCT CQECLUSTER 
            FROM CQI2025_CQX_CONTRIBUTION 
            WHERE CQECLUSTER IS NOT NULL 
            ORDER BY CQECLUSTER
        """)
        db_clusters = [row[0] for row in cur.fetchall()]

        cur.close()
        conn.close()

        csv_mapping = load_submarket_cluster_mapping()

        filters['submarkets'] = db_submarkets
        filters['cqeClusters'] = db_clusters

        if csv_mapping:
            filters['submarketClusters'] = csv_mapping
            logger.info(
                f"CSV mapping loaded: {len(csv_mapping)} submarkets with cluster relationships")

            db_submarkets_set = set(db_submarkets)
            csv_submarkets = set(csv_mapping.keys())
            unmapped_submarkets = db_submarkets_set - csv_submarkets

            if unmapped_submarkets:
                logger.info(
                    f"Submarkets without CSV mapping (will show all clusters): {unmapped_submarkets}")
        else:
            filters['submarketClusters'] = {}
            logger.info(
                "No CSV mapping file found - all clusters will be available for all submarkets")

        filters['metricNames'] = list(metric_mapping.values())
        filters['metricMapping'] = metric_mapping

        return jsonify(filters)

    except Exception as e:
        logger.error(f"Error fetching filter options: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/districts', methods=['GET'])
def get_districts():
    """Get available districts for a given submarket"""
    try:
        submarket = request.args.get('submarket', '')

        if not submarket:
            return jsonify({'districts': []})

        # Load district mapping from CSV file
        district_mapping = load_district_mapping(submarket)

        # Get unique districts
        districts = sorted(set(district_mapping.values())
                           ) if district_mapping else []

        logger.info(
            f"Returning {len(districts)} districts for submarket: {submarket}")

        return jsonify({'districts': districts})

    except Exception as e:
        logger.error(f"Error fetching districts: {str(e)}")
        return jsonify({'error': str(e), 'districts': []}), 500


@app.route('/api/data', methods=['GET'])
def get_cqi_data():
    """Get CQI data with FOCUSLEV-based contribution and failure values"""
    try:
        submarket = request.args.get('submarket', '')
        district_str = request.args.get('district', '')
        cqe_clusters_str = request.args.get('cqeClusters', '')
        period_start = request.args.get('periodStart', '')
        period_end = request.args.get('periodEnd', '')
        metric_name = request.args.get('metricName', '')
        usid = request.args.get('usid', '')
        sorting_criteria = request.args.get('sortingCriteria', 'contribution')

        # Parse multiple districts
        districts = []
        if district_str:
            districts = [d.strip()
                         for d in district_str.split(',') if d.strip()]

        cqe_clusters = []
        if cqe_clusters_str:
            cqe_clusters = [c.strip()
                            for c in cqe_clusters_str.split(',') if c.strip()]

        # IMPORTANT: Determine FOCUSLEV based on submarket filter
        # If submarket is selected, use FOCUSLEV = 3, otherwise use FOCUSLEV = 0
        focus_level = 3 if submarket else 0

        logger.info(f"Data request with sorting: {sorting_criteria}")
        logger.info(
            f"Using FOCUSLEV: {focus_level} ({'Submarket' if focus_level == 3 else 'National'})")
        logger.info(f"Selected Submarket: {submarket}")
        logger.info(f"Selected Districts: {districts}")
        logger.info(f"Selected CQE Clusters: {cqe_clusters}")

        metric_mapping = {
            'VOICE_CDR_RET_25': 'V-CDR',
            'LTE_IQI_NS_ESO_25': 'NS/ESO',
            'LTE_IQI_RSRP_25': 'Quality RSRP',
            'LTE_IQI_QUALITY_25': 'Quality RSRQ',
            'VOLTE_RAN_ACBACC_25_ALL': 'V-ACC',
            'VOLTE_CDR_MOMT_ACC_25': 'V-ACC-E2E',
            'ALLRAT_DACC_25': 'D-ACC',
            'ALLRAT_DL_TPUT_25': 'DLTPUT',
            'ALLRAT_UL_TPUT_25': 'ULTPUT',
            'ALLRAT_DDR_25': 'D-RET',
            'VOLTE_WIFI_CDR_25': 'WIFI-RET'
        }

        aggregate_all_metrics = not metric_name

        if aggregate_all_metrics:
            # Query with FOCUSLEV filter - extracting numeric part only
            query = """
                SELECT 
                    USID,
                    'ALL' as METRICNAME,
                    AVG(EXTRAFAILURES) as AVG_EXTRAFAILURES,
                    SUM(EXTRAFAILURES) as TOTAL_EXTRAFAILURES,
                    AVG(IDXCONTR) as AVG_IDXCONTR,
                    SUM(IDXCONTR) as TOTAL_IDXCONTR,
                    COUNT(*) as RECORD_COUNT,
                    ANY_VALUE(CQECLUSTER) as CQECLUSTER,
                    ANY_VALUE(SUBMKT) as SUBMKT,
                    AVG(FOCUSAREA_L1CQIACTUAL) as AVG_ACTUAL,
                    AVG(CQITARGET) as AVG_TARGET,
                    MIN(PERIODSTART) as EARLIEST_PERIOD,
                    MAX(PERIODEND) as LATEST_PERIOD,
                    MAX(LEFT(FOCUSLEV, 1)::INT) as FOCUSLEV
                FROM CQI2025_CQX_CONTRIBUTION
                WHERE LEFT(FOCUSLEV, 1)::INT = %s
            """
        else:
            query = """
                SELECT 
                    USID,
                    METRICNAME,
                    AVG(EXTRAFAILURES) as AVG_EXTRAFAILURES,
                    SUM(EXTRAFAILURES) as TOTAL_EXTRAFAILURES,
                    AVG(IDXCONTR) as AVG_IDXCONTR,
                    SUM(IDXCONTR) as TOTAL_IDXCONTR,
                    COUNT(*) as RECORD_COUNT,
                    ANY_VALUE(CQECLUSTER) as CQECLUSTER,
                    ANY_VALUE(SUBMKT) as SUBMKT,
                    AVG(FOCUSAREA_L1CQIACTUAL) as AVG_ACTUAL,
                    AVG(CQITARGET) as AVG_TARGET,
                    MIN(PERIODSTART) as EARLIEST_PERIOD,
                    MAX(PERIODEND) as LATEST_PERIOD,
                    MAX(LEFT(FOCUSLEV, 1)::INT) as FOCUSLEV
                FROM CQI2025_CQX_CONTRIBUTION
                WHERE LEFT(FOCUSLEV, 1)::INT = %s
            """

        # Start with FOCUSLEV parameter
        params = [focus_level]

        allowed_metrics = list(metric_mapping.keys())
        query += f" AND METRICNAME IN ({','.join(['%s'] * len(allowed_metrics))})"
        params.extend(allowed_metrics)

        if submarket:
            query += " AND SUBMKT = %s"
            params.append(submarket)

        if cqe_clusters:
            query += f" AND CQECLUSTER IN ({','.join(['%s'] * len(cqe_clusters))})"
            params.extend(cqe_clusters)

        if period_start:
            query += " AND PERIODSTART >= %s"
            params.append(f"{period_start} 00:00:00")

        if period_end:
            query += " AND PERIODEND <= %s"
            params.append(f"{period_end} 23:59:59")

        if metric_name:
            reverse_mapping = {v: k for k, v in metric_mapping.items()}
            actual_metric = reverse_mapping.get(metric_name, metric_name)
            query += " AND METRICNAME = %s"
            params.append(actual_metric)

        if usid:
            query += " AND USID = %s"
            params.append(usid)

        # Handle district filtering when both submarket and districts are selected
        if districts and submarket:
            district_mapping = load_district_mapping(submarket)
            if district_mapping:
                # Get USIDs that belong to any of the selected districts
                usids_in_districts = [
                    usid for usid, dist in district_mapping.items() if dist in districts]
                if usids_in_districts:
                    query += f" AND USID IN ({','.join(['%s'] * len(usids_in_districts))})"
                    params.extend(usids_in_districts)
                    logger.info(
                        f"Filtering {len(usids_in_districts)} USIDs for districts: {districts}")
                else:
                    logger.warning(
                        f"No USIDs found for districts: {districts}")

        if aggregate_all_metrics:
            query += " GROUP BY USID"
        else:
            query += " GROUP BY USID, METRICNAME"

        if sorting_criteria == 'contribution':
            query += " ORDER BY AVG_IDXCONTR ASC NULLS LAST"
        else:
            query += " ORDER BY TOTAL_EXTRAFAILURES DESC NULLS LAST"

        query += " LIMIT 1000"

        conn = get_snowflake_connection()
        cur = conn.cursor()

        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)

        columns = [desc[0] for desc in cur.description]
        data = cur.fetchall()

        cur.close()
        conn.close()

        # Load district mapping if submarket is selected
        district_mapping = {}
        if submarket:
            district_mapping = load_district_mapping(submarket)
            logger.info(
                f"Loaded {len(district_mapping)} district mappings for submarket: {submarket}")

        result = []
        for row in data:
            record = {}
            for i, col in enumerate(columns):
                value = row[i]

                if col in ['AVG_EXTRAFAILURES', 'TOTAL_EXTRAFAILURES']:
                    record[col] = clean_numeric_value(value)
                elif col in ['AVG_IDXCONTR', 'TOTAL_IDXCONTR']:
                    record[col] = clean_contribution_value(value)
                elif col in ['AVG_ACTUAL', 'AVG_TARGET']:
                    record[col] = clean_numeric_value(value)
                elif col in ['RECORD_COUNT', 'FOCUSLEV']:
                    record[col] = int(value) if value else 0
                elif col in ['EARLIEST_PERIOD', 'LATEST_PERIOD']:
                    if value is not None:
                        if isinstance(value, (datetime, pd.Timestamp)):
                            record[col] = value.isoformat()
                        else:
                            record[col] = None
                    else:
                        record[col] = None
                else:
                    record[col] = value

            if aggregate_all_metrics or record.get('METRICNAME') == 'ALL':
                record['METRIC_DISPLAY'] = 'All'
            elif record.get('METRICNAME') in metric_mapping:
                record['METRIC_DISPLAY'] = metric_mapping[record['METRICNAME']]
            else:
                record['METRIC_DISPLAY'] = record.get('METRICNAME', '')

            record['EXTRAFAILURES'] = record.get('AVG_EXTRAFAILURES', 0)
            record['IDXCONTR'] = record.get('AVG_IDXCONTR', 0)

            # Add district information if available
            if submarket and district_mapping:
                usid_str = str(record.get('USID', ''))
                record['DISTRICT'] = district_mapping.get(usid_str, '-')
                logger.debug(
                    f"USID {usid_str}: District = {record['DISTRICT']}")
            else:
                record['DISTRICT'] = None

            result.append(record)

        logger.info(
            f"Returning {len(result)} records with FOCUSLEV={focus_level}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error fetching CQI data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/summary', methods=['GET'])
def get_summary_stats():
    """Get summary statistics for the dashboard"""
    try:
        conn = get_snowflake_connection()
        cur = conn.cursor()

        end_date = datetime.now().strftime('%Y-%m-%d 23:59:59')
        start_date = (datetime.now() - timedelta(days=2)
                      ).strftime('%Y-%m-%d 00:00:00')

        metric_mapping = {
            'VOICE_CDR_RET_25': 'V-CDR',
            'LTE_IQI_NS_ESO_25': 'NS/ESO',
            'LTE_IQI_RSRP_25': 'Quality RSRP',
            'LTE_IQI_QUALITY_25': 'Quality RSRQ',
            'VOLTE_RAN_ACBACC_25_ALL': 'V-ACC',
            'VOLTE_CDR_MOMT_ACC_25': 'V-ACC-E2E',
            'ALLRAT_DACC_25': 'D-ACC',
            'ALLRAT_DL_TPUT_25': 'DLTPUT',
            'ALLRAT_UL_TPUT_25': 'ULTPUT',
            'ALLRAT_DDR_25': 'D-RET',
            'VOLTE_WIFI_CDR_25': 'WIFI-RET'
        }

        allowed_metrics = list(metric_mapping.keys())

        # Summary always uses FOCUSLEV = 0 (National) - extracting numeric part only
        query = f"""
            SELECT 
                COUNT(DISTINCT USID) as total_usids,
                COUNT(*) as total_records,
                SUM(EXTRAFAILURES) as total_failures,
                AVG(EXTRAFAILURES) as avg_failures,
                MAX(EXTRAFAILURES) as max_failures,
                COUNT(CASE WHEN EXTRAFAILURES > 10000 THEN 1 END) as critical_offenders,
                COUNT(CASE WHEN EXTRAFAILURES BETWEEN 1001 AND 10000 THEN 1 END) as high_offenders,
                COUNT(CASE WHEN EXTRAFAILURES BETWEEN 101 AND 1000 THEN 1 END) as medium_offenders,
                COUNT(CASE WHEN EXTRAFAILURES <= 100 THEN 1 END) as low_offenders
            FROM CQI2025_CQX_CONTRIBUTION
            WHERE PERIODSTART >= %s AND PERIODSTART <= %s
            AND METRICNAME IN ({','.join(['%s'] * len(allowed_metrics))})
            AND LEFT(FOCUSLEV, 1)::INT = 0
        """

        params = [start_date, end_date] + allowed_metrics
        cur.execute(query, params)
        result = cur.fetchone()

        summary = {
            'totalUsids': result[0] or 0,
            'totalRecords': result[1] or 0,
            'totalFailures': clean_numeric_value(result[2]),
            'avgFailures': clean_numeric_value(result[3]),
            'maxFailures': clean_numeric_value(result[4]),
            'criticalOffenders': result[5] or 0,
            'highOffenders': result[6] or 0,
            'mediumOffenders': result[7] or 0,
            'lowOffenders': result[8] or 0,
            'lastUpdated': datetime.now().isoformat()
        }

        cur.close()
        conn.close()

        return jsonify(summary)

    except Exception as e:
        logger.error(f"Error fetching summary stats: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/usid-detail', methods=['GET'])
def get_usid_detail():
    """Get detailed metric data for a specific USID over time - includes both EXTRAFAILURES and IDXCONTR"""
    try:
        usid = request.args.get('usid', '')
        period_start = request.args.get('periodStart', '')
        period_end = request.args.get('periodEnd', '')
        metric_name = request.args.get('metricName', '')

        # For USID detail, we should use the same FOCUSLEV as the main page context
        # Get this from the referrer or default to 0
        submarket = request.args.get('submarket', '')
        focus_level = 3 if submarket else 0

        if not usid:
            return jsonify({'error': 'USID is required'}), 400

        metric_mapping = {
            'VOICE_CDR_RET_25': 'V-CDR',
            'LTE_IQI_NS_ESO_25': 'NS/ESO',
            'LTE_IQI_RSRP_25': 'Quality RSRP',
            'LTE_IQI_QUALITY_25': 'Quality RSRQ',
            'VOLTE_RAN_ACBACC_25_ALL': 'V-ACC',
            'VOLTE_CDR_MOMT_ACC_25': 'V-ACC-E2E',
            'ALLRAT_DACC_25': 'D-ACC',
            'ALLRAT_DL_TPUT_25': 'DLTPUT',
            'ALLRAT_UL_TPUT_25': 'ULTPUT',
            'ALLRAT_DDR_25': 'D-RET',
            'VOLTE_WIFI_CDR_25': 'WIFI-RET'
        }

        # Updated query to include IDXCONTR and FOCUSLEV - extracting numeric part only
        query = """
            SELECT 
                USID,
                METRICNAME,
                DATE(PERIODSTART) as DATE,
                AVG(EXTRAFAILURES) as EXTRAFAILURES,
                AVG(IDXCONTR) as IDXCONTR,
                MAX(CQECLUSTER) as CQECLUSTER,
                MAX(SUBMKT) as SUBMKT,
                MAX(LEFT(FOCUSLEV, 1)::INT) as FOCUSLEV
            FROM CQI2025_CQX_CONTRIBUTION
            WHERE USID = %s
            AND LEFT(FOCUSLEV, 1)::INT = %s
        """

        allowed_metrics = list(metric_mapping.keys())
        query += f" AND METRICNAME IN ({','.join(['%s'] * len(allowed_metrics))})"
        params = [usid, focus_level] + allowed_metrics

        if period_start:
            query += " AND PERIODSTART >= %s"
            params.append(f"{period_start} 00:00:00")

        if period_end:
            query += " AND PERIODEND <= %s"
            params.append(f"{period_end} 23:59:59")

        if metric_name:
            reverse_mapping = {v: k for k, v in metric_mapping.items()}
            actual_metric = reverse_mapping.get(metric_name, metric_name)
            query += " AND METRICNAME = %s"
            params.append(actual_metric)

        query += """
            GROUP BY USID, METRICNAME, DATE(PERIODSTART)
            ORDER BY DATE(PERIODSTART), METRICNAME
        """

        conn = get_snowflake_connection()
        cur = conn.cursor()

        cur.execute(query, params)

        columns = [desc[0] for desc in cur.description]
        data = cur.fetchall()

        cur.close()
        conn.close()

        result = []
        for row in data:
            record = {}
            for i, col in enumerate(columns):
                value = row[i]

                if col == 'EXTRAFAILURES':
                    record[col] = clean_numeric_value(value)
                elif col == 'IDXCONTR':
                    # Use the contribution cleaning function that allows negative values
                    record[col] = clean_contribution_value(value)
                elif col == 'FOCUSLEV':
                    record[col] = int(value) if value else 0
                elif col == 'DATE':
                    if value:
                        if hasattr(value, 'isoformat'):
                            record['PERIODSTART'] = value.isoformat()
                        else:
                            record['PERIODSTART'] = str(value)
                    else:
                        record['PERIODSTART'] = None
                else:
                    record[col] = value

            if record.get('METRICNAME') in metric_mapping:
                record['METRIC_DISPLAY'] = metric_mapping[record['METRICNAME']]
            else:
                record['METRIC_DISPLAY'] = record.get('METRICNAME', '')

            result.append(record)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error fetching USID detail data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/market-targets', methods=['GET'])
def get_market_targets():
    """Get CQX targets data for market level visualization"""
    try:
        submarket = request.args.get('submarket', '')
        metric_filter = request.args.get('metric', '')
        week_range = int(request.args.get('weekRange', 12))

        if not submarket:
            return jsonify({'error': 'Submarket is required'}), 400

        logger.info(
            f"Fetching market targets for submarket: {submarket}, weeks: {week_range}")

        # Build the METRICREPORTINGKEY pattern for level 3 submarket
        # Format: "East,Florida,Tampa" where Tampa is the submarket
        reporting_key_pattern = f"%,{submarket}"

        query = """
            SELECT 
                WEEK,
                METRICREPORTINGLEVEL,
                METRICREPORTINGKEY,
                METRICNAME,
                RAW_GREEN_TARGET,
                RAW_YELLOW_TARGET,
                RAW_YOY_TARGET,
                CQI_GREEN_TARGET,
                CQI_YELLOW_TARGET,
                CQI_YOY_TARGET,
                CQI_METRICNAME,
                SCORECARD
            FROM PRD_MOBILITY.PRD_MOBILITYSCORECARD_VIEWS.CQI2025_TARGETS
            WHERE METRICREPORTINGLEVEL = 3
            AND METRICREPORTINGKEY LIKE %s
        """

        params = [reporting_key_pattern]

        if metric_filter:
            query += " AND CQI_METRICNAME = %s"
            params.append(metric_filter)

        # Fix: Use proper date arithmetic for WEEK column that contains dates
        # Get data from the last N weeks
        query += """
            AND WEEK >= DATEADD(WEEK, %s, CURRENT_DATE())
            AND WEEK <= CURRENT_DATE()
        """
        params.append(-week_range)

        query += " ORDER BY WEEK, CQI_METRICNAME"

        logger.info(f"Executing query with pattern: {reporting_key_pattern}")
        logger.info(f"Week range: last {week_range} weeks from current date")

        conn = get_snowflake_connection()
        cur = conn.cursor()

        cur.execute(query, params)

        columns = [desc[0] for desc in cur.description]
        data = cur.fetchall()

        logger.info(f"Retrieved {len(data)} rows from database")

        cur.close()
        conn.close()

        result = []
        for row in data:
            record = {}
            for i, col in enumerate(columns):
                value = row[i]

                # Handle different data types
                if col in ['RAW_GREEN_TARGET', 'RAW_YELLOW_TARGET', 'RAW_YOY_TARGET',
                           'CQI_GREEN_TARGET', 'CQI_YELLOW_TARGET', 'CQI_YOY_TARGET']:
                    record[col] = float(value) if value is not None else None
                elif col == 'WEEK':
                    # Convert date to string format (YYYY-MM-DD)
                    if value:
                        if hasattr(value, 'strftime'):
                            # Keep as date format for better display
                            record[col] = value.strftime('%Y-%m-%d')
                        else:
                            record[col] = str(value)
                    else:
                        record[col] = None
                else:
                    record[col] = value

            # Parse the region and state from METRICREPORTINGKEY
            if record.get('METRICREPORTINGKEY'):
                parts = record['METRICREPORTINGKEY'].split(',')
                if len(parts) >= 3:
                    record['REGION'] = parts[0]
                    record['STATE'] = parts[1]
                    record['SUBMARKET'] = parts[2]

            result.append(record)

        logger.info(
            f"Returning {len(result)} records for submarket: {submarket}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error fetching market targets: {str(e)}")
        logger.error(f"Query pattern was: %,{submarket}")
        return jsonify({'error': str(e), 'details': 'Check Flask console for more information'}), 500


if __name__ == '__main__':
    print("🚀 Starting CQI Dashboard API Server with FOCUSLEV Support...")
    print("📊 FOCUSLEV Mapping: 0=National, 1=Regional, 2=Market, 3=Submarket")
    print(f"📄 Looking for mapping file: {MAPPING_CSV_PATH}")
    print(f"📁 District CSV files directory: {DISTRICT_CSV_DIR}")

    if os.path.exists(MAPPING_CSV_PATH):
        print(f"✅ Mapping file found: {MAPPING_CSV_PATH}")
        mapping = load_submarket_cluster_mapping()
        if mapping:
            print(f"📊 Loaded mappings for {len(mapping)} submarkets")
            print(
                f"   Total cluster mappings: {sum(len(clusters) for clusters in mapping.values())}")
    else:
        print(
            f"⚠️  Mapping file not found. A sample file will be created at: {MAPPING_CSV_PATH}")
        print("   Please update it with your actual submarket-cluster mappings")

    # Check for district CSV files
    district_files = [f for f in os.listdir(
        DISTRICT_CSV_DIR) if f.endswith('.csv') and f != MAPPING_CSV_PATH]
    if district_files:
        print(f"📊 Found {len(district_files)} district CSV files:")
        for f in district_files[:5]:
            print(f"   - {f}")
        if len(district_files) > 5:
            print(f"   ... and {len(district_files) - 5} more")
    else:
        print(
            "⚠️  No district CSV files found. Add {SUBMARKET}.csv files for district mapping")

    if validate_config():
        print("✅ Configuration loaded from environment variables")
        print(
            f"📊 Connecting to Snowflake as user: {SNOWFLAKE_CONFIG.get('user')}")
    else:
        print("❌ Configuration incomplete. Please check environment variables.")

    print("\n📡 API will be available at: http://localhost:5000")
    print("✨ Features: FOCUSLEV-based filtering + CSV-based Submarket-Cluster filtering + District mapping!")
    print("-" * 50)

    app.run(debug=False, host='0.0.0.0', port=5000)
