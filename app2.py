"""
Merged CQI Dashboard + 5G Traffic Dashboard Flask API
Combines both dashboards into a single application
"""

from dotenv import load_dotenv
import json
import logging
from functools import lru_cache
import os
from datetime import datetime, timedelta
import csv
import re
import numpy as np
import pandas as pd
import snowflake.connector as sc
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress verbose logs
logging.getLogger('snowflake.connector').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Snowflake connection parameters
SNOWFLAKE_CONFIG = {
    'account': os.getenv('SNOWFLAKE_ACCOUNT', 'nsasprd.east-us-2.privatelink'),
    'user': os.getenv('SNOWFLAKE_USER'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', 'USR_REPORTING_WH'),
    'database': os.getenv('SNOWFLAKE_DATABASE', 'PRD_MOBILITY'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA', 'PRD_MOBILITYSCORECARD_VIEWS')
}

# Handle authentication
if os.getenv('SNOWFLAKE_PRIVATE_KEY_PATH'):
    SNOWFLAKE_CONFIG['private_key_file'] = os.getenv(
        'SNOWFLAKE_PRIVATE_KEY_PATH')
    if os.getenv('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE'):
        SNOWFLAKE_CONFIG['private_key_file_pwd'] = os.getenv(
            'SNOWFLAKE_PRIVATE_KEY_PASSPHRASE')
elif os.getenv('SNOWFLAKE_PASSWORD'):
    SNOWFLAKE_CONFIG['password'] = os.getenv('SNOWFLAKE_PASSWORD')

# CSV mapping files
MAPPING_CSV_PATH = 'submkt_cqecluster_mapping.csv'
DISTRICT_CSV_DIR = '.'


def get_snowflake_connection():
    """Create and return a Snowflake connection"""
    try:
        conn = sc.connect(**SNOWFLAKE_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {str(e)}")
        raise

# ============================================================================
# CQX DASHBOARD UTILITY FUNCTIONS
# ============================================================================


def clean_numeric_value(value):
    """Clean numeric values, handling NaN, Infinity, None, and negative values"""
    if value is None or pd.isna(value):
        return 0
    if isinstance(value, (float, np.floating)):
        if np.isinf(value) or np.isnan(value) or value < 0:
            return 0
        return int(value) if value == int(value) else float(value)
    if isinstance(value, (int, np.integer)):
        return max(0, int(value))
    try:
        num_val = float(value)
        if np.isnan(num_val) or np.isinf(num_val) or num_val < 0:
            return 0
        return int(num_val) if num_val == int(num_val) else num_val
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
        return 0 if np.isnan(num_val) or np.isinf(num_val) else num_val
    except (ValueError, TypeError):
        return 0


def load_submarket_cluster_mapping():
    """Load the submarket-cluster mapping from CSV file"""
    mapping = {}
    if not os.path.exists(MAPPING_CSV_PATH):
        return mapping
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
        logger.info(f"Loaded mapping for {len(mapping)} submarkets")
    except Exception as e:
        logger.error(f"Error reading mapping CSV: {str(e)}")
    return mapping


def load_district_mapping(submarket):
    """Load district mapping for a specific submarket from CSV file"""
    if not submarket:
        return {}
    district_file = os.path.join(DISTRICT_CSV_DIR, f"{submarket}.csv")
    if not os.path.exists(district_file):
        return {}
    district_mapping = {}
    try:
        with open(district_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        usid = str(row[0]).strip()
                        district = str(row[1]).strip()
                        district_mapping[usid] = district
                    except (ValueError, IndexError):
                        continue
        logger.info(
            f"Loaded {len(district_mapping)} district mappings for '{submarket}'")
    except Exception as e:
        logger.error(f"Error reading district file: {str(e)}")
    return district_mapping

# ============================================================================
# 5G TRAFFIC DASHBOARD UTILITY FUNCTIONS
# ============================================================================


def parse_cell_face_and_sector(nrcell):
    """Parse cell face (prefix) and sector letter from NRCELL name"""
    match = re.match(r'([^_]+)_(N\d{3}([A-Z]))_(\d)', nrcell)
    if match:
        return match.group(1), match.group(3)
    return None, None


def parse_band(nrcell):
    """Parse band information from NRCELL name"""
    match = re.search(r'_(N\d{3}[A-Z]?)_', nrcell)
    if not match:
        return 'Unknown'

    band_str = match.group(1)
    band_match = re.match(r'N(\d{3})([A-Z]?)', band_str)
    if not band_match:
        return 'Unknown'

    band_num = band_match.group(1)
    char = band_match.group(2)

    suffix_match = re.search(r'_' + re.escape(band_str) + r'_(\d)', nrcell)
    suffix = suffix_match.group(1) if suffix_match else None

    if band_num == '077' and char and suffix:
        if suffix == '1':
            return 'CBand'
        elif suffix == '2':
            return 'DOD'

    band_number = int(band_num)
    return f'Band {band_number}'


def expand_band_selection(selected_bands):
    """Expand Band 77 to include CBand and DOD"""
    expanded = []
    for band in selected_bands:
        expanded.append(band)
        if band == 'Band 77':
            if 'CBand' not in selected_bands:
                expanded.append('CBand')
            if 'DOD' not in selected_bands:
                expanded.append('DOD')
    return list(set(expanded))


def fetch_tput_data(start_date, end_date, nrcell_filter='ILXN%', gnodeb_filter=None):
    """Fetch TPUT data from Snowflake"""
    gnodeb_clause = f"AND GNODEB = '{gnodeb_filter}'" if gnodeb_filter else ""

    sql_query = f"""
    WITH
    NR_RTB_NRCELL_DY AS
    (SELECT DATETIMELOCAL, GNODEB, NRCELL,
    (SUM(NVL(NR_DL_MAC_KBYTES,0))/1000000) :: DECIMAL(38,4) AS DL_ACK_MAC_VOL_GB,
    (SUM(NVL(NR_UL_MAC_KBYTES,0))/1000000) :: DECIMAL(38,4) AS UL_ACK_MAC_VOL_GB,
    DIV0(SUM(NVL(NR_DL_DRB_TPUT_NUM_KBITS,0))/1000 , SUM(NVL(NR_DL_DRB_TPUT_DEN_SECS,0))) :: DECIMAL(38,4) AS DL_DRB_THPUT,
    DIV0(SUM(NVL(NR_UL_TPUT_NUM_KBITS,0))/1000 , SUM(NVL(NR_UL_TPUT_DEN_SECS,0))) :: DECIMAL(38,4) AS UL_DRB_THPUT,
    DIV0(SUM(NVL(NR_DL_CELL_TPUT_NUM_KBITS,0))/1000 , SUM(NVL(NR_DL_CELL_TPUT_DEN_SECS,0))) :: DECIMAL(38,4) AS DL_MAC_CELL_THPUT,
    DIV0(SUM(NVL(NR_UL_CELL_TPUT_NUM_KBITS,0))/1000 , SUM(NVL(NR_UL_CELL_TPUT_DEN_SECS,0))) :: DECIMAL(38,4) AS UL_MAC_CELL_THPUT,
    DIV0(SUM(NVL(NR_DL_TPUT_SINGLE_TTI_NUM_KBITS,0))/1000 , SUM(NVL(NR_DL_TPUT_SINGLE_TTI_DEN_SECS,0))) :: DECIMAL(38,4) AS DL_SINGLE_TTI_DRB_THPUT,
    DIV0(SUM(NVL(NR_DL_DRB_TOT_TPUT_NUM_KBITS,0))/1000 , SUM(NVL(NR_DL_DRB_TOT_TPUT_DEN_SECS,0))) :: DECIMAL(38,4) AS DL_TOTAL_DRB_THPUT,
    DIV0(0.001 * SUM(NR_NSA_DL_DRB_TPUT_NUM_KBITS),SUM(NR_NSA_DL_DRB_TPUT_DEN_SECS)) :: DECIMAL(38,4) AS NR_NSA_DL_DRB_TPUT,
    DIV0(0.001 * SUM(NR_SA_DL_DRB_TPUT_NUM_KBITS),SUM(NR_SA_DL_DRB_TPUT_DEN_SECS)) :: DECIMAL(38,4) AS NR_SA_DL_DRB_TPUT,
    SUM(NR_DL_DRB_TPUT_NUM_KBITS) AS NR_DL_DRB_TPUT_NUM_KBITS, 
    SUM(NR_DL_DRB_TPUT_DEN_SECS) AS NR_DL_DRB_TPUT_DEN_SECS,
    SUM(NR_UL_TPUT_NUM_KBITS) AS NR_UL_TPUT_NUM_KBITS, 
    SUM(NR_UL_TPUT_DEN_SECS) AS NR_UL_TPUT_DEN_SECS
    FROM PRD_MOBILITY.PRD_MOBILITYSCORECARD_VIEWS.NR_RTB_NRCELL_DY
    WHERE DATETIMELOCAL BETWEEN TO_TIMESTAMP('{start_date}') AND TO_TIMESTAMP('{end_date}')
    AND NRCELL LIKE '{nrcell_filter}'
    {gnodeb_clause}
    GROUP BY GNODEB, NRCELL, DATETIMELOCAL),
    NR_RTB_GNB_NRCELL_DY AS 
    (SELECT DATETIMELOCAL, GNODEB, NRCELL,
    DIV0(SUM(NVL(OV_ENDC_DL_DRB_TPUT_NUM_KBITS,0))/1000 , SUM(NVL(OV_ENDC_DL_DRB_TPUT_DEN_SECS,0))) :: DECIMAL(38,4) AS OV_ENDC_DL_DRB_TPUT,
    DIV0(SUM(NVL(ENDC_DL_DRB_TPUT_5GCOV_NUM_KBITS, 0)) / 1000, SUM(NVL(ENDC_DL_DRB_TPUT_5GCOV_DEN_SECS, 0))) :: DECIMAL(38, 4) AS ENDC_DL_DRB_TPUT_5GCOV
    FROM PRD_MOBILITY.PRD_MOBILITYSCORECARD_VIEWS.NR_RTB_GNB_NRCELL_DY
    WHERE DATETIMELOCAL BETWEEN TO_TIMESTAMP('{start_date}') AND TO_TIMESTAMP('{end_date}')
    AND NRCELL LIKE '{nrcell_filter}'
    {gnodeb_clause}
    GROUP BY GNODEB, NRCELL, DATETIMELOCAL)
    SELECT 
        TO_CHAR(A.DATETIMELOCAL,'YYYY-MM-DD') AS DATE_TIME_ID, 
        A.GNODEB, 
        A.NRCELL AS NR_CELL,
        A.DL_ACK_MAC_VOL_GB,
        A.UL_ACK_MAC_VOL_GB,
        A.DL_DRB_THPUT,
        A.UL_DRB_THPUT,
        A.DL_MAC_CELL_THPUT,
        A.UL_MAC_CELL_THPUT,
        A.DL_SINGLE_TTI_DRB_THPUT,
        A.DL_TOTAL_DRB_THPUT,
        B.OV_ENDC_DL_DRB_TPUT, 
        B.ENDC_DL_DRB_TPUT_5GCOV,
        A.NR_NSA_DL_DRB_TPUT, 
        A.NR_SA_DL_DRB_TPUT
    FROM NR_RTB_NRCELL_DY A
    LEFT OUTER JOIN NR_RTB_GNB_NRCELL_DY B 
        ON (A.DATETIMELOCAL = B.DATETIMELOCAL 
            AND A.GNODEB = B.GNODEB 
            AND A.NRCELL = B.NRCELL)
    ORDER BY A.GNODEB, A.NRCELL, A.DATETIMELOCAL
    """

    conn = get_snowflake_connection()
    df = pd.read_sql(sql_query, conn)
    conn.close()

    numeric_cols = ['DL_DRB_THPUT', 'UL_DRB_THPUT', 'DL_MAC_CELL_THPUT', 'UL_MAC_CELL_THPUT',
                    'DL_SINGLE_TTI_DRB_THPUT', 'DL_TOTAL_DRB_THPUT', 'OV_ENDC_DL_DRB_TPUT',
                    'ENDC_DL_DRB_TPUT_5GCOV', 'NR_NSA_DL_DRB_TPUT', 'NR_SA_DL_DRB_TPUT']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['BAND'] = df['NR_CELL'].apply(parse_band)

    return df

# ============================================================================
# HTML ROUTES
# ============================================================================


@app.route('/')
def index():
    """Main CQX dashboard"""
    return app.send_static_file('index.html')


@app.route('/5g')
def fiveg_dashboard():
    """5G Traffic dashboard"""
    return app.send_static_file('5g_dashboard.html')


@app.route('/5g/gnodeb/<gnodeb>')
def fiveg_gnodeb_detail(gnodeb):
    """5G GNODEB detail page"""
    start_date = request.args.get(
        'start_date', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    end_date = request.args.get(
        'end_date', datetime.now().strftime('%Y-%m-%d'))
    cell_face = request.args.get('cell_face', '')
    sector = request.args.get('sector', '')
    cell_filter = request.args.get('cell_filter', 'ILXN%')

    return render_template('5g_gnodeb_detail.html',
                           gnodeb=gnodeb,
                           start_date=start_date,
                           end_date=end_date,
                           cell_face=cell_face,
                           sector=sector,
                           cell_filter=cell_filter)

# ============================================================================
# CQX DASHBOARD API ENDPOINTS
# ============================================================================


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'cqx_enabled': True,
        'fiveg_enabled': True,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/filters', methods=['GET'])
def get_filter_options():
    """Get available filter options for CQX dashboard"""
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

        cur.execute(
            "SELECT DISTINCT SUBMKT FROM CQI2025_CQX_CONTRIBUTION WHERE SUBMKT IS NOT NULL ORDER BY SUBMKT")
        filters['submarkets'] = [row[0] for row in cur.fetchall()]

        cur.execute(
            "SELECT DISTINCT CQECLUSTER FROM CQI2025_CQX_CONTRIBUTION WHERE CQECLUSTER IS NOT NULL ORDER BY CQECLUSTER")
        filters['cqeClusters'] = [row[0] for row in cur.fetchall()]

        cur.close()
        conn.close()

        csv_mapping = load_submarket_cluster_mapping()
        filters['submarketClusters'] = csv_mapping if csv_mapping else {}
        filters['metricNames'] = list(metric_mapping.values())
        filters['metricMapping'] = metric_mapping

        return jsonify(filters)
    except Exception as e:
        logger.error(f"Error fetching filter options: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/districts', methods=['GET'])
def get_districts():
    """Get available districts for a submarket"""
    try:
        submarket = request.args.get('submarket', '')
        if not submarket:
            return jsonify({'districts': []})

        district_mapping = load_district_mapping(submarket)
        districts = sorted(set(district_mapping.values())
                           ) if district_mapping else []

        return jsonify({'districts': districts})
    except Exception as e:
        logger.error(f"Error fetching districts: {str(e)}")
        return jsonify({'error': str(e), 'districts': []}), 500


@app.route('/api/data', methods=['GET'])
def get_cqi_data():
    """Get CQI data with district information"""
    try:
        submarket = request.args.get('submarket', '')
        district_str = request.args.get('district', '')
        cqe_clusters_str = request.args.get('cqeClusters', '')
        period_start = request.args.get('periodStart', '')
        period_end = request.args.get('periodEnd', '')
        metric_name = request.args.get('metricName', '')
        usid = request.args.get('usid', '')
        sorting_criteria = request.args.get('sortingCriteria', 'contribution')

        districts = [d.strip() for d in district_str.split(',')
                     if d.strip()] if district_str else []
        cqe_clusters = [c.strip() for c in cqe_clusters_str.split(
            ',') if c.strip()] if cqe_clusters_str else []

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
            query = """
                SELECT 
                    USID, 'ALL' as METRICNAME,
                    AVG(EXTRAFAILURES) as AVG_EXTRAFAILURES,
                    SUM(EXTRAFAILURES) as TOTAL_EXTRAFAILURES,
                    AVG(IDXCONTR) as AVG_IDXCONTR,
                    SUM(IDXCONTR) as TOTAL_IDXCONTR,
                    COUNT(*) as RECORD_COUNT,
                    MAX(VENDOR) as VENDOR,
                    MAX(CQECLUSTER) as CQECLUSTER,
                    MAX(SUBMKT) as SUBMKT,
                    MIN(PERIODSTART) as EARLIEST_PERIOD,
                    MAX(PERIODEND) as LATEST_PERIOD
                FROM CQI2025_CQX_CONTRIBUTION WHERE 1=1
            """
        else:
            query = """
                SELECT 
                    USID, METRICNAME,
                    AVG(EXTRAFAILURES) as AVG_EXTRAFAILURES,
                    SUM(EXTRAFAILURES) as TOTAL_EXTRAFAILURES,
                    AVG(IDXCONTR) as AVG_IDXCONTR,
                    SUM(IDXCONTR) as TOTAL_IDXCONTR,
                    COUNT(*) as RECORD_COUNT,
                    MAX(VENDOR) as VENDOR,
                    MAX(CQECLUSTER) as CQECLUSTER,
                    MAX(SUBMKT) as SUBMKT,
                    MIN(PERIODSTART) as EARLIEST_PERIOD,
                    MAX(PERIODEND) as LATEST_PERIOD
                FROM CQI2025_CQX_CONTRIBUTION WHERE 1=1
            """

        allowed_metrics = list(metric_mapping.keys())
        query += f" AND METRICNAME IN ({','.join(['%s'] * len(allowed_metrics))})"
        params = allowed_metrics.copy()

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

        if districts and submarket:
            district_mapping = load_district_mapping(submarket)
            if district_mapping:
                usids_in_districts = [
                    u for u, d in district_mapping.items() if d in districts]
                if usids_in_districts:
                    query += f" AND USID IN ({','.join(['%s'] * len(usids_in_districts))})"
                    params.extend(usids_in_districts)

        query += " GROUP BY USID" if aggregate_all_metrics else " GROUP BY USID, METRICNAME"
        query += " ORDER BY AVG_IDXCONTR ASC NULLS LAST" if sorting_criteria == 'contribution' else " ORDER BY TOTAL_EXTRAFAILURES DESC NULLS LAST"
        query += " LIMIT 1000"

        conn = get_snowflake_connection()
        cur = conn.cursor()
        cur.execute(query, params)

        columns = [desc[0] for desc in cur.description]
        data = cur.fetchall()

        cur.close()
        conn.close()

        district_mapping = load_district_mapping(
            submarket) if submarket else {}

        result = []
        for row in data:
            record = {}
            for i, col in enumerate(columns):
                value = row[i]

                if col in ['AVG_EXTRAFAILURES', 'TOTAL_EXTRAFAILURES']:
                    record[col] = clean_numeric_value(value)
                elif col in ['AVG_IDXCONTR', 'TOTAL_IDXCONTR']:
                    record[col] = clean_contribution_value(value)
                elif col == 'RECORD_COUNT':
                    record[col] = int(value) if value else 0
                elif col in ['EARLIEST_PERIOD', 'LATEST_PERIOD']:
                    record[col] = value.isoformat() if value and hasattr(
                        value, 'isoformat') else None
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

            if submarket and district_mapping:
                usid_str = str(record.get('USID', ''))
                record['DISTRICT'] = district_mapping.get(usid_str, '-')
            else:
                record['DISTRICT'] = None

            result.append(record)

        return jsonify(result)
    except Exception as e:
        logger.error(f"Error fetching CQI data: {str(e)}")
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

@app.route('/api/market-targets', methods=['GET'])
def get_market_targets():
    """Get CQI targets data for market level visualization"""
    try:
        submarket = request.args.get('submarket', '')
        metric_filter = request.args.get('metric', '')
        week_range = int(request.args.get('weekRange', 12))

        if not submarket:
            return jsonify({'error': 'Submarket is required'}), 400

        reporting_key_pattern = f"%,{submarket}"

        query = """
            SELECT 
                WEEK, METRICREPORTINGLEVEL, METRICREPORTINGKEY, METRICNAME,
                RAW_GREEN_TARGET, RAW_YELLOW_TARGET, RAW_YOY_TARGET,
                CQI_GREEN_TARGET, CQI_YELLOW_TARGET, CQI_YOY_TARGET,
                CQI_METRICNAME, SCORECARD
            FROM PRD_MOBILITY.PRD_MOBILITYSCORECARD_VIEWS.CQI2025_TARGETS
            WHERE METRICREPORTINGLEVEL = 3 AND METRICREPORTINGKEY LIKE %s
        """

        params = [reporting_key_pattern]

        if metric_filter:
            query += " AND CQI_METRICNAME = %s"
            params.append(metric_filter)

        query += """
            AND WEEK >= (
                SELECT DATEADD(WEEK, -%s, MAX(WEEK))
                FROM PRD_MOBILITY.PRD_MOBILITYSCORECARD_VIEWS.CQI2025_TARGETS
                WHERE METRICREPORTINGLEVEL = 3
            )
        """
        params.append(week_range)

        query += " ORDER BY WEEK, CQI_METRICNAME"

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

                if col in ['RAW_GREEN_TARGET', 'RAW_YELLOW_TARGET', 'RAW_YOY_TARGET',
                           'CQI_GREEN_TARGET', 'CQI_YELLOW_TARGET', 'CQI_YOY_TARGET']:
                    record[col] = float(value) if value is not None else None
                elif col == 'WEEK':
                    record[col] = value.strftime(
                        '%Y-W%U') if value and hasattr(value, 'strftime') else str(value) if value else None
                else:
                    record[col] = value

            if record.get('METRICREPORTINGKEY'):
                parts = record['METRICREPORTINGKEY'].split(',')
                if len(parts) >= 3:
                    record['REGION'] = parts[0]
                    record['STATE'] = parts[1]
                    record['SUBMARKET'] = parts[2]

            result.append(record)

        return jsonify(result)
    except Exception as e:
        logger.error(f"Error fetching market targets: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# 5G TRAFFIC DASHBOARD API ENDPOINTS
# ============================================================================


@app.route('/api/5g/tput', methods=['GET'])
def get_5g_tput_data():
    """Main API endpoint for 5G TPUT data"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        cell_filter = request.args.get('cell_filter', 'ILXN%')
        top_count = int(request.args.get('top_count', 10))
        bands_param = request.args.get('bands', '')
        sectors_param = request.args.get('sectors', '')

        if not start_date or not end_date:
            return jsonify({'error': 'Missing required parameters'}), 400

        selected_bands = [b.strip() for b in bands_param.split(
            ',') if b.strip()] if bands_param else []
        selected_sectors = [s.strip() for s in sectors_param.split(
            ',') if s.strip()] if sectors_param else []

        if selected_bands:
            selected_bands = expand_band_selection(selected_bands)

        start_datetime = f"{start_date} 00:00:00"
        end_datetime = f"{end_date} 23:59:59"

        df = fetch_tput_data(start_datetime, end_datetime, cell_filter)

        if df.empty:
            return jsonify({'error': 'No data found'}), 404

        df['CELL_FACE'] = df['NR_CELL'].apply(
            lambda x: parse_cell_face_and_sector(x)[0])
        df['SECTOR'] = df['NR_CELL'].apply(
            lambda x: parse_cell_face_and_sector(x)[1])

        available_bands = sorted(df['BAND'].unique().tolist())
        available_sectors = sorted(
            df[df['SECTOR'].notna()]['SECTOR'].unique().tolist())

        if selected_bands:
            df = df[df['BAND'].isin(selected_bands)]
        if selected_sectors:
            df = df[df['SECTOR'].isin(selected_sectors)]

        summary = {
            'total_records': len(df),
            'unique_cells': df['NR_CELL'].nunique(),
            'avg_dl_drb': float(df['DL_DRB_THPUT'].mean()),
            'avg_ul_drb': float(df['UL_DRB_THPUT'].mean())
        }

        trend_data = df.groupby('DATE_TIME_ID').agg({
            'DL_DRB_THPUT': 'mean',
            'UL_DRB_THPUT': 'mean',
            'DL_MAC_CELL_THPUT': 'mean',
            'UL_MAC_CELL_THPUT': 'mean'
        }).reset_index()

        trending = {
            'dates': trend_data['DATE_TIME_ID'].tolist(),
            'dl_drb': trend_data['DL_DRB_THPUT'].round(2).tolist(),
            'ul_drb': trend_data['UL_DRB_THPUT'].round(2).tolist(),
            'dl_mac_cell': trend_data['DL_MAC_CELL_THPUT'].round(2).tolist(),
            'ul_mac_cell': trend_data['UL_MAC_CELL_THPUT'].round(2).tolist()
        }

        cell_stats = df.groupby(['NR_CELL', 'GNODEB', 'BAND']).agg({
            'DL_DRB_THPUT': 'mean',
            'UL_DRB_THPUT': 'mean',
            'DL_MAC_CELL_THPUT': 'mean',
            'DATE_TIME_ID': 'count'
        }).reset_index()

        cell_stats.columns = ['NR_CELL', 'GNODEB', 'BAND',
                              'avg_dl_drb', 'avg_ul_drb', 'avg_dl_cell', 'days_monitored']

        zero_days = df[df['DL_DRB_THPUT'] == 0].groupby(
            'NR_CELL').size().to_dict()
        cell_stats['zero_days'] = cell_stats['NR_CELL'].map(
            zero_days).fillna(0).astype(int)
        cell_stats = cell_stats[cell_stats['zero_days'] < 5]

        top_offenders = cell_stats.nsmallest(top_count, 'avg_dl_drb')
        offenders_list = top_offenders.to_dict('records')

        for offender in offenders_list:
            for key in ['avg_dl_drb', 'avg_ul_drb', 'avg_dl_cell']:
                offender[key] = float(offender[key])

        return jsonify({
            'summary': summary,
            'trending': trending,
            'top_offenders': offenders_list,
            'available_bands': available_bands,
            'available_sectors': available_sectors
        })
    except Exception as e:
        logger.error(f"Error in 5G TPUT endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/5g/delta-offenders', methods=['GET'])
def get_5g_delta_offenders():
    """API endpoint for 5G traffic delta offenders"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        cell_filter = request.args.get('cell_filter', 'ILXN%')
        top_count = int(request.args.get('top_count', 10))
        bands_param = request.args.get('bands', '')
        sectors_param = request.args.get('sectors', '')

        if not start_date or not end_date:
            return jsonify({'error': 'Missing required parameters'}), 400

        selected_bands = [b.strip() for b in bands_param.split(
            ',') if b.strip()] if bands_param else []
        selected_sectors = [s.strip() for s in sectors_param.split(
            ',') if s.strip()] if sectors_param else []

        if selected_bands:
            selected_bands = expand_band_selection(selected_bands)

        start_datetime = f"{start_date} 00:00:00"
        end_datetime = f"{end_date} 23:59:59"

        df = fetch_tput_data(start_datetime, end_datetime, cell_filter)

        if df.empty:
            return jsonify({'error': 'No data found'}), 404

        df['CELL_FACE'] = df['NR_CELL'].apply(
            lambda x: parse_cell_face_and_sector(x)[0])
        df['SECTOR'] = df['NR_CELL'].apply(
            lambda x: parse_cell_face_and_sector(x)[1])

        if selected_bands:
            df = df[df['BAND'].isin(selected_bands)]
        if selected_sectors:
            df = df[df['SECTOR'].isin(selected_sectors)]

        df['TOTAL_VOL'] = df['DL_ACK_MAC_VOL_GB'] + df['UL_ACK_MAC_VOL_GB']

        face_stats = df.groupby(['GNODEB', 'CELL_FACE', 'SECTOR', 'BAND']).agg({
            'TOTAL_VOL': 'sum',
            'NR_CELL': 'first'
        }).reset_index()

        delta_data = []

        for (gnodeb, face, sector), group in face_stats.groupby(['GNODEB', 'CELL_FACE', 'SECTOR']):
            if len(group) > 1:
                max_vol = group['TOTAL_VOL'].max()
                min_vol = group['TOTAL_VOL'].min()
                avg_vol = group['TOTAL_VOL'].mean()
                delta = max_vol - min_vol

                max_vol_row = group.loc[group['TOTAL_VOL'].idxmax()]
                max_vol_cell = max_vol_row['NR_CELL']
                max_vol_band = max_vol_row['BAND']

                min_vol_row = group.loc[group['TOTAL_VOL'].idxmin()]
                min_vol_cell = min_vol_row['NR_CELL']
                min_vol_band = min_vol_row['BAND']

                band_details = []
                for _, row in group.iterrows():
                    band_details.append({
                        'band': row['BAND'],
                        'volume': float(row['TOTAL_VOL']),
                        'cell': row['NR_CELL']
                    })

                delta_data.append({
                    'gnodeb': gnodeb,
                    'cell_face': face,
                    'max_vol_cell': max_vol_cell,
                    'max_vol_band': max_vol_band,
                    'min_vol_cell': min_vol_cell,
                    'min_vol_band': min_vol_band,
                    'sector': sector,
                    'band_count': len(group),
                    'max_vol': float(max_vol),
                    'min_vol': float(min_vol),
                    'avg_vol': float(avg_vol),
                    'delta': float(delta),
                    'bands': band_details
                })

        delta_df = pd.DataFrame(delta_data)
        if not delta_df.empty:
            delta_df = delta_df.sort_values(
                'delta', ascending=False).head(top_count)
            top_delta_offenders = delta_df.to_dict('records')
        else:
            top_delta_offenders = []

        return jsonify({'top_delta_offenders': top_delta_offenders})
    except Exception as e:
        logger.error(f"Error in 5G delta offenders: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/5g/gnodeb-comparison', methods=['GET'])
def get_5g_gnodeb_comparison():
    """API endpoint for 5G GNODEB comparison"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        cell_filter = request.args.get('cell_filter', 'ILXN%')
        bands_param = request.args.get('bands', '')
        sectors_param = request.args.get('sectors', '')

        if not start_date or not end_date:
            return jsonify({'error': 'Missing required parameters'}), 400

        selected_bands = [b.strip() for b in bands_param.split(
            ',') if b.strip()] if bands_param else []
        selected_sectors = [s.strip() for s in sectors_param.split(
            ',') if s.strip()] if sectors_param else []

        if selected_bands:
            selected_bands = expand_band_selection(selected_bands)

        start_datetime = f"{start_date} 00:00:00"
        end_datetime = f"{end_date} 23:59:59"

        df = fetch_tput_data(start_datetime, end_datetime, cell_filter)

        if df.empty:
            return jsonify({'error': 'No data found'}), 404

        df['CELL_FACE'] = df['NR_CELL'].apply(
            lambda x: parse_cell_face_and_sector(x)[0])
        df['SECTOR'] = df['NR_CELL'].apply(
            lambda x: parse_cell_face_and_sector(x)[1])

        if selected_bands:
            df = df[df['BAND'].isin(selected_bands)]
        if selected_sectors:
            df = df[df['SECTOR'].isin(selected_sectors)]

        df['DL_VOL'] = df['DL_ACK_MAC_VOL_GB']
        df['UL_VOL'] = df['UL_ACK_MAC_VOL_GB']
        df['TOTAL_VOL'] = df['DL_VOL'] + df['UL_VOL']

        cell_stats = df.groupby(['GNODEB', 'NR_CELL', 'BAND']).agg({
            'DL_VOL': 'sum',
            'UL_VOL': 'sum',
            'TOTAL_VOL': 'sum',
            'DL_DRB_THPUT': 'mean'
        }).reset_index()

        comparison_data = []
        for gnodeb, group in cell_stats.groupby('GNODEB'):
            if len(group) > 1:
                max_vol = group['TOTAL_VOL'].max()
                min_vol = group['TOTAL_VOL'].min()
                avg_vol = group['TOTAL_VOL'].mean()
                total_vol = group['TOTAL_VOL'].sum()
                delta = max_vol - min_vol

                cell_count = len(group)
                equal_share = total_vol / cell_count if cell_count > 0 else 0

                cells_data = []
                for _, row in group.iterrows():
                    cell_vol = row['TOTAL_VOL']
                    percent_of_total = float(
                        (cell_vol / total_vol * 100) if total_vol > 0 else 0)
                    delta_from_equal = float(cell_vol - equal_share)

                    cells_data.append({
                        'nr_cell': row['NR_CELL'],
                        'band': row['BAND'],
                        'dl_vol': float(row['DL_VOL']),
                        'ul_vol': float(row['UL_VOL']),
                        'total_vol': float(cell_vol),
                        'avg_tput': float(row['DL_DRB_THPUT']),
                        'percent_of_total': percent_of_total,
                        'delta_from_equal': delta_from_equal,
                        'equal_share': float(equal_share)
                    })

                cells_data.sort(key=lambda x: x['total_vol'], reverse=True)

                comparison_data.append({
                    'gnodeb': gnodeb,
                    'cell_count': len(group),
                    'max_vol': float(max_vol),
                    'min_vol': float(min_vol),
                    'avg_vol': float(avg_vol),
                    'total_vol': float(total_vol),
                    'delta': float(delta),
                    'cells': cells_data
                })

        comparison_data.sort(key=lambda x: x['delta'], reverse=True)

        return jsonify({'gnodeb_comparison': comparison_data[:20]})
    except Exception as e:
        logger.error(f"Error in 5G GNODEB comparison: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/5g/cell-face-comparison', methods=['GET'])
def get_5g_cell_face_comparison():
    """API endpoint for 5G cell face band comparison"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        cell_filter = request.args.get('cell_filter', 'ILXN%')
        bands_param = request.args.get('bands', '')
        sectors_param = request.args.get('sectors', '')

        if not start_date or not end_date:
            return jsonify({'error': 'Missing required parameters'}), 400

        selected_bands = [b.strip() for b in bands_param.split(
            ',') if b.strip()] if bands_param else []
        selected_sectors = [s.strip() for s in sectors_param.split(
            ',') if s.strip()] if sectors_param else []

        if selected_bands:
            selected_bands = expand_band_selection(selected_bands)

        start_datetime = f"{start_date} 00:00:00"
        end_datetime = f"{end_date} 23:59:59"

        df = fetch_tput_data(start_datetime, end_datetime, cell_filter)

        if df.empty:
            return jsonify({'error': 'No data found'}), 404

        df['CELL_FACE'] = df['NR_CELL'].apply(
            lambda x: parse_cell_face_and_sector(x)[0])
        df['SECTOR'] = df['NR_CELL'].apply(
            lambda x: parse_cell_face_and_sector(x)[1])

        if selected_bands:
            df = df[df['BAND'].isin(selected_bands)]
        if selected_sectors:
            df = df[df['SECTOR'].isin(selected_sectors)]

        df['DL_VOL'] = df['DL_ACK_MAC_VOL_GB']
        df['UL_VOL'] = df['UL_ACK_MAC_VOL_GB']
        df['TOTAL_VOL'] = df['DL_VOL'] + df['UL_VOL']

        cell_stats = df.groupby(['CELL_FACE', 'SECTOR', 'NR_CELL', 'GNODEB', 'BAND']).agg({
            'DL_VOL': 'sum',
            'UL_VOL': 'sum',
            'TOTAL_VOL': 'sum',
            'DL_DRB_THPUT': 'mean'
        }).reset_index()

        comparison_data = []
        for (face, sector), group in cell_stats.groupby(['CELL_FACE', 'SECTOR']):
            if len(group) > 1:
                max_vol = group['TOTAL_VOL'].max()
                min_vol = group['TOTAL_VOL'].min()
                avg_vol = group['TOTAL_VOL'].mean()
                total_vol = group['TOTAL_VOL'].sum()
                delta = max_vol - min_vol

                band_count = len(group)
                equal_share = total_vol / band_count if band_count > 0 else 0

                bands_data = []
                for _, row in group.iterrows():
                    band_vol = row['TOTAL_VOL']
                    percent_of_total = float(
                        (band_vol / total_vol * 100) if total_vol > 0 else 0)
                    delta_from_equal = float(band_vol - equal_share)

                    bands_data.append({
                        'band': row['BAND'],
                        'nr_cell': row['NR_CELL'],
                        'gnodeb': row['GNODEB'],
                        'dl_vol': float(row['DL_VOL']),
                        'ul_vol': float(row['UL_VOL']),
                        'total_vol': float(band_vol),
                        'avg_tput': float(row['DL_DRB_THPUT']),
                        'percent_of_total': percent_of_total,
                        'delta_from_equal': delta_from_equal,
                        'equal_share': float(equal_share)
                    })

                bands_data.sort(key=lambda x: x['total_vol'], reverse=True)

                comparison_data.append({
                    'cell_face': face,
                    'sector': sector,
                    'band_count': len(group),
                    'max_vol': float(max_vol),
                    'min_vol': float(min_vol),
                    'avg_vol': float(avg_vol),
                    'total_vol': float(total_vol),
                    'delta': float(delta),
                    'bands': bands_data
                })

        comparison_data.sort(key=lambda x: x['delta'], reverse=True)

        return jsonify({'face_comparison': comparison_data[:20]})
    except Exception as e:
        logger.error(f"Error in 5G cell face comparison: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/5g/gnodeb-detail/<gnodeb>', methods=['GET'])
def get_5g_gnodeb_detail(gnodeb):
    """API endpoint for 5G GNODEB detail data"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        cell_filter = request.args.get('cell_filter', 'ILXN%')
        bands_param = request.args.get('bands', '')
        sectors_param = request.args.get('sectors', '')

        if not start_date or not end_date:
            return jsonify({'error': 'Missing required parameters'}), 400

        selected_bands = [b.strip() for b in bands_param.split(
            ',') if b.strip()] if bands_param else []
        selected_sectors = [s.strip() for s in sectors_param.split(
            ',') if s.strip()] if sectors_param else []

        if selected_bands:
            selected_bands = expand_band_selection(selected_bands)

        start_datetime = f"{start_date} 00:00:00"
        end_datetime = f"{end_date} 23:59:59"

        df = fetch_tput_data(start_datetime, end_datetime,
                             cell_filter, gnodeb_filter=gnodeb)

        if df.empty:
            return jsonify({'error': f'No data found for GNODEB {gnodeb}'}), 404

        df['CELL_FACE'] = df['NR_CELL'].apply(
            lambda x: parse_cell_face_and_sector(x)[0])
        df['SECTOR'] = df['NR_CELL'].apply(
            lambda x: parse_cell_face_and_sector(x)[1])

        available_bands = sorted(df['BAND'].unique().tolist())
        available_sectors = sorted(
            df[df['SECTOR'].notna()]['SECTOR'].unique().tolist())

        if selected_bands:
            df = df[df['BAND'].isin(selected_bands)]
        if selected_sectors:
            df = df[df['SECTOR'].isin(selected_sectors)]

        summary = {
            'total_cells': df['NR_CELL'].nunique(),
            'total_volume': float(df['DL_ACK_MAC_VOL_GB'].sum() + df['UL_ACK_MAC_VOL_GB'].sum()),
            'avg_dl_tput': float(df['DL_DRB_THPUT'].mean()),
            'avg_ul_tput': float(df['UL_DRB_THPUT'].mean())
        }

        traffic_agg = df.groupby(['BAND', 'SECTOR']).agg({
            'DL_ACK_MAC_VOL_GB': 'sum',
            'UL_ACK_MAC_VOL_GB': 'sum'
        }).reset_index()

        unique_bands = sorted(traffic_agg['BAND'].unique().tolist())
        unique_sectors = sorted(traffic_agg['SECTOR'].unique().tolist())

        dl_by_sector = {}
        ul_by_sector = {}

        for sector in unique_sectors:
            sector_data = traffic_agg[traffic_agg['SECTOR'] == sector]
            dl_volumes = []
            ul_volumes = []

            for band in unique_bands:
                band_sector_data = sector_data[sector_data['BAND'] == band]
                if not band_sector_data.empty:
                    dl_volumes.append(
                        float(band_sector_data['DL_ACK_MAC_VOL_GB'].iloc[0]))
                    ul_volumes.append(
                        float(band_sector_data['UL_ACK_MAC_VOL_GB'].iloc[0]))
                else:
                    dl_volumes.append(0)
                    ul_volumes.append(0)

            dl_by_sector[sector] = dl_volumes
            ul_by_sector[sector] = ul_volumes

        traffic = {
            'labels': unique_bands,
            'sectors': unique_sectors,
            'dl_by_sector': dl_by_sector,
            'ul_by_sector': ul_by_sector
        }

        tput_trends_df = df.groupby(['DATE_TIME_ID', 'BAND']).agg({
            'DL_DRB_THPUT': 'mean',
            'UL_DRB_THPUT': 'mean'
        }).reset_index()

        dates = sorted(tput_trends_df['DATE_TIME_ID'].unique())
        bands = sorted(tput_trends_df['BAND'].unique())

        dl_tput_by_band = []
        ul_tput_by_band = []

        for band in bands:
            band_data = tput_trends_df[tput_trends_df['BAND'] == band].set_index(
                'DATE_TIME_ID')
            dl_tput_by_band.append([float(
                band_data.loc[d, 'DL_DRB_THPUT']) if d in band_data.index else 0 for d in dates])
            ul_tput_by_band.append([float(
                band_data.loc[d, 'UL_DRB_THPUT']) if d in band_data.index else 0 for d in dates])

        tput_trends = {
            'dates': dates,
            'bands': bands,
            'dl_tput': dl_tput_by_band,
            'ul_tput': ul_tput_by_band
        }

        return jsonify({
            'summary': summary,
            'traffic': traffic,
            'tput_trends': tput_trends,
            'available_bands': available_bands,
            'available_sectors': available_sectors
        })
    except Exception as e:
        logger.error(f"Error in 5G GNODEB detail: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================


if __name__ == '__main__':
    print("ðŸš€ Starting Merged CQX + 5G Traffic Dashboard API Server...")
    print("=" * 60)
    print("ðŸ“Š CQX Dashboard API: /api/*")
    print("ðŸ”§ 5G Traffic Dashboard API: /api/5g/*")
    print("=" * 60)
    print("ðŸŒ Dashboard URLs:")
    print("   CQX Dashboard: http://localhost:5000/")
    print("   5G Dashboard: http://localhost:5000/5g")
    print("=" * 60)

    app.run(debug=False, host='0.0.0.0', port=5000)