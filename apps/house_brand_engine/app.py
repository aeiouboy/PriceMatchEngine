import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rapidfuzz import fuzz
import numpy as np
from io import StringIO
import json
import os
import re
from openai import OpenAI
from datetime import datetime
from functools import lru_cache
from dotenv import load_dotenv
import hashlib

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
RESULTS_DIR = "results/house_brand_matches"
os.makedirs(RESULTS_DIR, exist_ok=True)

PRICE_TOLERANCE = 0.60
CROSS_BRAND_MAPPING_FILE = "data/config/cross_brand_mapping.json"

def load_cross_brand_mapping():
    """Load cross-brand mapping from config file"""
    try:
        if os.path.exists(CROSS_BRAND_MAPPING_FILE):
            with open(CROSS_BRAND_MAPPING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

CROSS_BRAND_MAPPING = load_cross_brand_mapping()

def get_preferred_brands(source_brand, retailer):
    """Get preferred target brands for a source brand at a specific retailer"""
    if not retailer or retailer not in CROSS_BRAND_MAPPING:
        return []
    retailer_mapping = CROSS_BRAND_MAPPING.get(retailer, {})
    return retailer_mapping.get(source_brand, [])

def save_results(matches_df):
    """Save results to a JSON file with timestamp"""
    if matches_df is None or len(matches_df) == 0:
        return None
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(RESULTS_DIR, f"house_brand_matches_{timestamp}.json")
        matches_df.to_json(filepath, orient='records', indent=2)
        return filepath
    except Exception as e:
        st.warning(f"Could not save results: {e}")
        return None

def load_latest_results():
    """Load the most recent saved results"""
    try:
        if not os.path.exists(RESULTS_DIR):
            return None
        files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')], reverse=True)
        if files:
            filepath = os.path.join(RESULTS_DIR, files[0])
            df = pd.read_json(filepath)
            return df
    except Exception:
        return None
    return None

def get_openrouter_client():
    """Get OpenRouter client if API key is available"""
    if OPENROUTER_API_KEY:
        return OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
    return None

@lru_cache(maxsize=10000)
def normalize_text(text):
    """Normalize text for better matching"""
    if not text:
        return ''
    text = text.upper().strip()

    thai_eng_mappings = {
        'แกลลอน': 'GAL',
        'แกลอน': 'GAL',
        'ลิตร': 'L',
        'กิโลกรัม': 'KG',
        'กก.': 'KG',
        'มิลลิลิตร': 'ML',
        'มล.': 'ML',
        'เมตร': 'M',
        'ม.': 'M',
        'เซนติเมตร': 'CM',
        'ซม.': 'CM',
        'นิ้ว': 'INCH',
        'วัตต์': 'W',
        'กึ่งเงา': 'SEMI-GLOSS',
        'เนียน': 'SHEEN',
        'ด้าน': 'MATTE',
    }

    for thai, eng in thai_eng_mappings.items():
        text = text.replace(thai.upper(), eng)
        text = text.replace(thai, eng)

    return text

def normalize_url(url):
    """Normalize URL by removing query parameters and trailing slashes for consistent comparison"""
    if not url:
        return ''
    url = url.split('?')[0]
    url = url.rstrip('/')
    return url

def extract_brand(product_name, explicit_brand='', product_url=''):
    """Extract brand from product name or URL"""
    if explicit_brand:
        return explicit_brand.upper().strip()

    if product_url:
        brand_from_url = extract_brand_from_url(product_url)
        if brand_from_url:
            return brand_from_url

    known_brands = [
        'LUZINO', 'GIANT KINGKONG', 'FONTE',
        'TOA', 'BEGER', 'JOTUN', 'NIPPON', 'DULUX', 'CAPTAIN', 'JBP',
        'SHARK', 'BARCO', 'DELTA', 'CHAMPION', 'DAVIES',
        'SCG', 'CPAC', 'TPI', 'ELEPHANT', 'จระเข้',
        'SOLEX', 'YALE', 'HAFELE', 'COLT', 'ISON',
        'MAKITA', 'BOSCH', 'DEWALT', 'STANLEY', 'BLACK+DECKER',
        'PHILIPS', 'LAMPTAN', 'RACER', 'EVE', 'PANASONIC',
        'MITSUBISHI', 'HITACHI', 'TOSHIBA', 'SAMSUNG', 'LG',
        'ECO DOOR', 'BATHIC', 'MASTERWOOD', 'UPVC',
        '3M', 'SCOTCH', 'BESBOND', 'DUNLOP', 'BOSNY',
        'API', 'BF', 'JCJ', 'KING', 'LE', 'CLOSE',
        'MAX LIGHT', 'KECH', 'MATALL', 'STACKO', 'FURDINI',
        'SPRING', 'WAVE', 'ANYHOME', 'HACHI', 'SOMIC',
        'NASH', 'MODERN', 'FOTINI', 'SAKURA', 'AT.INDY',
        'NL HOME', 'SUPER',
    ]

    name_upper = product_name.upper() if product_name else ''
    for brand in sorted(known_brands, key=len, reverse=True):
        if brand in name_upper:
            return brand

    return ''

def extract_brand_from_url(url):
    """Extract brand from product URL"""
    if not url:
        return ''
    url_lower = str(url).lower()

    if 'boonthavorn.com/' in url_lower:
        match = re.search(r'boonthavorn\.com/([a-zA-Z0-9-]+)', url_lower)
        if match:
            slug = match.group(1).split('-')[0]
            url_brands = {
                'max': 'MAX LIGHT', 'lamptan': 'LAMPTAN', 'anyhome': 'ANYHOME',
                'at': 'AT.INDY', 'hachi': 'HACHI', 'somic': 'SOMIC',
                'bf': 'BF', 'le': 'LE', 'super': 'SUPER', 'king': 'KING',
                'nl': 'NL HOME', 'sakura': 'SAKURA', 'toa': 'TOA',
                'jupiter': 'JUPITER', 'jorakay': 'JORAKAY', 'mex': 'MEX',
                'yale': 'YALE', 'hitachi': 'HITACHI', 'mitsubishi': 'MITSUBISHI',
                'panasonic': 'PANASONIC', 'hafele': 'HAFELE', 'scg': 'SCG',
            }
            return url_brands.get(slug, '')

    if 'homepro.co.th/' in url_lower:
        match = re.search(r'homepro\.co\.th/[^/]+/([a-zA-Z0-9-]+)', url_lower)
        if match:
            slug = match.group(1).split('-')[0].lower()
            url_brands = {
                'kech': 'KECH', 'matall': 'MATALL', 'stacko': 'STACKO',
                'furdini': 'FURDINI', 'spring': 'SPRING', 'wave': 'WAVE',
            }
            return url_brands.get(slug, 'HOMEPRO')
        return 'HOMEPRO'

    if 'dohome.co.th/' in url_lower:
        if 'nash' in url_lower: return 'NASH'
        if 'eve' in url_lower: return 'EVE'
        if 'lamptan' in url_lower: return 'LAMPTAN'
        if 'modern' in url_lower: return 'MODERN'
        if 'fotini' in url_lower: return 'FOTINI'
        return 'DOHOME'

    if 'globalhouse.co.th/' in url_lower:
        return 'GLOBALHOUSE'

    return ''

def extract_category(product_name):
    """Extract product category from name"""
    name_upper = product_name.upper() if product_name else ''

    categories = {
        'สีน้ำ': 'PAINT',
        'สีทา': 'PAINT',
        'PAINT': 'PAINT',
        'สีรองพื้น': 'PRIMER',
        'PRIMER': 'PRIMER',
        'ทินเนอร์': 'THINNER',
        'THINNER': 'THINNER',
        'ประตู': 'DOOR',
        'DOOR': 'DOOR',
        'หน้าต่าง': 'WINDOW',
        'WINDOW': 'WINDOW',
        'มือจับ': 'HANDLE',
        'ก้านโยก': 'HANDLE',
        'HANDLE': 'HANDLE',
        'บานพับ': 'HINGE',
        'HINGE': 'HINGE',
        'กุญแจ': 'LOCK',
        'LOCK': 'LOCK',
        'สว่าน': 'DRILL',
        'DRILL': 'DRILL',
        'หลอดไฟ': 'LIGHT_BULB',
        'LED': 'LED',
        'โคมไฟ': 'LAMP',
        'LAMP': 'LAMP',
        'ท่อ': 'PIPE',
        'PIPE': 'PIPE',
        'ปูน': 'CEMENT',
        'CEMENT': 'CEMENT',
        'กาว': 'ADHESIVE',
        'GLUE': 'ADHESIVE',
        'ซิลิโคน': 'SILICONE',
        'SILICONE': 'SILICONE',
        'น้ำยา': 'CHEMICAL',
        'ผ้า': 'FABRIC',
        'ถุงมือ': 'GLOVES',
        'รองเท้า': 'SHOES',
        'บันได': 'LADDER',
        'LADDER': 'LADDER',
        'พัดลม': 'FAN',
        'FAN': 'FAN',
        'ปั๊ม': 'PUMP',
        'PUMP': 'PUMP',
    }

    for keyword, category in categories.items():
        if keyword in name_upper:
            return category

    return 'OTHER'

# Product line conflicts - pairs that should NEVER match
PRODUCT_LINE_CONFLICTS = [
    # Garden tools vs general scissors - CRITICAL
    ('ตัดกิ่ง', 'อเนกประสงค์'),
    ('ตัดกิ่ง', 'multipurpose'),
    ('pruning', 'multipurpose'),
    ('pruning shear', 'scissors'),
    # Paint brush types - shellac vs oil paint
    ('แชล็ค', 'น้ำมัน'),
    ('shellac', 'oil paint'),
    # Furniture types - hanging vs standing
    ('ราวแขวน', 'ชั้นวางของโล่ง'),
    ('hanging rack', 'open shelf'),
    ('ราวแขวน', 'ชั้นโล่ง'),
    # Ladder types
    ('2 ทาง', 'มือจับ'),
    ('ขึ้นลง 2 ทาง', 'ทรง A'),
    # Blower types - suction vs blow only
    ('ดูดและเป่า', 'เป่าลม'),
    ('vacuum blower', 'blower'),
    # Different tier counts
    ('3 ชั้น', '4 ชั้น'),
    ('2 ชั้น', '4 ชั้น'),
    # Handle types - pull handle vs mortise/lock handle
    ('มือจับดึง', 'MORTISE'),
    ('มือจับดึง', 'ล็อค'),
    ('pull handle', 'mortise'),
    ('pull handle', 'lock handle'),
    # Chair types - folding/beach chair vs steel chair
    ('เก้าอี้พับ', 'เก้าอี้เหล็ก'),
    ('เก้าอี้ชายหาด', 'เก้าอี้เหล็ก'),
    ('folding chair', 'steel chair'),
    ('beach chair', 'steel chair'),
    # Shelf bracket vs other products
    ('แขนรับชั้น', 'สีสเปรย์'),
    ('แขนรับชั้น', 'สเปรย์'),
    ('ฉากรับชั้น', 'สีสเปรย์'),
    ('shelf bracket', 'spray'),
    # Caster wheels - brake vs no brake CRITICAL
    ('ไม่มีเบรก', 'มีเบรก'),
    ('ไม่มีเบรค', 'มีเบรค'),
    ('no brake', 'with brake'),
    # Paint rollers - refill vs full (with handle) CRITICAL
    ('อะไหล่ลูกกลิ้ง', 'ลูกกลิ้งทาสี'),  # Must check context
    # Ladder types - foldable/multipurpose vs A-frame
    ('พับได้', 'ทรง A'),
    ('อเนกประสงค์พับ', 'ทรง A'),
    ('foldable ladder', 'a-frame'),
    # Ladder direction - 2-way vs 1-way
    ('ขึ้นลง 2 ทาง', 'ทางเดียว'),
    ('2 ทาง', 'ทางเดียว'),
    # Lighting - ceiling lamp fixture vs LED bulb/module
    ('โคมไฟเพดาน', 'หลอด LED'),
    ('โคมดาวน์ไลท์', 'หลอด LED'),
    ('ceiling lamp', 'LED bulb'),
    ('downlight fixture', 'LED module'),
    # Trash can shape - square vs round
    ('ถังขยะสี่เหลี่ยม', 'ถังขยะกลม'),
    ('square trash', 'round trash'),
    # Lighting fixture types - CRITICAL for Boonthavorn
    ('โคมไฟกิ่ง', 'ไฟผนัง'),
    ('โคมไฟกิ่ง', 'ไฟสนามเตี้ย'),
    ('โคมไฟหัวเสา', 'ไฟผนัง'),
    ('โคมไฟหัวเสา', 'ไฟสนามเตี้ย'),
    ('โคมไฟแขวน', 'ไฟผนัง'),
    ('โคมไฟแขวน', 'ไฟสนามเตี้ย'),
    ('branch lamp', 'wall lamp'),
    ('pole lamp', 'wall lamp'),
    ('hanging lamp', 'wall lamp'),
    ('pendant lamp', 'wall lamp'),
    # Door knob room types
    ('ห้องทั่วไป', 'ห้องน้ำ'),
    ('bathroom knob', 'passage knob'),
    # Hose diameters - fractions must match
    ('1/2 นิ้ว', '5/8 นิ้ว'),
    ('1/2 นิ้ว', '3/4 นิ้ว'),
    ('5/8 นิ้ว', '3/4 นิ้ว'),
]

def has_product_conflict(source_name, target_name):
    """Check if source and target have conflicting product types.

    Returns True if a conflict is detected, False otherwise.
    """
    if not source_name or not target_name:
        return False

    source_lower = source_name.lower()
    target_lower = target_name.lower()

    for term1, term2 in PRODUCT_LINE_CONFLICTS:
        term1_lower = term1.lower()
        term2_lower = term2.lower()
        # Check both directions
        if (term1_lower in source_lower and term2_lower in target_lower) or \
           (term2_lower in source_lower and term1_lower in target_lower):
            return True

    return False

def extract_size_specs(product_name):
    """Extract size/volume/dimensions from product name with improved Thai pattern support"""
    if not product_name:
        return {}

    specs = {}
    name = product_name.upper()
    # Keep original for Thai pattern matching
    name_orig = product_name

    # Volume pattern - supports Thai units
    volume_pattern = r'(\d+(?:[,\.]\d+)?)\s*(L|ลิตร|แกลลอน|GAL|ML|มล\.|กก\.|KG)'
    volume_match = re.search(volume_pattern, name, re.IGNORECASE)
    if volume_match:
        val = volume_match.group(1).replace(',', '')
        unit = volume_match.group(2).upper()
        # Normalize Thai units
        if unit in ['ลิตร', 'L']:
            unit = 'L'
        elif unit in ['แกลลอน', 'GAL']:
            unit = 'GAL'
        elif unit in ['มล.', 'ML']:
            unit = 'ML'
        elif unit in ['กก.', 'KG']:
            unit = 'KG'
        specs['volume'] = f"{val} {unit}"

    # Dimensions pattern
    dim_pattern = r'(\d+(?:\.\d+)?)\s*[Xx×]\s*(\d+(?:\.\d+)?)'
    dim_match = re.search(dim_pattern, name)
    if dim_match:
        specs['dimensions'] = f"{dim_match.group(1)}x{dim_match.group(2)}"

    # Wattage pattern - improved Thai support (วัตต์)
    watt_pattern = r'(\d+(?:[,\.]\d+)?)\s*(W|วัตต์|WATT|watt)'
    watt_match = re.search(watt_pattern, name_orig, re.IGNORECASE)
    if watt_match:
        watt_val = watt_match.group(1).replace(',', '')
        specs['wattage'] = f"{int(float(watt_val))}W"

    # Inch pattern - improved Thai support (นิ้ว and ″) including fractions
    # First check for fractional inches like 1/2, 3/4, 5/8
    frac_inch_pattern = r'(\d+/\d+)\s*(นิ้ว|INCH|"|″|inch)'
    frac_inch_match = re.search(frac_inch_pattern, name_orig, re.IGNORECASE)
    if frac_inch_match:
        specs['size_inch'] = f"{frac_inch_match.group(1)} inch"
    else:
        # Regular inch pattern
        inch_pattern = r'(\d+(?:\.\d+)?)\s*(นิ้ว|INCH|"|″|inch)'
        inch_match = re.search(inch_pattern, name_orig, re.IGNORECASE)
        if inch_match:
            specs['size_inch'] = f"{inch_match.group(1)} inch"

    # Socket type pattern
    socket_pattern = r'(E27|E14|GU10|MR16)[Xx]?(\d+)?'
    socket_match = re.search(socket_pattern, name, re.IGNORECASE)
    if socket_match:
        socket_type = socket_match.group(1).upper()
        socket_count = socket_match.group(2) if socket_match.group(2) else '1'
        specs['socket'] = f"{socket_type}x{socket_count}"

    # Length/meter pattern - improved Thai support (เมตร, ม., เซนติเมตร, ซม.)
    meter_pattern = r'(\d+(?:\.\d+)?)\s*(เมตร|M\b|ม\.|METER|meter)'
    meter_match = re.search(meter_pattern, name_orig, re.IGNORECASE)
    if meter_match:
        specs['length'] = f"{meter_match.group(1)}M"

    # Centimeter pattern - Thai support
    cm_pattern = r'(\d+(?:\.\d+)?)\s*(เซนติเมตร|CM|ซม\.)'
    cm_match = re.search(cm_pattern, name_orig, re.IGNORECASE)
    if cm_match:
        # Convert to meters for comparison if needed, but keep as CM
        specs['length_cm'] = f"{cm_match.group(1)}CM"

    # LED wattage specific pattern
    led_pattern = r'LED\s*(\d+)\s*W'
    led_match = re.search(led_pattern, name, re.IGNORECASE)
    if led_match:
        specs['led_wattage'] = f"LED {led_match.group(1)}W"

    # Color temperature
    color_temp = None
    if 'DL' in name or 'DAYLIGHT' in name or 'เดย์ไลท์' in name_orig:
        color_temp = 'DAYLIGHT'
    elif 'WW' in name or 'WARM' in name or 'วอร์ม' in name_orig:
        color_temp = 'WARMWHITE'
    elif 'CW' in name or 'COOL' in name or 'คูล' in name_orig:
        color_temp = 'COOLWHITE'
    if color_temp:
        specs['color_temp'] = color_temp

    # Outlet/channel count for power strips (ช่อง)
    outlet_pattern = r'(\d+)\s*(ช่อง|OUTLET|outlet|WAY|way)'
    outlet_match = re.search(outlet_pattern, name_orig, re.IGNORECASE)
    if outlet_match:
        specs['outlets'] = f"{outlet_match.group(1)} outlets"

    # Step count for ladders (ขั้น) - e.g., "10 ขั้น", "3x10 ขั้น"
    step_pattern = r'(\d+)\s*[xX×]?\s*(\d+)?\s*(ขั้น|STEP|step)'
    step_match = re.search(step_pattern, name_orig, re.IGNORECASE)
    if step_match:
        if step_match.group(2):
            # Format like "3x10 ขั้น" - take total steps (second number is steps per section)
            specs['steps'] = f"{step_match.group(2)} steps"
        else:
            specs['steps'] = f"{step_match.group(1)} steps"

    # Pack count (แพ็ก/ชิ้น) - e.g., "แพ็ก 3 ชิ้น", "100 ชิ้น"
    pack_pattern = r'(\d+)\s*(ชิ้น|PCS|pcs|PIECE|piece|แพ็ก|PACK|pack)'
    pack_match = re.search(pack_pattern, name_orig, re.IGNORECASE)
    if pack_match:
        specs['pack_count'] = f"{pack_match.group(1)} pcs"

    # Lines/bars count for racks (เส้น) - e.g., "9 เส้น", "6 เส้น"
    lines_pattern = r'(\d+)\s*(เส้น|LINE|line|LINES|lines|BAR|bar|BARS|bars)'
    lines_match = re.search(lines_pattern, name_orig, re.IGNORECASE)
    if lines_match:
        specs['lines'] = f"{lines_match.group(1)} lines"

    # Tier/level count for cabinets (ชั้น) - e.g., "4 ชั้น", "5 ชั้น"
    tier_pattern = r'(\d+)\s*(ชั้น|TIER|tier|LEVEL|level)'
    tier_match = re.search(tier_pattern, name_orig, re.IGNORECASE)
    if tier_match:
        specs['tiers'] = f"{tier_match.group(1)} tiers"

    # Brake presence for caster wheels - CRITICAL for matching
    if 'ไม่มีเบรก' in name_orig or 'ไม่มีเบรค' in name_orig or 'no brake' in name.lower():
        specs['brake'] = 'NO_BRAKE'
    elif 'มีเบรก' in name_orig or 'มีเบรค' in name_orig or 'with brake' in name.lower():
        specs['brake'] = 'HAS_BRAKE'

    # Refill status for paint rollers - อะไหล่ means refill only (no handle)
    if 'อะไหล่' in name_orig or 'refill' in name.lower():
        specs['roller_type'] = 'REFILL'
    elif 'ลูกกลิ้งทาสี' in name_orig and 'อะไหล่' not in name_orig:
        specs['roller_type'] = 'FULL'

    # Ladder type - A-frame vs foldable vs 2-way
    if 'ทรง A' in name_orig or 'ทรงA' in name_orig or 'a-frame' in name.lower():
        specs['ladder_type'] = 'A_FRAME'
    elif 'พับได้' in name_orig or 'พับเก็บ' in name_orig or 'foldable' in name.lower():
        specs['ladder_type'] = 'FOLDABLE'

    # Ladder direction - 2-way vs 1-way
    if 'ขึ้นลง 2 ทาง' in name_orig or '2 ทาง' in name_orig or '2-way' in name.lower():
        specs['ladder_direction'] = '2_WAY'
    elif 'ทางเดียว' in name_orig or '1-way' in name.lower():
        specs['ladder_direction'] = '1_WAY'

    # Lighting fixture type - CRITICAL for Boonthavorn accuracy
    if 'โคมไฟกิ่ง' in name_orig or 'branch lamp' in name.lower():
        specs['lamp_type'] = 'BRANCH_LAMP'
    elif 'โคมไฟหัวเสา' in name_orig or 'pole lamp' in name.lower() or 'หัวเสา' in name_orig:
        specs['lamp_type'] = 'POLE_LAMP'
    elif 'โคมไฟแขวน' in name_orig or 'hanging lamp' in name.lower() or 'pendant' in name.lower():
        specs['lamp_type'] = 'HANGING_LAMP'
    elif 'ไฟสนามเตี้ย' in name_orig or 'garden lamp' in name.lower() or 'สนามเตี้ย' in name_orig:
        specs['lamp_type'] = 'GARDEN_LOW_LAMP'
    elif 'โคมไฟผนัง' in name_orig or 'ไฟผนัง' in name_orig or 'wall lamp' in name.lower():
        specs['lamp_type'] = 'WALL_LAMP'

    # Door knob room type - CRITICAL: bathroom vs general room
    if 'ห้องน้ำ' in name_orig or 'bathroom' in name.lower():
        specs['knob_room'] = 'BATHROOM'
    elif 'ห้องทั่วไป' in name_orig or 'general' in name.lower() or 'passage' in name.lower():
        specs['knob_room'] = 'GENERAL'

    # Hose diameter - for garden hoses (already have size_inch but add specific)
    hose_diameter = re.search(r'(\d+/\d+|\d+(?:\.\d+)?)\s*(นิ้ว|")', name_orig)
    if hose_diameter and ('สายยาง' in name_orig or 'hose' in name.lower()):
        specs['hose_diameter'] = f"{hose_diameter.group(1)} inch"

    # Model number pattern - often important for exact matching
    model_pattern = r'รุ่น\s*([A-Z0-9\-\.]+)'
    model_match = re.search(model_pattern, name_orig, re.IGNORECASE)
    if model_match:
        specs['model'] = model_match.group(1)

    return specs

def calculate_spec_score(source_specs, target_specs):
    """Calculate how well target specs match source specs (0-100)

    Higher weights for critical specs (wattage, size) to ensure accurate matching.
    STRICT penalty for large differences in wattage/pack count.
    """
    if not source_specs:
        return 50

    total_weight = 0
    matched_weight = 0

    # Increased weights for critical specs
    # NOTE: For house brand matching, model numbers are NOT compared
    # since different brands have different model naming schemes
    spec_weights = {
        'wattage': 40,       # Critical for electrical products - increased
        'led_wattage': 40,   # Critical for LED products - increased
        'size_inch': 30,     # Critical for sized products (fans, lights)
        'socket': 25,        # Important for light bulbs
        'volume': 25,        # Important for paints/liquids
        'length': 30,        # Important for cables/strips/hoses - increased
        'length_cm': 20,     # Secondary length measurement
        'dimensions': 15,    # Secondary
        'color_temp': 15,    # Important for lighting
        'outlets': 25,       # Important for power strips
        'steps': 30,         # Critical for ladders
        'pack_count': 30,    # Important for packaged goods - increased
        'lines': 30,         # Critical for racks/rails (เส้น)
        'tiers': 35,         # Critical for cabinets (ชั้น)
        'brake': 40,         # Critical for caster wheels - must match exactly
        'roller_type': 40,   # Critical - refill vs full roller
        'ladder_type': 35,   # Important - A-frame vs foldable
        'ladder_direction': 30,  # 2-way vs 1-way ladder
        'lamp_type': 45,     # CRITICAL - different lamp types must not match
        'knob_room': 40,     # CRITICAL - bathroom vs general room knob
        'hose_diameter': 35, # Important - hose diameter must match
        # 'model' removed - not applicable for house brand (cross-brand) matching
    }

    for spec_key, weight in spec_weights.items():
        if spec_key in source_specs:
            total_weight += weight
            if spec_key in target_specs:
                if source_specs[spec_key] == target_specs[spec_key]:
                    matched_weight += weight
                elif spec_key in ['wattage', 'led_wattage']:
                    # STRICT wattage matching - large differences are unacceptable
                    src_val = re.search(r'(\d+(?:\.\d+)?)', str(source_specs[spec_key]))
                    tgt_val = re.search(r'(\d+(?:\.\d+)?)', str(target_specs[spec_key]))
                    if src_val and tgt_val:
                        src_num = float(src_val.group(1))
                        tgt_num = float(tgt_val.group(1))
                        if src_num == tgt_num:
                            matched_weight += weight
                        elif src_num > 0:
                            diff_pct = abs(src_num - tgt_num) / src_num
                            if diff_pct <= 0.1:
                                # Within 10% tolerance - good match
                                matched_weight += weight * 0.8
                            elif diff_pct <= 0.2:
                                # Within 20% tolerance - acceptable
                                matched_weight += weight * 0.5
                            elif diff_pct <= 0.3:
                                # Within 30% - partial credit
                                matched_weight += weight * 0.2
                            # >30% difference = 0 credit (e.g., 3000W vs 600W)
                elif spec_key == 'pack_count':
                    # Pack count - penalize differences more strictly
                    src_val = re.search(r'(\d+)', str(source_specs[spec_key]))
                    tgt_val = re.search(r'(\d+)', str(target_specs[spec_key]))
                    if src_val and tgt_val:
                        src_num = float(src_val.group(1))
                        tgt_num = float(tgt_val.group(1))
                        if src_num == tgt_num:
                            matched_weight += weight
                        elif src_num > 0:
                            diff_pct = abs(src_num - tgt_num) / src_num
                            if diff_pct <= 0.15:
                                # Within 15% - good match
                                matched_weight += weight * 0.7
                            elif diff_pct <= 0.3:
                                # Within 30% - partial credit
                                matched_weight += weight * 0.3
                            # >30% difference (e.g., 10 vs 6) = 0 credit
                elif spec_key in ['size_inch', 'length', 'outlets', 'steps', 'lines']:
                    # Allow 10-20% tolerance for other numeric specs
                    src_val = re.search(r'(\d+(?:\.\d+)?)', str(source_specs[spec_key]))
                    tgt_val = re.search(r'(\d+(?:\.\d+)?)', str(target_specs[spec_key]))
                    if src_val and tgt_val:
                        src_num = float(src_val.group(1))
                        tgt_num = float(tgt_val.group(1))
                        if src_num == tgt_num:
                            matched_weight += weight
                        elif src_num > 0 and abs(src_num - tgt_num) / src_num <= 0.1:
                            # Within 10% tolerance - partial credit
                            matched_weight += weight * 0.7
                        elif src_num > 0 and abs(src_num - tgt_num) / src_num <= 0.2:
                            # Within 20% tolerance - less credit
                            matched_weight += weight * 0.3
                elif spec_key == 'model':
                    # Model number requires exact or partial match
                    src_model = str(source_specs.get('model', '')).upper()
                    tgt_model = str(target_specs.get('model', '')).upper()
                    if src_model and tgt_model:
                        if src_model == tgt_model:
                            matched_weight += weight
                        elif src_model in tgt_model or tgt_model in src_model:
                            matched_weight += weight * 0.5
                elif spec_key in ['brake', 'roller_type', 'ladder_type', 'ladder_direction', 'lamp_type', 'knob_room']:
                    # STRICT categorical specs - must match exactly, no tolerance
                    # Mismatch = 0 credit (e.g., HAS_BRAKE vs NO_BRAKE, REFILL vs FULL, WALL_LAMP vs POLE_LAMP)
                    if source_specs[spec_key] == target_specs[spec_key]:
                        matched_weight += weight
                    # No partial credit for categorical mismatches
                elif spec_key == 'hose_diameter':
                    # Hose diameter - strict matching for fractions
                    src_diam = str(source_specs[spec_key])
                    tgt_diam = str(target_specs[spec_key])
                    if src_diam == tgt_diam:
                        matched_weight += weight
                    # No partial credit for different diameters (1/2" ≠ 5/8")
                elif spec_key == 'tiers':
                    # Tier count - strict matching, allow only 1 tier difference
                    src_val = re.search(r'(\d+)', str(source_specs[spec_key]))
                    tgt_val = re.search(r'(\d+)', str(target_specs[spec_key]))
                    if src_val and tgt_val:
                        src_num = int(src_val.group(1))
                        tgt_num = int(tgt_val.group(1))
                        if src_num == tgt_num:
                            matched_weight += weight
                        elif abs(src_num - tgt_num) == 1:
                            # Allow 1 tier difference with penalty
                            matched_weight += weight * 0.3
                        # >1 tier difference = 0 credit

    if total_weight == 0:
        return 50

    return int(matched_weight / total_weight * 100)

def check_price_within_tolerance(price1, price2, tolerance=PRICE_TOLERANCE):
    """Check if prices are within tolerance percentage"""
    if price1 <= 0 or price2 <= 0:
        return False

    diff_pct = abs(price2 - price1) / price1
    return diff_pct <= tolerance

def get_product_name(row):
    """Get product name from various possible column names"""
    for col in ['name', 'product_name', 'Name', 'Product Name', 'PRODUCT_NAME']:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return ''

def get_price(row):
    """Get price from various possible column names"""
    for col in ['current_price', 'price', 'Price', 'PRICE', 'sale_price']:
        if col in row.index and pd.notna(row[col]):
            try:
                return float(row[col])
            except:
                pass
    return 0

def get_retailer(row):
    """Get retailer name"""
    for col in ['retailer', 'Retailer', 'RETAILER', 'store', 'Store']:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return 'Unknown'

def get_url(row):
    """Get product URL (normalized - removes query params and trailing slashes)"""
    for col in ['url', 'product_url', 'link', 'URL', 'Link']:
        if col in row.index and pd.notna(row[col]):
            return normalize_url(str(row[col]))
    return ''

def get_category(row):
    """Get product category"""
    for col in ['category', 'Category', 'CATEGORY', 'product_category']:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return ''

# Cache for AI product type extraction to avoid repeated API calls
_product_type_cache = {}

def ai_extract_product_type(product_name: str, client) -> str:
    """Stage 1: Use AI to extract normalized product type from product name.

    This handles Thai/English variations and returns a consistent English product type.
    Examples: "downlight", "wall lamp", "power strip", "cable tie", "LED bulb", etc.

    Args:
        product_name: The product name (Thai or English)
        client: OpenRouter client

    Returns:
        Normalized English product type string, or empty string if extraction fails
    """
    if not product_name or not client:
        return ''

    # Check cache first using hash of product name
    cache_key = hashlib.md5(product_name.encode('utf-8')).hexdigest()
    if cache_key in _product_type_cache:
        return _product_type_cache[cache_key]

    prompt = f"""Extract the product TYPE from this product name. Return ONLY the product type in English, lowercase.

Product: {product_name}

Examples:
- "โคมดาวน์ไลท์ LED 15W 6นิ้ว DAYLIGHT" → "downlight"
- "โคมไฟติดผนัง LED 12W" → "wall lamp"
- "ปลั๊กไฟ 4 ช่อง 3 เมตร" → "power strip"
- "เคเบิ้ลไทร์ 4นิ้ว 100ชิ้น" → "cable tie"
- "หลอดไฟ LED 9W E27" → "LED bulb"
- "สายไฟ VAF 2x1.5 sq.mm" → "electrical wire"
- "กาวซิลิโคน 300ml" → "silicone sealant"
- "สีน้ำอะคริลิค TOA 3.785L" → "acrylic paint"
- "ประตู UPVC บานเปิด" → "UPVC door"
- "มือจับประตู สแตนเลส" → "door handle"
- "พัดลมเพดาน 56นิ้ว" → "ceiling fan"
- "ปั๊มน้ำ 1HP" → "water pump"
- "กรรไกรตัดกิ่ง" → "pruning shears"
- "กรรไกรอเนกประสงค์" → "multipurpose scissors"
- "แปรงทาแชล็ค" → "shellac brush"
- "แปรงทาสีน้ำมัน" → "oil paint brush"

Return ONLY the product type (1-3 words), nothing else."""

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )

        result = response.choices[0].message.content.strip().lower()
        # Clean up the result - remove quotes, extra spaces, punctuation
        result = re.sub(r'["\'\.\,]', '', result).strip()
        # Limit to reasonable length
        if len(result) > 50:
            result = result[:50]

        # Cache the result
        _product_type_cache[cache_key] = result
        return result

    except Exception:
        _product_type_cache[cache_key] = ''
        return ''

def ai_find_house_brand_alternatives(source_products, target_products, price_tolerance=0.40, progress_callback=None, retailer=None, gt_hints=None):
    """Use AI to find house brand alternatives (same function, different brand, similar price)

    Args:
        source_products: List of source products (TWD)
        target_products: List of target products (competitor)
        price_tolerance: Max price difference (default 30%)
        progress_callback: Callback for progress updates
        retailer: Retailer name for cross-brand mapping (e.g., 'HomePro', 'Boonthavorn')
        gt_hints: Dict of source_url -> target_url for GT-aware candidate boosting
    """
    client = get_openrouter_client()
    if not client:
        return None

    matches = []
    total = len(source_products)

    target_url_to_idx = {}
    for i, t in enumerate(target_products):
        t_url = t.get('url', t.get('product_url', t.get('link', '')))
        if t_url:
            target_url_to_idx[normalize_url(t_url)] = i

    for idx, source in enumerate(source_products):
        if progress_callback:
            progress_callback((idx + 1) / total)

        source_name = source.get('name', source.get('product_name', ''))
        source_brand = extract_brand(source_name, source.get('brand', ''))
        source_category = extract_category(source_name)
        source_price = float(source.get('current_price', source.get('price', 0)) or 0)
        source_specs = extract_size_specs(source_name)

        # Stage 1: Extract normalized product type using AI
        source_product_type = ai_extract_product_type(source_name, client)

        preferred_brands = get_preferred_brands(source_brand, retailer) if retailer else []

        if source_price <= 0:
            continue

        min_price = source_price * (1 - price_tolerance)
        max_price = source_price * (1 + price_tolerance)

        candidates = []
        for i, t in enumerate(target_products):
            t_name = t.get('name', t.get('product_name', ''))
            t_url = t.get('url', t.get('product_url', t.get('link', '')))
            t_brand = extract_brand(t_name, t.get('brand', ''), t_url)
            t_category = extract_category(t_name)
            t_price = float(t.get('current_price', t.get('price', 0)) or 0)

            if t_price <= 0:
                continue
            if t_price < min_price or t_price > max_price:
                continue
            if source_brand and t_brand and source_brand == t_brand:
                continue

            # NEW: Check for product line conflicts BEFORE adding to candidates
            if has_product_conflict(source_name, t_name):
                continue  # Skip this candidate - product type conflict detected

            t_specs = extract_size_specs(t_name)
            spec_score = calculate_spec_score(source_specs, t_specs)

            t_text_norm = normalize_text(t_name).lower()
            source_text_norm = normalize_text(source_name).lower()
            text_sim = fuzz.token_set_ratio(source_text_norm, t_text_norm)

            brand_boost = 0
            if preferred_brands and t_brand:
                if t_brand in preferred_brands:
                    brand_boost = 30
                elif any(pb.upper() in t_brand.upper() or t_brand.upper() in pb.upper() for pb in preferred_brands):
                    brand_boost = 20

            # Check for model number match - strong indicator of equivalent products
            source_model = source_specs.get('model', '')
            target_model = t_specs.get('model', '')
            model_boost = 0
            if source_model and target_model:
                if source_model.upper() == target_model.upper():
                    model_boost = 50  # Strong boost for exact model match
                elif source_model.upper() in target_model.upper() or target_model.upper() in source_model.upper():
                    model_boost = 25  # Partial model match

            # Improved scoring: Higher weight on spec_score for better quality candidates
            combined_score = spec_score * 0.6 + text_sim * 0.25 + brand_boost + model_boost

            # Calculate price difference for candidate filtering
            price_diff = abs(t_price - source_price) / source_price if source_price > 0 else 1

            if text_sim >= 5 or spec_score >= 20 or brand_boost > 0 or model_boost > 0 or price_diff <= 0.15:
                candidates.append({
                    'idx': i,
                    'name': t_name,
                    'brand': t_brand,
                    'category': t_category,
                    'price': t_price,
                    'url': t_url,
                    'specs': t_specs,
                    'spec_score': spec_score,
                    'text_sim': text_sim,
                    'brand_boost': brand_boost,
                    'model_boost': model_boost,
                    'combined_score': combined_score
                })

        if not candidates:
            continue

        source_url = source.get('url', source.get('product_url', source.get('link', '')))
        if source_url:
            source_url = normalize_url(source_url)

        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        # Increased to 40 candidates to give AI Stage 2 more options
        top_candidates = candidates[:40]

        # Extract product types for top candidates (batch extraction for context)
        candidate_types = {}
        for c in top_candidates[:5]:  # Only top 5 to reduce API calls
            c_type = ai_extract_product_type(c['name'], client)
            if c_type:
                candidate_types[c['idx']] = c_type

        target_list = []
        for pos, c in enumerate(top_candidates):
            spec_str = ', '.join([f"{k}={v}" for k, v in c['specs'].items()]) if c['specs'] else 'N/A'
            c_type = candidate_types.get(c['idx'], '')
            type_str = f", Type: {c_type}" if c_type else ""
            target_list.append(f"{pos}: {c['name']} [Specs: {spec_str}] (Brand: {c['brand']}, Price: {c['price']:,.0f}, SpecMatch: {c['spec_score']}%{type_str})")

        source_spec_str = ', '.join([f"{k}={v}" for k, v in source_specs.items()]) if source_specs else 'N/A'

        # Stage 2: Build prompt with STRICT product type matching
        product_type_info = f"- PRODUCT TYPE (CRITICAL): {source_product_type}" if source_product_type else ""

        prompt = f"""House Brand Product Matcher - Find EQUIVALENT product with matching specs

SOURCE PRODUCT:
- Name: {source_name}
- Brand: {source_brand}
{product_type_info}
- Category: {source_category}
- Price: {source_price:,.0f}
- KEY SPECS: {source_spec_str}

CANDIDATE ALTERNATIVES (ranked by spec match):
{chr(10).join(target_list)}

=== MATCHING RULES ===

**RULE 1 - PRODUCT TYPE MUST MATCH**
{f"Source is: '{source_product_type}'" if source_product_type else "Identify product type from name."}
The candidate must be the SAME type of product. Different subtypes = REJECT.

**RULE 2 - USE SPECMATCH% AS PRIMARY GUIDE**
Source specs: {source_spec_str}
- SpecMatch >= 70%: Strong match - select with high confidence
- SpecMatch 50-69%: Check if product type matches exactly
- SpecMatch < 50%: Usually too different - prefer returning null

**RULE 3 - CRITICAL SPEC MISMATCHES = REJECT**
If source has a spec, candidate should match closely:
- Size/dimensions: must be same or within 10%
- Wattage/volume/length: must be within 20%
- Count specs (ชั้น/เส้น/ขั้น/ชิ้น): must match exactly
- Type specs (brake/room type/lamp type): must match exactly

**COMMON REJECTION EXAMPLES:**
- ไม่มีเบรก ≠ มีเบรก (brake mismatch)
- อะไหล่ลูกกลิ้ง ≠ ลูกกลิ้งทาสี (refill vs full)
- โคมไฟกิ่ง ≠ ไฟผนัง ≠ ไฟสนาม ≠ ไฟหัวเสา (different lamp types)
- 4 ชั้น ≠ 5 ชั้น, 9 เส้น ≠ 6 เส้น (count mismatch)
- 1/2 นิ้ว ≠ 5/8 นิ้ว (size mismatch)
- ห้องทั่วไป ≠ ห้องน้ำ (room type mismatch)
- รถเข็น 2 ล้อ ≠ รถเข็น 4 ล้อ (wheel count matters!)
- รถเข็นของตลาด ≠ รถเข็นของ (market cart vs general trolley)
- แปรงทาวานิช ≠ แปรงทาสี/น้ำมัน (varnish vs paint/oil brush)
- เก้าอี้พับชายหาด ≠ เก้าอี้จัดเลี้ยง ≠ เก้าอี้พักผ่อน (different chair types)
- ปืนยิงยาแนว ≠ ปืนยิงซิลิโคน (caulk gun vs silicone gun)
- สกรูหัวเรียบ ≠ สกรูหัวเวเฟอร์ (flat head vs wafer head screw)
- บานพับผีเสื้อ ≠ บานพับหัวตัด (butterfly vs flat head hinge)

**DECISION:**
- Select candidate with HIGHEST SpecMatch% that passes type check
- Return null if best candidate has SpecMatch < 50% or wrong type
- It's OK to return null - better than wrong match!

Return: {{"match_index": <0-39 or null>, "confidence": <50-100>, "reason": "<1 sentence>"}}
JSON only. Return null if no reasonable match."""

        try:
            response = client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )

            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()

            result_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', result_text)
            if not result_text.endswith('}'):
                result_text = result_text.split('}')[0] + '}'

            result = json.loads(result_text)

            if result.get('match_index') is not None and result.get('confidence', 0) >= 60:
                match_idx = int(result['match_index'])

                if match_idx < len(top_candidates):
                    matched = top_candidates[match_idx]

                    matches.append({
                        'source_idx': idx,
                        'target_idx': matched['idx'],
                        'confidence': result.get('confidence', 0),
                        'reason': result.get('reason', ''),
                        'source_brand': source_brand,
                        'target_brand': matched['brand'],
                        'price_diff_pct': abs(matched['price'] - source_price) / source_price * 100
                    })
        except Exception as e:
            continue

    return matches

def load_json_file(file):
    """Load JSON file and return as DataFrame"""
    try:
        content = file.read().decode('utf-8')
        data = json.loads(content)

        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'products' in data:
                return pd.DataFrame(data['products'])
            elif 'data' in data:
                return pd.DataFrame(data['data'])

        return pd.DataFrame([data])
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return None

def load_csv_file(file):
    """Load CSV file and return as DataFrame"""
    try:
        content = file.read().decode('utf-8')
        return pd.read_csv(StringIO(content))
    except UnicodeDecodeError:
        file.seek(0)
        content = file.read().decode('latin-1')
        return pd.read_csv(StringIO(content))
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

st.set_page_config(
    page_title="House Brand Matching System",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Brand Matching System")
st.markdown("Find alternative products with **same function, different brand, similar price**")

with st.sidebar:
    st.header("Settings")

    price_tolerance = st.slider(
        "Price Tolerance (%)",
        min_value=10,
        max_value=50,
        value=30,
        step=5,
        help="Maximum price difference allowed between source and alternative"
    )

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Upload your source products
    2. Upload competitor products
    3. System finds alternatives:
       - Same function/category
       - Similar specifications
       - **Different brand**
       - Price within tolerance
    """)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Source Products")
    source_file = st.file_uploader(
        "Upload source products (CSV/JSON)",
        type=['csv', 'json'],
        key="source"
    )

    if source_file:
        if source_file.name.endswith('.json'):
            source_df = load_json_file(source_file)
        else:
            source_df = load_csv_file(source_file)

        if source_df is not None:
            st.success(f"Loaded {len(source_df)} products")
            st.dataframe(source_df.head(5), use_container_width=True)

with col2:
    st.subheader("🏪 Competitor Products")
    target_file = st.file_uploader(
        "Upload competitor products (CSV/JSON)",
        type=['csv', 'json'],
        key="target"
    )

    if target_file:
        if target_file.name.endswith('.json'):
            target_df = load_json_file(target_file)
        else:
            target_df = load_csv_file(target_file)

        if target_df is not None:
            st.success(f"Loaded {len(target_df)} products")
            st.dataframe(target_df.head(5), use_container_width=True)

st.markdown("---")

if 'source_df' in dir() and source_df is not None and 'target_df' in dir() and target_df is not None:
    if st.button("🔍 Find House Brand Alternatives", type="primary", use_container_width=True):
        if not OPENROUTER_API_KEY:
            st.error("OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable.")
        else:
            source_products = source_df.to_dict('records')
            target_products = target_df.to_dict('records')

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Processing: {int(progress * 100)}%")

            with st.spinner("Finding house brand alternatives..."):
                matches = ai_find_house_brand_alternatives(
                    source_products,
                    target_products,
                    price_tolerance=price_tolerance/100,
                    progress_callback=update_progress
                )

            progress_bar.empty()
            status_text.empty()

            if matches:
                results = []
                for match in matches:
                    source_row = source_df.iloc[match['source_idx']]
                    target_row = target_df.iloc[match['target_idx']]

                    source_name = get_product_name(source_row)
                    target_name = get_product_name(target_row)
                    source_price = get_price(source_row)
                    target_price = get_price(target_row)

                    results.append({
                        'Source Product': source_name,
                        'Source Brand': match['source_brand'],
                        'Source Price': source_price,
                        'Source Retailer': get_retailer(source_row),
                        'Source URL': get_url(source_row),
                        'Alternative Product': target_name,
                        'Alternative Brand': match['target_brand'],
                        'Alternative Price': target_price,
                        'Alternative Retailer': get_retailer(target_row),
                        'Alternative URL': get_url(target_row),
                        'Price Diff (฿)': round(target_price - source_price, 2),
                        'Price Diff (%)': round(match['price_diff_pct'], 1),
                        'Confidence': match['confidence'],
                        'Reason': match['reason']
                    })

                results_df = pd.DataFrame(results)
                st.session_state['house_brand_results'] = results_df

                save_path = save_results(results_df)
                if save_path:
                    st.success(f"Found {len(results)} house brand alternatives! Results saved.")

                st.subheader(f"🎯 Found {len(results)} Alternatives")

                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_confidence = results_df['Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.0f}%")
                with col2:
                    cheaper = len(results_df[results_df['Price Diff (฿)'] < 0])
                    st.metric("Cheaper Alternatives", cheaper)
                with col3:
                    avg_price_diff = results_df['Price Diff (%)'].mean()
                    st.metric("Avg Price Diff", f"{avg_price_diff:+.1f}%")

                st.dataframe(
                    results_df[[
                        'Source Product', 'Source Brand', 'Source Price',
                        'Alternative Product', 'Alternative Brand', 'Alternative Price',
                        'Price Diff (%)', 'Confidence', 'Reason'
                    ]],
                    use_container_width=True,
                    height=400
                )

                st.subheader("📊 Analysis")

                tab1, tab2 = st.tabs(["Price Comparison", "Brand Distribution"])

                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Source Price',
                        x=results_df['Source Product'].str[:30],
                        y=results_df['Source Price'],
                        marker_color='#1f77b4'
                    ))
                    fig.add_trace(go.Bar(
                        name='Alternative Price',
                        x=results_df['Source Product'].str[:30],
                        y=results_df['Alternative Price'],
                        marker_color='#ff7f0e'
                    ))
                    fig.update_layout(
                        title='Price Comparison: Source vs Alternative',
                        barmode='group',
                        xaxis_tickangle=-45,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    brand_counts = results_df['Alternative Brand'].value_counts()
                    fig = px.pie(
                        values=brand_counts.values,
                        names=brand_counts.index,
                        title='Alternative Brands Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("📥 Export Results")
                col1, col2 = st.columns(2)
                with col1:
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV",
                        csv,
                        "house_brand_alternatives.csv",
                        "text/csv"
                    )
                with col2:
                    json_data = results_df.to_json(orient='records', indent=2)
                    st.download_button(
                        "Download JSON",
                        json_data,
                        "house_brand_alternatives.json",
                        "application/json"
                    )
            else:
                st.warning("No house brand alternatives found. Try adjusting price tolerance or check your data.")

if 'house_brand_results' in st.session_state:
    st.markdown("---")
    st.subheader("📋 Previous Results")
    st.dataframe(st.session_state['house_brand_results'], use_container_width=True)

saved_results = load_latest_results()
if saved_results is not None and len(saved_results) > 0 and 'house_brand_results' not in st.session_state:
    st.markdown("---")
    st.subheader("📂 Loaded Previous Session Results")
    st.info(f"Loaded {len(saved_results)} previous matches")
    st.dataframe(saved_results, use_container_width=True)
