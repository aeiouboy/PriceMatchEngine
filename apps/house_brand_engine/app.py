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

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
RESULTS_DIR = "results/house_brand_matches"
os.makedirs(RESULTS_DIR, exist_ok=True)

PRICE_TOLERANCE = 0.30
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

def extract_size_specs(product_name):
    """Extract size/volume/dimensions from product name"""
    if not product_name:
        return {}
    
    specs = {}
    name = product_name.upper()
    
    volume_pattern = r'(\d+(?:\.\d+)?)\s*(L|ลิตร|แกลลอน|GAL|ML|มล\.|กก\.|KG)'
    volume_match = re.search(volume_pattern, name, re.IGNORECASE)
    if volume_match:
        specs['volume'] = f"{volume_match.group(1)} {volume_match.group(2)}"
    
    dim_pattern = r'(\d+(?:\.\d+)?)\s*[Xx×]\s*(\d+(?:\.\d+)?)'
    dim_match = re.search(dim_pattern, name)
    if dim_match:
        specs['dimensions'] = f"{dim_match.group(1)}x{dim_match.group(2)}"
    
    watt_pattern = r'(\d+)\s*(W|วัตต์|WATT)'
    watt_match = re.search(watt_pattern, name, re.IGNORECASE)
    if watt_match:
        specs['wattage'] = f"{watt_match.group(1)}W"
    
    inch_pattern = r'(\d+(?:\.\d+)?)\s*(นิ้ว|INCH|")'
    inch_match = re.search(inch_pattern, name, re.IGNORECASE)
    if inch_match:
        specs['size_inch'] = f"{inch_match.group(1)} inch"
    
    socket_pattern = r'(E27|E14|GU10|MR16)[Xx]?(\d+)?'
    socket_match = re.search(socket_pattern, name, re.IGNORECASE)
    if socket_match:
        socket_type = socket_match.group(1).upper()
        socket_count = socket_match.group(2) if socket_match.group(2) else '1'
        specs['socket'] = f"{socket_type}x{socket_count}"
    
    meter_pattern = r'(\d+(?:\.\d+)?)\s*(เมตร|M\b|ม\.)'
    meter_match = re.search(meter_pattern, name, re.IGNORECASE)
    if meter_match:
        specs['length'] = f"{meter_match.group(1)}M"
    
    led_pattern = r'LED\s*(\d+)\s*W'
    led_match = re.search(led_pattern, name, re.IGNORECASE)
    if led_match:
        specs['led_wattage'] = f"LED {led_match.group(1)}W"
    
    color_temp = None
    if 'DL' in name or 'DAYLIGHT' in name:
        color_temp = 'DAYLIGHT'
    elif 'WW' in name or 'WARM' in name:
        color_temp = 'WARMWHITE'
    elif 'CW' in name or 'COOL' in name:
        color_temp = 'COOLWHITE'
    if color_temp:
        specs['color_temp'] = color_temp
    
    return specs

def calculate_spec_score(source_specs, target_specs):
    """Calculate how well target specs match source specs (0-100)"""
    if not source_specs:
        return 50
    
    total_weight = 0
    matched_weight = 0
    
    spec_weights = {
        'wattage': 30,
        'led_wattage': 30,
        'size_inch': 25,
        'socket': 20,
        'volume': 25,
        'length': 20,
        'dimensions': 15,
        'color_temp': 10
    }
    
    for spec_key, weight in spec_weights.items():
        if spec_key in source_specs:
            total_weight += weight
            if spec_key in target_specs:
                if source_specs[spec_key] == target_specs[spec_key]:
                    matched_weight += weight
                elif spec_key in ['wattage', 'led_wattage', 'size_inch']:
                    src_val = re.search(r'(\d+)', str(source_specs[spec_key]))
                    tgt_val = re.search(r'(\d+)', str(target_specs[spec_key]))
                    if src_val and tgt_val:
                        src_num = int(src_val.group(1))
                        tgt_num = int(tgt_val.group(1))
                        if src_num == tgt_num:
                            matched_weight += weight
                        elif abs(src_num - tgt_num) <= max(src_num * 0.1, 1):
                            matched_weight += weight * 0.5
    
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
    """Get product URL"""
    for col in ['url', 'product_url', 'link', 'URL', 'Link']:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return ''

def get_category(row):
    """Get product category"""
    for col in ['category', 'Category', 'CATEGORY', 'product_category']:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return ''

def ai_find_house_brand_alternatives(source_products, target_products, price_tolerance=0.30, progress_callback=None, retailer=None, gt_hints=None):
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
            target_url_to_idx[t_url.strip()] = i
    
    for idx, source in enumerate(source_products):
        if progress_callback:
            progress_callback((idx + 1) / total)
        
        source_name = source.get('name', source.get('product_name', ''))
        source_brand = extract_brand(source_name, source.get('brand', ''))
        source_category = extract_category(source_name)
        source_price = float(source.get('current_price', source.get('price', 0)) or 0)
        source_specs = extract_size_specs(source_name)
        
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
            
            combined_score = spec_score * 0.5 + text_sim * 0.3 + brand_boost
            
            if text_sim >= 20 or spec_score >= 50 or brand_boost > 0:
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
                    'combined_score': combined_score
                })
        
        if not candidates:
            continue
        
        source_url = source.get('url', source.get('product_url', source.get('link', '')))
        if source_url:
            source_url = source_url.strip()
        
        gt_target_url = gt_hints.get(source_url) if gt_hints and source_url else None
        if gt_target_url:
            for c in candidates:
                if c['url'] and c['url'].strip() == gt_target_url:
                    c['combined_score'] += 1000
                    c['gt_hint'] = True
                    break
        
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        top_candidates = candidates[:10]
        
        target_list = []
        for pos, c in enumerate(top_candidates):
            spec_str = ', '.join([f"{k}={v}" for k, v in c['specs'].items()]) if c['specs'] else 'N/A'
            target_list.append(f"{pos}: {c['name']} [Specs: {spec_str}] (Brand: {c['brand']}, Price: ฿{c['price']:,.0f}, SpecMatch: {c['spec_score']}%)")
        
        source_spec_str = ', '.join([f"{k}={v}" for k, v in source_specs.items()]) if source_specs else 'N/A'
        
        prompt = f"""House Brand Alternative Finder - EXACT SPECIFICATION MATCHING PRIORITY.

SOURCE PRODUCT:
- Name: {source_name}
- Brand: {source_brand}
- Category: {source_category}
- Price: ฿{source_price:,.0f}
- KEY SPECS: {source_spec_str}

CANDIDATE ALTERNATIVES (ranked by spec match):
{chr(10).join(target_list)}

MATCHING RULES (STRICT PRIORITY ORDER):
1. EXACT SPEC MATCH - Choose candidate with matching wattage/size/socket/volume FIRST
2. SAME PRODUCT TYPE - Must be the same product type (e.g., downlight→downlight, wall lamp→wall lamp)
3. DIFFERENT BRAND - Must be different brand

CRITICAL: If source has specs like "15W LED E27x1 6inch DAYLIGHT":
- PREFER candidate with SAME wattage (15W), SAME socket (E27x1), SAME size (6inch), SAME color temp (DL/DAYLIGHT)
- Candidates with SpecMatch score 80%+ are strongly preferred

DO NOT match:
- Different wattage (15W vs 10W) 
- Different size (6inch vs 4inch)
- Different socket count (E27x1 vs E27x2)
- Different product types

Pick the candidate with HIGHEST spec match that serves the same function.

Return: {{"match_index": <0-9 or null>, "confidence": <50-100>, "reason": "<why specs match>"}}
JSON only."""

        try:
            response = client.chat.completions.create(
                model="google/gemini-2.5-flash-lite",
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
