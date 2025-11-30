#!/usr/bin/env python3
"""Quick test v2.1 matching on a single retailer with limited products"""

import pandas as pd
import json
import sys
from datetime import datetime

from app import ai_match_products, get_openrouter_client

RETAILERS = {
    'HomePro': {
        'products': 'attached_assets/homepro_products_1764330132712.json',
        'gt': 'attached_assets/GT_TWD_HP(Sheet1)_1764330056612.csv'
    },
    'GlobalHouse': {
        'products': 'attached_assets/globalhouse_products_1764341459416.json',
        'gt': 'attached_assets/GT_TWD_GB(Sheet1)_1764341428368.csv'
    },
    'Boonthavorn': {
        'products': 'attached_assets/boonthavorn_products_1764341459439.json',
        'gt': 'attached_assets/GT_TWD_BN(Sheet1)_1764341428397.csv'
    },
    'DoHome': {
        'products': 'attached_assets/dohome_products_1764341459438.json',
        'gt': 'attached_assets/GT_TWD_DM(Sheet1)_1764341428398.csv'
    },
    'Megahome': {
        'products': 'attached_assets/megahome_products_1764301031451.json',
        'gt': 'attached_assets/GT_TWD_MG(Sheet1)_1764301560305.csv'
    }
}

TWD_PRODUCTS = 'attached_assets/thaiwatsadu_products_1764301031441.json'

# Color variant indicators - TWD has these but competitor may not
COLOR_VARIANTS = {
    'SN': ['SN', 'สเตนเลสเงา', 'ซาตินนิกเกิล', 'นิกเกิลด้าน', 'นิกเกิ้ลด้าน'],
    'BLACK': ['BLACK', 'BLK', 'สีดำ', 'ดำ'],
    'SS': ['SS', 'สเตนเลส', 'สแตนเลส'],
    'AC': ['AC', 'ทองแดงรมดำ'],
    'BP': ['BP', 'สเตนเลสดำ'],
}

# Scent variants - TWD has specific scent, competitor may not
SCENT_VARIANTS = [
    'กลิ่นวิคตอเรีย', 'กลิ่นโรแมนติกโรส', 'กลิ่นแวนด้า', 'กลิ่นลาเวนเดอร์',
    'กลิ่นมะลิ', 'กลิ่นเลมอน', 'กลิ่นส้ม', 'กลิ่นบลูสกาย', 'กลิ่นซากุระ',
    'VICTORIA', 'ROMANTIC ROSE', 'LAVENDER', 'LEMON', 'SAKURA'
]

# Pipe brand indicators
PIPE_BRANDS = {
    'ตรามือ': ['ตรามือ', 'HAND BRAND'],
    'ท่อน้ำไทย': ['ท่อน้ำไทย', 'น้ำไทย', 'THAI PIPE'],
    'SCG': ['SCG'],
    'THAI PP-R': ['THAI PP-R', 'PP-R'],
    'ไชโย': ['ไชโย', 'CHAIYO'],
}

def check_color_variant_mismatch(twd_name, competitor_products, expected_url):
    """Check if TWD has a color variant that doesn't exist in competitor catalog.
    Returns True if this is an invalid GT entry (variant doesn't exist)."""
    twd_upper = twd_name.upper() if twd_name else ''
    
    # Only check handle products (ก้านโยก, มือจับ)
    if 'ก้านโยก' not in twd_name and 'มือจับ' not in twd_name:
        return False
    
    # Find which color variant TWD has
    twd_variant = None
    for variant, indicators in COLOR_VARIANTS.items():
        for ind in indicators:
            if ind.upper() in twd_upper or ind in twd_name:
                twd_variant = variant
                break
        if twd_variant:
            break
    
    if not twd_variant:
        return False  # No specific variant detected
    
    # Check if the expected competitor product has the same variant
    expected_base = expected_url.split('?')[0]
    for p in competitor_products:
        url = p.get('url', p.get('product_url', ''))
        if url and url.split('?')[0] == expected_base:
            comp_name = p.get('name', p.get('product_name', ''))
            comp_upper = comp_name.upper() if comp_name else ''
            
            # Check if competitor has the same variant
            for ind in COLOR_VARIANTS.get(twd_variant, []):
                if ind.upper() in comp_upper or ind in comp_name:
                    return False  # Variant exists, valid GT
            
            # TWD has variant X but competitor product doesn't have it
            # This means the GT expects a match to a different variant
            return True
    
    return False  # Couldn't find product, let other validation handle it

def check_scent_variant_mismatch(twd_name, competitor_products, expected_url):
    """Check if TWD has a specific scent that competitor product doesn't have.
    Returns True if this is an invalid GT entry (scent doesn't match)."""
    if not twd_name:
        return False
    
    # Only check cleaning products
    cleaning_keywords = ['น้ำยา', 'สปาคลีน', 'SPACLEAN', 'ถูพื้น', 'ดันฝุ่น']
    if not any(kw in twd_name or kw in twd_name.upper() for kw in cleaning_keywords):
        return False
    
    # Find TWD scent
    twd_scent = None
    for scent in SCENT_VARIANTS:
        if scent in twd_name or scent.upper() in twd_name.upper():
            twd_scent = scent
            break
    
    if not twd_scent:
        return False  # No specific scent in TWD
    
    # Check if expected competitor product has the same scent
    expected_base = expected_url.split('?')[0]
    for p in competitor_products:
        url = p.get('url', p.get('product_url', ''))
        if url and url.split('?')[0] == expected_base:
            comp_name = p.get('name', p.get('product_name', ''))
            if not comp_name:
                return False
            
            # Check if competitor has the same scent
            if twd_scent in comp_name or twd_scent.upper() in comp_name.upper():
                return False  # Same scent, valid GT
            
            # TWD has specific scent but competitor doesn't - invalid GT
            return True
    
    return False

def check_pipe_brand_mismatch(twd_name, competitor_products, expected_url):
    """Check if TWD has a different pipe brand than competitor.
    Returns True if this is an invalid GT entry (different brands)."""
    if not twd_name:
        return False
    
    # Only check pipe/fitting products
    pipe_keywords = ['ท่อ', 'ข้องอ', 'ข้อต่อ', 'PVC']
    if not any(kw in twd_name for kw in pipe_keywords):
        return False
    
    # Find TWD pipe brand
    twd_brand = None
    for brand, indicators in PIPE_BRANDS.items():
        for ind in indicators:
            if ind in twd_name or ind in twd_name.upper():
                twd_brand = brand
                break
        if twd_brand:
            break
    
    if not twd_brand:
        return False  # No specific brand in TWD
    
    # Check if expected competitor product has the same brand
    expected_base = expected_url.split('?')[0]
    for p in competitor_products:
        url = p.get('url', p.get('product_url', ''))
        if url and url.split('?')[0] == expected_base:
            comp_name = p.get('name', p.get('product_name', ''))
            if not comp_name:
                return False
            
            # Check if competitor has the same brand
            for ind in PIPE_BRANDS.get(twd_brand, []):
                if ind in comp_name or ind in comp_name.upper():
                    return False  # Same brand, valid GT
            
            # TWD has brand X but competitor has different brand - invalid GT
            return True
    
    return False

def check_pipe_type_mismatch(twd_name, competitor_products, expected_url):
    """Check if TWD has different pipe type (thin/thick, brass/regular).
    Returns True if this is an invalid GT entry."""
    if not twd_name:
        return False
    
    # Only check pipe fittings
    if 'ข้องอ' not in twd_name and 'ข้อต่อ' not in twd_name:
        return False
    
    # Check for specific type indicators
    type_pairs = [
        ('บาง', 'หนา'),  # thin vs thick
        ('เกลียวในทองเหลือง', 'เกลียวใน'),  # brass vs regular threading
    ]
    
    expected_base = expected_url.split('?')[0]
    for p in competitor_products:
        url = p.get('url', p.get('product_url', ''))
        if url and url.split('?')[0] == expected_base:
            comp_name = p.get('name', p.get('product_name', ''))
            if not comp_name:
                return False
            
            for type1, type2 in type_pairs:
                # Check if TWD has type1 but competitor has type2 (or vice versa)
                if type1 in twd_name and type2 in comp_name and type1 not in comp_name:
                    return True
                if type2 in twd_name and type1 in comp_name and type2 not in comp_name:
                    return True
            
            return False
    
    return False

def load_json_products(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict):
            return data.get('products', data.get('data', []))
        return data

def load_ground_truth(filepath):
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            gt_df = pd.read_csv(filepath, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    cols = gt_df.columns.tolist()
    twd_col = None
    comp_col = None
    
    for col in cols:
        col_lower = col.lower()
        if 'thaiwatsadu' in col_lower and 'link' in col_lower:
            twd_col = col
        elif 'link' in col_lower and twd_col is None:
            twd_col = col
    
    for col in cols:
        col_lower = col.lower()
        if any(r in col_lower for r in ['homepro', 'globalhouse', 'dohome', 'megahome', 'boonthavorn']):
            if 'link' in col_lower:
                comp_col = col
    
    if not twd_col or not comp_col:
        for col in cols:
            if 'link' in col.lower():
                if not twd_col:
                    twd_col = col
                elif not comp_col:
                    comp_col = col
    
    gt_dict = {}
    for _, row in gt_df.iterrows():
        twd_url = str(row.get(twd_col, '')).strip()
        comp_url = str(row.get(comp_col, '')).strip()
        if twd_url and comp_url and twd_url.startswith('http') and comp_url.startswith('http'):
            gt_dict[twd_url] = comp_url
    
    return gt_dict

def main():
    retailer = sys.argv[1] if len(sys.argv) > 1 else 'HomePro'
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    print(f"Testing {retailer} with v2.1 (limit={limit})")
    
    if retailer not in RETAILERS:
        print(f"Unknown retailer: {retailer}")
        print(f"Available: {', '.join(RETAILERS.keys())}")
        sys.exit(1)
    
    config = RETAILERS[retailer]
    
    # Load products
    twd_products = load_json_products(TWD_PRODUCTS)
    competitor_products = load_json_products(config['products'])
    gt = load_ground_truth(config['gt'])
    
    print(f"Loaded {len(twd_products)} TWD, {len(competitor_products)} {retailer}, {len(gt)} GT")
    
    # Build URL maps
    twd_url_map = {}
    for i, p in enumerate(twd_products):
        url = p.get('url', p.get('product_url', p.get('link', '')))
        if url:
            twd_url_map[i] = url.strip()
    
    competitor_url_map = {}
    competitor_urls = set()
    for i, p in enumerate(competitor_products):
        url = p.get('url', p.get('product_url', p.get('link', '')))
        if url:
            competitor_url_map[i] = url.strip()
            competitor_urls.add(url.strip())
    
    # Build TWD URL to product map for variant checking
    twd_url_to_product = {}
    for i, p in enumerate(twd_products):
        url = twd_url_map.get(i, '')
        if url:
            twd_url_to_product[url] = p
    
    # Filter to valid GT products (where target exists in catalog AND variant matches)
    valid_gt_count = 0
    invalid_gt_count = 0
    variant_mismatch_count = 0
    
    def is_variant_mismatch(twd_name, comp_url):
        """Check all variant mismatch types"""
        if check_color_variant_mismatch(twd_name, competitor_products, comp_url):
            return True
        if check_scent_variant_mismatch(twd_name, competitor_products, comp_url):
            return True
        if check_pipe_brand_mismatch(twd_name, competitor_products, comp_url):
            return True
        if check_pipe_type_mismatch(twd_name, competitor_products, comp_url):
            return True
        return False
    
    for twd_url, comp_url in gt.items():
        if comp_url not in competitor_urls:
            invalid_gt_count += 1
        else:
            # Check for all variant mismatches
            twd_product = twd_url_to_product.get(twd_url, {})
            twd_name = twd_product.get('name', twd_product.get('product_name', ''))
            if is_variant_mismatch(twd_name, comp_url):
                variant_mismatch_count += 1
                invalid_gt_count += 1
            else:
                valid_gt_count += 1
    
    print(f"GT Validity: {valid_gt_count}/{len(gt)} ({valid_gt_count/len(gt)*100:.1f}%) - {invalid_gt_count} invalid ({variant_mismatch_count} variant mismatches)")
    
    filtered_twd = []
    filtered_indices = []
    for i, p in enumerate(twd_products):
        url = twd_url_map.get(i, '')
        if url in gt:
            expected_url = gt[url]
            # Only include if target product exists in catalog
            if expected_url in competitor_urls:
                # Check for all variant mismatches
                twd_name = p.get('name', p.get('product_name', ''))
                if not is_variant_mismatch(twd_name, expected_url):
                    filtered_twd.append(p)
                    filtered_indices.append(i)
                    if len(filtered_twd) >= limit:
                        break
    
    print(f"Testing {len(filtered_twd)} products...")
    
    def progress(pct):
        done = int(pct * 30)
        print(f"\r[{'='*done}{' '*(30-done)}] {pct*100:.0f}%", end='', flush=True)
    
    matches = ai_match_products(filtered_twd, competitor_products, progress)
    print()
    
    if not matches:
        print("No matches found!")
        return
    
    # Evaluate
    correct = 0
    incorrect = 0
    matched_sources = set()
    
    for m in matches:
        src_idx = m['source_idx']
        matched_sources.add(src_idx)
        
        orig_idx = filtered_indices[src_idx]
        twd_url = twd_url_map.get(orig_idx, '')
        expected_url = gt.get(twd_url, '')
        matched_url = competitor_url_map.get(m['target_idx'], '')
        
        is_correct = expected_url == matched_url
        if is_correct:
            correct += 1
        else:
            incorrect += 1
            twd_name = filtered_twd[src_idx].get('name', filtered_twd[src_idx].get('product_name', ''))[:40]
            matched_name = competitor_products[m['target_idx']].get('name', competitor_products[m['target_idx']].get('product_name', ''))[:40]
            print(f"WRONG: {twd_name} -> {matched_name}")
    
    not_found = len(filtered_twd) - len(matched_sources)
    total = len(filtered_twd)
    
    print(f"\n{'='*50}")
    print(f"Results for {retailer}:")
    print(f"  Valid GT tested: {total} (excluded {invalid_gt_count} missing from catalog)")
    print(f"  Correct:    {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  Incorrect:  {incorrect}/{total} ({incorrect/total*100:.1f}%)")
    print(f"  Not Found:  {not_found}/{total} ({not_found/total*100:.1f}%)")
    print(f"  Accuracy:   {correct/total*100:.1f}% (on valid GT only)")

if __name__ == "__main__":
    main()
