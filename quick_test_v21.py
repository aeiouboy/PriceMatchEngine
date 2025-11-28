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
    for i, p in enumerate(competitor_products):
        url = p.get('url', p.get('product_url', p.get('link', '')))
        if url:
            competitor_url_map[i] = url.strip()
    
    # Filter to GT products
    filtered_twd = []
    filtered_indices = []
    for i, p in enumerate(twd_products):
        url = twd_url_map.get(i, '')
        if url in gt:
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
    print(f"  Correct:    {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  Incorrect:  {incorrect}/{total} ({incorrect/total*100:.1f}%)")
    print(f"  Not Found:  {not_found}/{total} ({not_found/total*100:.1f}%)")
    print(f"  Accuracy:   {correct/total*100:.1f}%")

if __name__ == "__main__":
    main()
