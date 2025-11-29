#!/usr/bin/env python3
"""Detailed test showing each item match result"""

import pandas as pd
import json
import sys

from app import ai_match_products

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

def get_product_url(p):
    return p.get('url', p.get('product_url', p.get('link', '')))

def get_product_name(p):
    return p.get('name', p.get('product_name', 'Unknown'))

def test_retailer(retailer, limit=50):
    print(f"\n{'='*80}")
    print(f"RETAILER: {retailer}")
    print(f"{'='*80}")
    
    config = RETAILERS[retailer]
    
    twd_products = load_json_products(TWD_PRODUCTS)
    competitor_products = load_json_products(config['products'])
    gt_dict = load_ground_truth(config['gt'])
    
    # Build URL to product index mappings
    twd_url_to_idx = {}
    for i, p in enumerate(twd_products):
        url = get_product_url(p)
        if url:
            twd_url_to_idx[url] = i
    
    comp_url_to_idx = {}
    comp_idx_to_url = {}
    for i, p in enumerate(competitor_products):
        url = get_product_url(p)
        if url:
            comp_url_to_idx[url] = i
            comp_idx_to_url[i] = url
    
    # Filter to valid GT (products that exist in both catalogs)
    valid_gt = {}
    for twd_url, comp_url in gt_dict.items():
        if twd_url in twd_url_to_idx and comp_url in comp_url_to_idx:
            valid_gt[twd_url] = comp_url
    
    print(f"Total GT: {len(gt_dict)}, Valid GT: {len(valid_gt)} ({100*len(valid_gt)/len(gt_dict):.1f}%)")
    print(f"Products: {len(twd_products)} TWD, {len(competitor_products)} {retailer}")
    print()
    
    # Get test items
    test_items = list(valid_gt.items())[:limit]
    
    # Build filtered source products and their indices
    filtered_twd = []
    filtered_indices = []
    twd_idx_map = {}
    
    for twd_url, _ in test_items:
        original_idx = twd_url_to_idx[twd_url]
        filtered_indices.append(original_idx)
        twd_idx_map[len(filtered_twd)] = original_idx
        filtered_twd.append(twd_products[original_idx])
    
    # Run AI matching
    def progress(p):
        done = int(p * 30)
        print(f"\r[{'=' * done}{' ' * (30-done)}] {int(p*100)}%", end='', flush=True)
    
    matches = ai_match_products(filtered_twd, competitor_products, progress)
    print()
    
    # Build match lookup
    match_lookup = {}
    if matches:
        for m in matches:
            src_idx = m['source_idx']
            match_lookup[src_idx] = m
    
    correct = []
    wrong = []
    not_found = []
    
    # Evaluate each test item
    for i, (twd_url, expected_url) in enumerate(test_items):
        twd_name = get_product_name(twd_products[twd_url_to_idx[twd_url]])[:50]
        expected_name = get_product_name(competitor_products[comp_url_to_idx[expected_url]])[:50]
        
        if i in match_lookup:
            m = match_lookup[i]
            matched_idx = m['target_idx']
            matched_url = comp_idx_to_url.get(matched_idx, '')
            matched_name = get_product_name(competitor_products[matched_idx])[:50]
            confidence = m.get('confidence', 0)
            
            if matched_url == expected_url:
                correct.append((twd_name, matched_name, confidence))
                status = "✓"
            else:
                wrong.append((twd_name, matched_name, expected_name, confidence))
                status = "✗"
        else:
            not_found.append((twd_name, expected_name))
            status = "?"
        
        print(f"[{i+1:3d}/{limit}] {status} {twd_name[:45]}")
    
    # Summary
    print(f"\n{'-'*80}")
    print(f"SUMMARY for {retailer}:")
    print(f"  Correct:   {len(correct):3d}/{limit} ({100*len(correct)/limit:.1f}%)")
    print(f"  Wrong:     {len(wrong):3d}/{limit} ({100*len(wrong)/limit:.1f}%)")
    print(f"  Not Found: {len(not_found):3d}/{limit} ({100*len(not_found)/limit:.1f}%)")
    print(f"  ACCURACY:  {100*len(correct)/limit:.1f}%")
    
    if wrong:
        print(f"\n  WRONG MATCHES ({len(wrong)}):")
        for twd, matched, expected, conf in wrong:
            print(f"    - TWD: {twd[:45]}")
            print(f"      Got: {matched[:45]} ({conf}%)")
            print(f"      Expected: {expected[:45]}")
            print()
    
    if not_found:
        print(f"\n  NOT FOUND ({len(not_found)}):")
        for twd, expected in not_found:
            print(f"    - TWD: {twd[:50]}")
            print(f"      Expected: {expected[:50]}")
            print()
    
    return {
        'retailer': retailer,
        'correct': len(correct),
        'wrong': len(wrong),
        'not_found': len(not_found),
        'total': limit,
        'accuracy': 100*len(correct)/limit
    }

def main():
    retailers = sys.argv[1:] if len(sys.argv) > 1 else list(RETAILERS.keys())
    limit = 50
    
    results = []
    for retailer in retailers:
        if retailer in RETAILERS:
            result = test_retailer(retailer, limit)
            results.append(result)
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - ALL RETAILERS")
    print(f"{'='*80}")
    print(f"{'Retailer':<15} {'Correct':>8} {'Wrong':>8} {'Not Found':>10} {'Accuracy':>10}")
    print("-"*55)
    for r in results:
        print(f"{r['retailer']:<15} {r['correct']:>8} {r['wrong']:>8} {r['not_found']:>10} {r['accuracy']:>9.1f}%")
    
    avg_accuracy = sum(r['accuracy'] for r in results) / len(results) if results else 0
    print("-"*55)
    print(f"{'AVERAGE':<15} {'':<8} {'':<8} {'':<10} {avg_accuracy:>9.1f}%")

if __name__ == "__main__":
    main()
