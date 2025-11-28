#!/usr/bin/env python3
"""Test v2.1 matching algorithm on all 5 retailers"""

import pandas as pd
import json
import os
import sys
from datetime import datetime

# Import from main app
from app import (
    ai_match_products, normalize_text, AttributeExtractor,
    get_openrouter_client
)

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
    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            gt_df = pd.read_csv(filepath, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    gt_dict = {}
    # Detect columns based on filename pattern
    cols = gt_df.columns.tolist()
    
    # Find TWD URL column
    twd_col = None
    comp_col = None
    
    for col in cols:
        col_lower = col.lower()
        if 'thaiwatsadu' in col_lower and 'link' in col_lower:
            twd_col = col
        elif 'twd' in col_lower and 'link' in col_lower:
            twd_col = col
        elif 'link' in col_lower and twd_col is None:
            twd_col = col
    
    for col in cols:
        col_lower = col.lower()
        if any(r in col_lower for r in ['homepro', 'globalhouse', 'dohome', 'megahome', 'boonthavorn']):
            if 'link' in col_lower:
                comp_col = col
    
    # Fallback
    if not twd_col:
        for col in cols:
            if 'link' in col.lower():
                twd_col = col
                break
    if not comp_col:
        for col in cols:
            if 'link' in col.lower() and col != twd_col:
                comp_col = col
                break
    
    print(f"  GT columns: TWD={twd_col}, Comp={comp_col}")
    
    for _, row in gt_df.iterrows():
        twd_url = str(row.get(twd_col, '')).strip()
        comp_url = str(row.get(comp_col, '')).strip()
        if twd_url and comp_url and twd_url.startswith('http') and comp_url.startswith('http'):
            gt_dict[twd_url] = comp_url
    
    return gt_dict

def run_matching(retailer_name, retailer_config, twd_products):
    print(f"\n{'='*60}")
    print(f"Testing {retailer_name} with v2.1 algorithm")
    print(f"{'='*60}")
    
    # Load retailer products
    competitor_products = load_json_products(retailer_config['products'])
    print(f"Loaded {len(competitor_products)} {retailer_name} products")
    
    # Load ground truth
    gt = load_ground_truth(retailer_config['gt'])
    print(f"Loaded {len(gt)} ground truth pairs")
    
    # Build URL lookup for competitor products
    competitor_url_map = {}
    for i, p in enumerate(competitor_products):
        url = p.get('url', p.get('product_url', p.get('link', '')))
        if url:
            competitor_url_map[i] = url.strip()
    
    # Build URL lookup for TWD products
    twd_url_map = {}
    for i, p in enumerate(twd_products):
        url = p.get('url', p.get('product_url', p.get('link', '')))
        if url:
            twd_url_map[i] = url.strip()
    
    # Filter TWD products to only those in ground truth
    filtered_twd = []
    filtered_indices = []
    for i, p in enumerate(twd_products):
        url = twd_url_map.get(i, '')
        if url in gt:
            filtered_twd.append(p)
            filtered_indices.append(i)
    
    print(f"Testing {len(filtered_twd)} TWD products that have ground truth")
    
    # Run matching
    def progress(pct):
        done = int(pct * 40)
        print(f"\r  Progress: [{'='*done}{' '*(40-done)}] {pct*100:.0f}%", end='', flush=True)
    
    matches = ai_match_products(filtered_twd, competitor_products, progress)
    print()
    
    if not matches:
        print(f"  No matches found!")
        return None
    
    # Evaluate results
    correct = 0
    incorrect = 0
    not_found = 0
    
    matched_sources = set()
    results = []
    
    for m in matches:
        src_idx = m['source_idx']
        matched_sources.add(src_idx)
        
        orig_twd_idx = filtered_indices[src_idx]
        twd_url = twd_url_map.get(orig_twd_idx, '')
        expected_url = gt.get(twd_url, '')
        
        target_idx = m['target_idx']
        matched_url = competitor_url_map.get(target_idx, '')
        
        twd_name = filtered_twd[src_idx].get('name', filtered_twd[src_idx].get('product_name', ''))
        target_name = competitor_products[target_idx].get('name', competitor_products[target_idx].get('product_name', ''))
        
        is_correct = expected_url and matched_url and expected_url == matched_url
        
        results.append({
            'twd_product': twd_name,
            'twd_url': twd_url,
            'matched_product': target_name,
            'matched_url': matched_url,
            'expected_url': expected_url,
            'is_correct': is_correct,
            'confidence': m['confidence']
        })
        
        if is_correct:
            correct += 1
        else:
            incorrect += 1
    
    # Count not found
    for i in range(len(filtered_twd)):
        if i not in matched_sources:
            orig_twd_idx = filtered_indices[i]
            twd_url = twd_url_map.get(orig_twd_idx, '')
            twd_name = filtered_twd[i].get('name', filtered_twd[i].get('product_name', ''))
            
            results.append({
                'twd_product': twd_name,
                'twd_url': twd_url,
                'matched_product': 'NOT FOUND',
                'matched_url': '',
                'expected_url': gt.get(twd_url, ''),
                'is_correct': False,
                'confidence': 0
            })
            not_found += 1
    
    total = len(filtered_twd)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n  Results for {retailer_name}:")
    print(f"  - Correct:    {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  - Incorrect:  {incorrect}/{total} ({incorrect/total*100:.1f}%)")
    print(f"  - Not Found:  {not_found}/{total} ({not_found/total*100:.1f}%)")
    print(f"  - Accuracy:   {accuracy:.1f}%")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results_v21_{retailer_name.lower()}.csv', index=False)
    
    return {
        'retailer': retailer_name,
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'not_found': not_found,
        'accuracy': accuracy
    }

def main():
    print("="*60)
    print("V2.1 Algorithm Test - All Retailers")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Check API key
    if not get_openrouter_client():
        print("ERROR: OPENROUTER_API_KEY not set!")
        sys.exit(1)
    
    # Load TWD products
    print(f"\nLoading Thaiwatsadu products...")
    twd_products = load_json_products(TWD_PRODUCTS)
    print(f"Loaded {len(twd_products)} TWD products")
    
    # Test each retailer
    all_results = []
    
    for retailer_name, config in RETAILERS.items():
        result = run_matching(retailer_name, config, twd_products)
        if result:
            all_results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - V2.1 Algorithm Results")
    print("="*60)
    
    total_all = sum(r['total'] for r in all_results)
    correct_all = sum(r['correct'] for r in all_results)
    incorrect_all = sum(r['incorrect'] for r in all_results)
    not_found_all = sum(r['not_found'] for r in all_results)
    
    print(f"\n{'Retailer':<15} {'Total':>8} {'Correct':>10} {'Incorrect':>10} {'Not Found':>10} {'Accuracy':>10}")
    print("-"*65)
    for r in all_results:
        print(f"{r['retailer']:<15} {r['total']:>8} {r['correct']:>10} {r['incorrect']:>10} {r['not_found']:>10} {r['accuracy']:>9.1f}%")
    print("-"*65)
    overall_acc = (correct_all / total_all * 100) if total_all > 0 else 0
    print(f"{'TOTAL':<15} {total_all:>8} {correct_all:>10} {incorrect_all:>10} {not_found_all:>10} {overall_acc:>9.1f}%")
    
    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv('v21_accuracy_summary.csv', index=False)
    
    with open('v21_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'overall_accuracy': overall_acc,
            'total_products': total_all,
            'correct': correct_all,
            'incorrect': incorrect_all,
            'not_found': not_found_all,
            'by_retailer': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to v21_accuracy_summary.csv and v21_results.json")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
