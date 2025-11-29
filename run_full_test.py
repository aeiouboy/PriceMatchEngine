#!/usr/bin/env python3
"""Full production test with progress tracking - runs in batches"""

import pandas as pd
import json
import sys
import os
import time
from datetime import datetime

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

def test_retailer(retailer_name, batch_size=50):
    """Test all products for a single retailer with progress tracking"""
    from app import ai_match_products
    
    print(f"\n{'='*70}")
    print(f"TESTING: {retailer_name} (FULL RUN)")
    print(f"{'='*70}")
    
    config = RETAILERS[retailer_name]
    
    twd_products = load_json_products(TWD_PRODUCTS)
    competitor_products = load_json_products(config['products'])
    gt = load_ground_truth(config['gt'])
    
    print(f"Loaded {len(twd_products)} TWD, {len(competitor_products)} {retailer_name}, {len(gt)} GT")
    
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
    
    valid_gt_count = sum(1 for comp_url in gt.values() if comp_url in competitor_urls)
    invalid_gt_count = len(gt) - valid_gt_count
    
    print(f"GT Validity: {valid_gt_count}/{len(gt)} ({100*valid_gt_count/len(gt):.1f}%) - {invalid_gt_count} invalid entries excluded")
    
    twd_indices_to_test = []
    for i, url in twd_url_map.items():
        if url in gt:
            expected_url = gt[url]
            if expected_url in competitor_urls:
                twd_indices_to_test.append(i)
    
    total_to_test = len(twd_indices_to_test)
    print(f"Testing {total_to_test} products...")
    
    correct = 0
    incorrect = 0
    not_found = 0
    wrong_matches = []
    
    start_time = time.time()
    
    for batch_start in range(0, total_to_test, batch_size):
        batch_end = min(batch_start + batch_size, total_to_test)
        batch_indices = twd_indices_to_test[batch_start:batch_end]
        
        test_products = [twd_products[i] for i in batch_indices]
        
        try:
            matches = ai_match_products(test_products, competitor_products)
        except Exception as e:
            print(f"Error in batch {batch_start}-{batch_end}: {e}")
            continue
        
        for idx, twd_idx in enumerate(batch_indices):
            twd_url = twd_url_map[twd_idx]
            expected_url = gt[twd_url]
            
            match = None
            if matches:
                for m in matches:
                    if m.get('source_idx') == idx:
                        match = m
                        break
            
            if match is None or match.get('target_idx') is None:
                not_found += 1
            else:
                match_idx = match['target_idx']
                matched_url = competitor_url_map.get(match_idx, '')
                
                if matched_url == expected_url:
                    correct += 1
                else:
                    incorrect += 1
                    twd_name = test_products[idx].get('name', test_products[idx].get('product_name', ''))[:50]
                    got_name = competitor_products[match_idx].get('name', competitor_products[match_idx].get('product_name', ''))[:50]
                    wrong_matches.append(f"{twd_name} -> {got_name}")
        
        elapsed = time.time() - start_time
        tested = batch_end
        rate = tested / elapsed if elapsed > 0 else 0
        remaining = (total_to_test - tested) / rate if rate > 0 else 0
        
        current_accuracy = 100 * correct / tested if tested > 0 else 0
        print(f"  [{tested:4d}/{total_to_test}] Accuracy: {current_accuracy:.1f}% | Rate: {rate:.1f}/s | ETA: {remaining/60:.1f}min")
        sys.stdout.flush()
    
    total_tested = correct + incorrect + not_found
    accuracy = 100 * correct / total_tested if total_tested > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS for {retailer_name}:")
    print(f"  Total tested: {total_tested}")
    print(f"  Correct:    {correct} ({100*correct/total_tested:.1f}%)")
    print(f"  Incorrect:  {incorrect} ({100*incorrect/total_tested:.1f}%)")
    print(f"  Not Found:  {not_found} ({100*not_found/total_tested:.1f}%)")
    print(f"  ACCURACY:   {accuracy:.1f}%")
    print(f"  Time: {time.time()-start_time:.1f}s")
    print(f"{'='*70}")
    
    os.makedirs('test_results', exist_ok=True)
    result = {
        'retailer': retailer_name,
        'timestamp': datetime.now().isoformat(),
        'total_tested': total_tested,
        'correct': correct,
        'incorrect': incorrect,
        'not_found': not_found,
        'accuracy': accuracy,
        'wrong_matches_sample': wrong_matches[:30]
    }
    
    filename = f"test_results/{retailer_name}_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {filename}")
    
    return result

def main():
    if len(sys.argv) > 1:
        retailer = sys.argv[1]
        if retailer == 'all':
            results = []
            for r in RETAILERS.keys():
                result = test_retailer(r)
                if result:
                    results.append(result)
            
            print("\n" + "="*70)
            print("FINAL SUMMARY - ALL RETAILERS")
            print("="*70)
            print(f"{'Retailer':<15} {'Tested':>8} {'Correct':>8} {'Wrong':>8} {'NotFound':>10} {'Accuracy':>10}")
            print("-"*70)
            
            total_correct = 0
            total_tested = 0
            
            for r in results:
                print(f"{r['retailer']:<15} {r['total_tested']:>8} {r['correct']:>8} {r['incorrect']:>8} {r['not_found']:>10} {r['accuracy']:>9.1f}%")
                total_correct += r['correct']
                total_tested += r['total_tested']
            
            print("-"*70)
            overall = 100 * total_correct / total_tested if total_tested > 0 else 0
            print(f"{'OVERALL':<15} {total_tested:>8} {total_correct:>8} {'':<8} {'':<10} {overall:>9.1f}%")
        else:
            if retailer not in RETAILERS:
                print(f"Unknown retailer: {retailer}")
                print(f"Available: {', '.join(RETAILERS.keys())}, or 'all'")
                sys.exit(1)
            test_retailer(retailer)
    else:
        print("Usage: python run_full_test.py <retailer|all>")
        print(f"Available: {', '.join(RETAILERS.keys())}, or 'all'")

if __name__ == '__main__':
    main()
