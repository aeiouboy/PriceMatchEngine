#!/usr/bin/env python3
"""Full production test - runs ALL SKUs for all 5 retailers"""

import pandas as pd
import json
import sys
import os
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps', 'price_match_engine'))
from app import ai_match_products

RETAILERS = {
    'HomePro': {
        'products': 'data/products/homepro.json',
        'gt': 'data/ground_truth/GT_TWD_HP.csv'
    },
    'GlobalHouse': {
        'products': 'data/products/globalhouse.json',
        'gt': 'data/ground_truth/GT_TWD_GB.csv'
    },
    'Boonthavorn': {
        'products': 'data/products/boonthavorn.json',
        'gt': 'data/ground_truth/GT_TWD_BN.csv'
    },
    'DoHome': {
        'products': 'data/products/dohome.json',
        'gt': 'data/ground_truth/GT_TWD_DM.csv'
    },
    'Megahome': {
        'products': 'data/products/megahome.json',
        'gt': 'data/ground_truth/GT_TWD_MG.csv'
    }
}

TWD_PRODUCTS = 'data/products/thaiwatsadu.json'

def load_json_products(filepath):
    """Load products from JSON or Excel file"""
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
        return df.to_dict('records')
    else:
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

def test_retailer(retailer_name, save_details=True):
    """Test all products for a single retailer"""
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
    
    valid_gt_count = 0
    invalid_gt_count = 0
    for twd_url, comp_url in gt.items():
        if comp_url in competitor_urls:
            valid_gt_count += 1
        else:
            invalid_gt_count += 1
    
    print(f"GT Validity: {valid_gt_count}/{len(gt)} ({100*valid_gt_count/len(gt):.1f}%) - {invalid_gt_count} invalid entries excluded")
    
    twd_indices_to_test = []
    for i, url in twd_url_map.items():
        if url in gt:
            expected_url = gt[url]
            if expected_url in competitor_urls:
                twd_indices_to_test.append(i)
    
    total_to_test = len(twd_indices_to_test)
    print(f"Testing {total_to_test} products (all valid GT)...")

    test_products = [twd_products[i] for i in twd_indices_to_test]

    # Process in batches with progress
    BATCH_SIZE = 50
    all_matches = []

    try:
        for batch_start in range(0, len(test_products), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(test_products))
            batch_products = test_products[batch_start:batch_end]

            # Progress display
            progress = (batch_end / total_to_test) * 100
            print(f"\r  Progress: {batch_end}/{total_to_test} ({progress:.1f}%) ", end='', flush=True)

            batch_matches = ai_match_products(batch_products, competitor_products)

            # Adjust source_idx for the batch offset
            if batch_matches:
                for m in batch_matches:
                    m['source_idx'] += batch_start
                all_matches.extend(batch_matches)

        print(f"\r  Progress: {total_to_test}/{total_to_test} (100.0%) - Done!       ")
        matches = all_matches

    except Exception as e:
        print(f"\nError during matching: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    correct = 0
    incorrect = 0
    not_found = 0
    
    wrong_matches = []
    not_found_list = []
    correct_list = []
    
    # Build match lookup by source_idx
    match_by_source = {m['source_idx']: m for m in matches} if matches else {}

    for idx, twd_idx in enumerate(twd_indices_to_test):
        twd_url = twd_url_map[twd_idx]
        expected_url = gt[twd_url]

        match = match_by_source.get(idx)

        if match is None or match.get('target_idx') is None:
            not_found += 1
            # Find expected name from competitor products
            exp_name = 'N/A'
            for ci, curl in competitor_url_map.items():
                if curl == expected_url:
                    exp_name = competitor_products[ci].get('name', competitor_products[ci].get('product_name', ''))[:60]
                    break
            not_found_list.append({
                'twd_name': test_products[idx].get('name', test_products[idx].get('product_name', ''))[:60],
                'twd_url': twd_url,
                'expected_name': exp_name,
                'expected_url': expected_url
            })
        else:
            match_idx = match['target_idx']
            matched_url = competitor_url_map.get(match_idx, '')
            
            if matched_url == expected_url:
                correct += 1
                correct_list.append({
                    'twd_name': test_products[idx].get('name', test_products[idx].get('product_name', ''))[:60],
                    'twd_url': twd_url,
                    'matched_name': competitor_products[match_idx].get('name', competitor_products[match_idx].get('product_name', ''))[:60],
                    'matched_url': matched_url,
                    'confidence': match.get('confidence', 0),
                    'reason': match.get('reason', '')
                })
            else:
                incorrect += 1
                expected_name = 'N/A'
                for ci, curl in competitor_url_map.items():
                    if curl == expected_url:
                        expected_name = competitor_products[ci].get('name', competitor_products[ci].get('product_name', ''))[:60]
                        break

                wrong_matches.append({
                    'twd_name': test_products[idx].get('name', test_products[idx].get('product_name', ''))[:60],
                    'twd_url': twd_url,
                    'got_name': competitor_products[match_idx].get('name', competitor_products[match_idx].get('product_name', ''))[:60],
                    'got_url': matched_url,
                    'expected_name': expected_name,
                    'expected_url': expected_url,
                    'confidence': match.get('confidence', 0),
                    'reason': match.get('reason', '')
                })
    
    total = len(twd_indices_to_test)
    accuracy = 100 * correct / total if total > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"RESULTS for {retailer_name}:")
    print(f"  Total GT tested: {total}")
    print(f"  Correct:    {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  Incorrect:  {incorrect}/{total} ({100*incorrect/total:.1f}%)")
    print(f"  Not Found:  {not_found}/{total} ({100*not_found/total:.1f}%)")
    print(f"  ACCURACY:   {accuracy:.1f}%")
    print(f"{'='*70}")
    
    if save_details:
        result = {
            'retailer': retailer_name,
            'timestamp': datetime.now().isoformat(),
            'total_tested': total,
            'correct': correct,
            'incorrect': incorrect,
            'not_found': not_found,
            'accuracy': accuracy,
            'correct_matches': correct_list,
            'wrong_matches': wrong_matches,
            'not_found_list': not_found_list
        }
        
        os.makedirs('results/matches', exist_ok=True)
        filename = f"results/matches/{retailer_name}_full_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {filename}")
    
    return {
        'retailer': retailer_name,
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'not_found': not_found,
        'accuracy': accuracy
    }

def main():
    if len(sys.argv) > 1:
        retailer = sys.argv[1]
        if retailer not in RETAILERS:
            print(f"Unknown retailer: {retailer}")
            print(f"Available: {', '.join(RETAILERS.keys())}")
            sys.exit(1)
        test_retailer(retailer)
    else:
        print("="*70)
        print("FULL PRODUCTION TEST - ALL RETAILERS")
        print("="*70)
        
        results = []
        for retailer in RETAILERS.keys():
            result = test_retailer(retailer)
            if result:
                results.append(result)
        
        print("\n" + "="*70)
        print("FINAL SUMMARY - ALL RETAILERS (FULL RUN)")
        print("="*70)
        print(f"{'Retailer':<15} {'Total':>8} {'Correct':>8} {'Wrong':>8} {'NotFound':>10} {'Accuracy':>10}")
        print("-"*70)
        
        total_correct = 0
        total_tested = 0
        
        for r in results:
            print(f"{r['retailer']:<15} {r['total']:>8} {r['correct']:>8} {r['incorrect']:>8} {r['not_found']:>10} {r['accuracy']:>9.1f}%")
            total_correct += r['correct']
            total_tested += r['total']
        
        print("-"*70)
        overall_accuracy = 100 * total_correct / total_tested if total_tested > 0 else 0
        print(f"{'TOTAL':<15} {total_tested:>8} {total_correct:>8} {'':<8} {'':<10} {overall_accuracy:>9.1f}%")
        print("="*70)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'total_tested': total_tested,
            'total_correct': total_correct,
            'overall_accuracy': overall_accuracy
        }
        
        os.makedirs('results/matches', exist_ok=True)
        filename = f"results/matches/full_production_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nFull summary saved to: {filename}")

if __name__ == '__main__':
    main()
