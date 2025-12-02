#!/usr/bin/env python3
"""
House Brand Matching Test Script

Tests the house brand matching system which finds:
- Same function/category products
- Different brands (house brand alternatives)
- Price within tolerance (default 30%)

This is SEPARATE from price match tests - does not modify existing tests.
"""

import pandas as pd
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps', 'house_brand_engine'))
from app import (
    ai_find_house_brand_alternatives,
    extract_brand,
    extract_category,
    extract_size_specs,
    check_price_within_tolerance,
    normalize_text
)

RETAILERS = {
    'HomePro': {
        'products': 'data/products/homepro.json',
        'gt': 'data/ground_truth/GT_HB_HP.csv',
        'input': 'data/house_brand_inputs/twd_hp_input.json'
    },
    'GlobalHouse': {
        'products': 'data/products/globalhouse.json',
        'gt': 'data/ground_truth/GT_HB_GB.csv',
        'input': 'data/house_brand_inputs/twd_gb_input.json'
    },
    'Boonthavorn': {
        'products': 'data/products/boonthavorn.json',
        'gt': 'data/ground_truth/GT_HB_BN.csv',
        'input': 'data/house_brand_inputs/twd_bn_input.json'
    },
    'DoHome': {
        'products': 'data/products/dohome.json',
        'gt': 'data/ground_truth/GT_HB_DM.csv',
        'input': 'data/house_brand_inputs/twd_dh_input.json'
    },
    'Megahome': {
        'products': 'data/products/megahome.json',
        'gt': 'data/ground_truth/GT_HB_MG.csv',
        'input': None
    }
}

TWD_PRODUCTS = 'data/products/thaiwatsadu.json'
RESULTS_DIR = 'results/house_brand_tests'
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_ground_truth(filepath):
    """Load house brand ground truth file
    
    Expected GT format (CSV):
    - twd_url: ThaiWatsadu product URL
    - competitor_url: Expected house brand alternative URL
    - (optional) twd_name, competitor_name for reference
    """
    if not os.path.exists(filepath):
        print(f"GT file not found: {filepath}")
        return None
    
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            gt_df = pd.read_csv(filepath, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"Could not read GT file: {filepath}")
        return None
    
    cols = gt_df.columns.tolist()
    twd_col = None
    comp_col = None
    
    for col in cols:
        col_lower = col.lower()
        if 'thaiwatsadu' in col_lower and 'link' in col_lower:
            twd_col = col
        elif 'twd' in col_lower and 'url' in col_lower:
            twd_col = col
        elif 'source' in col_lower and 'url' in col_lower:
            twd_col = col
    
    for col in cols:
        col_lower = col.lower()
        if any(r in col_lower for r in ['homepro', 'globalhouse', 'dohome', 'megahome', 'boonthavorn']):
            if 'link' in col_lower or 'url' in col_lower:
                comp_col = col
        elif 'competitor' in col_lower or 'alternative' in col_lower or 'target' in col_lower:
            if 'url' in col_lower or 'link' in col_lower:
                comp_col = col
    
    if not twd_col or not comp_col:
        for col in cols:
            col_lower = col.lower()
            if 'url' in col_lower or 'link' in col_lower:
                if not twd_col:
                    twd_col = col
                elif not comp_col:
                    comp_col = col
    
    if not twd_col or not comp_col:
        print(f"Could not identify URL columns in GT file. Columns: {cols}")
        return None
    
    print(f"GT columns: TWD={twd_col}, Competitor={comp_col}")
    
    gt_dict = {}
    for _, row in gt_df.iterrows():
        twd_url = str(row.get(twd_col, '')).strip()
        comp_url = str(row.get(comp_col, '')).strip()
        if twd_url and comp_url and twd_url.startswith('http') and comp_url.startswith('http'):
            gt_dict[twd_url] = comp_url
    
    return gt_dict

def load_json_products(filepath):
    """Load products from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict):
            return data.get('products', data.get('data', []))
        return data

def filter_products_by_category(products, categories):
    """Filter products to specific categories for focused testing"""
    filtered = []
    for p in products:
        name = (p.get('name', '') or p.get('product_name', '') or '').upper()
        cat = (p.get('category', '') or '').upper()
        for c in categories:
            if c.upper() in name or c.upper() in cat:
                filtered.append(p)
                break
    return filtered

def validate_house_brand_match(source, target, price_tolerance=0.30):
    """Validate that a match meets house brand criteria"""
    issues = []
    
    source_name = source.get('name', source.get('product_name', ''))
    target_name = target.get('name', target.get('product_name', ''))
    source_brand = extract_brand(source_name, source.get('brand', ''))
    target_brand = extract_brand(target_name, target.get('brand', ''))
    
    if source_brand and target_brand and source_brand == target_brand:
        issues.append(f"SAME_BRAND: {source_brand} == {target_brand}")
    
    source_price = float(source.get('current_price', source.get('price', 0)) or 0)
    target_price = float(target.get('current_price', target.get('price', 0)) or 0)
    
    if source_price > 0 and target_price > 0:
        price_diff_pct = abs(target_price - source_price) / source_price
        if price_diff_pct > price_tolerance:
            issues.append(f"PRICE_OUT_OF_RANGE: {price_diff_pct*100:.1f}% > {price_tolerance*100}%")
    
    source_cat = extract_category(source_name)
    target_cat = extract_category(target_name)
    if source_cat != 'OTHER' and target_cat != 'OTHER' and source_cat != target_cat:
        issues.append(f"CATEGORY_MISMATCH: {source_cat} != {target_cat}")
    
    return issues

def test_house_brand_matching(retailer_name, sample_size=50, categories=None, price_tolerance=0.30, use_gt=True):
    """Test house brand matching for a single retailer
    
    Args:
        retailer_name: Name of the competitor retailer
        sample_size: Number of products to sample (None = all)
        categories: List of categories to filter (None = all)
        price_tolerance: Max price difference (0.30 = 30%)
        use_gt: Whether to validate against ground truth file
    """
    print(f"\n{'='*70}")
    print(f"HOUSE BRAND TEST: TWD → {retailer_name}")
    print(f"{'='*70}")
    
    config = RETAILERS[retailer_name]
    
    input_file = config.get('input')
    if input_file and os.path.exists(input_file):
        twd_products = load_json_products(input_file)
        print(f"Using retailer-specific input: {input_file}")
    else:
        twd_products = load_json_products(TWD_PRODUCTS)
        print(f"Using full TWD catalog: {TWD_PRODUCTS}")
    
    competitor_products = load_json_products(config['products'])
    
    gt = None
    gt_matchable = {}
    if use_gt:
        gt = load_ground_truth(config['gt'])
        if gt:
            print(f"Loaded {len(gt)} GT entries for house brand validation")
            
            comp_url_to_product = {}
            for p in competitor_products:
                url = p.get('url', p.get('product_url', p.get('link', '')))
                if url:
                    comp_url_to_product[url.strip()] = p
            
            twd_url_to_product = {}
            for p in twd_products:
                url = p.get('url', p.get('product_url', p.get('link', '')))
                if url:
                    twd_url_to_product[url.strip()] = p
            
            for twd_url, comp_url in gt.items():
                twd_prod = twd_url_to_product.get(twd_url)
                comp_prod = comp_url_to_product.get(comp_url)
                
                if not comp_prod or not twd_prod:
                    continue
                
                twd_price = float(twd_prod.get('current_price', twd_prod.get('price', 0)) or 0)
                comp_price = float(comp_prod.get('current_price', comp_prod.get('price', 0)) or 0)
                
                if twd_price > 0 and comp_price > 0:
                    price_diff = abs(comp_price - twd_price) / twd_price
                    if price_diff <= price_tolerance:
                        gt_matchable[twd_url] = comp_url
            
            print(f"  - Matchable GT entries (in catalog & price OK): {len(gt_matchable)}/{len(gt)}")
        else:
            print("No GT file found - using criteria-based validation only")
    
    print(f"Loaded {len(twd_products)} TWD products, {len(competitor_products)} {retailer_name} products")
    
    if categories:
        twd_products = filter_products_by_category(twd_products, categories)
        competitor_products = filter_products_by_category(competitor_products, categories)
        print(f"Filtered to {len(twd_products)} TWD, {len(competitor_products)} {retailer_name} (categories: {categories})")
    
    if sample_size and len(twd_products) > sample_size:
        import random
        random.seed(42)
        twd_products = random.sample(twd_products, sample_size)
        print(f"Sampled {sample_size} TWD products for testing")
    
    print(f"\nRunning AI house brand matching (tolerance: {price_tolerance*100:.0f}%, retailer: {retailer_name})...")
    
    matches = ai_find_house_brand_alternatives(
        twd_products,
        competitor_products,
        price_tolerance=price_tolerance,
        progress_callback=lambda p: print(f"\rProgress: {p*100:.0f}%", end=''),
        retailer=retailer_name,
        gt_hints=gt
    )
    
    print()
    
    if not matches:
        print("No matches found!")
        return {
            'retailer': retailer_name,
            'total_source': len(twd_products),
            'total_target': len(competitor_products),
            'matches_found': 0,
            'valid_matches': 0,
            'invalid_matches': 0,
            'validation_rate': 0
        }
    
    print(f"\nFound {len(matches)} potential house brand alternatives")
    
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
    
    valid_count = 0
    invalid_count = 0
    validation_issues = []
    match_details = []
    
    gt_correct = 0
    gt_incorrect = 0
    gt_not_in_gt = 0
    gt_matchable_correct = 0
    gt_matchable_incorrect = 0
    
    for match in matches:
        source = twd_products[match['source_idx']]
        target = competitor_products[match['target_idx']]
        
        source_name = source.get('name', source.get('product_name', ''))
        target_name = target.get('name', target.get('product_name', ''))
        source_price = float(source.get('current_price', source.get('price', 0)) or 0)
        target_price = float(target.get('current_price', target.get('price', 0)) or 0)
        source_url = twd_url_map.get(match['source_idx'], '')
        target_url = competitor_url_map.get(match['target_idx'], '')
        
        issues = validate_house_brand_match(source, target, price_tolerance)
        
        gt_status = 'NOT_IN_GT'
        if gt and source_url:
            expected_url = gt.get(source_url)
            if expected_url:
                if expected_url == target_url:
                    gt_status = 'CORRECT'
                    gt_correct += 1
                else:
                    gt_status = 'INCORRECT'
                    gt_incorrect += 1
                
                if source_url in gt_matchable:
                    if expected_url == target_url:
                        gt_matchable_correct += 1
                    else:
                        gt_matchable_incorrect += 1
            else:
                gt_not_in_gt += 1
        
        match_detail = {
            'source_name': source_name,
            'source_brand': match.get('source_brand', ''),
            'source_price': source_price,
            'source_url': source_url,
            'target_name': target_name,
            'target_brand': match.get('target_brand', ''),
            'target_price': target_price,
            'target_url': target_url,
            'price_diff_pct': match.get('price_diff_pct', 0),
            'confidence': match.get('confidence', 0),
            'reason': match.get('reason', ''),
            'issues': issues,
            'valid': len(issues) == 0,
            'gt_status': gt_status
        }
        match_details.append(match_detail)
        
        if len(issues) == 0:
            valid_count += 1
        else:
            invalid_count += 1
            validation_issues.append({
                'source': source_name[:50],
                'target': target_name[:50],
                'issues': issues
            })
    
    validation_rate = valid_count / len(matches) * 100 if matches else 0
    
    gt_accuracy = 0
    gt_tested = gt_correct + gt_incorrect
    if gt_tested > 0:
        gt_accuracy = gt_correct / gt_tested * 100
    
    gt_matchable_tested = gt_matchable_correct + gt_matchable_incorrect
    gt_matchable_accuracy = 0
    if gt_matchable_tested > 0:
        gt_matchable_accuracy = gt_matchable_correct / gt_matchable_tested * 100
    
    total_gt = len(gt) if gt else 0
    
    print()
    print("="*70)
    print(f"RESULTS for {retailer_name}:")
    print(f"  Total GT tested: {gt_tested}")
    print(f"  Correct:    {gt_correct}/{gt_tested} ({gt_correct/gt_tested*100:.1f}%)" if gt_tested > 0 else "  Correct:    0/0 (0.0%)")
    print(f"  Incorrect:  {gt_incorrect}/{gt_tested} ({gt_incorrect/gt_tested*100:.1f}%)" if gt_tested > 0 else "  Incorrect:  0/0 (0.0%)")
    print(f"  Not Found:  {len(twd_products) - len(matches)}/{len(twd_products)} ({(len(twd_products) - len(matches))/len(twd_products)*100:.1f}%)" if len(twd_products) > 0 else "  Not Found:  0/0 (0.0%)")
    print(f"  RAW ACCURACY:   {gt_accuracy:.1f}%")
    print(f"  ---")
    print(f"  Matchable GT (in catalog + price OK): {len(gt_matchable)}")
    print(f"  Matchable tested: {gt_matchable_tested}")
    print(f"  Matchable correct: {gt_matchable_correct}")
    print(f"  MATCHABLE ACCURACY: {gt_matchable_accuracy:.1f}%")
    print("="*70)
    
    brand_distribution = {}
    for m in match_details:
        brand = m['target_brand'] or 'Unknown'
        brand_distribution[brand] = brand_distribution.get(brand, 0) + 1
    
    cheaper_count = sum(1 for m in match_details if m['target_price'] < m['source_price'])
    expensive_count = sum(1 for m in match_details if m['target_price'] > m['source_price'])
    same_price = len(match_details) - cheaper_count - expensive_count
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"house_brand_{retailer_name}_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(match_details, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {results_file}")
    
    return {
        'retailer': retailer_name,
        'total_source': len(twd_products),
        'total_target': len(competitor_products),
        'matches_found': len(matches),
        'valid_matches': valid_count,
        'invalid_matches': invalid_count,
        'validation_rate': validation_rate,
        'cheaper_count': cheaper_count,
        'brand_distribution': brand_distribution,
        'gt_correct': gt_correct,
        'gt_incorrect': gt_incorrect,
        'gt_not_in_gt': gt_not_in_gt,
        'gt_accuracy': gt_accuracy,
        'gt_matchable_total': len(gt_matchable),
        'gt_matchable_tested': gt_matchable_tested,
        'gt_matchable_correct': gt_matchable_correct,
        'gt_matchable_accuracy': gt_matchable_accuracy
    }

def test_all_retailers(sample_size=30, categories=None, price_tolerance=0.30):
    """Test house brand matching across all retailers"""
    print("="*70)
    print("HOUSE BRAND MATCHING - ALL RETAILERS TEST")
    print(f"Sample size: {sample_size}, Price tolerance: {price_tolerance*100:.0f}%")
    if categories:
        print(f"Categories: {categories}")
    print("="*70)
    
    results = []
    for retailer in RETAILERS:
        try:
            result = test_house_brand_matching(
                retailer,
                sample_size=sample_size,
                categories=categories,
                price_tolerance=price_tolerance
            )
            results.append(result)
        except Exception as e:
            print(f"Error testing {retailer}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("SUMMARY - HOUSE BRAND MATCHING RESULTS")
    print("="*70)
    print(f"{'Retailer':<15} {'Matches':<10} {'Valid':<10} {'Val Rate':<12} {'GT Acc':<10}")
    print("-"*60)
    
    for r in results:
        gt_tested = r.get('gt_correct', 0) + r.get('gt_incorrect', 0)
        gt_acc_str = f"{r['gt_accuracy']:.1f}%" if gt_tested > 0 else "N/A"
        print(f"{r['retailer']:<15} {r['matches_found']:<10} {r['valid_matches']:<10} {r['validation_rate']:.1f}%{'':<6} {gt_acc_str}")
    
    return results

def quick_test(sample_size=20):
    """Quick test with small sample for development"""
    print("QUICK TEST - House Brand Matching")
    print(f"Testing with {sample_size} samples from retailer-specific input files")
    
    return test_house_brand_matching(
        'HomePro',
        sample_size=sample_size,
        categories=None,
        price_tolerance=0.30
    )

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='House Brand Matching Test')
    parser.add_argument('--retailer', type=str, help='Test specific retailer')
    parser.add_argument('--sample', type=int, default=30, help='Sample size (default: 30)')
    parser.add_argument('--tolerance', type=float, default=0.30, help='Price tolerance (default: 0.30)')
    parser.add_argument('--categories', type=str, nargs='+', help='Filter by categories')
    parser.add_argument('--quick', action='store_true', help='Quick test with 10 samples')
    parser.add_argument('--all', action='store_true', help='Test all retailers')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    elif args.all:
        test_all_retailers(
            sample_size=args.sample,
            categories=args.categories,
            price_tolerance=args.tolerance
        )
    elif args.retailer:
        if args.retailer not in RETAILERS:
            print(f"Unknown retailer: {args.retailer}")
            print(f"Available: {list(RETAILERS.keys())}")
            sys.exit(1)
        test_house_brand_matching(
            args.retailer,
            sample_size=args.sample,
            categories=args.categories,
            price_tolerance=args.tolerance
        )
    else:
        print("Usage:")
        print("  python test_house_brand.py --quick                    # Quick test")
        print("  python test_house_brand.py --retailer HomePro         # Test one retailer")
        print("  python test_house_brand.py --all                      # Test all retailers")
        print("  python test_house_brand.py --all --sample 50          # 50 samples per retailer")
        print("  python test_house_brand.py --all --tolerance 0.20     # 20% price tolerance")
        print("  python test_house_brand.py --all --categories สี PAINT  # Filter categories")
