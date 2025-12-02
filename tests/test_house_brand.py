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
        'gt': 'data/ground_truth/GT_HB_HP.csv'
    },
    'GlobalHouse': {
        'products': 'data/products/globalhouse.json',
        'gt': 'data/ground_truth/GT_HB_GB.csv'
    },
    'Boonthavorn': {
        'products': 'data/products/boonthavorn.json',
        'gt': 'data/ground_truth/GT_HB_BN.csv'
    },
    'DoHome': {
        'products': 'data/products/dohome.json',
        'gt': 'data/ground_truth/GT_HB_DM.csv'
    },
    'Megahome': {
        'products': 'data/products/megahome.json',
        'gt': 'data/ground_truth/GT_HB_MG.csv'
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

def test_house_brand_matching(retailer_name, sample_size=50, categories=None, price_tolerance=0.30):
    """Test house brand matching for a single retailer"""
    print(f"\n{'='*70}")
    print(f"HOUSE BRAND TEST: TWD → {retailer_name}")
    print(f"{'='*70}")
    
    twd_products = load_json_products(TWD_PRODUCTS)
    competitor_products = load_json_products(RETAILERS[retailer_name])
    
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
    
    print(f"\nRunning AI house brand matching (tolerance: {price_tolerance*100:.0f}%)...")
    
    matches = ai_find_house_brand_alternatives(
        twd_products,
        competitor_products,
        price_tolerance=price_tolerance,
        progress_callback=lambda p: print(f"\rProgress: {p*100:.0f}%", end='')
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
    
    valid_count = 0
    invalid_count = 0
    validation_issues = []
    match_details = []
    
    for match in matches:
        source = twd_products[match['source_idx']]
        target = competitor_products[match['target_idx']]
        
        source_name = source.get('name', source.get('product_name', ''))
        target_name = target.get('name', target.get('product_name', ''))
        source_price = float(source.get('current_price', source.get('price', 0)) or 0)
        target_price = float(target.get('current_price', target.get('price', 0)) or 0)
        
        issues = validate_house_brand_match(source, target, price_tolerance)
        
        match_detail = {
            'source_name': source_name,
            'source_brand': match.get('source_brand', ''),
            'source_price': source_price,
            'target_name': target_name,
            'target_brand': match.get('target_brand', ''),
            'target_price': target_price,
            'price_diff_pct': match.get('price_diff_pct', 0),
            'confidence': match.get('confidence', 0),
            'reason': match.get('reason', ''),
            'issues': issues,
            'valid': len(issues) == 0
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
    
    print(f"\n--- VALIDATION RESULTS ---")
    print(f"Valid matches: {valid_count}/{len(matches)} ({validation_rate:.1f}%)")
    print(f"Invalid matches: {invalid_count}")
    
    if validation_issues:
        print(f"\n--- SAMPLE ISSUES (first 5) ---")
        for issue in validation_issues[:5]:
            print(f"  Source: {issue['source']}")
            print(f"  Target: {issue['target']}")
            print(f"  Issues: {issue['issues']}")
            print()
    
    brand_distribution = {}
    for m in match_details:
        brand = m['target_brand'] or 'Unknown'
        brand_distribution[brand] = brand_distribution.get(brand, 0) + 1
    
    print(f"\n--- ALTERNATIVE BRANDS FOUND ---")
    for brand, count in sorted(brand_distribution.items(), key=lambda x: -x[1])[:10]:
        print(f"  {brand}: {count}")
    
    cheaper_count = sum(1 for m in match_details if m['target_price'] < m['source_price'])
    expensive_count = sum(1 for m in match_details if m['target_price'] > m['source_price'])
    same_price = len(match_details) - cheaper_count - expensive_count
    
    print(f"\n--- PRICE COMPARISON ---")
    print(f"  Cheaper alternatives: {cheaper_count}")
    print(f"  Same price: {same_price}")
    print(f"  More expensive: {expensive_count}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"house_brand_{retailer_name}_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(match_details, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return {
        'retailer': retailer_name,
        'total_source': len(twd_products),
        'total_target': len(competitor_products),
        'matches_found': len(matches),
        'valid_matches': valid_count,
        'invalid_matches': invalid_count,
        'validation_rate': validation_rate,
        'cheaper_count': cheaper_count,
        'brand_distribution': brand_distribution
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
    print(f"{'Retailer':<15} {'Matches':<10} {'Valid':<10} {'Rate':<10}")
    print("-"*45)
    
    for r in results:
        print(f"{r['retailer']:<15} {r['matches_found']:<10} {r['valid_matches']:<10} {r['validation_rate']:.1f}%")
    
    return results

def quick_test(sample_size=10):
    """Quick test with small sample for development"""
    print("QUICK TEST - House Brand Matching")
    print("Testing paint products only with 10 samples")
    
    return test_house_brand_matching(
        'HomePro',
        sample_size=sample_size,
        categories=['สี', 'PAINT', 'สีทา', 'สีน้ำ'],
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
