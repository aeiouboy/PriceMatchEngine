#!/usr/bin/env python3
"""
House Brand Matching - Accuracy Report Generator

This script generates a comprehensive accuracy report that shows:
- TRUE accuracy (including failed-to-match products)
- Clear breakdown of why products were not found
- ACCEPTABLE reasons (data issues) vs CRITICAL reasons (algorithm issues)

Usage:
    python scripts/generate_accuracy_report.py
"""

import json
import glob
import pandas as pd
from datetime import datetime


def analyze_retailer(retailer: str, input_file: str, gt_file: str, catalog_file: str, chunk_pattern: str) -> dict:
    """Analyze a retailer's matching results with detailed breakdown."""

    # Load TWD products (input)
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    twd_products = {p.get('url', '').lower().strip(): p for p in input_data.get('products', [])}

    # Get matched URLs and counts from chunk results
    matched_urls = set()
    correct_count = incorrect_count = gt_missing_count = 0

    chunk_files = sorted(glob.glob(chunk_pattern))
    for cf in chunk_files:
        with open(cf, 'r') as f:
            data = json.load(f)
        correct_count += data.get('correct', 0)
        incorrect_count += data.get('incorrect', 0)
        gt_missing_count += data.get('gt_missing', 0)
        for m in data.get('matches', []):
            matched_urls.add(m.get('source_url', '').lower().strip())

    # Load GT file
    gt_df = pd.read_csv(gt_file)
    gt_df['twd_url'] = gt_df['Thaiwatsadu_link'].str.lower().str.strip()

    # Find competitor columns
    comp_cols = [c for c in gt_df.columns if 'link' in c.lower() and 'twd' not in c.lower() and 'thai' not in c.lower()]
    comp_col = comp_cols[0] if comp_cols else None
    price_cols = [c for c in gt_df.columns if 'price' in c.lower() and 'twd' not in c.lower() and 'thai' not in c.lower()]
    price_col = price_cols[0] if price_cols else None

    # Load competitor catalog
    try:
        with open(catalog_file, 'r') as f:
            cat_data = json.load(f)
        cat_products = cat_data.get('products', cat_data) if isinstance(cat_data, dict) else cat_data
        cat_urls = {p.get('url', '').lower().strip() for p in cat_products if p.get('url')}
        cat_size = len(cat_products)
    except:
        cat_urls = set()
        cat_size = 0

    # Analyze not-found products
    no_gt = missing_in_catalog = price_filtered = low_similarity = 0

    for twd_url, twd_prod in twd_products.items():
        if twd_url in matched_urls:
            continue

        gt_row = gt_df[gt_df['twd_url'] == twd_url]

        # Check if GT entry exists
        if gt_row.empty or comp_col is None or pd.isna(gt_row[comp_col].values[0]):
            no_gt += 1
            continue

        expected_url = str(gt_row[comp_col].values[0]).lower().strip()

        # Check if expected product is in catalog
        if expected_url not in cat_urls:
            missing_in_catalog += 1
        else:
            # Check price difference
            twd_price = twd_prod.get('current_price', 0)
            comp_price = gt_row[price_col].values[0] if price_col else None

            if twd_price and comp_price and pd.notna(comp_price):
                try:
                    diff = abs(float(twd_price) - float(comp_price)) / float(twd_price) * 100
                    if diff > 30:
                        price_filtered += 1
                    else:
                        low_similarity += 1
                except:
                    low_similarity += 1
            else:
                low_similarity += 1

    return {
        'total': len(twd_products),
        'catalog_size': cat_size,
        'gt_size': len(gt_df),
        'matched': len(matched_urls),
        'correct': correct_count,
        'incorrect': incorrect_count,
        'gt_missing': gt_missing_count,
        'no_gt': no_gt,
        'missing_in_catalog': missing_in_catalog,
        'price_filtered': price_filtered,
        'low_similarity': low_similarity
    }


def print_retailer_report(name: str, r: dict):
    """Print detailed report for a single retailer."""

    not_found = r['total'] - r['matched']
    acceptable = r['no_gt'] + r['missing_in_catalog']
    critical = r['price_filtered'] + r['low_similarity']

    matchable = r['correct'] + r['incorrect'] + r['price_filtered'] + r['low_similarity']
    failed = r['incorrect'] + r['price_filtered'] + r['low_similarity']
    true_accuracy = r['correct'] / matchable * 100 if matchable > 0 else 0

    print(f'''
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ {name.upper():^85} ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ INPUT                                                                         ┃
┃   TWD Products:              {r['total']:>5}                                            ┃
┃   {name} Catalog:            {r['catalog_size']:>5} products                                   ┃
┃   GT Entries:                {r['gt_size']:>5}                                            ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ MATCHED (Found in Catalog):  {r['matched']:>5} ({r['matched']/r['total']*100:>5.1f}%)                                  ┃
┃   ├─ ✓ Correct:              {r['correct']:>5}                                            ┃
┃   ├─ ✗ Incorrect:            {r['incorrect']:>5}                                            ┃
┃   └─ ? No GT to validate:    {r['gt_missing']:>5}                                            ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ NOT FOUND:                   {not_found:>5} ({not_found/r['total']*100:>5.1f}%)                                  ┃
┃                                                                               ┃
┃   ACCEPTABLE (Data Issues):  {acceptable:>5}                                            ┃
┃     ├─ No GT entry:          {r['no_gt']:>5}                                            ┃
┃     └─ Missing in catalog:   {r['missing_in_catalog']:>5}                                            ┃
┃                                                                               ┃
┃   CRITICAL (Algorithm):      {critical:>5}  ← Must Fix!                               ┃
┃     ├─ Price filtered >30%:  {r['price_filtered']:>5}                                            ┃
┃     └─ Low similarity:       {r['low_similarity']:>5}                                            ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ ACCURACY                                                                      ┃
┃   Matchable Products:        {matchable:>5} (has GT + in catalog)                      ┃
┃   Correct:                   {r['correct']:>5}                                            ┃
┃   Failed:                    {failed:>5} (Incorrect + Price + Similarity)             ┃
┃                                                                               ┃
┃   ★ TRUE ACCURACY:           {r['correct']:>5} / {matchable:<5} = {true_accuracy:>5.1f}%                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛''')


def generate_report():
    """Generate the full accuracy report."""

    retailers = {
        'DoHome': {
            'input': 'data/house_brand_inputs/twd_dh_input.json',
            'gt': 'data/ground_truth/GT_HB_DM.csv',
            'catalog': 'data/products/dohome.json',
            'chunks': 'results/chunked_tests/DoHome_chunk_*.json'
        },
        'GlobalHouse': {
            'input': 'data/house_brand_inputs/twd_gb_input.json',
            'gt': 'data/ground_truth/GT_HB_GB.csv',
            'catalog': 'data/products/globalhouse.json',
            'chunks': 'results/chunked_tests/GlobalHouse_chunk_*.json'
        },
        'Boonthavorn': {
            'input': 'data/house_brand_inputs/twd_bn_input.json',
            'gt': 'data/ground_truth/GT_HB_BN.csv',
            'catalog': 'data/products/boonthavorn.json',
            'chunks': 'results/chunked_tests/Boonthavorn_chunk_*.json'
        },
        'HomePro': {
            'input': 'data/house_brand_inputs/twd_hp_input.json',
            'gt': 'data/ground_truth/GT_HB_HP.csv',
            'catalog': 'data/products/homepro.json',
            'chunks': 'results/chunked_tests/HomePro_chunk_*.json'
        },
    }

    print('=' * 90)
    print(f'HOUSE BRAND MATCHING - ACCURACY REPORT')
    print(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 90)

    results = {}
    for name, config in retailers.items():
        try:
            r = analyze_retailer(name, config['input'], config['gt'], config['catalog'], config['chunks'])
            results[name] = r
            print_retailer_report(name, r)
        except Exception as e:
            print(f'\n{name}: Error - {e}')

    # Print summary table
    print()
    print('=' * 90)
    print('SUMMARY')
    print('=' * 90)
    print()
    print(f'{"Retailer":<12} | {"Input":<6} | {"Catalog":<8} | {"Matchable":<10} | {"Correct":<8} | {"Failed":<8} | {"TRUE Acc":<10}')
    print('-' * 90)

    total_input = total_matchable = total_correct = total_failed = 0

    for name, r in results.items():
        matchable = r['correct'] + r['incorrect'] + r['price_filtered'] + r['low_similarity']
        failed = r['incorrect'] + r['price_filtered'] + r['low_similarity']
        true_accuracy = r['correct'] / matchable * 100 if matchable > 0 else 0

        total_input += r['total']
        total_matchable += matchable
        total_correct += r['correct']
        total_failed += failed

        print(f'{name:<12} | {r["total"]:<6} | {r["catalog_size"]:<8} | {matchable:<10} | {r["correct"]:<8} | {failed:<8} | {true_accuracy:<8.1f}%')

    print('-' * 90)
    overall_acc = total_correct / total_matchable * 100 if total_matchable > 0 else 0
    print(f'{"TOTAL":<12} | {total_input:<6} | {"":8} | {total_matchable:<10} | {total_correct:<8} | {total_failed:<8} | {overall_acc:<8.1f}%')
    print('=' * 90)

    # Print legend
    print()
    print('LEGEND:')
    print('  Input      = TWD products to match')
    print('  Catalog    = Competitor products available')
    print('  Matchable  = Products with GT + exists in catalog (algorithm SHOULD match)')
    print('  Correct    = Matched correctly')
    print('  Failed     = Incorrect + Price Filtered + Low Similarity')
    print('  TRUE Acc   = Correct / Matchable (real accuracy)')
    print()
    print('NOT FOUND REASONS:')
    print('  ACCEPTABLE (Data Issues - not algorithm fault):')
    print('    - No GT entry: Unknown expected match')
    print('    - Missing in catalog: Need to re-scrape competitor')
    print()
    print('  CRITICAL (Algorithm Issues - must fix):')
    print('    - Price filtered >30%: Found product but price too different')
    print('    - Low similarity: Found product but name matching failed')


if __name__ == '__main__':
    generate_report()
