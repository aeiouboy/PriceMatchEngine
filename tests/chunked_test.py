#!/usr/bin/env python3
"""
Chunked House Brand Test - Processes products in batches to avoid timeouts.
Saves progress after each chunk and can resume from checkpoint.

Usage:
  python tests/chunked_test.py --retailer HomePro --chunk 1    # Run chunk 1 (products 0-29)
  python tests/chunked_test.py --retailer HomePro --chunk 2    # Run chunk 2 (products 30-59)
  python tests/chunked_test.py --retailer HomePro --summary    # Show summary of all chunks
  python tests/chunked_test.py --summary-all                   # Summary all retailers
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from apps.house_brand_engine.app import (
    ai_find_house_brand_alternatives,
    get_openrouter_client,
    extract_brand,
    get_url,
    normalize_url
)

CHUNK_SIZE = 15
RESULTS_DIR = 'results/chunked_tests'

RETAILERS = {
    'HomePro': {
        'products': 'data/products/homepro.json',
        'gt': 'data/ground_truth/GT_HB_HP.csv',
        'input': 'data/house_brand_inputs/twd_hp_input.json',
        'gt_cols': ('Thaiwatsadu_link', 'HomePro_link')
    },
    'GlobalHouse': {
        'products': 'data/products/globalhouse.json',
        'gt': 'data/ground_truth/GT_HB_GB.csv',
        'input': 'data/house_brand_inputs/twd_gb_input.json',
        'gt_cols': ('Thaiwatsadu_link', 'GlobalHouse_link')
    },
    'DoHome': {
        'products': 'data/products/dohome.json',
        'gt': 'data/ground_truth/GT_HB_DM.csv',
        'input': 'data/house_brand_inputs/twd_dh_input.json',
        'gt_cols': ('Thaiwatsadu_link', 'DoHome_link')
    },
    'Boonthavorn': {
        'products': 'data/products/boonthavorn.json',
        'gt': 'data/ground_truth/GT_HB_BN.csv',
        'input': 'data/house_brand_inputs/twd_bn_input.json',
        'gt_cols': ('Thaiwatsadu_link', 'Boonthavorn_link')
    }
}

def load_json_products(filepath):
    """Load products from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict):
            return data.get('products', data.get('data', []))
        return data

def load_data(retailer):
    """Load TWD and competitor products"""
    config = RETAILERS[retailer]
    
    twd_products = load_json_products(config['input'])
    competitor_products = load_json_products(config['products'])
    
    return twd_products, competitor_products

def load_ground_truth(retailer):
    """Load ground truth mapping"""
    config = RETAILERS[retailer]
    gt_df = pd.read_csv(config['gt'])
    twd_col, comp_col = config['gt_cols']
    
    gt_dict = {}
    for _, row in gt_df.iterrows():
        twd_url = row.get(twd_col, '')
        comp_url = row.get(comp_col, '')
        if pd.notna(twd_url) and pd.notna(comp_url) and twd_url and comp_url:
            gt_dict[normalize_url(str(twd_url))] = normalize_url(str(comp_url))
    
    return gt_dict

def get_chunk_file(retailer, chunk_num):
    """Get path for chunk results file"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f'{retailer}_chunk_{chunk_num:03d}.json')

def run_chunk(retailer, chunk_num):
    """Run a single chunk of products"""
    print(f"\n{'='*60}")
    print(f"CHUNK TEST: {retailer} - Chunk {chunk_num}")
    print(f"{'='*60}")
    
    twd_products, competitor_products = load_data(retailer)
    gt_dict = load_ground_truth(retailer)
    
    total_products = len(twd_products)
    start_idx = (chunk_num - 1) * CHUNK_SIZE
    end_idx = min(start_idx + CHUNK_SIZE, total_products)
    
    if start_idx >= total_products:
        print(f"Chunk {chunk_num} is out of range. Total products: {total_products}")
        print(f"Max chunk number: {(total_products + CHUNK_SIZE - 1) // CHUNK_SIZE}")
        return None
    
    chunk_products = twd_products[start_idx:end_idx]
    
    print(f"Products: {start_idx+1} to {end_idx} of {total_products}")
    print(f"Chunk size: {len(chunk_products)}")
    print(f"Competitor catalog: {len(competitor_products)} products")
    print(f"Ground truth entries: {len(gt_dict)}")
    
    print(f"\nRunning AI matching...")
    
    def progress_callback(p):
        pct = int(p * 100)
        if pct % 10 == 0:
            print(f"  Progress: {pct}%")
    
    matches = ai_find_house_brand_alternatives(
        chunk_products,
        competitor_products,
        price_tolerance=0.30,
        progress_callback=progress_callback,
        retailer=retailer
    )
    
    results = []
    correct = 0
    incorrect = 0
    
    if matches:
        for m in matches:
            source_idx = m['source_idx']
            actual_source_idx = start_idx + source_idx
            
            source = chunk_products[source_idx]
            source_url = normalize_url(source.get('url', source.get('product_url', source.get('link', ''))))
            
            target = competitor_products[m['target_idx']]
            target_url = normalize_url(target.get('url', target.get('product_url', target.get('link', ''))))
            
            expected_url = gt_dict.get(source_url, '')
            
            if expected_url:
                is_correct = (target_url == expected_url)
                gt_status = 'CORRECT' if is_correct else 'INCORRECT'
                if is_correct:
                    correct += 1
                else:
                    incorrect += 1
            else:
                gt_status = 'NO_GT'
            
            results.append({
                'chunk': chunk_num,
                'source_idx': actual_source_idx,
                'source_name': source.get('name', source.get('product_name', '')),
                'source_brand': m['source_brand'],
                'source_url': source_url,
                'target_name': target.get('name', target.get('product_name', '')),
                'target_brand': m['target_brand'],
                'target_url': target_url,
                'expected_url': expected_url,
                'confidence': m['confidence'],
                'reason': m['reason'],
                'gt_status': gt_status
            })
    
    total_gt_tested = correct + incorrect
    accuracy = (correct / total_gt_tested * 100) if total_gt_tested > 0 else 0
    
    chunk_result = {
        'retailer': retailer,
        'chunk': chunk_num,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'products_tested': len(chunk_products),
        'matches_found': len(matches) if matches else 0,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat(),
        'matches': results
    }
    
    chunk_file = get_chunk_file(retailer, chunk_num)
    with open(chunk_file, 'w') as f:
        json.dump(chunk_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"CHUNK {chunk_num} RESULTS:")
    print(f"  Products tested: {len(chunk_products)}")
    print(f"  Matches found: {len(matches) if matches else 0}")
    print(f"  GT tested: {total_gt_tested}")
    print(f"  Correct: {correct}/{total_gt_tested} ({accuracy:.1f}%)")
    print(f"  Saved to: {chunk_file}")
    print(f"{'='*60}")
    
    return chunk_result

def show_summary(retailer):
    """Show summary of all completed chunks for a retailer"""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {retailer}")
    print(f"{'='*60}")
    
    twd_products, _ = load_data(retailer)
    total_products = len(twd_products)
    total_chunks = (total_products + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    print(f"Total products: {total_products}")
    print(f"Total chunks needed: {total_chunks}")
    print()
    
    all_correct = 0
    all_incorrect = 0
    all_matches = 0
    completed_chunks = []
    
    for chunk_num in range(1, total_chunks + 1):
        chunk_file = get_chunk_file(retailer, chunk_num)
        if os.path.exists(chunk_file):
            with open(chunk_file, 'r') as f:
                data = json.load(f)
            completed_chunks.append(chunk_num)
            all_correct += data['correct']
            all_incorrect += data['incorrect']
            all_matches += data['matches_found']
            print(f"  Chunk {chunk_num}: {data['correct']}/{data['correct']+data['incorrect']} correct ({data['accuracy']:.1f}%)")
        else:
            print(f"  Chunk {chunk_num}: NOT RUN")
    
    print()
    print(f"Completed: {len(completed_chunks)}/{total_chunks} chunks")
    
    if completed_chunks:
        total_gt = all_correct + all_incorrect
        overall_accuracy = (all_correct / total_gt * 100) if total_gt > 0 else 0
        print(f"Total matches: {all_matches}")
        print(f"Overall accuracy: {all_correct}/{total_gt} ({overall_accuracy:.1f}%)")
        
        missing = [c for c in range(1, total_chunks + 1) if c not in completed_chunks]
        if missing:
            print(f"\nMissing chunks: {missing}")
            print(f"Run: python tests/chunked_test.py --retailer {retailer} --chunk <num>")

def show_summary_all():
    """Show summary for all retailers"""
    print(f"\n{'='*60}")
    print("SUMMARY - ALL RETAILERS")
    print(f"{'='*60}\n")
    
    grand_correct = 0
    grand_incorrect = 0
    
    for retailer in RETAILERS:
        twd_products, _ = load_data(retailer)
        total_products = len(twd_products)
        total_chunks = (total_products + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        correct = 0
        incorrect = 0
        completed = 0
        
        for chunk_num in range(1, total_chunks + 1):
            chunk_file = get_chunk_file(retailer, chunk_num)
            if os.path.exists(chunk_file):
                with open(chunk_file, 'r') as f:
                    data = json.load(f)
                correct += data['correct']
                incorrect += data['incorrect']
                completed += 1
        
        total_gt = correct + incorrect
        accuracy = (correct / total_gt * 100) if total_gt > 0 else 0
        
        status = "COMPLETE" if completed == total_chunks else f"{completed}/{total_chunks}"
        print(f"{retailer:<15} {correct:>3}/{total_gt:<3} ({accuracy:>5.1f}%)  [{status}]")
        
        grand_correct += correct
        grand_incorrect += incorrect
    
    grand_total = grand_correct + grand_incorrect
    grand_accuracy = (grand_correct / grand_total * 100) if grand_total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"{'OVERALL':<15} {grand_correct:>3}/{grand_total:<3} ({grand_accuracy:>5.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Chunked House Brand Test')
    parser.add_argument('--retailer', type=str, help='Retailer name')
    parser.add_argument('--chunk', type=int, help='Chunk number to run (1-based)')
    parser.add_argument('--summary', action='store_true', help='Show summary for retailer')
    parser.add_argument('--summary-all', action='store_true', help='Show summary for all retailers')
    parser.add_argument('--list', action='store_true', help='List all retailers and chunk counts')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable retailers and chunk counts:")
        print("-" * 40)
        for retailer in RETAILERS:
            twd_products, _ = load_data(retailer)
            total = len(twd_products)
            chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
            print(f"  {retailer}: {total} products ({chunks} chunks)")
        return
    
    if args.summary_all:
        show_summary_all()
        return
    
    if not args.retailer:
        print("Usage:")
        print("  python tests/chunked_test.py --list")
        print("  python tests/chunked_test.py --retailer HomePro --chunk 1")
        print("  python tests/chunked_test.py --retailer HomePro --summary")
        print("  python tests/chunked_test.py --summary-all")
        return
    
    if args.retailer not in RETAILERS:
        print(f"Unknown retailer: {args.retailer}")
        print(f"Available: {list(RETAILERS.keys())}")
        return
    
    if args.summary:
        show_summary(args.retailer)
        return
    
    if args.chunk:
        run_chunk(args.retailer, args.chunk)
    else:
        print(f"Please specify --chunk <num> or --summary")
        show_summary(args.retailer)

if __name__ == '__main__':
    main()
