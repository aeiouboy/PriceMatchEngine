#!/usr/bin/env python3
"""Analyze error patterns from quick test runs."""

import sys
import os
import json
import re

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps', 'price_match_engine'))

retailer = sys.argv[1] if len(sys.argv) > 1 else 'HomePro'

# Load data
with open('data/products/thaiwatsadu.json') as f:
    twd_data = json.load(f)
if isinstance(twd_data, list):
    twd_products = {p['name']: p for p in twd_data}
else:
    twd_products = {p['name']: p for p in twd_data.get('products', twd_data)}

# Load retailer GT and products
GT_FILES = {
    'HomePro': 'data/ground_truth/GT_TWD_HP.csv',
    'GlobalHouse': 'data/ground_truth/GT_TWD_GB.csv',
    'Megahome': 'data/ground_truth/GT_TWD_MG.csv',
    'Boonthavorn': 'data/ground_truth/GT_TWD_BN.csv',
    'DoHome': 'data/ground_truth/GT_TWD_DM.csv'
}

RETAILER_FILES = {
    'HomePro': 'data/products/homepro.json',
    'GlobalHouse': 'data/products/globalhouse.json',
    'Megahome': 'data/products/megahome.json',
    'Boonthavorn': 'data/products/boonthavorn.json',
    'DoHome': 'data/products/dohome.json'
}

import pandas as pd
gt_df = pd.read_csv(GT_FILES[retailer])
with open(RETAILER_FILES[retailer]) as f:
    retailer_data = json.load(f)
if isinstance(retailer_data, list):
    retailer_products = {p['name']: p for p in retailer_data}
else:
    retailer_products = {p['name']: p for p in retailer_data.get('products', retailer_data)}

# Check GT validity
print(f"=== {retailer} GT Analysis ===")
print(f"GT entries: {len(gt_df)}")

col_name = retailer.lower()
if retailer == 'GlobalHouse':
    col_name = 'globalhouse'

valid_entries = []
for _, row in gt_df.iterrows():
    twd_name = row['thaiwatsadu']
    target_name = row[col_name]
    
    if twd_name in twd_products and target_name in retailer_products:
        valid_entries.append((twd_name, target_name))
        
print(f"Valid GT entries: {len(valid_entries)} / {len(gt_df)} ({100*len(valid_entries)/len(gt_df):.1f}%)")

# Look at product categories
import re
from collections import Counter

categories = Counter()
brands = Counter()

for twd_name, target_name in valid_entries[:200]:
    # Extract brand if present
    for brand in ['TOA', 'STANLEY', 'จระเข้', 'JOTUN', 'NIPPON', 'SHARK']:
        if brand in twd_name.upper():
            brands[brand] += 1
            
print(f"\nTop brands in sample:")
for brand, count in brands.most_common(10):
    print(f"  {brand}: {count}")
