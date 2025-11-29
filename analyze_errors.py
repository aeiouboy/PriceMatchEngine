#!/usr/bin/env python3
"""Analyze error patterns from quick test runs."""

import sys
import json
import re

retailer = sys.argv[1] if len(sys.argv) > 1 else 'HomePro'

# Load data
with open('attached_assets/thaiwatsadu_products_1764301031441.json') as f:
    twd_data = json.load(f)
if isinstance(twd_data, list):
    twd_products = {p['name']: p for p in twd_data}
else:
    twd_products = {p['name']: p for p in twd_data.get('products', twd_data)}

# Load retailer GT and products
GT_FILES = {
    'HomePro': 'attached_assets/GT_TWD_HP(Sheet1)_1764330056612.csv',
    'GlobalHouse': 'attached_assets/GT_TWD_GH(Sheet1)_1764329959316.csv',
    'Megahome': 'attached_assets/GT_TWD_MH(Sheet1)_1764329979605.csv',
    'Boonthavorn': 'attached_assets/GT_TWD_BTV(Sheet1)_1764329930044.csv',
    'DoHome': 'attached_assets/GT_TWD_DH(Sheet1)_1764329912879.csv'
}

RETAILER_FILES = {
    'HomePro': 'attached_assets/homepro_products_1764330132712.json',
    'GlobalHouse': 'attached_assets/globalhouse_products_1764330141174.json',
    'Megahome': 'attached_assets/megahome_products_1764330153371.json',
    'Boonthavorn': 'attached_assets/boonthavorn_products_1764330167050.json',
    'DoHome': 'attached_assets/dohome_products_1764330180247.json'
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
