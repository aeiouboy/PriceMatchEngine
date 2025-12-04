#!/usr/bin/env python3
"""
Analyze failed matches from chunked test results to identify patterns for optimization.
"""

import os
import json
from collections import defaultdict

RESULTS_DIR = 'results/chunked_tests'

def load_all_results():
    """Load all chunk results and extract failures"""
    all_matches = []

    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(RESULTS_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        retailer = data.get('retailer', filename.split('_')[0])
        for match in data.get('matches', []):
            match['retailer'] = retailer
            all_matches.append(match)

    return all_matches

def analyze_failures(matches):
    """Analyze failure patterns"""

    correct = [m for m in matches if m.get('gt_status') == 'CORRECT']
    incorrect = [m for m in matches if m.get('gt_status') == 'INCORRECT']
    no_gt = [m for m in matches if m.get('gt_status') == 'NO_GT']

    print("="*80)
    print("FAILURE ANALYSIS REPORT")
    print("="*80)
    print(f"\nTotal matches: {len(matches)}")
    print(f"Correct: {len(correct)}")
    print(f"Incorrect: {len(incorrect)}")
    print(f"No GT (not in ground truth): {len(no_gt)}")
    print(f"Accuracy: {len(correct)/(len(correct)+len(incorrect))*100:.1f}%")

    # Group failures by retailer
    print("\n" + "="*80)
    print("FAILURES BY RETAILER")
    print("="*80)

    by_retailer = defaultdict(list)
    for m in incorrect:
        by_retailer[m['retailer']].append(m)

    for retailer, failures in sorted(by_retailer.items()):
        print(f"\n### {retailer}: {len(failures)} failures ###")

    # Analyze failure patterns
    print("\n" + "="*80)
    print("DETAILED FAILURE ANALYSIS")
    print("="*80)

    # Categorize failures
    categories = {
        'size_mismatch': [],
        'type_mismatch': [],
        'brand_confusion': [],
        'spec_mismatch': [],
        'wrong_variant': [],
        'other': []
    }

    for m in incorrect:
        source = m.get('source_name', '').lower()
        target = m.get('target_name', '').lower()
        reason = m.get('reason', '').lower()

        # Check for size/dimension mismatches
        import re
        source_sizes = re.findall(r'(\d+(?:\.\d+)?)\s*(นิ้ว|cm|mm|m|ซม\.|เมตร|inch|")', source)
        target_sizes = re.findall(r'(\d+(?:\.\d+)?)\s*(นิ้ว|cm|mm|m|ซม\.|เมตร|inch|")', target)

        # Check for wattage mismatches
        source_watts = re.findall(r'(\d+)\s*w', source)
        target_watts = re.findall(r'(\d+)\s*w', target)

        # Check for volume mismatches
        source_vol = re.findall(r'(\d+(?:\.\d+)?)\s*(l|ลิตร|ml|gal|แกลลอน)', source)
        target_vol = re.findall(r'(\d+(?:\.\d+)?)\s*(l|ลิตร|ml|gal|แกลลอน)', target)

        # Check for tier/count mismatches
        source_tiers = re.findall(r'(\d+)\s*(ชั้น|เส้น|ขั้น|tier)', source)
        target_tiers = re.findall(r'(\d+)\s*(ชั้น|เส้น|ขั้น|tier)', target)

        categorized = False

        # Size mismatch
        if source_sizes and target_sizes:
            if source_sizes != target_sizes:
                categories['size_mismatch'].append(m)
                categorized = True

        # Wattage mismatch
        if not categorized and source_watts and target_watts:
            if source_watts != target_watts:
                categories['spec_mismatch'].append(m)
                categorized = True

        # Volume mismatch
        if not categorized and source_vol and target_vol:
            if source_vol != target_vol:
                categories['spec_mismatch'].append(m)
                categorized = True

        # Tier/count mismatch
        if not categorized and source_tiers and target_tiers:
            if source_tiers != target_tiers:
                categories['spec_mismatch'].append(m)
                categorized = True

        # Type keywords that indicate different products
        type_keywords = [
            ('โคมไฟกิ่ง', 'ไฟผนัง'), ('โคมไฟกิ่ง', 'ไฟสนาม'),
            ('โคมไฟหัวเสา', 'ไฟผนัง'), ('โคมไฟแขวน', 'ไฟผนัง'),
            ('ดาวน์ไลท์', 'โคมไฟ'), ('หลอดไฟ', 'โคมไฟ'),
            ('เก้าอี้พับ', 'เก้าอี้เหล็ก'), ('เก้าอี้จัดเลี้ยง', 'เก้าอี้บาร์'),
            ('ตัดกิ่ง', 'อเนกประสงค์'), ('แชล็ค', 'น้ำมัน'),
            ('มีเบรก', 'ไม่มีเบรก'), ('อะไหล่', 'ลูกกลิ้ง'),
        ]

        if not categorized:
            for kw1, kw2 in type_keywords:
                if (kw1 in source and kw2 in target) or (kw2 in source and kw1 in target):
                    categories['type_mismatch'].append(m)
                    categorized = True
                    break

        if not categorized:
            categories['other'].append(m)

    # Print category summaries
    print("\n### FAILURE CATEGORIES ###\n")
    for cat, items in categories.items():
        print(f"{cat.upper()}: {len(items)} failures")

    # Print detailed failures for each category
    for cat, items in categories.items():
        if not items:
            continue
        print(f"\n{'='*60}")
        print(f"CATEGORY: {cat.upper()} ({len(items)} failures)")
        print('='*60)

        for i, m in enumerate(items[:10]):  # Show first 10
            print(f"\n--- Failure {i+1} [{m['retailer']}] ---")
            print(f"SOURCE: {m.get('source_name', '')[:80]}")
            print(f"MATCHED: {m.get('target_name', '')[:80]}")
            print(f"EXPECTED: (URL: {m.get('expected_url', '')[-50:]})")
            print(f"REASON: {m.get('reason', '')}")
            print(f"CONFIDENCE: {m.get('confidence', 0)}")

        if len(items) > 10:
            print(f"\n... and {len(items) - 10} more")

    return categories, incorrect

def suggest_optimizations(categories, incorrect):
    """Suggest optimizations based on failure patterns"""

    print("\n" + "="*80)
    print("OPTIMIZATION SUGGESTIONS")
    print("="*80)

    total_failures = len(incorrect)

    suggestions = []

    # Size mismatch suggestions
    if categories['size_mismatch']:
        pct = len(categories['size_mismatch']) / total_failures * 100
        suggestions.append({
            'category': 'Size Mismatch',
            'count': len(categories['size_mismatch']),
            'pct': pct,
            'suggestion': 'Stricter size extraction and comparison in extract_size_specs(). Add size tolerance rules.',
            'potential_gain': pct * 0.7  # Assume 70% of these can be fixed
        })

    # Spec mismatch suggestions
    if categories['spec_mismatch']:
        pct = len(categories['spec_mismatch']) / total_failures * 100
        suggestions.append({
            'category': 'Spec Mismatch (Wattage/Volume/Tiers)',
            'count': len(categories['spec_mismatch']),
            'pct': pct,
            'suggestion': 'Increase weight for wattage/volume/tier specs in calculate_spec_score(). Add STRICT matching for these.',
            'potential_gain': pct * 0.8
        })

    # Type mismatch suggestions
    if categories['type_mismatch']:
        pct = len(categories['type_mismatch']) / total_failures * 100
        suggestions.append({
            'category': 'Product Type Mismatch',
            'count': len(categories['type_mismatch']),
            'pct': pct,
            'suggestion': 'Add more entries to PRODUCT_LINE_CONFLICTS. Improve AI Stage 1 product type extraction.',
            'potential_gain': pct * 0.9
        })

    # Other suggestions
    if categories['other']:
        pct = len(categories['other']) / total_failures * 100
        suggestions.append({
            'category': 'Other/Uncategorized',
            'count': len(categories['other']),
            'pct': pct,
            'suggestion': 'Manual review needed. May require better cross-brand mapping or prompt tuning.',
            'potential_gain': pct * 0.3
        })

    # Print suggestions
    print("\n### PRIORITIZED SUGGESTIONS ###\n")
    suggestions.sort(key=lambda x: x['potential_gain'], reverse=True)

    current_accuracy = 68.7
    potential_accuracy = current_accuracy

    for s in suggestions:
        gain = s['potential_gain'] * (100 - current_accuracy) / 100
        potential_accuracy += gain
        print(f"\n{s['category']}:")
        print(f"  Failures: {s['count']} ({s['pct']:.1f}% of failures)")
        print(f"  Suggestion: {s['suggestion']}")
        print(f"  Potential accuracy gain: +{gain:.1f}%")

    print(f"\n### PROJECTED ACCURACY ###")
    print(f"Current: {current_accuracy:.1f}%")
    print(f"Potential (if all fixes applied): {min(potential_accuracy, 95):.1f}%")
    print(f"Target: 85%")
    print(f"Gap to close: {85 - current_accuracy:.1f}%")

def main():
    matches = load_all_results()
    categories, incorrect = analyze_failures(matches)
    suggest_optimizations(categories, incorrect)

    # Also output the raw incorrect matches for detailed review
    print("\n" + "="*80)
    print("FULL INCORRECT MATCHES (for manual review)")
    print("="*80)

    output_file = 'results/failure_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(incorrect, f, ensure_ascii=False, indent=2)
    print(f"\nFull failure data saved to: {output_file}")

if __name__ == '__main__':
    main()
