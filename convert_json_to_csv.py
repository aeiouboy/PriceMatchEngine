#!/usr/bin/env python3
"""Convert JSON test results to CSV format"""

import json
import pandas as pd
import sys

def convert_json_to_csv(json_file):
    """Convert JSON test results to CSV"""

    # Load JSON
    print(f"Loading {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract metadata
    retailer = data.get('retailer', 'Unknown')
    timestamp = data.get('timestamp', '')
    total_tested = data.get('total_tested', 0)
    correct = data.get('correct', 0)
    incorrect = data.get('incorrect', 0)
    not_found = data.get('not_found', 0)
    accuracy = data.get('accuracy', 0)

    print(f"\nRetailer: {retailer}")
    print(f"Timestamp: {timestamp}")
    print(f"Total: {total_tested}, Correct: {correct}, Incorrect: {incorrect}, Not Found: {not_found}")
    print(f"Accuracy: {accuracy:.1f}%")

    # Convert each section to CSV
    output_base = json_file.replace('.json', '')

    # 1. Correct matches
    if 'correct_matches' in data and data['correct_matches']:
        correct_df = pd.DataFrame(data['correct_matches'])
        correct_csv = f"{output_base}_correct.csv"
        correct_df.to_csv(correct_csv, index=False, encoding='utf-8')
        print(f"\n✓ Saved {len(correct_df)} correct matches to: {correct_csv}")

    # 2. Incorrect matches
    if 'wrong_matches' in data and data['wrong_matches']:
        incorrect_df = pd.DataFrame(data['wrong_matches'])
        incorrect_csv = f"{output_base}_incorrect.csv"
        incorrect_df.to_csv(incorrect_csv, index=False, encoding='utf-8')
        print(f"✗ Saved {len(incorrect_df)} incorrect matches to: {incorrect_csv}")

    # 3. Not found
    if 'not_found_list' in data and data['not_found_list']:
        notfound_df = pd.DataFrame(data['not_found_list'])
        notfound_csv = f"{output_base}_notfound.csv"
        notfound_df.to_csv(notfound_csv, index=False, encoding='utf-8')
        print(f"? Saved {len(notfound_df)} not found items to: {notfound_csv}")

    # 4. Combined summary
    all_results = []

    if 'correct_matches' in data:
        for item in data['correct_matches']:
            all_results.append({
                'status': 'CORRECT',
                'twd_name': item.get('twd_name', ''),
                'twd_url': item.get('twd_url', ''),
                'matched_name': item.get('matched_name', ''),
                'matched_url': item.get('matched_url', ''),
                'expected_name': item.get('matched_name', ''),
                'expected_url': item.get('matched_url', ''),
                'confidence': item.get('confidence', 0),
                'reason': item.get('reason', '')
            })

    if 'wrong_matches' in data:
        for item in data['wrong_matches']:
            all_results.append({
                'status': 'INCORRECT',
                'twd_name': item.get('twd_name', ''),
                'twd_url': item.get('twd_url', ''),
                'matched_name': item.get('got_name', ''),
                'matched_url': item.get('got_url', ''),
                'expected_name': item.get('expected_name', ''),
                'expected_url': item.get('expected_url', ''),
                'confidence': item.get('confidence', 0),
                'reason': item.get('reason', '')
            })

    if 'not_found_list' in data:
        for item in data['not_found_list']:
            all_results.append({
                'status': 'NOT_FOUND',
                'twd_name': item.get('twd_name', ''),
                'twd_url': item.get('twd_url', ''),
                'matched_name': '',
                'matched_url': '',
                'expected_name': item.get('expected_name', ''),
                'expected_url': item.get('expected_url', ''),
                'confidence': 0,
                'reason': ''
            })

    if all_results:
        all_df = pd.DataFrame(all_results)
        all_csv = f"{output_base}_all.csv"
        all_df.to_csv(all_csv, index=False, encoding='utf-8')
        print(f"📊 Saved {len(all_df)} total results to: {all_csv}")

    print(f"\n✅ Conversion complete!")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_json_to_csv.py <json_file>")
        sys.exit(1)

    convert_json_to_csv(sys.argv[1])
