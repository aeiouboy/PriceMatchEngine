#!/usr/bin/env python3
"""
Batch test script for House Brand Matching across all retailers.
Runs each retailer sequentially and saves comprehensive results.
"""

import subprocess
import json
import os
import sys
from datetime import datetime

RETAILERS = ['Boonthavorn', 'GlobalHouse', 'DoHome', 'HomePro']
RESULTS_DIR = 'results/house_brand_tests'

def run_retailer_test(retailer, sample_size=0):
    """Run test for a single retailer and return results"""
    print(f"\n{'='*60}")
    print(f"Testing: {retailer} (sample: {'ALL' if sample_size == 0 else sample_size})")
    print(f"{'='*60}")
    
    cmd = [
        'uv', 'run', 'python', 'tests/test_house_brand.py',
        '--retailer', retailer,
        '--sample', str(sample_size)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        output = result.stdout + result.stderr
        
        correct = 0
        total = 0
        accuracy = 0
        not_found = 0
        
        for line in output.split('\n'):
            if 'Correct:' in line and '/' in line:
                parts = line.split()
                for p in parts:
                    if '/' in p and p[0].isdigit():
                        correct, total = map(int, p.split('/'))
                        break
            if 'RAW ACCURACY:' in line:
                for p in line.split():
                    if '%' in p:
                        accuracy = float(p.replace('%', ''))
                        break
            if 'Not Found:' in line:
                for p in line.split():
                    if '/' in p and p[0].isdigit():
                        not_found = int(p.split('/')[0])
                        break
        
        print(output)
        
        return {
            'retailer': retailer,
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'not_found': not_found,
            'status': 'completed'
        }
        
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {retailer} test took too long")
        return {
            'retailer': retailer,
            'status': 'timeout'
        }
    except Exception as e:
        print(f"ERROR: {retailer} - {e}")
        return {
            'retailer': retailer,
            'status': 'error',
            'error': str(e)
        }

def main():
    sample_size = 50
    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            sample_size = 0
        else:
            try:
                sample_size = int(sys.argv[1])
            except:
                pass
    
    print("="*60)
    print("HOUSE BRAND MATCHING - COMPREHENSIVE TEST")
    print(f"Sample size: {'ALL' if sample_size == 0 else sample_size}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    all_results = []
    
    for retailer in RETAILERS:
        result = run_retailer_test(retailer, sample_size)
        all_results.append(result)
    
    print("\n")
    print("="*60)
    print("SUMMARY - ALL RETAILERS")
    print("="*60)
    print(f"{'Retailer':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<12} {'Status'}")
    print("-"*60)
    
    total_correct = 0
    total_tested = 0
    
    for r in all_results:
        if r['status'] == 'completed':
            print(f"{r['retailer']:<15} {r['correct']:<10} {r['total']:<10} {r['accuracy']:.1f}%{'':<7} {r['status']}")
            total_correct += r['correct']
            total_tested += r['total']
        else:
            print(f"{r['retailer']:<15} {'-':<10} {'-':<10} {'-':<12} {r['status']}")
    
    print("-"*60)
    if total_tested > 0:
        overall_accuracy = (total_correct / total_tested) * 100
        print(f"{'OVERALL':<15} {total_correct:<10} {total_tested:<10} {overall_accuracy:.1f}%")
    
    summary_file = os.path.join(RESULTS_DIR, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'sample_size': sample_size,
            'results': all_results,
            'overall': {
                'total_correct': total_correct,
                'total_tested': total_tested,
                'accuracy': (total_correct / total_tested * 100) if total_tested > 0 else 0
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {summary_file}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
