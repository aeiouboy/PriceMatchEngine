#!/usr/bin/env python3
"""
Run all chunks for all retailers sequentially.
"""
import subprocess
import sys
import os

RETAILERS = {
    'HomePro': 33,
    'GlobalHouse': 7,
    'DoHome': 23,
    'Boonthavorn': 5
}

def run_all_chunks():
    """Run all chunks for all retailers"""
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    for retailer, num_chunks in RETAILERS.items():
        print(f"\n{'='*70}")
        print(f"STARTING: {retailer} ({num_chunks} chunks)")
        print(f"{'='*70}\n")

        for chunk in range(1, num_chunks + 1):
            print(f"\n--- {retailer} chunk {chunk}/{num_chunks} ---")
            cmd = [sys.executable, 'tests/chunked_test.py', '--retailer', retailer, '--chunk', str(chunk)]
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                print(f"Warning: Chunk {chunk} may have had issues")

        # Show summary for this retailer
        cmd = [sys.executable, 'tests/chunked_test.py', '--retailer', retailer, '--summary']
        subprocess.run(cmd, capture_output=False)

    # Show overall summary
    print("\n\n")
    cmd = [sys.executable, 'tests/chunked_test.py', '--summary-all']
    subprocess.run(cmd, capture_output=False)

if __name__ == '__main__':
    run_all_chunks()
