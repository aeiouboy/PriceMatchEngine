#!/usr/bin/env python3
"""Quick test for enhanced prompt with few-shot examples"""
import json
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# Test cases based on real error patterns
test_cases = [
    {
        "name": "Size mismatch test",
        "source": "สีน้ำทาภายนอก ชนิดกึ่งเงา JOTUN TOUGH SHIELD BASE B สีขาว 9 ลิตร",
        "targets": [
            "0: สีน้ำภายนอกกึ่งเงา JOTUN รุ่น TOUGH SHIELD ขนาด 2.5 แกลลอน (Brand: JOTUN, Model: TOUGH SHIELD, Size: 2.5GL)",
            "1: สีน้ำภายนอก ชนิดกึ่งเงา JOTUN รุ่น JOTASHIELD INFINITY ขนาด 9 ลิตร (Brand: JOTUN, Model: JOTASHIELD, Size: 9L)",
            "2: สีน้ำภายนอกกึ่งเงา JOTUN รุ่น TOUGH SHIELD ขนาด 9 ลิตร (Brand: JOTUN, Model: TOUGH SHIELD, Size: 9L)",
        ],
        "expected_match": 2,  # Should match same size (9L)
        "expected_not_match": [0, 1],  # Should NOT match 2.5GL or different line
    },
    {
        "name": "Product line test (SUPERMATEX vs SUPERSHIELD)",
        "source": "สีน้ำทาภายนอก ชนิดกึ่งเงา TOA SUPERMATEX BASE A 9 ลิตร",
        "targets": [
            "0: TOA สีน้ำกึ่งเงา ภายนอก รุ่น Supershield Advance ขนาด 9 ลิตร (Brand: TOA, Model: SUPERSHIELD, Size: 9L)",
            "1: สีน้ำกึ่งเงาภายนอก ซุปเปอร์เมเทค รุ่น แอดวานซ์ ขนาด 9 ลิตร (Brand: TOA, Model: SUPERMATEX, Size: 9L)",
        ],
        "expected_match": 1,  # Should match SUPERMATEX not SUPERSHIELD
        "expected_not_match": [0],
    },
    {
        "name": "Thai-English test (เวเธอร์บอนด์=WEATHERBOND)",
        "source": "สีน้ำทาภายนอก NIPPON PAINT WEATHERBONDSHEEN BASE A 9L",
        "targets": [
            "0: สีน้ำภายนอก NIPPON รุ่น เวเธอร์บอนด์ กึ่งเงา ขนาด 9 ลิตร (Brand: NIPPON, Model: WEATHERBOND, Size: 9L)",
            "1: สีน้ำภายนอก NIPPON รุ่น ไฮบริดชีลด์ ขนาด 9 ลิตร (Brand: NIPPON, Model: HYBRIDSHIELD, Size: 9L)",
        ],
        "expected_match": 0,  # Should match WEATHERBOND
        "expected_not_match": [1],
    },
    {
        "name": "JOTASHIELD vs TOUGH SHIELD test",
        "source": "สีน้ำทาภายนอก ชนิดกึ่งเงา JOTUN JOTASHIELD ANTIFADE BASE A 2.5 แกลลอน",
        "targets": [
            "0: สีน้ำภายนอกกึ่งเงา JOTUN รุ่น TOUGH SHIELD ขนาด 2.5 แกลลอน (Brand: JOTUN, Model: TOUGH SHIELD, Size: 2.5GL)",
            "1: สีน้ำภายนอกเนียน JOTUN รุ่น โจตาชิลด์เฟล็กซ์ ขนาด 2.5 แกลลอน (Brand: JOTUN, Model: JOTASHIELD FLEX, Size: 2.5GL)",
            "2: สีน้ำภายนอกกึ่งเงา JOTUN รุ่น JOTASHIELD AF ขนาด 2.5 แกลลอน (Brand: JOTUN, Model: JOTASHIELD AF, Size: 2.5GL)",
        ],
        "expected_match": 2,  # Should match JOTASHIELD AF (ANTIFADE=AF)
        "expected_not_match": [0, 1],  # NOT TOUGH SHIELD or FLEX
    },
]

def run_test(test_case):
    """Run a single test case"""
    prompt = f"""Thai retail product matcher. Find the SAME product in targets.

SOURCE: {test_case['source']}

TARGETS:
{chr(10).join(test_case['targets'])}

=== EXAMPLES ===
Ex1: SOURCE: สีน้ำภายนอกกึ่งเงา JOTUN TOUGH SHIELD 9 ลิตร
0: JOTUN TOUGH SHIELD 2.5GL → ❌ WRONG SIZE (9L≠2.5GL)
1: JOTUN JOTASHIELD 9L → ❌ WRONG LINE (TOUGH SHIELD≠JOTASHIELD)
2: JOTUN TOUGH SHIELD 9L → ✓ CORRECT
Answer: {{"match_index":2,"confidence":95,"reason":"same line+size"}}

Ex2: SOURCE: สีน้ำกึ่งเงาภายนอก TOA SUPERMATEX 9 ลิตร
0: TOA Supershield 9L → ❌ WRONG LINE (SUPERMATEX≠SUPERSHIELD)
1: TOA SUPERMATEX 9L → ✓ CORRECT
Answer: {{"match_index":1,"confidence":95,"reason":"exact match"}}

Ex3: SOURCE: สีน้ำภายนอก NIPPON เวเธอร์บอนด์ 9L
0: NIPPON WEATHERBOND 9L → ✓ (เวเธอร์บอนด์=WEATHERBOND)
Answer: {{"match_index":0,"confidence":95,"reason":"Thai=English"}}

Ex4: SOURCE: TOA SUPERSHIELD กึ่งเงา 5GL
0: TOA Supershield เนียน 5GL → ✓ (finish differs OK, same product)
Answer: {{"match_index":0,"confidence":90,"reason":"finish can differ"}}

=== RULES ===
1. SIZE must match: 9L→9L, 2.5GL→2.5GL, 5GL→5GL (CRITICAL!)
2. PRODUCT LINE must match exactly:
   - SUPERMATEX ≠ SUPERSHIELD ≠ SUPERSHIELD ADVANCE
   - JOTASHIELD ≠ JOTASHIELD FLEX ≠ TOUGH SHIELD
   - FLEXISEAL ≠ QUICK SEALER (ควิกซิลเลอร์)
3. Thai=English: วีนิเลกซ์=VINILEX, โจตาชิลด์=JOTASHIELD, เวเธอร์บอนด์=WEATHERBOND
4. Finish type CAN differ (กึ่งเงา/ด้าน/เนียน OK)
5. Find BEST match, not perfect match

Return JSON: {{"match_index": <0-14 or null>, "confidence": <50-100>, "reason": "<brief>"}}"""

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Handle markdown code blocks
        if "```" in result_text:
            parts = result_text.split("```")
            for part in parts:
                if part.strip().startswith("json"):
                    result_text = part.strip()[4:].strip()
                    break
                elif part.strip().startswith("{"):
                    result_text = part.strip()
                    break
        
        # Extract JSON object from text
        import re
        json_match = re.search(r'\{[^{}]*\}', result_text)
        if json_match:
            result_text = json_match.group()
        
        # Clean up control characters
        result_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', result_text)
        
        result = json.loads(result_text)
        return result
    except Exception as e:
        return {"error": str(e), "raw": result_text if 'result_text' in dir() else "N/A"}

def main():
    print("=" * 60)
    print("Testing Enhanced Prompt with Few-Shot Examples")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"  Source: {test_case['source'][:60]}...")
        
        result = run_test(test_case)
        
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            failed += 1
            continue
            
        match_idx = result.get('match_index')
        confidence = result.get('confidence', 0)
        reason = result.get('reason', 'N/A')
        
        print(f"  AI Result: match_index={match_idx}, confidence={confidence}")
        print(f"  Reason: {reason}")
        
        if match_idx == test_case['expected_match']:
            print(f"  ✓ PASSED - Correctly matched index {match_idx}")
            passed += 1
        elif match_idx in test_case.get('expected_not_match', []):
            print(f"  ✗ FAILED - Matched wrong index {match_idx} (expected {test_case['expected_match']})")
            failed += 1
        else:
            print(f"  ? UNEXPECTED - Matched index {match_idx} (expected {test_case['expected_match']})")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(test_cases)} passed, {failed} failed")
    print("=" * 60)
    
    return passed, failed

if __name__ == "__main__":
    main()
