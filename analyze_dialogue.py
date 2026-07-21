"""Analyze a ComPilot results JSON and report how productive the system/LLM
dialogue was: the distribution of compiler feedback categories, the fraction of
runnable schedules, how many exchanges were 'unproductive' (invalid/illegal/crash),
and the best speedup found per benchmark.

Usage: python analyze_dialogue.py <results.json> [<results2.json> ...]
"""
import json
import re
import sys


def categorize(msg):
    """Map a compiler feedback message to a coarse category."""
    m = msg.lower()
    if 'yielded a speedup' in m:
        return 'ran'
    if 'is illegal' in m or 'was not legal' in m:
        return 'illegal'
    # New-style precise validity errors
    if 'is invalid and was not applied' in m:
        return 'invalid'
    # Old-style vague, content-free creation crash (the message we set out to fix)
    if 'crashed at creation for unclear reasons' in m:
        return 'crash_unclear'
    if 'crashed at creation with' in m or 'could not be applied' in m:
        return 'invalid'  # old-style but with a message
    # Genuine backend crashes (named error / segfault / abort) with guidance
    if ('segmentation fault' in m or 'crashed at runtime' in m or 'crashed during' in m
            or 'made the compiler crash' in m or 'could not be built due to an unexpected' in m):
        return 'crash'
    if 'could not parse' in m or 'could not find any transformation' in m:
        return 'parse_fail'
    if 'already suggested' in m:
        return 'duplicate'
    # continue-exploration nudges / start / analysis prompts
    return 'other'


def analyze(path):
    with open(path) as f:
        data = json.load(f)
    print(f"\n{'='*80}\n{path}\n{'='*80}")
    grand = {}
    for name, prog in data['programs'].items():
        if 'error' in prog and 'Optimization_dialogue' not in prog:
            print(f"  {name:32s}  [FAILED: {prog['error'][:60]}]")
            continue
        cats = {}
        for msg in prog.get('Optimization_dialogue', []):
            if msg['role'] != 'user':
                continue
            c = categorize(msg['content'])
            cats[c] = cats.get(c, 0) + 1
        for k, v in cats.items():
            grand[k] = grand.get(k, 0) + v
        # best speedup
        init = prog.get('Initial_execution_time')
        best = prog.get('Best_schedule', ['', init])
        speedup = init / best[1] if best and best[1] else 1.0
        # count schedule-carrying feedback (exclude prompts/nudges/other)
        sched_cats = {k: v for k, v in cats.items()
                      if k in ('ran', 'illegal', 'invalid', 'crash', 'crash_unclear', 'duplicate')}
        total_sched = sum(sched_cats.values())
        ran = sched_cats.get('ran', 0)
        frac = (ran / total_sched) if total_sched else 0.0
        print(f"  {name:32s} best_speedup={speedup:6.2f}x  "
              f"ran={ran:2d}/{total_sched:2d} ({frac*100:4.0f}%)  {sched_cats}")
    print(f"  {'-'*76}")
    print(f"  TOTAL compiler-message categories: {grand}")
    return grand


if __name__ == '__main__':
    for p in sys.argv[1:]:
        analyze(p)
