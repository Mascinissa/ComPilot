"""Summarize a ComPilot results JSON: for each benchmark, how many proposed
schedules were runnable vs. rejected (invalid/illegal) vs. crashed, the runnable
rate, and the best speedup found. Reads the recorded per-schedule outcomes, so it
works for both conversational and compact runs.

Usage: python evaluation/analyze_dialogue.py <results.json> [<results2.json> ...]
"""
import json
import sys


def analyze(path):
    with open(path) as f:
        data = json.load(f)
    print(f"\n{'='*80}\n{path}\n{'='*80}")
    grand = {}
    for name, prog in data['programs'].items():
        if 'Explored_schedules' not in prog:
            print(f"  {name:32s}  [FAILED: {str(prog.get('error'))[:60]}]")
            continue
        # Skip the injected baseline entry (empty schedule with outcome "run").
        scheds = [e for e in prog['Explored_schedules']
                  if not (e.get('schedule') == '' and e.get('outcome') == 'run')]
        cats = {}
        for e in scheds:
            cats[e['outcome']] = cats.get(e['outcome'], 0) + 1
        for k, v in cats.items():
            grand[k] = grand.get(k, 0) + v

        init = prog.get('Initial_execution_time')
        best = prog.get('Best_schedule', ['', init])
        speedup = init / best[1] if best and best[1] else 1.0
        total = len(scheds)
        ran = cats.get('ran', 0)
        frac = (ran / total) if total else 0.0
        print(f"  {name:32s} best_speedup={speedup:6.2f}x  "
              f"ran={ran:2d}/{total:2d} ({frac*100:4.0f}%)  {cats}")
    print(f"  {'-'*76}")
    print(f"  TOTAL explored-schedule outcomes: {grand}")
    return grand


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        analyze(p)
