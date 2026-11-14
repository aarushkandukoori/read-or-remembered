"""RecallAudit v0 inference: flag reasoning steps that explicitly recall training knowledge.

Usage:
  .venv/bin/python -I scripts/recall_audit.py "I recall the upstream Django fix added a check"
  .venv/bin/python -I scripts/recall_audit.py --traj path/to/trajectory.traj.json   # mini-swe-agent format
Prints the probability of explicit recall (>= 0.5 = flagged).
"""
import json, pickle, sys
from pathlib import Path

MODEL = Path(__file__).resolve().parent.parent / 'models/recall-audit/recall_audit_v0.pkl'


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__); return
    model = pickle.load(open(MODEL, 'rb'))
    if sys.argv[1] == '--traj':
        msgs = json.load(open(sys.argv[2])).get('messages', [])
        for i, m in enumerate(msgs):
            if m.get('role') != 'assistant':
                continue
            for channel in ('reasoning_content', 'content'):
                text = m.get(channel) or (m.get('extra', {}) or {}).get(channel)
                if isinstance(text, str) and text.strip():
                    p = model.predict_proba([text])[0, 1]
                    if p >= 0.5:
                        print(f'message {i} {channel}: p={p:.2f}  {text[:120]!r}')
    else:
        text = ' '.join(sys.argv[1:])
        print(f'p(explicit recall) = {model.predict_proba([text])[0, 1]:.2f}')


if __name__ == '__main__':
    main()
