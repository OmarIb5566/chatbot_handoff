import sys; sys.path.insert(0,'backend')
from contextualize import looks_like_followup
tests = [
  ('who signs after that', True),
  ('what happens before that', True),
  ('which form number is that', False),
  ('that is not all', True),
  ('what is the next step in the RFQ process', False),
]
for q, want in tests:
    got = looks_like_followup(q, has_history=True).is_followup
    print(f"{'OK ' if got==want else 'FAIL'}  {q!r:42} got={got} want={want}")
