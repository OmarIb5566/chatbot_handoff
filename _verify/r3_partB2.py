import sys; sys.path.insert(0,'backend')
from chatbot import drop_unreadable, load_corpus
import json
w = json.load(open('data/workflow_chunks.json', encoding='utf-8'))
kept = drop_unreadable(w)
dropped = len(w) - len(kept)
print(f'dropped {dropped} of {len(w)} (expect exactly 1)')
c = load_corpus()
print(f'production corpus: {len(c)} chunks (expect 1199)')
# which chunk went, and confirm the two round-2 casualties came back
gone = [x for x in w if x not in kept]
for g in gone:
    print(f"  DROPPED: {g['filename']} | {g.get('section')}")
dam = [x for x in kept if 'Damietta' in x['filename']]
print(f"  Damietta chunks retained: {[x.get('section') for x in dam]}")
