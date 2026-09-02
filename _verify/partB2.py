import sys; sys.path.insert(0,'backend')
from chatbot import drop_unreadable, load_corpus
import json
w = json.load(open('data/workflow_chunks.json', encoding='utf-8'))
kept = drop_unreadable(w)
print(f'dropped {len(w)-len(kept)} of {len(w)} workflow chunks (expect 3)')
c = load_corpus()
print(f'production corpus: {len(c)} chunks (expect 1200 - dropped)')
