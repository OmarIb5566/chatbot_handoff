import ast
[ast.parse(open(f,encoding='utf-8').read()) for f in ['backend/retriever.py','backend/chatbot.py','backend/contextualize.py','evals/eval_retrieval.py']]
print('syntax OK')
