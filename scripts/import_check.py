import importlib

mods = [
    'backend.parser',
    'backend.embeddings',
    'backend.vectorstore',
    'backend.rag_pipeline',
    'backend.llm',
]

all_ok = True
for m in mods:
    try:
        importlib.import_module(m)
        print('OK', m)
    except Exception as e:
        all_ok = False
        print('FAIL', m, type(e).__name__, e)

if not all_ok:
    raise SystemExit(1)
else:
    print('All imports OK')
