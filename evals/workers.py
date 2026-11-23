"""Second process for the concurrency checks: python workers.py <store> <tag> <n>"""
import sys
from pathlib import Path
from hippocamp.embedders import HashEmbedder
from hippocamp.memory import Memory
d, tag, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = Memory(d, embedder=HashEmbedder(), device_id="dev-a", cache_dir=str(Path(d).parent / "cache"))
errs = 0
for i in range(n):
    try: m.assert_fact(f"{tag} fact {i} " + "x" * 600)
    except Exception as e: errs += 1
print(errs)
