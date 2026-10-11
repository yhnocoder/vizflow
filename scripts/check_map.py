import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
text = (root / "CLAUDE.md").read_text(encoding="utf-8")
section = re.search(r"^## 代码地图\n(.*?)^## ", text, re.S | re.M)
if section is None:
    sys.exit("CLAUDE.md has no 代码地图 section")
paths = re.findall(r"`((?:src|test|examples|scripts|docs)/[^`\s]*)`", section.group(1))
missing = [p for p in paths if not (root / p).exists()]
for p in missing:
    print(f"CLAUDE.md 代码地图: {p} does not exist")
if missing:
    sys.exit(1)
print(f"代码地图: {len(paths)} paths exist")
