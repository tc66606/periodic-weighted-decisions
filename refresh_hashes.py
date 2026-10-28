# refresh_hashes.py - run from the repository root after editing README.md and/or CITATION.cff:
#   python refresh_hashes.py README.md CITATION.cff
# Updates the SHA-256 and byte count of each named file in manifest.json and manifest.sha256,
# then the manifest.json line in manifest.sha256. Standard library only. Do not commit this script.
import hashlib, re, sys
from pathlib import Path

names = sys.argv[1:] or ["README.md"]
manifest = Path("manifest.json")
text = manifest.read_bytes().decode("utf-8")
for name in names:
    data = Path(name).read_bytes()
    digest, size = hashlib.sha256(data).hexdigest(), len(data)
    pattern = re.compile(r'("artifact_id": "%s",\s*"sha256": ")[0-9a-f]{64}(",\s*"bytes": )\d+' % re.escape(name))
    text, count = pattern.subn(lambda m: m.group(1) + digest + m.group(2) + str(size), text)
    assert count == 1, "not found in manifest.json: " + name
manifest.write_bytes(text.encode("utf-8"))

sums = Path("manifest.sha256")
lines = sums.read_bytes().decode("utf-8").splitlines(keepends=True)
for name in names + ["manifest.json"]:
    digest = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    hits = [i for i, line in enumerate(lines) if line.rstrip("\r\n").endswith("  " + name)]
    assert len(hits) == 1, "not found in manifest.sha256: " + name
    lines[hits[0]] = digest + "  " + name + "\n"
sums.write_bytes("".join(lines).encode("utf-8"))
print("updated:", ", ".join(names + ["manifest.json"]))
