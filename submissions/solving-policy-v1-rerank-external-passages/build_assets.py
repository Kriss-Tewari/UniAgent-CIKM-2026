import hashlib
import json
from pathlib import Path

import pyterrier as pt

ROOT = Path(__file__).resolve().parent
config = json.loads((ROOT / "submission-config.json").read_text())

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

pt.terrier.set_version("5.11")
pt.terrier.set_helper_version("0.0.8")
pt.java.set_memory_limit(2048)
pt.java.init()

home = Path(pt.io.pyterrier_home())
jars = [
    home / "terrier-assemblies-5.11-jar-with-dependencies.jar",
    home / "terrier-python-helper-0.0.8.jar",
]
for path in jars:
    if not path.is_file():
        raise FileNotFoundError(path)

manifest = {
    "terrier": "5.11",
    "terrier_helper": "0.0.8",
    "jars": {path.name: digest(path) for path in jars},
}

if config["external_mode"] == "qwen3":
    from huggingface_hub import snapshot_download

    revision = config["reranker_revision"]
    target = ROOT / "models/qwen3-reranker"

    snapshot_download(
        repo_id="Qwen/Qwen3-Reranker-0.6B",
        revision=revision,
        local_dir=target,
        max_workers=4,
        allow_patterns=[
            "*.json", "*.safetensors", "*.model", "*.txt",
            "*.jinja", "*.tiktoken", "LICENSE*", "README*",
        ],
    )

    if not list(target.glob("*.safetensors")):
        raise RuntimeError("Reranker weights are missing.")

    provenance = {
        "repo_id": "Qwen/Qwen3-Reranker-0.6B",
        "resolved_commit": revision,
    }
    (target / "download_source.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    manifest["reranker"] = {
        **provenance,
        "files": {
            str(path.relative_to(target)): digest(path)
            for path in sorted(target.rglob("*"))
            if path.is_file() and ".cache" not in path.parts
        },
    }

(ROOT / "build-assets.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
print("Build-time assets prepared.")
