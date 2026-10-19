"""Explicit online preparation. Retrieval/reranking never downloads at runtime."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='models/qwen3-reranker')
parser.add_argument('--revision', default='main', help='Prefer a fixed commit for reproducibility')
args = parser.parse_args()
resolved = HfApi().model_info('Qwen/Qwen3-Reranker-0.6B', revision=args.revision).sha
snapshot_download(repo_id='Qwen/Qwen3-Reranker-0.6B', revision=resolved, local_dir=args.output,
                  allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken','*.jinja','LICENSE*','README.md'])
Path(args.output,'download_source.json').write_text(json.dumps({'repo_id':'Qwen/Qwen3-Reranker-0.6B',
    'requested_revision':args.revision, 'resolved_commit':resolved},indent=2))
print('Downloaded to', Path(args.output).resolve())
