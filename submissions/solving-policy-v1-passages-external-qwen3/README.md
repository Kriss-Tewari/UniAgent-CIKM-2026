# solving-policy-v1-passages-external-qwen3

- Local case passage mode: `passages`.
- External corpus passage mode: `qwen3`.
- Original case documents remain available to the decision model.
- Policy-v1 administrative rules are preserved.
- Decision generation uses the TIRA-provided OpenAI-compatible endpoint.
- Final decision output budget: 4096 tokens.
- GPT-OSS defaults to low reasoning effort.
- Terrier 5.11 and helper 0.0.8 are cached during the image build.
- Qwen3 reranker weights are bundled only when external mode is qwen3.

TIRA command:

```sh
/app/run.sh --input $inputDataset --output $outputDir
```

The workflow tests indexing, retrieval, and any local reranker with container networking disabled before uploading.
