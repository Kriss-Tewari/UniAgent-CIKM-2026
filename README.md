# UniAgent CIKM 2026 — RhineGanga

Task 2 custom submissions.

## solving-smolagents-policy-v1

Derived from the official solving-smolagents baseline, with two additional instructions: cross-document evidence verification and checks that distinguish financial roles and net/gross amounts. The original policies, tools, prediction format, and event logging are preserved.

Previous native public spot-check results using Qwen/Qwen2.5-7B-Instruct: travel 5/5, expenses 7/8, procurement 6/8. These are development results.

Runtime model configuration comes from OPENAI_BASE_URL, OPENAI_MODEL, and OPENAI_API_KEY.

The manually triggered GitHub workflow checks container packaging and registers a Docker submission with TIRA. It does not run an LLM accuracy evaluation. Hosted-model runs are performed by the organizers.
