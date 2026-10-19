"""Upload one fixed Task 2 external-evidence configuration."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

from tira.rest_api_client import Client
from tira.tira_run import push_image

ALLOWED = {
    "solving-policy-v1-passages-external-passages",
    "solving-policy-v1-passages-external-qwen3",
    "solving-policy-v1-rerank-external-passages",
    "solving-policy-v1-rerank-external-qwen3",
}

def main():
    folder = Path(os.environ["SUBMISSION_DIRECTORY"])
    config = json.loads((folder / "submission-config.json").read_text())
    variant = config["variant"]
    if variant not in ALLOWED:
        raise ValueError(f"Unexpected variant: {variant}")

    checksums = json.loads((folder / "package-checksums.json").read_text())
    for name, expected in checksums.items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Package changed: {name}")

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    client = Client(api_key=os.environ["TIRA_CLIENT_TOKEN"])
    client.fail_if_api_key_is_invalid()

    uploaded_image = push_image(
        client,
        os.environ["SUBMISSION_IMAGE"],
        "uniagent-2026",
        "rhineganga",
    )

    result = client.add_docker_software(
        image=uploaded_image,
        command="/app/run.sh --input $inputDataset --output $outputDir",
        tira_vm_id="rhineganga",
        tira_task_id="uniagent-2026",
        code_repository_id=None,
        source_code_remotes={
            "origin": "https://github.com/Kriss-Tewari/UniAgent-CIKM-2026.git"
        },
        source_code_commit=commit,
        source_code_active_branch="main",
        forward_environment_variable=[
            "OPENAI_BASE_URL",
            "OPENAI_MODEL",
            "OPENAI_API_KEY",
        ],
        build_environment={
            "GITHUB_REPOSITORY": os.environ.get("GITHUB_REPOSITORY", ""),
            "GITHUB_SHA": commit,
            "variant": variant,
            "source_directory": str(folder),
            "local_mode": config["local_mode"],
            "external_mode": config["external_mode"],
            "max_output_tokens": "4096",
            "gpt_oss_default_reasoning_effort": "low",
            "reranker_revision": config.get("reranker_revision") or "-",
            "validation": (
                "native public spot-check comparisons; "
                "offline container indexing/retrieval; "
                "local CPU reranker check where applicable"
            ),
        },
    )

    name = result.get("display_name", "(see TIRA submission page)")
    record = {
        "variant": variant,
        "submission_name": name,
        "image": uploaded_image,
        "source_commit": commit,
    }
    Path("tira-registration.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )

    print("Registered TIRA submission:", name)
    print("Variant:", variant)
    print("Uploaded image:", uploaded_image)
    print("Source commit:", commit)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a") as stream:
            stream.write(
                f"\nTIRA submission: {name}\n\n"
                f"Variant: {variant}\n\n"
                f"Source commit: {commit}\n\n"
                "Hosted-model evaluation is performed by the organizers.\n"
            )

if __name__ == "__main__":
    main()
