"""Register the previously tested policy-v1 rerank agent as a TIRA Docker submission."""
import os
import subprocess
from pathlib import Path

from tira.rest_api_client import Client
from tira.tira_run import push_image


def main():
    token = os.environ["TIRA_CLIENT_TOKEN"]
    image = os.environ["SUBMISSION_IMAGE"]

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    client = Client(api_key=token)
    client.fail_if_api_key_is_invalid()


    expected_local_image = "lrv1:" + commit + "-rerank"
    if image != expected_local_image:
        raise RuntimeError("Unexpected local image name: " + image)

    registry_prefix = (
        client.docker_registry().rstrip("/")
        + "/code-research/tira/tira-user-rhineganga/"
    )
    expected_registry_image = client.local_execution.normalize_image_name(
        image, registry_prefix
    )
    if not expected_registry_image.endswith("/" + expected_local_image):
        raise RuntimeError("Registry normalization changed the unique image name.")
    print("Verified distinct registry image:", expected_registry_image)

    uploaded_image = push_image(
        client, image, "uniagent-2026", "rhineganga"
    )

    if uploaded_image != expected_registry_image:
        raise RuntimeError("Uploaded image differs from the verified name.")

    command = (
        "/run.sh --input $inputDataset --output $outputDir"
    )

    result = client.add_docker_software(
        image=uploaded_image,
        command=command,
        tira_vm_id="rhineganga",
        tira_task_id="uniagent-2026",
        code_repository_id=None,
        source_code_remotes={
            "origin": "https://github.com/Kriss-Tewari/UniAgent-CIKM-2026.git"
        },
        source_code_commit=commit,
        source_code_active_branch="main",
        forward_environment_variable=[
            "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY"
        ],
        build_environment={
            "GITHUB_REPOSITORY": os.environ.get("GITHUB_REPOSITORY", ""),
            "GITHUB_SHA": commit,
            "variant": "solving-policy-v1-rerank",
            "source_directory": "submissions/solving-policy-v1-rerank",
            "validation": "container checks; previous native public spot-check runs",
        },
    )

    name = result.get("display_name", "(see TIRA submission page)")
    print("Registered TIRA submission:", name)
    print("Uploaded image:", uploaded_image)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a") as stream:
            stream.write(
                f"\nTIRA submission: {name}\n\n"
                f"Variant: solving-policy-v1-rerank\n\n"
                f"Source commit: {commit}\n\n"
                "Hosted-model evaluation is performed by the organizers.\n"
            )


if __name__ == "__main__":
    main()
