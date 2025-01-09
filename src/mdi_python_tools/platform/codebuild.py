import json
from hashlib import sha256
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory
from typing import Dict, List
from zipfile import ZipFile

import boto3
from botocore.exceptions import ClientError

from mdi_python_tools.log import logger
from mdi_python_tools.mdi_sdk import download_resource


def validate_recipe(zip_path: Path, required_paths: set = None) -> None:
    """
    Validates the structure of a zip archive.
    Ensures the presence of specific files and directories.

    Args:
        zip_path (Path): Path to the zip archive.
        required_paths (set): A set of required paths relative to the archive root.

    Raises:
        FileNotFoundError: If any required file or directory is missing.
    """
    required_paths = (
        {"src/lib/", "src/scripts/", "Dockerfile"} if required_paths is None else required_paths
    )
    with ZipFile(zip_path, "r") as archive:
        archive_contents = {Path(file).as_posix() for file in archive.namelist()}
        missing_paths = {
            path
            for path in required_paths
            if not any(entry.startswith(path) for entry in archive_contents)
        }

        if missing_paths:
            raise FileNotFoundError(
                f"The following required paths are missing in the zip archive: {missing_paths}"
            )


def clean_up(files: List[Path], dirs: List[Path], prefix: str = "__") -> None:
    """
    Clean up a list of files first, and then remove all folders starting with a specific prefix.

    Args:
        paths (list[Path]): List of file and directory paths to clean up.
        prefix (str, optional): Prefix to match for folder removal. Defaults to "__".
    """
    for path in files:
        if path.exists() and path.is_file():
            path.unlink()

    for path in dirs:
        if path.exists() and path.is_dir():
            for sub_dir in path.glob(f"**/{prefix}*"):
                if sub_dir.is_dir():
                    rmtree(sub_dir)


def get_recipe_content(recipe_url: str, output_dir: str, state_file: str) -> Dict:
    """
    Retrieves and validates the recipe content from an S3 location and extracts it.

    Args:
        recipe_uuid (str): The unique identifier of the recipe.
        output_dir (str): Directory to extract the recipe content.
        state_file (str): File to save the state information.

    Returns:
        Dict: Metadata about the recipe for further processing.
    """
    result = download_resource(recipe_url, output_dir)
    recipe_path = Path(result["downloaded_files"][0])

    validate_recipe(recipe_path)

    with ZipFile(recipe_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    tag = sha256(recipe_path.read_bytes()).hexdigest()[:8]

    output_dir = Path(output_dir)
    scripts_dir = output_dir / "src/scripts"
    valid_commands = [
        str(f.name)[:-3] for f in scripts_dir.iterdir() if f.is_file() and f.suffix.lower() == ".py"
    ]

    if not valid_commands:
        raise ValueError("No valid Python script commands found in the 'src/scripts' directory.")

    state = {
        "user_data": (output_dir / "src").as_posix(),
        "docker_context": output_dir.as_posix(),
        "dockerfile": (output_dir / "Dockerfile").as_posix(),
        "docker_tag": f"runtime:{tag}",
        "hash": tag,
        "valid_commands": valid_commands,
    }

    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception as e:
        raise RuntimeError(f"Failed to write state file: {e}")

    clean_up([recipe_path], [output_dir])

    return state


def build_dockerfile(
    dockerfile: str, docker_context: str, docker_tag: str, secrets: dict = None
) -> None:
    """
    Build a Docker image using the provided Dockerfile.

    Args:
        dockerfile (Path): Path to the Dockerfile.
        docker_context (str): Path to the build context.
        docker_tag (str): Tag for the Docker image.
        secrets (dict, optional): Dictionary of secrets to pass to docker build.
            Each key-value pair will be passed as --secret id=key,env=value
    """

    import subprocess

    if not Path(dockerfile).exists():
        raise FileNotFoundError("Dockerfile not found.")
    logger.info(f"Building Docker image {docker_tag} from {dockerfile}")
    build_cmd = [
        "docker",
        "build",
        "-t",
        docker_tag,
        "-f",
        dockerfile,
    ]
    if secrets:
        for secret_id, env_var in secrets.items():
            build_cmd.extend(["--secret", f"id={secret_id},env={env_var}"])

    build_cmd.append(docker_context)
    subprocess.run(build_cmd, check=True)


def build_mdi_runtime(
    docker_context: Path,
    user_data: str,
    user_runtime_tag: str,
    mdi_tag: str,
    exec_cmd: str,
) -> None:
    DOCKERFILE = f"""
FROM --platform=linux/amd64 {user_runtime_tag}

ENV RECIPE_DIR=/opt/recipe

RUN mkdir -p $RECIPE_DIR
COPY src $RECIPE_DIR

RUN --mount=type=secret,id=pypi_index_url \
    pip install \
        --index-url $(cat /run/secrets/pypi_index_url) \
        --no-cache-dir \
        mdi-cli

RUN mkdir -p /opt/mdi \
    && echo "#!/bin/bash\\neval \$(mdi mount $RECIPE_DIR)\\nmdi launch {exec_cmd}" > /opt/mdi/entrypoint.sh \
    && chmod +x /opt/mdi/entrypoint.sh
"""
    dockerfile = docker_context / "mdi.Dockerfile"
    with open(dockerfile, "w") as f:
        f.write(DOCKERFILE.strip())
    secrets = {"pypi_index_url": "PYPI_INDEX_URL"}
    build_dockerfile(dockerfile.as_posix(), docker_context.as_posix(), mdi_tag, secrets=secrets)


def check_ecr(repository_name: str, image_tag: str) -> bool:
    ecr_client = boto3.client("ecr")
    try:
        response = ecr_client.describe_images(
            repositoryName=repository_name, imageIds=[{"imageTag": image_tag}]
        )
        if response["imageDetails"]:
            logger.info(f"Image {image_tag} already exists in ECR.")
            return True
        else:
            return False
    except ClientError as e:
        if e.response["Error"]["Code"] == "ImageNotFoundException":
            logger.info(f"Image {image_tag} does not exist in ECR.")
            return False
        else:
            raise e


# def get_api_key() -> str:
#     """
#     Retrieves the MDI API key from local config or AWS Secrets Manager.

#     The function will:
#       1. Attempt to get the key from local configuration via CONFIG.get_api_key().
#       2. If not found, it will look for the Secrets Manager name in the environment
#          variable 'MDI_API_KEY'.
#       3. Fetch the secret value from AWS Secrets Manager using the provided secret name
#          and the region from the environment variable 'AWS_REGION' (defaulting to 'us-west-1').

#     Returns:
#         str: The MDI API key.

#     Raises:
#         ValueError: If the environment variable 'MDI_API_KEY' is not set or is empty.
#         ClientError: If fetching the secret from AWS Secrets Manager fails.
#     """
#     local_api_key = CONFIG.get_api_key()
#     if local_api_key:
#         return local_api_key

#     api_key_name = os.getenv("MDI_API_KEY")
#     if not api_key_name:
#         raise ValueError(
#             "Environment variable 'MDI_API_KEY' is not set or is empty. "
#             "Update it or use CLI to configure 'mdi sys configure'."
#         )

#     aws_region = os.getenv("AWS_REGION", "us-west-1")

#     client = boto3.client("secretsmanager", region_name=aws_region)
#     try:
#         response = client.get_secret_value(SecretId=api_key_name)
#         secret_value = response.get("SecretString")
#         if not secret_value:
#             # Secrets Manager can return the secret in 'SecretBinary' instead if it's not a string
#             raise ValueError("Secret string was not found in the AWS Secrets Manager response.")
#         return secret_value
#     except (BotoCoreError, ClientError) as e:
#         logger.error(f"Failed to retrieve secret from AWS Secrets Manager: {e}")
#         raise ClientError(
#             error_response={"Error": {"Message": str(e)}},
#             operation_name="get_secret_value",
#         )


def run_build(
    recipe_url: str, exec_cmd: str, state_file: str = "", ecr_repository: str = ""
) -> None:
    state_file = state_file or "state.json"
    with TemporaryDirectory() as tmp_dir:
        get_recipe_content(recipe_url, tmp_dir, state_file)
        state = json.loads(Path(state_file).read_text())
        prefix = f"{ecr_repository}/" if ecr_repository else ""
        state["mdi_tag"] = f"{prefix}mdi-runtime:{state['hash']}"
        state["image_exists"] = check_ecr("mdi-runtime", state["hash"]) if ecr_repository else False
        build_dockerfile(state["dockerfile"], state["docker_context"], state["mdi_tag"])
        with open(state_file.as_posix(), "w") as f:
            json.dump(state, f)
