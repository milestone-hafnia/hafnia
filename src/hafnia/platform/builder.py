import json
import os
from hashlib import sha256
from pathlib import Path
from shutil import rmtree
from typing import Dict, List, Optional
from zipfile import ZipFile

import boto3
from botocore.exceptions import ClientError

from hafnia.log import logger
from hafnia.platform import download_resource


def validate_recipe(zip_path: Path, required_paths: Optional[set] = None) -> None:
    """
    Validates the structure of a zip archive.
    Ensures the presence of specific files and directories.

    Args:
        zip_path (Path): Path to the zip archive.
        required_paths (set): A set of required paths relative to the archive root.

    Raises:
        FileNotFoundError: If any required file or directory is missing.
    """
    required_paths = {"src/lib/", "src/scripts/", "Dockerfile"} if required_paths is None else required_paths
    with ZipFile(zip_path, "r") as archive:
        archive_contents = {Path(file).as_posix() for file in archive.namelist()}
        missing_paths = {
            path for path in required_paths if not any(entry.startswith(path) for entry in archive_contents)
        }

        if missing_paths:
            raise FileNotFoundError(f"The following required paths are missing in the zip archive: {missing_paths}")

        script_files = [f for f in archive_contents if f.startswith("src/scripts/") and f.endswith(".py")]

        if not script_files:
            raise ValueError("No Python script files found in the 'src/scripts/' directory.")


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


def get_recipe_content(recipe_url: str, output_dir: Path, state_file: str, api_key: str) -> Dict:
    """
    Retrieves and validates the recipe content from an S3 location and extracts it.

    Args:
        recipe_uuid (str): The unique identifier of the recipe.
        output_dir (str): Directory to extract the recipe content.
        state_file (str): File to save the state information.

    Returns:
        Dict: Metadata about the recipe for further processing.
    """
    result = download_resource(recipe_url, output_dir, api_key)
    recipe_path = Path(result["downloaded_files"][0])

    validate_recipe(recipe_path)

    with ZipFile(recipe_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    tag = sha256(recipe_path.read_bytes()).hexdigest()[:8]

    scripts_dir = output_dir / "src/scripts"
    valid_commands = [str(f.name)[:-3] for f in scripts_dir.iterdir() if f.is_file() and f.suffix.lower() == ".py"]

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


def build_dockerfile(dockerfile: str, docker_context: str, docker_tag: str, secrets: Optional[Dict] = None) -> None:
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
    build_cmd = [
        "docker",
        "build",
        "--platform=linux/amd64",
        "-t",
        docker_tag,
        "-f",
        dockerfile,
    ]
    build_cmd.append(docker_context)
    logger.info(f"Building Docker image: {' '.join(build_cmd)}")
    subprocess.run(build_cmd, check=True)


def check_ecr(repository_name: str, image_tag: str) -> bool:
    aws_region = os.getenv("AWS_REGION", None)
    if aws_region is None:
        logger.warning("ECR registry region is not provided can not look up in the registry.")
        return False
    session = boto3.Session(region_name=aws_region)
    ecr_client = session.client("ecr")
    try:
        response = ecr_client.describe_images(repositoryName=repository_name, imageIds=[{"imageTag": image_tag}])
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


def prepare_recipe(recipe_url: str, output_dir: Path, api_key: str) -> Dict:
    state_file = output_dir / "state.json"
    get_recipe_content(recipe_url, output_dir, state_file.as_posix(), api_key)
    with open(state_file.as_posix(), "r") as f:
        return json.loads(f.read())


def build_image(image_info: Dict, ecr_prefix: str, state_file: str = "state.json") -> None:
    hafnia_tag = f"{ecr_prefix}/{image_info['name']}:{image_info['hash']}"
    image_exists = False
    if "localhost" not in ecr_prefix:
        image_exists = check_ecr(image_info["name"], image_info["hash"])

    image_info.update({"mdi_tag": hafnia_tag, "image_exists": image_exists})
    state_path = Path(state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if image_exists:
        logger.info(f"Image {hafnia_tag} already exists in ECR. Skipping build.")
    else:
        build_dockerfile(image_info["dockerfile"], image_info["docker_context"], hafnia_tag)
    with open(state_path.as_posix(), "w") as f:
        json.dump(image_info, f, indent=4)
