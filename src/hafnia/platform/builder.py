import json
import os
import subprocess
import tempfile
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
    required_paths = {"src", "scripts", "Dockerfile"} if required_paths is None else required_paths
    with ZipFile(zip_path, "r") as archive:
        archive_contents = {Path(file).as_posix() for file in archive.namelist()}
        missing_paths = {
            path for path in required_paths if not any(entry.startswith(path) for entry in archive_contents)
        }

        if missing_paths:
            raise FileNotFoundError(f"The following required paths are missing in the zip archive: {missing_paths}")

        script_files = [f for f in archive_contents if f.startswith("scripts/") and f.endswith(".py")]

        if not script_files:
            raise ValueError("No Python script files found in the 'scripts' directory.")


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

    scripts_dir = output_dir / "scripts"
    valid_commands = [str(f.name)[:-3] for f in scripts_dir.iterdir() if f.is_file() and f.suffix.lower() == ".py"]

    if not valid_commands:
        raise ValueError("No valid Python script commands found in the 'scripts' directory.")

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


def prepare_recipe(recipe_url: str, output_dir: Path, api_key: str) -> Dict:
    state_file = output_dir / "state.json"
    get_recipe_content(recipe_url, output_dir, state_file.as_posix(), api_key)
    with open(state_file.as_posix(), "r") as f:
        return json.loads(f.read())


def buildx_available() -> bool:
    try:
        result = subprocess.run(["docker", "buildx", "version"], capture_output=True, text=True, check=True)
        return "buildx" in result.stdout.lower()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def build_dockerfile(dockerfile: str, docker_context: str, docker_tag: str, meta_file: str) -> None:
    """
    Build a Docker image using the provided Dockerfile.

    Args:
        dockerfile (str): Path to the Dockerfile.
        docker_context (str): Path to the build context.
        docker_tag (str): Tag for the Docker image.
        meta_file (Optional[str]): File to store build metadata.
    """

    if not Path(dockerfile).exists():
        raise FileNotFoundError("Dockerfile not found.")

    if buildx_available():
        cmd = [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64",
            "--build-arg",
            "BUILDKIT_INLINE_CACHE=1",
            "--load",
            f"--metadata-file={meta_file}",
            "-t",
            docker_tag,
            "-f",
            dockerfile,
            docker_context,
        ]
        logger.info(
            "Building Docker image with BuildKit (buildx), using --load (image will be available for docker push)…"
        )
    else:
        cmd = [
            "docker",
            "build",
            "-t",
            docker_tag,
            "-f",
            dockerfile,
            docker_context,
        ]
        logger.warning("Docker buildx is not available. Falling back to classic docker build (no cache, no metadata).")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker build failed: {e}")
        raise RuntimeError(f"Docker build failed: {e}")


def check_ecr(repository: str, image_tag: str) -> Optional[str]:
    """
    Returns the remote digest for TAG if it exists, otherwise None.
    """
    if "localhost" in repository:
        return None

    region = os.getenv("AWS_REGION")
    if not region:
        logger.warning("AWS_REGION environment variable not set. Skip image exist check.")
        return None

    repo_name = repository.split("/")[-1]
    ecr = boto3.client("ecr", region_name=region)
    try:
        out = ecr.describe_images(
            repositoryName=repo_name,
            imageIds=[{"imageTag": image_tag}],
        )
        return out["imageDetails"][0]["imageDigest"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "ImageNotFoundException":
            return None
        raise


def build_image(info: Dict, ecr_repo: str, state_file: str = "state.json") -> None:
    tag = f"{ecr_repo}/{info['name']}:{info['hash']}"
    info["image_tag"] = tag

    remote_digest = check_ecr(info["name"], info["hash"])
    info["image_exists"] = False #remote_digest is not None

    if info["image_exists"]:
        logger.info("Tag already in ECR – skipping build.")
    else:
        with tempfile.NamedTemporaryFile() as meta_tmp:
            meta_file = meta_tmp.name
            build_dockerfile(info["dockerfile"], info["docker_context"], tag, meta_file)
            with open(meta_file) as m:
                try:
                    build_meta = json.load(m)
                    info["local_digest"] = build_meta["containerimage.digest"]
                except Exception:
                    info["local_digest"] = ""

    Path(state_file).write_text(json.dumps(info, indent=2))
