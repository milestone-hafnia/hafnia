import json
import os
import subprocess
import tempfile
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional

import boto3
from botocore.exceptions import ClientError

from hafnia.log import sys_logger, user_logger
from hafnia.platform import download_resource


def prepare_recipe(recipe_url: str, output_dir: Path, api_key: str) -> Dict:
    state_file = output_dir / "state.json"
    resource = download_resource(recipe_url, output_dir.as_posix(), api_key)
    recipe_path = Path(resource["downloaded_files"][0])
    with zipfile.ZipFile(recipe_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    required = {"src", "scripts", "Dockerfile"} if required is None else required
    errors = 0
    for rp in required:
        if not (output_dir / rp).exists():
            user_logger.error(f"Required path {rp} not found in recipe.")
            errors += 1
    if errors > 0:
        raise Exception("Wrong recipe structure")

    tag = sha256(recipe_path.read_bytes()).hexdigest()[:8]
    scripts_dir = output_dir / "scripts"
    if not any(scripts_dir.iterdir()):
        user_logger.warning("Scripts folder is empty")

    metadata = {
        "user_data": (output_dir / "src").as_posix(),
        "docker_context": output_dir.as_posix(),
        "dockerfile": (output_dir / "Dockerfile").as_posix(),
        "docker_tag": f"runtime:{tag}",
        "hash": tag,
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return metadata


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

    cmd = ["docker", "build", "--platform", "linux/amd64", "-t", docker_tag, "-f", dockerfile]

    remote_cache = os.getenv("REMOTE_CACHE_REPO")
    cloud_mode = os.getenv("HAFNIA_CLOUD", "false").lower() in ["true", "1", "yes"]

    if buildx_available():
        cmd.insert(1, "buildx")
        cmd += ["--build-arg", "BUILDKIT_INLINE_CACHE=1", "--metadata-file", meta_file]
        if cloud_mode:
            cmd += ["--push"]
        if remote_cache:
            cmd += [
                "--cache-from",
                f"type=registry,ref={remote_cache}:buildcache",
                "--cache-to",
                f"type=registry,ref={remote_cache}:buildcache,mode=max",
            ]
    cmd.append(docker_context)
    sys_logger.debug("Build cmd: `{}`".format(" ".join(cmd)))
    sys_logger.info(f"Building and pushing Docker image with BuildKit (buildx); cache repo: {remote_cache or 'none'}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys_logger.error(f"Docker build failed: {e}")
        raise RuntimeError(f"Docker build failed: {e}")


def check_ecr(repository: str, image_tag: str) -> Optional[str]:
    """
    Returns the remote digest for TAG if it exists, otherwise None.
    """
    if "localhost" in repository:
        return None

    region = os.getenv("AWS_REGION")
    if not region:
        sys_logger.warning("AWS_REGION environment variable not set. Skip image exist check.")
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
    tag = f"{ecr_repo}:{info['hash']}"
    info["image_tag"] = tag

    remote_digest = check_ecr(info["name"], info["hash"])
    info["image_exists"] = remote_digest is not None

    if info["image_exists"]:
        sys_logger.info("Tag already in ECR – skipping build.")
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
