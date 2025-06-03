from pathlib import Path
from typing import Dict

import pytest
from click.testing import CliRunner

import cli.__main__ as cli
import cli.consts as consts
from cli.config import Config, ConfigSchema


@pytest.fixture
def cli_runner(tmp_path: Path) -> CliRunner:
    env = {"MDI_CONFIG_PATH": str(tmp_path / "config.json")}
    return CliRunner(env=env)


@pytest.fixture
def api_key() -> str:
    return "test-api-key-12345678"


@pytest.fixture
def test_config_path(tmp_path: Path) -> Path:
    """Return a temporary config file path for testing."""
    return tmp_path / "config.json"


@pytest.fixture()
def profile_data(api_key: str) -> Dict:
    """Base profile data that can be reused across different profiles."""
    return {"platform_url": consts.DEFAULT_API_URL, "api_key": api_key}


@pytest.fixture
def empty_config(test_config_path: Path) -> Config:
    return Config(config_path=test_config_path)


@pytest.fixture(scope="function")
def config_with_profiles(test_config_path: Path, profile_data: dict) -> Config:
    config = Config(config_path=test_config_path)
    config.add_profile("default", ConfigSchema(**profile_data), set_active=True)
    config.add_profile("staging", ConfigSchema(**profile_data))
    config.add_profile("production", ConfigSchema(**profile_data))
    return config


def test_configure(cli_runner: CliRunner, empty_config: Config, api_key: str) -> None:
    inputs = f"default\ntest-api-key\n{consts.DEFAULT_API_URL}\n"
    result = cli_runner.invoke(cli.main, ["configure"], input="".join(inputs))
    assert result.exit_code == 0
    assert f"{consts.PROFILE_TABLE_HEADER} default" in result.output


class TestProfile:
    def test_list_profiles(
        self,
        cli_runner: CliRunner,
        empty_config: Config,
        config_with_profiles: Config,
    ) -> None:
        """Test list of profiles functionality."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cli.__main__.Config", lambda *args, **kwargs: empty_config)
            result = cli_runner.invoke(cli.main, ["profile", "ls"])
            assert result.exit_code != 0
            assert consts.ERROR_CONFIGURE in result.output

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cli.__main__.Config", lambda *args, **kwargs: config_with_profiles)
            result = cli_runner.invoke(cli.main, ["profile", "ls"])
            assert result.exit_code == 0
            assert "default" in result.output
            assert "staging" in result.output
            assert "production" in result.output
            assert "Active profile: default" in result.output

    def test_switch_profile(self, cli_runner: CliRunner, empty_config: Config, config_with_profiles: Config) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cli.__main__.Config", lambda *args, **kwargs: empty_config)
            result = cli_runner.invoke(cli.main, ["profile", "use", "default"])
            assert result.exit_code != 0
            assert f"Error: {consts.ERROR_CONFIGURE}" in result.output

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cli.__main__.Config", lambda *args, **kwargs: config_with_profiles)
            result = cli_runner.invoke(cli.main, ["profile", "active"])
            assert result.exit_code == 0
            assert f"{consts.PROFILE_TABLE_HEADER} default" in result.output

            result = cli_runner.invoke(cli.main, ["profile", "use", "staging"])
            assert result.exit_code == 0
            assert f"{consts.PROFILE_SWITCHED_SUCCESS} staging" in result.output

            result = cli_runner.invoke(cli.main, ["profile", "active"])
            assert result.exit_code == 0
            assert f"{consts.PROFILE_TABLE_HEADER} staging" in result.output

            result = cli_runner.invoke(cli.main, ["profile", "use", "nonexistent"])
            assert result.exit_code != 0
            assert consts.ERROR_PROFILE_NOT_EXIST in result.output

    def test_remove_profile(self, cli_runner: CliRunner, empty_config: Config, config_with_profiles: Config) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cli.__main__.Config", lambda *args, **kwargs: empty_config)
            result = cli_runner.invoke(cli.main, ["profile", "rm", "default"])
            assert result.exit_code != 0
            assert consts.ERROR_CONFIGURE in result.output

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("cli.__main__.Config", lambda *args, **kwargs: config_with_profiles)
            result = cli_runner.invoke(cli.main, ["profile", "rm", "staging"])
            assert result.exit_code == 0
            assert f"{consts.PROFILE_REMOVED_SUCCESS} staging" in result.output

            result = cli_runner.invoke(cli.main, ["profile", "ls"])
            assert result.exit_code == 0
            assert "staging" not in result.output
            assert "production" in result.output
            assert "default" in result.output

            result = cli_runner.invoke(cli.main, ["profile", "rm", "nonexistent"])
            assert result.exit_code != 0
            assert consts.ERROR_PROFILE_NOT_EXIST in result.output

            result = cli_runner.invoke(cli.main, ["profile", "rm", "default"])
            assert result.exit_code != 0
            assert consts.ERROR_PROFILE_REMOVE_ACTIVE in result.output


class TestData:
    @pytest.fixture
    def data_endpoint(self) -> str:
        return f"{consts.DEFAULT_API_URL}/api/v1/datasets/my-dataset"

    @pytest.fixture
    def destination(self, tmp_path: Path) -> str:
        return str(tmp_path / "data")

    def test_get_data_failure(self, cli_runner: CliRunner, data_endpoint: str, destination: str) -> None:
        """Test data get command when download fails"""

        def mock_download_failure(*args, **kwargs):
            raise Exception("Download failed")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("hafnia.platform.download_resource", mock_download_failure)
            result = cli_runner.invoke(cli.main, ["data", "get", data_endpoint, destination])
            assert result.exit_code != 0
            assert consts.ERROR_GET_RESOURCE in result.output

    def test_get_data_success(
        self,
        cli_runner: CliRunner,
        config_with_profiles: Config,
        data_endpoint: str,
        destination: str,
        tmp_path: Path,
    ) -> None:
        """Test data get command when download succeeds"""
        dummy_file = tmp_path / "data" / "downloaded.txt"

        def mock_download_success(*args, **kwargs):
            return {"status": "success", "downloaded_files": [dummy_file.as_posix()]}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("hafnia.platform.download_resource", mock_download_success)
            mp.setattr("cli.__main__.Config", lambda *args, **kwargs: config_with_profiles)

            result = cli_runner.invoke(cli.main, ["data", "get", data_endpoint, destination])
            assert result.exit_code == 0
            assert "success" in result.output
            assert "downloaded_files" in result.output
