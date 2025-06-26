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
