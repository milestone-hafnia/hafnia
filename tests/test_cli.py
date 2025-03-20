from typing import Dict
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import src.cli.__main__ as cli
import src.cli.consts as consts
from mdi_python_tools.config import Config, ConfigFileSchema, ConfigSchema


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def organization_id() -> str:
    return "org-123"


@pytest.fixture
def api_key() -> str:
    return "test-api-key-12345678"


@pytest.fixture()
def profile_data(organization_id: str, api_key: str) -> Dict:
    """Base profile data that can be reused across different profiles."""
    return {
        "organization_id": organization_id,
        "platform_url": "https://api.mdi.milestonesys.com",
        "api_key": api_key,
        "api_mapping": {
            "organizations": "https://api.mdi.milestonesys.com/api/v1/organizations",
            "recipes": "https://api.mdi.milestonesys.com/api/v1/recipes",
            "experiments": "https://api.mdi.milestonesys.com/api/v1/experiments",
            "experiment_environments": "https://api.mdi.milestonesys.com/api/v1/experiment-environments",
            "experiment_runs": "https://api.mdi.milestonesys.com/api/v1/experiment-runs",
            "runs": "https://api.mdi.milestonesys.com/api/v1/experiments-runs",
            "datasets": "https://api.mdi.milestonesys.com/api/v1/datasets",
        },
    }


@pytest.fixture(scope="function")
def empty_config() -> Config:
    return Config()


@pytest.fixture(scope="function")
def config_with_profiles(profile_data: Dict) -> Config:
    active_profile = "default"
    default = ConfigSchema(**profile_data)
    staging = ConfigSchema(**profile_data)
    production = ConfigSchema(**profile_data)

    config = Config()
    config.config_data = ConfigFileSchema(
        active_profile=active_profile,
        profiles={active_profile: default, "staging": staging, "production": production},
    )
    return config


def test_configure(
    cli_runner: CliRunner, empty_config: Config, api_key: str, organization_id: str
) -> None:
    with patch("mdi_python_tools.config.CONFIG", empty_config):
        with patch("mdi_python_tools.platform.get_organization_id", return_value="org-123"):
            with patch("mdi_python_tools.config.Config.save_config"):
                inputs = "default\n" "test-api-key\n" "https://api.mdi.milestonesys.com\n"
                result = cli_runner.invoke(cli.configure, input="".join(inputs))
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
        with patch("mdi_python_tools.config.CONFIG", empty_config):
            result = cli_runner.invoke(cli.profile, ["ls"])
            assert result.exit_code == 0
            assert consts.ERROR_CONFIGURE in result.output

        with patch("mdi_python_tools.config.CONFIG", config_with_profiles):
            result = cli_runner.invoke(cli.profile, ["ls"])
            assert "default" in result.output
            assert "staging" in result.output
            assert "production" in result.output
            assert "Active profile: default" in result.output

    def test_switch_profile(
        self, cli_runner: CliRunner, empty_config: Config, config_with_profiles: Config
    ) -> None:
        with patch("mdi_python_tools.config.CONFIG", empty_config):
            result = cli_runner.invoke(cli.profile, ["use", "default"])
            assert result.exit_code == 0
            assert consts.ERROR_CONFIGURE in result.output

        with patch("mdi_python_tools.config.CONFIG", config_with_profiles):
            with patch("mdi_python_tools.config.Config.save_config"):
                result = cli_runner.invoke(cli.profile, ["active"])
                assert result.exit_code == 0
                assert f"{consts.PROFILE_TABLE_HEADER} default" in result.output

                result = cli_runner.invoke(cli.profile, ["use", "staging"])
                assert result.exit_code == 0
                assert f"{consts.PROFILE_SWITCHED_SUCCESS} staging" in result.output

                result = cli_runner.invoke(cli.profile, ["active"])
                assert result.exit_code == 0
                assert f"{consts.PROFILE_TABLE_HEADER} staging" in result.output

                result = cli_runner.invoke(cli.profile, ["use", "nonexistent"])
                assert result.exit_code == 0
                assert consts.ERROR_PROFILE_NOT_EXIST in result.output

    def test_remove_profile(
        self, cli_runner: CliRunner, empty_config: Config, config_with_profiles: Config
    ) -> None:
        with patch("mdi_python_tools.config.CONFIG", empty_config):
            result = cli_runner.invoke(cli.profile, ["rm", "default"])
            assert result.exit_code == 0
            assert consts.ERROR_CONFIGURE in result.output

        with patch("mdi_python_tools.config.CONFIG", config_with_profiles):
            with patch("mdi_python_tools.config.Config.save_config"):
                result = cli_runner.invoke(cli.profile, ["rm", "staging"])
                assert result.exit_code == 0
                assert f"{consts.PROFILE_REMOVED_SUCCESS} staging" in result.output

                result = cli_runner.invoke(cli.profile, ["ls"])
                assert result.exit_code == 0
                assert "staging" not in result.output
                assert "production" in result.output
                assert "default" in result.output

                result = cli_runner.invoke(cli.profile, ["rm", "nonexistent"])
                assert result.exit_code == 0
                assert consts.ERROR_PROFILE_NOT_EXIST in result.output

                result = cli_runner.invoke(cli.profile, ["rm", "default"])
                assert result.exit_code == 0
                assert consts.ERROR_PROFILE_REMOVE_ACTIVE in result.output
