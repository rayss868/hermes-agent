"""Tests for gateway.run._check_unavailable_skill.

Regression coverage for the dir-name-vs-frontmatter-name drift bug.
The hint function used to compare the skill's parent-directory name
against the typed command and the disabled list. That silently missed
every skill whose directory name differs from its declared frontmatter
name (~19 skills on a standard install), so users typing a real slug
like ``/stable-diffusion-image-generation`` got a generic "unknown
command" response instead of the intended "disabled — enable with …"
or "not installed — install with …" hint.

These tests pin the fixed behavior:

* Slug is derived from the frontmatter ``name:`` (exactly matching
  :func:`agent.skill_commands.scan_skill_commands`), so the slug differs
  from the directory name when the declared name is multi-word.
* ``disabled`` membership is checked by the declared name, because that
  is what :func:`hermes_cli.skills_config.save_disabled_skills` stores.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def tmp_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated skills dir + HERMES_HOME so the real user config is untouched."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home / "skills"


def _write_skill(skills_dir: Path, rel: str, frontmatter_name: str) -> Path:
    """Create a SKILL.md at ``<skills_dir>/<rel>/SKILL.md``."""
    skill_dir = skills_dir / rel
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {frontmatter_name}\ndescription: test skill\n---\nBody.\n",
        encoding="utf-8",
    )
    return skill_md


def test_frontmatter_slug_matched_even_when_dir_name_differs(
    tmp_skills: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directory ``stable-diffusion`` + frontmatter ``Stable Diffusion Image Generation``.

    Command typed: ``stable-diffusion-image-generation`` (the slug the
    agent actually registers). The old dir-name-based check would have
    compared ``stable-diffusion`` to the typed command and missed.
    """
    from gateway import run as gateway_run

    _write_skill(tmp_skills, "mlops/stable-diffusion", "Stable Diffusion Image Generation")

    # Config disables by declared name (matches what `hermes skills config` writes).
    monkeypatch.setattr(
        "gateway.run._get_disabled_skill_names",
        lambda: {"Stable Diffusion Image Generation"},
        raising=False,
    )
    with patch(
        "tools.skills_tool._get_disabled_skill_names",
        return_value={"Stable Diffusion Image Generation"},
    ), patch(
        "agent.skill_utils.get_all_skills_dirs",
        return_value=[tmp_skills],
    ):
        msg = gateway_run._check_unavailable_skill("stable-diffusion-image-generation")

    assert msg is not None, (
        "expected a 'disabled' hint for the frontmatter-derived slug; "
        "the old code compared the dir name 'stable-diffusion' and returned None"
    )
    assert "disabled" in msg.lower()
    assert "hermes skills config" in msg


def test_unknown_command_still_returns_none(
    tmp_skills: Path,
) -> None:
    """A command that matches no on-disk skill still returns None."""
    from gateway import run as gateway_run

    _write_skill(tmp_skills, "creative/ascii-art", "ascii-art")

    with patch(
        "tools.skills_tool._get_disabled_skill_names", return_value=set()
    ), patch(
        "agent.skill_utils.get_all_skills_dirs", return_value=[tmp_skills]
    ):
        assert gateway_run._check_unavailable_skill("no-such-skill") is None


def test_optional_skill_installed_after_gateway_start_is_not_hinted_as_missing(
    tmp_skills: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An optional skill that IS installed must not suggest re-installing itself.

    Regression: ``hermes skills install`` runs in a separate process that clears the skills
    *prompt* cache but not the gateway process's in-memory slash-command map, so a skill
    installed while the gateway was already running was absent from that stale map, fell
    through to the optional-skills branch, and the user was told to install a skill they
    had just installed (observed with ``/godmode``). The hub lock file is authoritative.
    """
    from gateway import run as gateway_run

    # An optional-source copy exists in the repo but the skill is not in the (stale) installed
    # skills dirs scan, and is not disabled — exactly the stale-gateway situation.
    optional_dir = tmp_path / "optional-skills" / "security" / "some-skill"
    optional_dir.mkdir(parents=True)
    _write_skill(optional_dir, "some-skill", "Some Skill")
    monkeypatch.setattr(
        "hermes_constants.get_optional_skills_dir", lambda default=None: optional_dir.parent
    )

    installed = {"Some Skill": {"install_path": "security/some-skill"}}

    class _FakeHubLockFile:
        def get_installed(self, name: str):
            return installed.get(name)

    with patch(
        "tools.skills_tool._get_disabled_skill_names", return_value=set()
    ), patch(
        "agent.skill_utils.get_all_skills_dirs", return_value=[tmp_skills]
    ), patch(
        "tools.skills_hub.HubLockFile", _FakeHubLockFile
    ):
        msg = gateway_run._check_unavailable_skill("some-skill")

    assert msg is None, (
        "an installed optional skill was reported as 'not installed'; the gateway's stale "
        "command map made it fall through to the optional-skills branch"
    )


def test_optional_skill_genuinely_missing_still_hinted(
    tmp_skills: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The install hint still fires for an optional skill that is NOT installed."""
    from gateway import run as gateway_run

    optional_dir = tmp_path / "optional-skills" / "security" / "cool-skill"
    optional_dir.mkdir(parents=True)
    _write_skill(optional_dir, "cool-skill", "Cool Skill")
    monkeypatch.setattr(
        "hermes_constants.get_optional_skills_dir", lambda default=None: optional_dir.parent
    )

    class _FakeHubLockFile:
        def get_installed(self, name: str):
            return None

    with patch(
        "tools.skills_tool._get_disabled_skill_names", return_value=set()
    ), patch(
        "agent.skill_utils.get_all_skills_dirs", return_value=[tmp_skills]
    ), patch(
        "tools.skills_hub.HubLockFile", _FakeHubLockFile
    ):
        msg = gateway_run._check_unavailable_skill("cool-skill")

    assert msg is not None and "not installed" in msg.lower()
    assert "official/cool-skill/cool-skill" in msg


