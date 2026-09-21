"""Tests for dbn init."""

from pathlib import Path

from typer.testing import CliRunner

from decisionbrain.cli.app import app

runner = CliRunner()


def test_init_new_directory(tmp_path: Path):
    """Initialize a new directory."""
    target = tmp_path / "my-project"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0
    assert (target / "problem.md").is_file()
    assert not (target / "dbn.yaml").exists()
    assert (target / ".gitignore").is_file()
    assert (target / "data").is_dir()


def test_init_current_directory_default(tmp_path: Path):
    """Initialize the current directory explicitly or by default."""
    # Pass the path directly.
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "problem.md").is_file()


def test_init_existing_empty_directory(tmp_path: Path):
    """Initialize an existing empty directory."""
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "problem.md").is_file()


def test_init_refuses_overwrite_without_force(tmp_path: Path):
    """Reject existing files without --force."""
    # Create problem.md first.
    (tmp_path / "problem.md").write_text("original content")
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code != 0
    # Content must remain unchanged.
    assert (tmp_path / "problem.md").read_text() == "original content"


def test_init_force_overwrites_files(tmp_path: Path):
    """--force overwrites existing template files."""
    (tmp_path / "problem.md").write_text("old")

    result = runner.invoke(app, ["init", str(tmp_path), "--force"])
    assert result.exit_code == 0
    content = (tmp_path / "problem.md").read_text()
    assert "Optimization Problem" in content
    assert "old" not in content


def test_init_force_preserves_other_files(tmp_path: Path):
    """--force does not delete unrelated files."""
    (tmp_path / "problem.md").write_text("old")
    (tmp_path / "my_data.csv").write_text("col1,col2")

    result = runner.invoke(app, ["init", str(tmp_path), "--force"])
    assert result.exit_code == 0
    assert (tmp_path / "my_data.csv").is_file()


def test_init_output_mentions_followup_commands(tmp_path: Path):
    """The output suggests the next command."""
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0
    assert "dbn run" in result.stdout
    assert "dbn chat" in result.stdout


def test_init_path_is_file(tmp_path: Path):
    """Reject a target path that is an existing file."""
    f = tmp_path / "file.txt"
    f.write_text("hello")
    result = runner.invoke(app, ["init", str(f)])
    assert result.exit_code != 0


def test_init_creates_nonexistent_parent(tmp_path: Path):
    """Create missing parent directories."""
    target = tmp_path / "nested" / "project"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0
    assert (target / "problem.md").is_file()


def test_init_path_with_spaces_and_unicode(tmp_path: Path):
    """Support paths containing spaces and Unicode."""
    target = tmp_path / "我的 项目"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0
    assert (target / "problem.md").is_file()
