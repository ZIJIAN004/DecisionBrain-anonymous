"""Run 输入工作区测试。"""

from pathlib import Path

from decisionbrain.run_storage import FolderWorkspace, InputWorkspace


def test_from_path_creates_folder_workspace(tmp_path: Path):
    """从路径创建 FolderWorkspace，自动探测 problem.md 和 data/。"""
    (tmp_path / "problem.md").write_text("# Test Problem")
    (tmp_path / "data").mkdir()

    workspace = FolderWorkspace.from_path(tmp_path)

    assert workspace.root == tmp_path.resolve()
    assert workspace.problem_md == tmp_path / "problem.md"
    assert workspace.data_dir == tmp_path / "data"


def test_problem_description_reads_file(tmp_path: Path):
    """读取 problem.md 内容。"""
    (tmp_path / "problem.md").write_text("# 我的优化问题\n\n最小化成本。")

    workspace = FolderWorkspace.from_path(tmp_path)
    assert workspace.problem_description == "# 我的优化问题\n\n最小化成本。"
    assert workspace.has_problem is True


def test_has_problem_false_when_missing(tmp_path: Path):
    """无 problem.md 时 has_problem 返回 False。"""
    workspace = FolderWorkspace.from_path(tmp_path)
    assert workspace.problem_description == ""
    assert workspace.has_problem is False


def test_has_problem_false_when_empty(tmp_path: Path):
    """空 problem.md 时 has_problem 返回 False。"""
    (tmp_path / "problem.md").write_text("   \n")

    workspace = FolderWorkspace.from_path(tmp_path)
    assert workspace.has_problem is False


def test_validate_empty_workspace(tmp_path: Path):
    """空目录校验无误。"""
    workspace = FolderWorkspace.from_path(tmp_path)
    issues = workspace.validate()
    assert issues == []


def test_validate_missing_root():
    """A missing path reports one error."""
    workspace = FolderWorkspace(root=Path("/nonexistent/path/for/testing"))
    issues = workspace.validate()
    assert len(issues) >= 1
    assert any("does not exist" in issue for issue in issues)


def test_validate_root_is_file(tmp_path: Path):
    """路径是文件而非目录时报告错误。"""
    file_path = tmp_path / "file.txt"
    file_path.write_text("hello")
    workspace = FolderWorkspace(root=file_path)
    issues = workspace.validate()
    assert len(issues) >= 1


def test_generate_input_summary(tmp_path: Path):
    """The generated input summary contains the expected fields."""
    (tmp_path / "problem.md").write_text("test content")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "input.csv").write_text("a,b,c")

    workspace = FolderWorkspace.from_path(tmp_path)
    summary = workspace.generate_input_summary()

    assert summary["problem_file"] == "problem.md"
    assert "config_file" not in summary
    assert "data/input.csv" in summary["data_files"]
    assert summary["problem_length"] > 0


def test_path_with_spaces_and_unicode(tmp_path: Path):
    """路径含空格和 Unicode 字符。"""
    workspace_dir = tmp_path / "项目 测试"
    workspace_dir.mkdir()
    (workspace_dir / "problem.md").write_text("# 优化问题")
    (workspace_dir / "data").mkdir()

    workspace = FolderWorkspace.from_path(workspace_dir)

    assert workspace.root == workspace_dir.resolve()
    assert workspace.problem_description == "# 优化问题"
    issues = workspace.validate()
    assert issues == []


def test_folder_workspace_from_str_path(tmp_path: Path):
    """from_path 接受字符串参数。"""
    (tmp_path / "problem.md").write_text("test")
    workspace = FolderWorkspace.from_path(str(tmp_path))
    assert workspace.root == tmp_path.resolve()


def test_input_workspace_writes_problem_description(tmp_path: Path):
    """无 workspace 输入也能落盘为根目录 problem.md。"""
    workspace = InputWorkspace(tmp_path)

    path = workspace.write_problem_description("  最小化配送成本  ")

    assert path == tmp_path / "problem.md"
    assert path.read_text(encoding="utf-8") == "最小化配送成本\n"
    assert workspace.list_files() == []
