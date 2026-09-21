"""Run input-workspace adapters."""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import tempfile
from pathlib import Path

from ..exceptions import UserInputError


@dataclass
class FolderWorkspace:
    """Existing local input directory with problem.md and optional data/."""

    root: Path
    problem_md: Path | None = None
    data_dir: Path | None = None

    @classmethod
    def from_path(cls, path: Path | str) -> "FolderWorkspace":
        """Describe an input directory and discover its files."""
        root = Path(path).expanduser().resolve()
        return cls(
            root=root,
            problem_md=cls._find_file(root, "problem.md"),
            data_dir=cls._find_dir(root, "data"),
        )

    @property
    def problem_description(self) -> str:
        """Read problem.md, returning an empty string when unavailable."""
        if self.problem_md is None:
            return ""
        try:
            return self.problem_md.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    @property
    def has_problem(self) -> bool:
        return bool(self.problem_description)

    def validate(self) -> list[str]:
        """Validate the input directory and return detected issues."""
        issues: list[str] = []
        if not self.root.exists():
            issues.append(f"Workspace path does not exist: {self.root}")
            return issues
        if not self.root.is_dir():
            issues.append(f"Path is not a directory: {self.root}")
            return issues

        if self.problem_md is not None:
            if not self.problem_md.is_file():
                issues.append(f"problem.md is not a file: {self.problem_md}")
            else:
                try:
                    self.problem_md.read_text(encoding="utf-8")
                except OSError as exc:
                    issues.append(f"Cannot read problem.md: {exc}")

        if self.data_dir is not None:
            if not self.data_dir.is_dir():
                issues.append(f"data is not a directory: {self.data_dir}")
            else:
                for path in self.data_dir.rglob("*"):
                    if not path.is_file():
                        continue
                    try:
                        path.read_bytes()
                    except OSError as exc:
                        issues.append(f"Unreadable file under data/: {path.relative_to(self.root)} - {exc}")

        return issues

    def generate_input_summary(self) -> dict:
        """Build the input summary consumed by RunRepository.create_run_record()."""
        data_files: list[str] = []
        if self.data_dir is not None and self.data_dir.is_dir():
            data_files = sorted(
                str(path.relative_to(self.root))
                for path in self.data_dir.rglob("*")
                if path.is_file()
            )
        return {
            "problem_file": str(self.problem_md.relative_to(self.root))
            if self.problem_md is not None
            else None,
            "data_files": data_files,
            "problem_length": len(self.problem_description),
        }

    @staticmethod
    def _find_file(root: Path, name: str) -> Path | None:
        candidate = root / name
        return candidate if candidate.is_file() else None

    @staticmethod
    def _find_dir(root: Path, name: str) -> Path | None:
        candidate = root / name
        return candidate if candidate.is_dir() else None


class InputWorkspace:
    """Stage Run inputs before Runtime creates an immutable snapshot."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.data_dir = self.root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, *, prefix: str = "decisionbrain-input-") -> "InputWorkspace":
        return cls(Path(tempfile.mkdtemp(prefix=prefix)))

    def add_local_file(self, path: Path | str, *, base_dir: Path | None = None) -> Path:
        """Validate and copy one file, resolving relative paths from base_dir."""
        source = Path(path).expanduser()
        if not source.is_absolute():
            source = (base_dir or Path.cwd()) / source
        source = source.resolve()
        if not source.exists():
            raise UserInputError(f"Data file does not exist: {source}")
        if not source.is_file():
            raise UserInputError(f"Data input must be one file: {source}")
        try:
            with source.open("rb") as stream:
                stream.read(1)
        except OSError as exc:
            raise UserInputError(f"Data file is not readable: {source} ({exc})") from exc

        destination = self._available_destination(source.name)
        try:
            shutil.copy2(source, destination)
        except OSError as exc:
            raise UserInputError(f"Cannot stage data file {source}: {exc}") from exc
        return destination

    def add_bytes(self, filename: str, content: bytes) -> Path:
        """Write uploaded content under a safe filename."""
        name = Path(filename or "uploaded").name or "uploaded"
        destination = self._available_destination(name)
        destination.write_bytes(content)
        return destination

    def write_problem_description(self, content: str) -> Path:
        """Write a workspace-free problem description to root problem.md."""
        text = content.strip()
        if not text:
            raise UserInputError("problem.md cannot be empty")
        destination = self.root / "problem.md"
        destination.write_text(text + "\n", encoding="utf-8")
        return destination

    def list_files(self) -> list[Path]:
        return sorted(path for path in self.data_dir.iterdir() if path.is_file())

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _available_destination(self, name: str) -> Path:
        candidate = self.data_dir / name
        if not candidate.exists():
            return candidate
        path = Path(name)
        index = 2
        while True:
            candidate = self.data_dir / f"{path.stem}-{index}{path.suffix}"
            if not candidate.exists():
                return candidate
            index += 1
