from pathlib import Path

import pytest

from decisionbrain.algorithm_library import AlgorithmCatalogError, LocalAlgorithmCatalog
from decisionbrain.algorithm_library.management import install_algorithm_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = PROJECT_ROOT / "algorithms" / "manifests" / "pyjobshop.yaml"


def test_install_manifest_connects_package_to_catalog(tmp_path):
    result = install_algorithm_manifest(SOURCE_MANIFEST, tmp_path)

    assert result.manifest.id == "pyjobshop"
    assert result.path == tmp_path / "pyjobshop.yaml"
    assert result.replaced is False
    assert LocalAlgorithmCatalog(tmp_path).get("pyjobshop").id == "pyjobshop"


def test_install_manifest_requires_explicit_replace(tmp_path):
    install_algorithm_manifest(SOURCE_MANIFEST, tmp_path)

    with pytest.raises(AlgorithmCatalogError, match="already exists"):
        install_algorithm_manifest(SOURCE_MANIFEST, tmp_path)

    result = install_algorithm_manifest(SOURCE_MANIFEST, tmp_path, replace=True)
    assert result.replaced is True


def test_invalid_manifest_does_not_modify_catalog(tmp_path):
    invalid = tmp_path / "invalid.yaml"
    target = tmp_path / "catalog"
    invalid.write_text(
        'schema_version: "1.3"\nid: invalid\ninterface:\n  profile: unsupported\n',
        encoding="utf-8",
    )

    with pytest.raises(AlgorithmCatalogError, match="invalid algorithm manifest"):
        install_algorithm_manifest(invalid, target)

    assert not target.exists()
