from scripts.audit_dependencies import audit


def test_all_runtime_imports_are_versioned_and_installed_by_notebook():
    report = audit()

    assert report["undeclared_in_requirements"] == []
    assert report["missing_from_notebook_install"] == []
    assert report["unused_declarations"] == []
    assert report["unused_colab_declarations"] == []
    assert report["unversioned_declarations"] == []
    assert report["unversioned_colab_declarations"] == []
