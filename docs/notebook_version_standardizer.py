import json
import os

import click

DOCS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")


def _get_ipython_notebooks(docs_source):
    directories_to_skip = ["_templates", "generated", ".ipynb_checkpoints"]
    notebooks = []
    for root, _, filenames in os.walk(docs_source):
        if any(dir_ in root for dir_ in directories_to_skip):
            continue
        for filename in filenames:
            if filename.endswith(".ipynb"):
                notebooks.append(os.path.join(root, filename))
    return notebooks


def _check_delete_empty_cell(notebook, delete=True):
    with open(notebook, "r") as f:
        source = json.load(f)
    cell = source["cells"][-1]
    if cell["cell_type"] == "code" and cell["source"] == []:
        # this is an empty cell, which we should delete
        if delete:
            source["cells"] = source["cells"][:-1]
        else:
            return False
    if delete:
        with open(notebook, "w") as f:
            json.dump(source, f, ensure_ascii=False, indent=1)
    else:
        return True


def _check_execution_and_output(notebook):
    with open(notebook, "r") as f:
        source = json.load(f)
    for cells in source["cells"]:
        if cells["cell_type"] == "code" and (
            cells["execution_count"] is not None or cells["outputs"] != []
        ):
            return False
    return True


def _check_python_version(notebook, default_version):
    with open(notebook, "r") as f:
        source = json.load(f)
    if source["metadata"]["language_info"]["version"] != default_version:
        return False
    return True


def _fix_python_version(notebook, default_version):
    with open(notebook, "r") as f:
        source = json.load(f)
    source["metadata"]["language_info"]["version"] = default_version
    with open(notebook, "w") as f:
        json.dump(source, f, ensure_ascii=False, indent=1)


def _fix_execution_and_output(notebook):
    with open(notebook, "r") as f:
        source = json.load(f)
    for cells in source["cells"]:
        if cells["cell_type"] == "code" and cells["execution_count"] is not None:
            cells["execution_count"] = None
            cells["outputs"] = []
    source["metadata"]["kernelspec"]["display_name"] = "Python 3"
    source["metadata"]["kernelspec"]["name"] = "python3"
    with open(notebook, "w") as f:
        json.dump(source, f, ensure_ascii=False, indent=1)


def _get_notebooks_with_executions_and_empty(notebooks, default_version="3.9.2"):
    executed = []
    empty_last_cell = []
    versions = []
    for notebook in notebooks:
        if not _check_execution_and_output(notebook):
            executed.append(notebook)
        if not _check_delete_empty_cell(notebook, delete=False):
            empty_last_cell.append(notebook)
        if not _check_python_version(notebook, default_version):
            versions.append(notebook)
    return (executed, empty_last_cell, versions)


def _fix_versions(notebooks, default_version="3.9.2"):
    for notebook in notebooks:
        _fix_python_version(notebook, default_version)


def _remove_notebook_empty_last_cell(notebooks):
    for notebook in notebooks:
        _check_delete_empty_cell(notebook, delete=True)


def _standardize_outputs(notebooks):
    for notebook in notebooks:
        _fix_execution_and_output(notebook)


@click.group()
def cli():
    """no-op"""


@cli.command()
def standardize():
    notebooks = _get_ipython_notebooks(DOCS_PATH)
    (
        executed_notebooks,
        empty_cells,
        versions,
    ) = _get_notebooks_with_executions_and_empty(notebooks)
    if executed_notebooks:
        _standardize_outputs(executed_notebooks)
        executed_notebooks = ["\t" + notebook for notebook in executed_notebooks]
        executed_notebooks = "\n".join(executed_notebooks)
        click.echo(f"Removed the outputs for:\n {executed_notebooks}")
    if empty_cells:
        _remove_notebook_empty_last_cell(empty_cells)
        empty_cells = ["\t" + notebook for notebook in empty_cells]
        empty_cells = "\n".join(empty_cells)
        click.echo(f"Removed the empty cells for:\n {empty_cells}")
    if versions:
        _fix_versions(versions)
        versions = ["\t" + notebook for notebook in versions]
        versions = "\n".join(versions)
        click.echo(f"Fixed python versions for:\n {versions}")


@cli.command()
def check_execution():
    notebooks = _get_ipython_notebooks(DOCS_PATH)
    (
        executed_notebooks,
        empty_cells,
        versions,
    ) = _get_notebooks_with_executions_and_empty(notebooks)
    if executed_notebooks:
        executed_notebooks = ["\t" + notebook for notebook in executed_notebooks]
        executed_notebooks = "\n".join(executed_notebooks)
        raise SystemExit(
            f"The following notebooks have executed outputs:\n {executed_notebooks}\n"
            "Please run make lint-fix to fix this.",
        )
    if empty_cells:
        empty_cells = ["\t" + notebook for notebook in empty_cells]
        empty_cells = "\n".join(empty_cells)
        raise SystemExit(
            f"The following notebooks have empty cells at the end:\n {empty_cells}\n"
            "Please run make lint-fix to fix this.",
        )
    if versions:
        versions = ["\t" + notebook for notebook in versions]
        versions = "\n".join(versions)
        raise SystemExit(
            f"The following notebooks have the wrong Python version: \n {versions}\n"
            "Please run make lint-fix to fix this.",
        )


if __name__ == "__main__":
    cli()
