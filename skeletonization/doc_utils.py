import os
import warnings
from tempfile import TemporaryDirectory

from sphinx import cmdline
from sphinx.ext.apidoc import main as apidoc_main

import skeleton.io_tools


def build_api_docs(input_path: str, output_path: str, ignore="*_tests.py"):
    """
    Autogenerate the Python API documentation for the module found at
    input_path.

    :raises AssertionError: If input directory does not exist.
    :param input_path: Folder to read module contents from.
    :param output_path: Folder to output .rst files to.
    :param ignore: Ignore files pattern.
    """
    skeleton.io_tools.assert_directory_exists(input_path)
    skeleton.io_tools.force_directory(output_path)

    apidoc_main(["-f", "-o", output_path, input_path, ignore])


def _build_docs(source_dir: str, output_dir: str, format: str) -> int:
    """
    Build html sphinx documentation.

    :param source_dir: The input directory to build docs from
    :param output_dir: The place to output docs to
    :return: integer number of warnings issued during docs build
    """

    assert format in ["html", "pdf"]
    with TemporaryDirectory() as td:
        warnings_file = os.path.join(td, "docs_warnings.txt")
        cmdline.main(["-w", warnings_file, "-b", format, source_dir, output_dir])

        warns = open(warnings_file).readlines()
        for w in warns:
            warnings.warn(w.rstrip())

    return len(warns)


def build_html_docs(source_dir: str, output_dir: str) -> int:
    """
    Build html sphinx documentation.

    :param source_dir: The input directory to build docs from
    :param output_dir: The place to output docs to
    :return: integer number of warnings issued during docs build
    """
    return _build_docs(source_dir, output_dir, "html")


def build_pdf_docs(source_dir: str, output_dir: str) -> int:
    """
    Build pdf sphinx documentation.

    :param source_dir: The input directory to build docs from
    :param output_dir: The place to output docs to
    :return: integer number of warnings issued during docs build
    """
    return _build_docs(source_dir, output_dir, "pdf")
