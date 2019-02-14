#!/usr/bin/env python3

import argparse
import os
from tempfile import TemporaryDirectory

import skeletonization
import skeletonization.doc_utils as docs

ROOT_SKELETONIZATION_DIR = os.path.dirname(os.path.dirname(skeletonization.__file__))

"""Generate Sphinx documentation for the skeletonization module"""
DOCS_FOLDER = os.path.join(ROOT_SKELETONIZATION_DIR, "docs")
DOC_WARNING_RATCHET = 200


def main():
    """
    Generate Sphinx docs for the skeletonization module.
    """

    parser = argparse.ArgumentParser(
        description="Generate sphinx documentation")
    parser.add_argument(
        "--output_folder",
        help="""The folder to output the html documentation tree into.
        If unspecified the docs will be only be generated to check
        for errors/warnings.""",
        type=str)

    docs.build_api_docs(
        os.path.join(ROOT_SKELETONIZATION_DIR, ""),
        os.path.join(DOCS_FOLDER, "api_python"))

    args = parser.parse_args()

    with TemporaryDirectory() as td:
        output = args.output_folder if (args.output_folder is not None) else td
        warning_count = docs.build_html_docs(DOCS_FOLDER, output)

    print("Documentation written to: {}", output)
    few_enough_warnings = warning_count <= DOC_WARNING_RATCHET
    print(
        "Documentation Build Test: {} ({}/{})",
        "PASSED" if few_enough_warnings else "FAILED",
        warning_count,
        DOC_WARNING_RATCHET)

    return 0 if few_enough_warnings else 1


if __name__ == '__main__':
    main()
