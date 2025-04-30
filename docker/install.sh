#!/bin/bash
python3 -m venv $1
source $1/bin/activate
pip install --upgrade pip setuptools wheel
pip install poetry pre-commit pytest
poetry lock
poetry install --no-root