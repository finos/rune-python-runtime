#!/bin/bash
# Applies the Apache 2.0 SPDX license header to all tracked source files.
set -e
TMPL="$(dirname "$0")/header.txt"
licenseheaders -t "$TMPL" --dir src --dir test --dir .github
licenseheaders -t "$TMPL" --additional-extensions python=.toml -f build_wheel.sh dev_clean_setup.sh netlify.toml pyproject.toml safety-policy.yml .pre-commit-config.yaml
