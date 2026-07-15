#!/bin/bash
type -P python > /dev/null && PY_EXE=python || PY_EXE=python3
if ! $PY_EXE -c 'import sys; assert sys.version_info >= (3,10)' > /dev/null 2>&1; then
        echo "Found $($PY_EXE -V)"
        echo "Expecting at least python 3.10 - exiting!"
        exit 1
fi

export PYTHONDONTWRITEBYTECODE=1

MY_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $MY_PATH/..

$PY_EXE -m venv --clear .pytest
AC_DIR=$($PY_EXE -c "import sys;print('Scripts' if sys.platform.startswith('win') else 'bin')")
source .pytest/$AC_DIR/activate
# after the above, use directly python as it will be on the path

python -m pip install pytest pytest-mock
python -m pip install rune_runtime*-py3-*.whl

# run tests
python -m pytest -p no:cacheprovider test/
rm -rf .pytest