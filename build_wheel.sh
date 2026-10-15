#!/bin/bash
##
## Copyright (c) 2023-2026 CLOUDRISK Limited and FT Advisory LLC
## SPDX-License-Identifier: Apache-2.0
##

function error
{
    echo
    echo "***************************************************************************"
    echo "*                                                                         *"
    echo "*                          WHEEL Build FAILED!                            *"
    echo "*                                                                         *"
    echo "***************************************************************************"
    echo
    exit -1
}

MY_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${MY_PATH} || error

if [ -d ".pydevenv" ]; then
    export PATH=`echo $PATH | tr ":" "\n" | grep -v ".pydevenv" | tr "\n" ":"`
fi
type -P python > /dev/null && PY_EXE=python || PY_EXE=python3

${PY_EXE} -m pip wheel --no-deps --only-binary :all: . || error