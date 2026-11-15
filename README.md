[![FINOS - Incubating](https://cdn.jsdelivr.net/gh/finos/contrib-toolbox@master/images/badge-incubating.svg)](https://community.finos.org/docs/governance/Software-Projects/stages/incubating)
[![Run Unit Tests](https://github.com/finos/rune-python-runtime/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/finos/rune-python-runtime/actions/workflows/run-tests.yml)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10733/badge)](https://www.bestpractices.dev/projects/10733)
[![Python CI](https://github.com/finos/rune-python-runtime/actions/workflows/cve-scanning-python.yml/badge.svg)](https://github.com/finos/rune-python-runtime/actions/workflows/cve-scanning-python.yml)
<img align="right" width="15%" src="https://www.finos.org/hubfs/FINOS/finos-logo/FINOS_Icon_Wordmark_Name_RGB_horizontal.png">

# Rune Python Runtime

The Rune Python Runtime provides shared support for Python models and functions
generated from [Rune DSL](https://github.com/finos/rune-dsl), including the
[Common Domain Model (CDM)](https://github.com/finos/common-domain-model).

The [Rune Python Code Generator](https://github.com/finos/rune-python-generator)
translates Rune models into Python packages. This runtime supplies their shared
validation, JSON serialization, metadata, reference handling, and function helpers.

## Installation

Python 3.11 or later is required. The runtime is normally installed as a dependency
of a generated model package. To install it directly:

```sh
python -m pip install rune-runtime
```

## Design

Start with the [runtime design overview](https://github.com/finos/rune-python-runtime/blob/main/docs/design.md) for the generator/runtime
boundary, the main components, and links to the detailed design documents.

## Development setup

With Python 3.11 or later and Bash available (for example, Git Bash on Windows):

```sh
git clone https://github.com/finos/rune-python-runtime.git
cd rune-python-runtime
./dev_clean_setup.sh
```

The [setup script](https://github.com/finos/rune-python-runtime/blob/main/dev_clean_setup.sh) creates `.pydevenv`, installs development
dependencies, and registers pre-commit hooks, including license-header checks.

Build a wheel, then run the tests against it in an isolated environment:

```sh
./build_wheel.sh
./test/run_runtime_tests.sh
```

See [build_wheel.sh](https://github.com/finos/rune-python-runtime/blob/main/build_wheel.sh) and the
[test runner](https://github.com/finos/rune-python-runtime/blob/main/test/run_runtime_tests.sh) for details.

## Roadmap

The roadmap is aligned with Rune DSL and the
[CDM roadmap](https://github.com/finos/common-domain-model/blob/master/ROADMAP.md).

## Contributors

- [CLOUDRISK Limited](https://www.cloudrisk.uk), email: info@cloudrisk.co.uk
- [FT Advisory LLC](https://www.ftadvisory.co), email: info@ftadvisory.co
- [TradeHeader SL](https://www.tradeheader.com), email: info@tradeheader.com

## Contributing

For questions, bug reports, or feature requests, open a
[GitHub issue](https://github.com/finos/rune-python-runtime/issues).

Read the [contribution guidelines](https://github.com/finos/rune-python-runtime/blob/main/CONTRIBUTING.md) for the contribution process
and FINOS contributor license agreement requirements, and follow the
[Community Code of Conduct](https://github.com/finos/rune-python-runtime/blob/main/.github/CODE_OF_CONDUCT.md).
To contribute, [fork the runtime](https://github.com/finos/rune-python-runtime/fork),
create a branch, and submit a pull request.

You can also get in touch through
[FINOS Slack](https://www.finos.org/blog/finos-announces-new-community-slack).

## Governance

This project follows the
[FINOS governance framework](https://community.finos.org/docs/governance/#open-source-software-projects).

## License

Copyright 2023-2026 CLOUDRISK Limited and FT Advisory LLC

Distributed under the [Apache License, Version 2.0](https://github.com/finos/rune-python-runtime/blob/main/LICENSE).

SPDX-License-Identifier: [Apache-2.0](https://spdx.org/licenses/Apache-2.0)
