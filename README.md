[![FINOS - Incubating](https://cdn.jsdelivr.net/gh/finos/contrib-toolbox@master/images/badge-incubating.svg)](https://community.finos.org/docs/governance/Software-Projects/stages/incubating)
[![Run Unit Tests](https://github.com/regnosys/rune-python-runtime/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/regnosys/rune-python-runtime/actions/workflows/run-tests.yml)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10733/badge)](https://www.bestpractices.dev/projects/10733)[![Python CI](https://github.com/regnosys/rune-python-runtime/actions/workflows/cve-scanning-python.yml/badge.svg)](https://github.com/regnosys/rune-python-runtime/actions/workflows/cve-scanning-python.yml)
<img align="right" width="15%" src="https://www.finos.org/hubfs/FINOS/finos-logo/FINOS_Icon_Wordmark_Name_RGB_horizontal.png">
# Rune Python Runtime

*Rune Python Runtime* - the runtime supports and is an integral part of Python code generated from a [Rune DSL](https://github.com/finos/rune-dsl) defined model.  Rune DSL is a Domain-Specific Language used to model Financial Markets activities including the [Common Domain Model (CDM)](https://github.com/finos/common-domain-model).  

The Rune Python Runtime is used in collaboration with the [Rune Python Code Generator](https://github.com/REGnosys/rune-python-generator) to translate a Rune DSL model into a fully usable Python package.

## Installation

The runtime is not generally installed on a stand alone basis but rather comes as part of a generated package such as CDM.
However, pip install `rune-runtime` would install it from PyPi.

Regardless, to install the package standalone:

1. Fetch the latest release data from the GitHub API
```sh
release_data=$(curl -s https://api.github.com/repos/REGnosys/rune-python-runtime/releases/latest)
```
2. Extract the download URL of the first asset
```sh
download_url=$(echo "$release_data" | grep '"browser_download_url":' | head -n 1 | sed -E 's/.*"([^"]+)".*/\1/')
```
3. Download the artifact using wget or curl
```sh
wget "$download_url"
```

4. Install the Runtime
```sh
python -m pip install rune.runtime*-py3-*.whl
```

## Development setup
### Setup for developers
This guide is meant for everyone who wants to contribute to the Rune Python Runtime and needs to get things up and running.
### 1. Clone the Project
Start by cloning the project: `git clone https://github.com/regnosys/rune-python-generator`

### 2. Setup the Development Environment
Use [dev_clean_setup.sh](https://github.com/Cloudrisk/rune-python-runtime/blob/main/dev_clean_setup.sh) to setup a development environment.

```sh
./dev_clean_setup.sh
```
### 3. Build
Use [build_wheel.sh](https://github.com/Cloudrisk/rune-python-runtime/blob/main/build_wheel.sh) to build the package
```sh
./build_wheel.sh
```
To run the unit tests:
```sh
test/run_runtime_tests.sh
```

## Roadmap

The Roadmap will be aligned to the Rune DSL and [CDM](https://github.com/finos/common-domain-model/blob/master/ROADMAP.md) roadmaps.

In addition, the intention is to make future releases available at [PyPi](https://pypi.org)

## Contributors
- [CLOUDRISK Limited](https://www.cloudrisk.uk), email: info@cloudrisk.com
- [FT Advisory LLC](https://www.ftadvisory.co), email: info@ftadvisory.co
- [TradeHeader SL](https://www.tradeheader.com), email: info@tradeheader.com

## Contributing
For any questions, bugs or feature requests please open an [issue](https://github.com/regnosys/rune-python-runtime/issues)
For anything else please send an email to {project mailing list}.

To submit a contribution:
1. Fork it (<https://github.com/regnosys/rune-python-runtime/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Read our [contribution guidelines](.github/CONTRIBUTING.md) and [Community Code of Conduct](https://www.finos.org/code-of-conduct)
4. Commit your changes (`git commit -am 'Add some fooBar'`)
5. Push to the branch (`git push origin feature/fooBar`)
6. Create a new Pull Request

_NOTE:_ Commits and pull requests to FINOS repositories will only be accepted from those contributors with an active, executed Individual Contributor License Agreement (ICLA) with FINOS OR who are covered under an existing and active Corporate Contribution License Agreement (CCLA) executed with FINOS. Commits from individuals not covered under an ICLA or CCLA will be flagged and blocked by the FINOS Clabot tool (or [EasyCLA](https://community.finos.org/docs/governance/Software-Projects/easycla)). Please note that some CCLAs require individuals/employees to be explicitly named on the CCLA.

*Need an ICLA? Unsure if you are covered under an existing CCLA? Email [help@finos.org](mailto:help@finos.org)*

## Get in touch with the Rune Python Runtime Team

 Get in touch with the Rune team by creating a [GitHub issue](https://github.com/REGnosys/rune-python-runtime/issues/new) and labelling it with "help wanted".

 We encourage the community to get in touch via the [FINOS Slack](https://www.finos.org/blog/finos-announces-new-community-slack).

## Governance

This project implements https://community.finos.org/docs/governance/#open-source-software-projects

## License

Copyright 2023-2025 CLOUDRISK Limited and FT Advisory LLC

Distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

SPDX-License-Identifier: [Apache-2.0](https://spdx.org/licenses/Apache-2.0)
