# Copyright (c) 2023-2026 CLOUDRISK Limited and FT Advisory LLC
# SPDX-License-Identifier: Apache-2.0

'''Compare reference-list overhead using the same models and installed dependencies.

Run with the project interpreter. --source selects an extracted revision's src
directory; omit it to measure the current checkout. Timings exclude model schema
construction and report the best of --repeat runs independently for each round.
'''
import argparse
import cProfile
import gc
import json
import platform
import sys
import timeit
from pathlib import Path
from typing import Annotated


def main() -> None:
    '''Measure public read, write and validation operations on large model graphs.'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--label', default='working-tree')
    parser.add_argument('--size', type=int, default=2000)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--profile', type=Path)
    args = parser.parse_args()
    source = args.source or Path(__file__).resolve().parents[1] / 'src'
    sys.path.insert(0, str(source.resolve()))

    import pydantic
    from rune.runtime.base_data_class import BaseDataClass
    from rune.runtime.metadata import BaseReference

    class Party(BaseDataClass):
        _ALLOWED_METADATA = {'@key', '@key:external'}
        name: str

    class PlainBook(BaseDataClass):
        parties: Annotated[list[Party], Party.serializer(), Party.validator(('@key:external',))]
        items: Annotated[list[Party], Party.serializer(), Party.validator()]

    class ReferenceBook(BaseDataClass):
        parties: Annotated[list[Party], Party.serializer(), Party.validator(('@key:external',))]
        items: Annotated[list[Party | BaseReference], Party.serializer(), Party.validator(('@ref:external',))]
        _KEY_REF_CONSTRAINTS = {'items': {'@ref:external'}}

    data = {
        'parties': [{'@key:external': f'p{i}', 'name': f'Party {i}'} for i in range(args.size)],
        'items': [{'name': f'Inline {i}'} for i in range(args.size)],
    }
    cases = [('no-references', PlainBook, data),
             ('reference-capable-inline', ReferenceBook, data),
             ('references', ReferenceBook, data | {
                 'items': [{'@ref:external': f'p{i}'} for i in range(args.size)],
             })]
    result = {'label': args.label, 'python': platform.python_version(),
              'pydantic': pydantic.__version__, 'source': str(source.resolve()),
              'size': args.size, 'repeat': args.repeat, 'rounds': args.rounds, 'cases': {}}
    for name, model_type, payload in cases:
        encoded = json.dumps(payload)
        model = model_type.rune_deserialize(encoded, validate_model=False)
        operations = {
            'read': lambda: model_type.rune_deserialize(encoded, validate_model=False),
            'write': lambda: model.rune_serialize(validate_model=False),
            'validate': model.validate_model,
        }
        measurements = {}
        for operation, call in operations.items():
            try:
                call()
            except Exception as error:
                measurements[operation] = {'error': f'{type(error).__name__}: {error}'}
                continue
            timings = []
            for _ in range(args.rounds):
                gc.collect()
                timings.append(min(timeit.repeat(call, number=1, repeat=args.repeat)))
            measurements[operation] = {'seconds': timings}
            if args.profile:
                args.profile.mkdir(parents=True, exist_ok=True)
                profile = cProfile.Profile()
                profile.runcall(call)
                profile.dump_stats(str(args.profile / f'{args.label}-{name}-{operation}.prof'))
        result['cases'][name] = measurements
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
