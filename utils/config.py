"""JSON presets translated into validated CLI arguments; explicit flags win."""
import argparse
import json
import sys
from pathlib import Path


def parse_with_config(parser, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config', type=Path)
    selected, remaining = probe.parse_known_args(argv)
    parser.add_argument('--config', type=Path, help='JSON preset; explicit CLI options override it')
    if selected.config is None:
        return parser.parse_args(argv)
    try:
        config = json.loads(selected.config.read_text())
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not isinstance(config, dict):
        parser.error('Config must be a JSON object')
    actions = {a.dest: a for a in parser._actions if a.option_strings and a.dest not in ('config', 'help')}
    tokens = []
    for key, value in config.items():
        if key not in actions:
            parser.error(f'Unknown config key: {key}')
        action = actions[key]
        option = next(s for s in action.option_strings if s.startswith('--'))
        if isinstance(action, argparse._StoreTrueAction):
            if not isinstance(value, bool):
                parser.error(f'{key} must be boolean')
            if value:
                tokens.append(option)
        elif value is not None:
            tokens.append(option)
            tokens.extend(str(v) for v in value) if isinstance(value, list) else tokens.append(str(value))
    # argparse validates types, choices and required options for both sources.
    result = parser.parse_args([*tokens, *remaining])
    result.config = str(selected.config)
    return result
