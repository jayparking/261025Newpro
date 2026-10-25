"""CPU tests, optionally treating dependency skips as server validation failures."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-no-skips', action='store_true')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.require_no_skips and result.skipped:
        print('Server gate failed: install missing test dependencies; skips are not accepted.', file=sys.stderr)
    return int(not result.wasSuccessful() or (args.require_no_skips and bool(result.skipped)))


if __name__ == '__main__':
    raise SystemExit(main())
