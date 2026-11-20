#!/usr/bin/env python3
"""Exercise actual CLI parsing without opening a GPU context (Linux and Windows)."""
import pathlib
import subprocess
import sys

for argument in sys.argv[1:]:
    binary = str(pathlib.Path(argument).resolve())
    count = 0
    def run(args, expected=0, contains=None):
        global count
        p = subprocess.run([binary, *args], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=15)
        assert p.returncode == expected, (args, p.returncode, p.stdout)
        if contains:
            assert contains in p.stdout, (args, p.stdout)
        count += 1
    run(['--help'], contains='--no-autotune')
    for args in [
        ['--no-autotune', '--list-variants'],
        ['--list-variants', '--no-autotune'],
        ['--autotune', '0', '--no-autotune', '--list-variants'],
        ['--no-autotune', '--autotune', '0', '--list-variants'],
    ]:
        run(args, contains='fe')
    for args in [
        ['--no-autotune', '--autotune'],
        ['--autotune', '--no-autotune'],
        ['--no-autotune', '--autotune', '6'],
        ['--autotune', '6', '--no-autotune'],
    ]:
        run(args, expected=1, contains='--no-autotune conflicts with --autotune')
    print(pathlib.Path(binary).name, count, 'CPU-only startup flag checks passed')
assert len(sys.argv) > 1, 'provide at least one executable'
