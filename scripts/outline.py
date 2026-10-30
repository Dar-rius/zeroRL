#!/usr/bin/env python3
"""Token-frugal file outliner: print only defs/classes/decorators with lines.

Usage:
  scripts/outline.py FILE [FILE ...]          # top-level + methods
  scripts/outline.py -d NAME FILE             # just line numbers of NAME
  scripts/outline.py -a -120 FILE             # tail slice (like tail -n 120)
"""
from __future__ import annotations

import ast
import sys


def outline(path: str) -> None:
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    print(f"== {path} ({len(src.splitlines())} lines) ==")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decs = [f"@{ast.unparse(d)}" if isinstance(d, ast.Name) else "@..." for d in node.decorator_list]
            head = f"{type(node).__name__[5:].lower()} {node.name}({ast.unparse(node.args)})"
            if isinstance(node, ast.ClassDef):
                head = f"class {node.name}" + (f"({ast.unparse(node.bases)})" if node.bases else "")
            for d in decs:
                print(f"{node.lineno:>5}  {d}")
            print(f"{node.lineno:>5}  {head}  # end {getattr(node, 'end_lineno', '?')}")
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        print(f"{sub.lineno:>5}    def {sub.name}({ast.unparse(sub.args)})  # end {sub.end_lineno}")


def main(argv: list[str]) -> int:
    if argv and argv[0] == "-d":
        name, path = argv[1], argv[2]
        for i, line in enumerate(open(path, encoding="utf-8"), 1):
            if name in line:
                print(f"{i}: {line.rstrip()}")
        return 0
    if argv and argv[0] == "-a":
        n, path = int(argv[1]), argv[2]
        lines = open(path, encoding="utf-8").readlines()
        for i, line in enumerate(lines[-n:], len(lines) - n + 1):
            print(f"{i}: {line.rstrip()}")
        return 0
    for path in argv:
        outline(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
