


from __future__ import annotations

import re
import sys


def strip_inactive_blocks(source: str) -> str:

    token_pattern = re.compile(r"\\iffalse\b|\\fi\b")
    output: list[str] = []
    depth = 0
    cursor = 0

    for match in token_pattern.finditer(source):
        if depth == 0:
            output.append(source[cursor : match.start()])
        if match.group() == r"\iffalse":
            depth += 1
        elif depth > 0:
            depth -= 1
        else:
            output.append(source[cursor : match.end()])
        cursor = match.end()

    if depth != 0:
        raise ValueError("Unbalanced \\iffalse/\\fi block in LaTeX input")
    output.append(source[cursor:])
    return "".join(output)


if __name__ == "__main__":
    sys.stdout.write(strip_inactive_blocks(sys.stdin.read()))
