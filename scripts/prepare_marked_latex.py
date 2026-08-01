


from __future__ import annotations

import re
import sys


BEGIN_DOCUMENT = r"\begin{document}"
DELETE_SPANS = (
    (r"\DIFdelbeginFL", r"\DIFdelendFL"),
    (r"\DIFdelbegin", r"\DIFdelend"),
)
DELETE_COMMANDS = (r"\DIFdelFL", r"\DIFdel")


def skip_braced_group(text: str, opening: int) -> int:

    if opening >= len(text) or text[opening] != "{":
        return opening
    depth = 0
    index = opening
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise ValueError("Unbalanced latexdiff deletion group")


def strip_deletions(body: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(body):
        matched = False
        for begin, end in DELETE_SPANS:
            if body.startswith(begin, index):
                closing = body.find(end, index + len(begin))
                if closing < 0:
                    raise ValueError(f"Missing {end} for deletion span")
                index = closing + len(end)
                matched = True
                break
        if matched:
            continue
        for command in DELETE_COMMANDS:
            prefix = command + "{"
            if body.startswith(prefix, index):
                index = skip_braced_group(body, index + len(command))
                matched = True
                break
        if matched:
            continue
        output.append(body[index])
        index += 1
    return "".join(output)


def polish_diff_boundaries(body: str) -> str:


    body = re.sub(
        r"(\\DIFaddbeginFL)\s*(\\hline)",
        r"\2 \1",
        body,
    )


    body = re.sub(
        r"(?<=[A-Za-z0-9}])\s+(\\DIFaddbegin(?:FL)?\s+\\DIFadd(?:FL)?\{(?=[A-Za-z0-9]))",
        r"\\ \1",
        body,
    )
    body = re.sub(
        r"(\}\\DIFaddend(?:FL)?)\s+(?=[A-Za-z0-9])",
        r"\1\\ ",
        body,
    )
    body = re.sub(
        r"\s+(\\DIFaddbegin(?:FL)?\s+\\DIFadd(?:FL)?\{(?=[,.;:?!]))",
        r"\1",
        body,
    )
    body = re.sub(r"[ \t]+([,.;:?!])", r"\1", body)
    body = re.sub(r"\([ \t]+", "(", body)
    body = re.sub(r"[ \t]+\)", ")", body)
    return body


def main() -> None:
    source = sys.stdin.read()
    position = source.find(BEGIN_DOCUMENT)
    if position < 0:
        raise SystemExit("No \\begin{document} found")
    preamble = source[:position]
    body = polish_diff_boundaries(strip_deletions(source[position:]))
    overrides = r"""
% Journal-facing marked copy: additions are red and deletions are omitted.
% Revised figures are displayed without colored frames.
\renewcommand{\DIFaddtex}[1]{{\protect\color{red}#1}}
\renewcommand{\DIFaddincludegraphics}[2][]{\DIFOincludegraphics[#1]{#2}}
\renewcommand{\DIFdelincludegraphics}[2][]{}

"""
    sys.stdout.write(preamble + overrides + body)


if __name__ == "__main__":
    main()
