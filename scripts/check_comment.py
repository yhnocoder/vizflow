import ast
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

WHY = re.compile(r"(?://|#)\s*why\(#\d+\): \S")
SLASH_DIRECTIVE = re.compile(
    r"(?:///?|/\*)\s*(?:eslint-|@ts-|prettier-ignore|biome-ignore|deno-lint-ignore|deno-fmt-ignore"
    r"|istanbul ignore|c8 ignore|NOLINT|clang-format |swiftlint:|go:|\+build|nolint|<reference )"
)
HASH_DIRECTIVE = re.compile(
    r"#\s*(?:noqa|type:|pragma|pyright:|mypy:|fmt:|pylint:|ruff:|isort:|-\*-|coding[:=]|shellcheck\s)"
)
SCRIPT_METADATA_START = re.compile(r"#\s*/// [a-z]")
SCRIPT_METADATA_END = "# ///"
HEREDOC = re.compile(r"<<(-?)[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
RUST_RAW_STRING = re.compile(r"(?:br|cr|r)(#*)\"")
RUST_CHAR = re.compile(r"'(?:[^'\\\n]|\\(?:[nrt0\\'\"]|x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]{1,6}\}))'")
QUOTED_CHAR = re.compile(r"'(?:[^'\\\n]|\\.)*'")
SHEBANG = re.compile(r"#!\s*(\S+)(?:\s+(?:-\S+\s+)*(\S+))?")

SLASH_LANGUAGES = {
    ".rs": "rust",
    ".swift": "nested",
    ".kt": "nested",
    ".kts": "nested",
    ".scala": "nested",
    ".c": "c",
    ".h": "c",
    ".cc": "c",
    ".cpp": "c",
    ".hpp": "c",
    ".java": "c",
    ".cs": "c",
    ".go": "go",
    ".dart": "js",
    ".js": "js",
    ".jsx": "js",
    ".mjs": "js",
    ".cjs": "js",
    ".ts": "js",
    ".tsx": "js",
    ".mts": "js",
    ".cts": "js",
}
PYTHON_EXTENSIONS = {".py", ".pyi"}
SHELL_EXTENSIONS = {".sh", ".bash", ".zsh"}
INTERPRETERS = {
    "python": "python",
    "uv": "python",
    "sh": "shell",
    "bash": "shell",
    "zsh": "shell",
    "dash": "shell",
    "node": "js",
}


def slash_comments(source, flavor):
    found = []
    i = 0
    line = 1
    n = len(source)
    while i < n:
        c = source[i]
        if c == "\n":
            line += 1
            i += 1
        elif source.startswith("//", i):
            end = line_end(source, i)
            found.append((line, source[i:end]))
            i = end
        elif source.startswith("/*", i):
            end = block_end(source, i, nested=flavor in ("rust", "nested"))
            text = source[i:end]
            found.append((line, text.splitlines()[0]))
            line += text.count("\n")
            i = end
        elif flavor == "rust" and (raw := RUST_RAW_STRING.match(source, i)) and not is_identifier_char(source, i - 1):
            closing = '"' + raw.group(1)
            end = source.find(closing, raw.end())
            end = n if end == -1 else end + len(closing)
            line += source.count("\n", i, end)
            i = end
        elif c == '"' or (c in "'`" and flavor == "js") or (c == "`" and flavor == "go"):
            end = quoted_end(source, i, c)
            line += source.count("\n", i, end)
            i = end
        elif c == "'" and (char := (RUST_CHAR if flavor == "rust" else QUOTED_CHAR).match(source, i)):
            i = char.end()
        else:
            i += 1
    return [(number, text) for number, text in found if not SLASH_DIRECTIVE.match(text)]


def line_end(source, start):
    end = source.find("\n", start)
    return len(source) if end == -1 else end


def block_end(source, start, nested):
    if not nested:
        end = source.find("*/", start + 2)
        return len(source) if end == -1 else end + 2
    depth = 0
    j = start
    while j < len(source):
        if source.startswith("/*", j):
            depth += 1
            j += 2
        elif source.startswith("*/", j):
            depth -= 1
            j += 2
            if depth == 0:
                return j
        else:
            j += 1
    return j


def quoted_end(source, start, quote):
    j = start + 1
    while j < len(source) and source[j] != quote:
        j += 2 if source[j] == "\\" else 1
    return min(j + 1, len(source))


def is_identifier_char(source, index):
    return index >= 0 and (source[index].isalnum() or source[index] == "_")


def python_comments(source):
    found = []
    in_metadata = False
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        line, text = token.start[0], token.string
        if in_metadata:
            in_metadata = text.rstrip() != SCRIPT_METADATA_END
        elif SCRIPT_METADATA_START.match(text):
            in_metadata = True
        elif not (line == 1 and text.startswith("#!")) and not HASH_DIRECTIVE.match(text):
            found.append((line, text))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            found.append((node.lineno, "docstring or string statement"))
    return sorted(found)


def shell_comments(source):
    found = []
    heredocs = []
    i = 0
    line = 1
    n = len(source)
    while i < n:
        c = source[i]
        if c == "\n":
            line += 1
            i += 1
            for delimiter, strip_tabs in heredocs:
                i, line = heredoc_end(source, i, line, delimiter, strip_tabs)
            heredocs = []
        elif c == "\\":
            line += source.count("\n", i, i + 2)
            i += 2
        elif c == "'":
            end = source.find("'", i + 1)
            end = n if end == -1 else end + 1
            line += source.count("\n", i, end)
            i = end
        elif c in '"`':
            end = quoted_end(source, i, c)
            line += source.count("\n", i, end)
            i = end
        elif c == "#" and (i == 0 or source[i - 1] in " \t\n;&|()"):
            end = line_end(source, i)
            text = source[i:end]
            if not (line == 1 and text.startswith("#!")) and not HASH_DIRECTIVE.match(text):
                found.append((line, text))
            i = end
        elif (heredoc := HEREDOC.match(source, i)) and not source.startswith("<<<", i):
            heredocs.append((heredoc.group(3), heredoc.group(1) == "-"))
            i = heredoc.end()
        else:
            i += 1
    return found


def heredoc_end(source, i, line, delimiter, strip_tabs):
    while i < len(source):
        end = line_end(source, i)
        text = source[i:end]
        i = end + 1
        line += 1
        if (text.lstrip("\t") if strip_tabs else text) == delimiter:
            break
    return i, line


def language(path, source):
    if path.suffix in SLASH_LANGUAGES:
        return SLASH_LANGUAGES[path.suffix]
    if path.suffix in PYTHON_EXTENSIONS:
        return "python"
    if path.suffix in SHELL_EXTENSIONS:
        return "shell"
    if path.suffix or not (shebang := SHEBANG.match(source)):
        return None
    program = shebang.group(2) if shebang.group(1).endswith("/env") else shebang.group(1)
    name = re.sub(r"[\d.]+$", "", Path(program or "").name)
    return INTERPRETERS.get(name)


def comments(path):
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    kind = language(path, source)
    if kind == "python":
        return python_comments(source)
    if kind == "shell":
        return shell_comments(source)
    if kind:
        return slash_comments(source, kind)
    return []


def source_files(args):
    if args:
        return [Path(arg) for arg in args]
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return [Path(path) for path in listed if Path(path).is_file()]


def main():
    violations = []
    for path in source_files(sys.argv[1:]):
        try:
            found = comments(path)
        except (SyntaxError, tokenize.TokenError) as error:
            violations.append(f"{path}: cannot parse: {error}")
            continue
        for line, text in found:
            if not WHY.match(text):
                violations.append(f"{path}:{line}: {text.strip()}")
    for violation in violations:
        print(violation)
    if violations:
        print("comments are not allowed; see CLAUDE.md, the only exception is why(#N): reason")
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
