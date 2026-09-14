from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

UPSTREAM_REPO = "https://github.com/oreilly-japan/deep-learning-from-scratch-6.git"
UPSTREAM_COMMIT = "c9b6e2ed531b08dd9f451a091a34e9645148e2e2"
ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / ".build_upstream"
NOTEBOOKS = ROOT / "notebooks"

CHAPTERS = {
    1: ("Tokenizer from characters to BPE", "ch01_tokenizer.ipynb"),
    2: ("Attention, Transformer and GPT", "ch02_transformer_gpt.ipynb"),
    3: ("Pretraining, generation, SFT and GRPO", "ch03_pretrain_sft_grpo.ipynb"),
    4: ("BPE optimization and parallel tokenization", "ch04_tokenizer_optimization.ipynb"),
    5: ("RoPE, SwiGLU, RMSNorm and KV cache", "ch05_modern_llm_blocks.ipynb"),
    6: ("Training optimization and DPO", "ch06_training_optimization_dpo.ipynb"),
    7: ("Tokenizer scaling", "ch07_tokenizer_scaling.ipynb"),
}


def clone_upstream() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    subprocess.run(["git", "init", str(BUILD)], check=True)
    subprocess.run(["git", "-C", str(BUILD), "remote", "add", "origin", UPSTREAM_REPO], check=True)
    subprocess.run(
        ["git", "-C", str(BUILD), "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT],
        check=True,
    )
    subprocess.run(["git", "-C", str(BUILD), "checkout", "FETCH_HEAD"], check=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    if text and not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def node_kind(node: ast.AST) -> str:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "imports"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return "definitions"
    return "execution"


def split_source_exact(source: str) -> list[tuple[str, str]]:
    """Split a Python file into readable notebook cells without changing any byte of source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [("source", source)]

    if not tree.body:
        return [("source", source)]

    lines = source.splitlines(keepends=True)
    chunks: list[tuple[str, str]] = []
    cursor = 0
    current_kind: str | None = None
    current_start = 0

    for node in tree.body:
        node_start = node.lineno - 1
        kind = node_kind(node)
        if current_kind is None:
            current_kind = kind
            current_start = cursor
        elif kind != current_kind:
            chunks.append((current_kind, "".join(lines[current_start:node_start])))
            current_kind = kind
            current_start = node_start
        cursor = getattr(node, "end_lineno", node.lineno)

    chunks.append((current_kind or "source", "".join(lines[current_start:])))
    chunks = [(kind, text) for kind, text in chunks if text]

    rebuilt = "".join(text for _, text in chunks)
    assert rebuilt == source, "Notebook split changed source text"
    return chunks


def module_to_path(module: str) -> Path | None:
    parts = module.split(".")
    file_path = BUILD.joinpath(*parts).with_suffix(".py")
    if file_path.exists():
        return file_path
    init_path = BUILD.joinpath(*parts, "__init__.py")
    if init_path.exists():
        return init_path
    return None


def local_imports(source: str) -> set[Path]:
    found: set[Path] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue

        for name in names:
            path = module_to_path(name)
            if path is not None:
                found.add(path)
    return found


def dependency_closure(chapter_files: list[Path]) -> list[Path]:
    queue: list[Path] = []
    seen: set[Path] = set(chapter_files)

    for path in chapter_files:
        queue.extend(local_imports(path.read_text(encoding="utf-8")))

    deps: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in seen or path in deps:
            continue
        deps.add(path)
        source = path.read_text(encoding="utf-8")
        queue.extend(local_imports(source))

    return sorted(deps, key=lambda p: str(p.relative_to(BUILD)))


def relative(path: Path) -> str:
    return path.relative_to(BUILD).as_posix()


def make_notebook(chapter: int, title: str, filename: str) -> dict:
    chapter_dir = BUILD / f"ch{chapter:02d}"
    chapter_files = sorted(chapter_dir.glob("*.py"), key=lambda p: p.name)
    deps = dependency_closure(chapter_files)

    badge = (
        "https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/"
        f"blob/main/notebooks/{filename}"
    )

    cells: list[dict] = [
        md(
            f"# Chapter {chapter}: {title}\n\n"
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({badge})\n\n"
            "This notebook is generated from the complete chapter source. Nothing is replaced by a toy implementation. "
            "Python files are only split into notebook cells for readability; concatenating those cells reproduces the original source exactly.\n\n"
            f"**Pinned upstream commit:** `{UPSTREAM_COMMIT}`\n"
        ),
        md(
            "## Notebook architecture\n\n"
            "The notebook follows the chapter as a readable pipeline rather than hiding implementation behind `%run` calls. "
            "Shared local modules used by the chapter are written from visible cells first, then every chapter script is presented in source order. "
            "Original model dimensions, algorithms, and training hyperparameters are preserved.\n"
        ),
        code(
            "from pathlib import Path\n"
            "import os\n"
            "import subprocess\n\n"
            "UPSTREAM_COMMIT = '" + UPSTREAM_COMMIT + "'\n"
            "WORKDIR = Path('/content/deep-learning-from-scratch-6')\n\n"
            "if not WORKDIR.exists():\n"
            "    subprocess.run(['git', 'clone', '--quiet', '" + UPSTREAM_REPO + "', str(WORKDIR)], check=True)\n"
            "    subprocess.run(['git', '-C', str(WORKDIR), 'checkout', '--quiet', UPSTREAM_COMMIT], check=True)\n\n"
            "os.chdir(WORKDIR)\n"
            "print('working directory:', Path.cwd())\n"
            "try:\n"
            "    import torch\n"
            "    print('torch:', torch.__version__)\n"
            "    print('cuda:', torch.cuda.is_available())\n"
            "    if torch.cuda.is_available():\n"
            "        print('gpu:', torch.cuda.get_device_name(0))\n"
            "except Exception as exc:\n"
            "    print('torch check:', exc)\n"
        ),
    ]

    if deps:
        cells.append(md("## Shared modules used by this chapter\n\nThese cells keep shared architecture visible while preserving the original package layout for imports.\n"))
        for path in deps:
            src = path.read_text(encoding="utf-8")
            rel = relative(path)
            digest = hashlib.sha256(src.encode()).hexdigest()
            cells.append(md(f"### `{rel}`\n\nSHA-256: `{digest}`\n"))
            cells.append(code(f"%%writefile {rel}\n{src}"))

    cells.append(md("## Complete chapter source\n"))

    for path in chapter_files:
        src = path.read_text(encoding="utf-8")
        rel = relative(path)
        digest = hashlib.sha256(src.encode()).hexdigest()
        parts = split_source_exact(src)

        cells.append(md(f"## `{rel}`\n\nSHA-256: `{digest}`\n"))
        for kind, part in parts:
            cells.append(md(f"**{kind.capitalize()}**\n"))
            cells.append(code(part))

    cells.append(
        md(
            "## T4 execution note\n\n"
            "The implementation above keeps the upstream code and hyperparameters intact. "
            "For chapters with long training loops, a Colab T4 can execute the implementation, but completing the full training schedule may take substantial wall-clock time. "
            "No reduced model, shortened algorithm, or toy substitute is enabled by default in this notebook.\n"
        )
    )

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def build_readme() -> str:
    rows = []
    for chapter, (title, filename) in CHAPTERS.items():
        badge_url = "https://colab.research.google.com/assets/colab-badge.svg"
        colab = (
            "https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/"
            f"blob/main/notebooks/{filename}"
        )
        rows.append(f"| {chapter} | {title} | [![Open In Colab]({badge_url})]({colab}) |")

    return "\n".join(
        [
            "# zerokaraLLM",
            "",
            "Colab notebooks for `oreilly-japan/deep-learning-from-scratch-6`.",
            "",
            "The notebooks for Chapters 1-7 contain the complete upstream chapter implementations. Source files are reorganized into readable notebook cells without dropping algorithms or replacing them with toy versions. Shared `codebot` / `storybot` / `webbot` modules needed by a chapter are also exposed in visible cells.",
            "",
            f"Pinned upstream commit: `{UPSTREAM_COMMIT}`",
            "",
            "Chapters 8 and 9 are intentionally excluded for now.",
            "",
            "| Chapter | Topic | Colab |",
            "|---|---|---|",
            *rows,
            "",
            "## Policy",
            "",
            "- Preserve upstream architecture, algorithms, dimensions, and default training hyperparameters.",
            "- Split `.py` files into notebook cells only for readability; generation fails if concatenation differs from the original source.",
            "- Keep shared local modules visible instead of hiding important implementation behind an external import.",
            "- Use the upstream checkout only for data/assets and the original package layout required by chapter scripts.",
            "- Do not introduce reduced/toy defaults merely to make a notebook finish quickly on T4.",
            "",
        ]
    )


def main() -> None:
    clone_upstream()
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)

    for chapter, (title, filename) in CHAPTERS.items():
        notebook = make_notebook(chapter, title, filename)
        target = NOTEBOOKS / filename
        target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", target)

    (ROOT / "README.md").write_text(build_readme(), encoding="utf-8")


if __name__ == "__main__":
    main()
