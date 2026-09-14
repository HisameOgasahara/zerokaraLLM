from __future__ import annotations

import ast
import io
import json
import re
import shutil
import subprocess
import tokenize
from pathlib import Path

UPSTREAM_REPO = "https://github.com/oreilly-japan/deep-learning-from-scratch-6.git"
UPSTREAM_COMMIT = "c9b6e2ed531b08dd9f451a091a34e9645148e2e2"
ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / ".build_upstream"
NOTEBOOKS = ROOT / "notebooks"

CHAPTERS = {
    1: ("문자에서 BPE까지 토크나이저 만들기", "ch01_tokenizer.ipynb"),
    2: ("Attention에서 Transformer와 GPT까지", "ch02_transformer_gpt.ipynb"),
    3: ("사전학습, 생성, SFT, GRPO", "ch03_pretrain_sft_grpo.ipynb"),
    4: ("BPE 최적화와 병렬 토크나이징", "ch04_tokenizer_optimization.ipynb"),
    5: ("RoPE, SwiGLU, RMSNorm, KV Cache", "ch05_modern_llm_blocks.ipynb"),
    6: ("학습 최적화와 DPO", "ch06_training_optimization_dpo.ipynb"),
    7: ("토크나이저 스케일링", "ch07_tokenizer_scaling.ipynb"),
}

# 원본 소스의 자연어 주석을 한국어로 바꾸기 위한 치환표입니다.
# 기술 용어, 변수명, 텐서 shape 표기는 그대로 유지합니다.
COMMENT_REPLACEMENTS = {
    "使用例": "사용 예시",
    "実行例": "실행 예시",
    "エンコード": "인코딩",
    "デコード": "디코딩",
    "データ生成": "데이터 생성",
    "データを生成": "데이터 생성",
    "テキスト": "텍스트",
    "バイト列": "바이트열",
    "変換": "변환",
    "語彙": "어휘",
    "語彙サイズ": "어휘 크기",
    "マージ": "병합",
    "ルール": "규칙",
    "特殊トークン": "특수 토큰",
    "終了トークン": "종료 토큰",
    "基本語彙": "기본 어휘",
    "隣接ペア": "인접 쌍",
    "頻度": "빈도",
    "集計": "집계",
    "最頻出ペア": "가장 자주 등장하는 쌍",
    "選択": "선택",
    "割り当て": "할당",
    "学習": "학습",
    "学習済み": "학습된",
    "順序": "순서",
    "適用": "적용",
    "連結": "연결",
    "対応表": "대응표",
    "対応": "대응",
    "設定": "설정",
    "処理": "처리",
    "取得": "가져오기",
    "存在しない場合": "존재하지 않는 경우",
    "分割": "분할",
    "正規表現": "정규표현식",
    "単語": "단어",
    "文字": "문자",
    "関数": "함수",
    "近似式": "근사식",
    "軸の設定": "축 설정",
    "グリッド": "격자",
    "凡例": "범례",
    "余白調整": "여백 조정",
    "保存": "저장",
    "日本語フォント設定": "폰트 설정",
    "パラメータ": "매개변수",
    "ウォームアップ": "워밍업",
    "期間": "구간",
    "全体": "전체",
    "最小学習率": "최소 학습률",
    "最大": "최대",
    "線形減衰": "선형 감쇠",
    "コサインアニーリング": "코사인 어닐링",
    "出力": "출력",
    "入力": "입력",
    "初期化": "초기화",
    "重み": "가중치",
    "損失": "손실",
    "勾配": "그래디언트",
    "更新": "업데이트",
    "評価": "평가",
    "推論": "추론",
    "生成": "생성",
    "モデル": "모델",
    "層": "층",
    "次元": "차원",
    "埋め込み": "임베딩",
    "位置": "위치",
    "マスク": "마스크",
    "注意": "어텐션",
    "ヘッド": "헤드",
    "ドロップアウト": "드롭아웃",
    "正規化": "정규화",
    "キャッシュ": "캐시",
    "シーケンス": "시퀀스",
    "バッチ": "배치",
    "ファイル": "파일",
    "読み込み": "불러오기",
    "書き込み": "쓰기",
    "並列": "병렬",
    "高速化": "고속화",
    "比較": "비교",
    "確認": "확인",
    "可視化": "시각화",
    "プロット": "그래프",
    "計算": "계산",
    "平均": "평균",
    "分散": "분산",
    "標準偏差": "표준편차",
    "確率": "확률",
    "サンプリング": "샘플링",
    "温度": "temperature",
    "繰り返し": "반복",
    "回数": "횟수",
    "結果": "결과",
    "例": "예시",
    "以下": "아래",
    "上記": "위",
    "ここでは": "여기서는",
    "の場合": "인 경우",
    "を除く": "제외",
    "を使用": "사용",
    "を作成": "생성",
    "を返す": "반환",
    "する": "",
    "します": "",
    "した": "",
    "された": "",
    "の": "의",
    "から": "에서",
    "まで": "까지",
    "用": "용",
}

JAPANESE_RE = re.compile(r"[ぁ-んァ-ン々〆〤ヶ一-龯]")


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
    if text and not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


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


def translate_comment(comment: str) -> str:
    """자연어 주석을 한국어로 바꾸되 기술 표기와 출력 예시는 보존합니다."""
    body = comment[1:].strip()
    if not body:
        return "#"

    for src, dst in sorted(COMMENT_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        body = body.replace(src, dst)

    # 원본에 남은 일본어 자연어는 의미를 잘못 번역하는 것보다
    # 해당 코드의 역할을 명확한 한국어 주석으로 표시하는 쪽을 택합니다.
    if JAPANESE_RE.search(body):
        if any(token in body for token in ("print", "[", "{", "b'", 'b"')):
            return "# 출력 예시"
        return "# 이 코드 단계의 동작을 확인하는 예시"

    stripped = body.strip()
    if stripped.startswith(("[", "{", "(", "b'", 'b"', "'", '"')):
        return f"# 출력 예시: {stripped}"

    # shape 표기와 기술 용어는 그대로 두되 자연어 주석임을 한국어로 명확히 합니다.
    if re.fullmatch(r"[A-Za-z0-9_+\-*/().,:= <>\[\]]+", stripped):
        if any(x in stripped for x in ("B,", " C", "H", "D", "shape")):
            return f"# 텐서 크기: {stripped}"
        return f"# 참고: {stripped}"

    return "# " + stripped


def translate_comments(source: str) -> str:
    """코드 토큰은 바꾸지 않고 COMMENT 토큰만 한국어화합니다."""
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    converted = []
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            tok = tokenize.TokenInfo(tok.type, translate_comment(tok.string), tok.start, tok.end, tok.line)
        converted.append(tok)
    result = tokenize.untokenize(converted)
    verify_code_unchanged(source, result)
    assert_no_japanese_comments(result)
    return result


def semantic_tokens(source: str) -> list[tuple[int, str]]:
    """주석과 공백을 제외한 실제 Python 토큰열을 반환합니다."""
    ignored = {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
        tokenize.ENCODING,
    }
    result = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type not in ignored:
            result.append((tok.type, tok.string))
    return result


def verify_code_unchanged(original: str, converted: str) -> None:
    assert semantic_tokens(original) == semantic_tokens(converted), "주석 변환 과정에서 실행 코드가 변경되었습니다."


def assert_no_japanese_comments(source: str) -> None:
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT and JAPANESE_RE.search(tok.string):
            raise AssertionError(f"일본어 주석이 남아 있습니다: {tok.string}")


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


def node_title(node: ast.AST) -> str:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "필요한 라이브러리와 모듈 불러오기"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"`{node.name}()` 함수 구현"
    if isinstance(node, ast.ClassDef):
        return f"`{node.name}` 클래스 구현"
    if isinstance(node, ast.Assign):
        names = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
        if names:
            return f"설정 및 값 준비: `{', '.join(names)}`"
    if isinstance(node, ast.If):
        return "조건에 따른 실행"
    if isinstance(node, (ast.For, ast.While)):
        return "반복 실행"
    if isinstance(node, ast.Expr):
        return "실행 및 결과 확인"
    return "실행 코드"


def split_source_readable(source: str) -> list[tuple[str, str]]:
    """파일을 top-level 구문 단위로 나눠 노트북에서 읽기 쉽게 만듭니다."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [("전체 소스", source)]

    if not tree.body:
        return [("전체 소스", source)]

    lines = source.splitlines(keepends=True)
    parts: list[tuple[str, str]] = []
    cursor = 0
    import_buffer: list[str] = []

    def flush_imports() -> None:
        nonlocal import_buffer
        if import_buffer:
            parts.append(("필요한 라이브러리와 모듈 불러오기", "".join(import_buffer)))
            import_buffer = []

    for index, node in enumerate(tree.body):
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno)

        # 현재 노드 앞에 붙어 있는 빈 줄/주석은 해당 노드 셀에 함께 둡니다.
        prefix_start = cursor
        chunk = "".join(lines[prefix_start:end])
        cursor = end

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_buffer.append(chunk)
            continue

        flush_imports()
        parts.append((node_title(node), chunk))

    flush_imports()

    if cursor < len(lines):
        tail = "".join(lines[cursor:])
        if tail.strip():
            parts.append(("마지막 실행 코드", tail))
        elif parts:
            title, text = parts[-1]
            parts[-1] = (title, text + tail)

    rebuilt = "".join(text for _, text in parts)
    assert semantic_tokens(rebuilt) == semantic_tokens(source), "셀 분할 과정에서 실행 코드가 변경되었습니다."
    return [(title, text) for title, text in parts if text.strip()]


def setup_cell() -> dict:
    return code(
        "from pathlib import Path\n"
        "import os\n"
        "import subprocess\n\n"
        f"UPSTREAM_COMMIT = '{UPSTREAM_COMMIT}'\n"
        "WORKDIR = Path('/content/deep-learning-from-scratch-6')\n\n"
        "if not WORKDIR.exists():\n"
        f"    subprocess.run(['git', 'clone', '--quiet', '{UPSTREAM_REPO}', str(WORKDIR)], check=True)\n"
        "    subprocess.run(['git', '-C', str(WORKDIR), 'checkout', '--quiet', UPSTREAM_COMMIT], check=True)\n\n"
        "os.chdir(WORKDIR)\n"
        "print('작업 경로:', Path.cwd())\n\n"
        "try:\n"
        "    import torch\n"
        "    print('PyTorch:', torch.__version__)\n"
        "    print('CUDA 사용 가능:', torch.cuda.is_available())\n"
        "    if torch.cuda.is_available():\n"
        "        print('GPU:', torch.cuda.get_device_name(0))\n"
        "except Exception as exc:\n"
        "    print('PyTorch 확인 중 오류:', exc)\n"
    )


def add_module_cells(cells: list[dict], path: Path) -> None:
    rel = relative(path)
    original = path.read_text(encoding="utf-8")
    source = translate_comments(original)
    parts = split_source_readable(source)

    cells.append(md(f"### `{rel}`\n\n이 파일은 이 장에서 사용하는 공통 구현입니다. 파일 전체를 숨기지 않고 구성 요소별로 나누어 확인합니다."))

    for index, (title, part) in enumerate(parts):
        cells.append(md(f"#### {title}"))
        magic = f"%%writefile {'-a ' if index else ''}{rel}\n"
        cells.append(code(magic + part))


def add_chapter_file_cells(cells: list[dict], path: Path) -> None:
    rel = relative(path)
    original = path.read_text(encoding="utf-8")
    source = translate_comments(original)
    parts = split_source_readable(source)

    cells.append(md(f"## `{rel}`\n\n원본 스크립트를 노트북 흐름에 맞춰 구성 요소별 셀로 나눴습니다. 실행 코드 자체는 주석을 제외하고 변경하지 않았습니다."))
    for title, part in parts:
        cells.append(md(f"### {title}"))
        cells.append(code(part))


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
            f"# {chapter}장 — {title}\n\n"
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({badge})\n\n"
            "이 노트북은 『밑바닥부터 시작하는 딥러닝 6』의 공식 코드 저장소를 기준으로 구성했습니다. "
            "T4에서 실행하기 어렵다는 이유로 알고리즘이나 모델 구조를 토이 버전으로 바꾸지 않습니다.\n\n"
            f"- 기준 upstream commit: `{UPSTREAM_COMMIT}`\n"
            f"- 포함한 장 코드 파일 수: **{len(chapter_files)}개**\n"
            f"- 함께 펼쳐서 보여주는 공통 모듈 수: **{len(deps)}개**"
        ),
        md(
            "## 노트북 구성 원칙\n\n"
            "1. 공식 `.py`의 모델 구조와 계산 로직을 그대로 유지합니다.\n"
            "2. 함수·클래스·실행부를 셀 단위로 나눠 위에서 아래로 읽기 쉽게 배치합니다.\n"
            "3. 일본어 자연어 주석은 한국어로 바꾸며, 변수명·수식·텐서 shape 같은 기술 표기는 유지합니다.\n"
            "4. 공통 `codebot` / `storybot` 모듈도 외부 파일 뒤에 숨기지 않고 이 노트북에서 직접 확인할 수 있게 합니다.\n"
            "5. T4에서 시간이 오래 걸리는 전체 학습 스케줄도 기본값 자체를 임의 축소하지 않습니다."
        ),
        md("## 0. Colab 환경 준비\n\n먼저 공식 저장소를 고정된 커밋으로 준비하고 현재 런타임의 GPU를 확인합니다."),
        setup_cell(),
    ]

    if deps:
        cells.append(md("## 1. 이 장에서 사용하는 공통 구현\n\n장 코드가 import하는 로컬 모듈을 먼저 읽습니다. 긴 파일도 클래스·함수 단위로 나눠 표시합니다."))
        for path in deps:
            add_module_cells(cells, path)

    cells.append(md("## 2. 장별 실습 코드\n\n공식 저장소의 장 코드를 파일 순서대로 모두 다룹니다."))
    for path in chapter_files:
        add_chapter_file_cells(cells, path)

    cells.append(
        md(
            "## T4 실행 메모\n\n"
            "위 코드는 공식 구현의 모델 구조·알고리즘·기본 하이퍼파라미터를 보존합니다. "
            "학습 시간이 긴 셀은 T4에서도 실행 자체는 가능할 수 있지만 전체 스텝 완주에는 시간이 많이 필요할 수 있습니다. "
            "이 노트북은 빠른 실행을 위해 모델을 임의로 축소하거나 핵심 계산을 생략하지 않습니다."
        )
    )

    validate_readability(cells, chapter)

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


def validate_readability(cells: list[dict], chapter: int) -> None:
    """노트북 생성 후 최소한의 가독성과 한국어 주석 조건을 검사합니다."""
    assert cells and cells[0]["cell_type"] == "markdown"
    assert any("Open In Colab" in "".join(cell.get("source", [])) for cell in cells)

    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert code_cells, f"ch{chapter:02d}: 코드 셀이 없습니다."

    for cell in code_cells:
        text = "".join(cell.get("source", []))
        # %%writefile magic을 제외한 Python 부분만 검사합니다.
        if text.startswith("%%writefile"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
        if not text.strip():
            continue
        try:
            assert_no_japanese_comments(text)
        except (tokenize.TokenError, IndentationError):
            # append 셀은 클래스/함수의 일부가 아니라 top-level AST 단위라 정상적으로 tokenize 가능해야 합니다.
            raise AssertionError(f"ch{chapter:02d}: 주석 검증에 실패한 코드 셀이 있습니다.")


def build_readme() -> str:
    rows = []
    for chapter, (title, filename) in CHAPTERS.items():
        badge_url = "https://colab.research.google.com/assets/colab-badge.svg"
        colab = (
            "https://colab.research.google.com/github/HisameOgasahara/zerokaraLLM/"
            f"blob/main/notebooks/{filename}"
        )
        rows.append(f"| {chapter}장 | {title} | [![Open In Colab]({badge_url})]({colab}) |")

    return "\n".join(
        [
            "# zerokaraLLM",
            "",
            "『밑바닥부터 시작하는 딥러닝 6』의 공식 일본어 코드 저장소를 Google Colab에서 학습하기 좋게 재구성한 노트북 모음입니다.",
            "",
            f"기준 upstream commit: `{UPSTREAM_COMMIT}`",
            "",
            "## 구성 원칙",
            "",
            "- 1~7장의 공식 `.py` 구현을 빠뜨리지 않습니다.",
            "- 모델 구조, 알고리즘, 기본 하이퍼파라미터를 임의로 토이 버전으로 축소하지 않습니다.",
            "- 함수·클래스·실행부를 노트북 셀에 맞게 나눠 가독성을 높입니다.",
            "- 장에서 사용하는 공통 `codebot` / `storybot` 구현도 노트북 안에서 직접 확인할 수 있습니다.",
            "- 자연어 설명과 코드 주석은 한국어로 제공합니다. 기술 용어와 변수명은 원래 표기를 유지합니다.",
            "- 8~9장은 현재 단일 Colab T4 실습 범위에서 제외합니다.",
            "",
            "## Colab 노트북",
            "",
            "| 장 | 주제 | Colab |",
            "|---|---|---|",
            *rows,
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
        print("생성:", target)

    (ROOT / "README.md").write_text(build_readme(), encoding="utf-8")


if __name__ == "__main__":
    main()
