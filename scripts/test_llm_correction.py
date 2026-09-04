"""One-off evaluation: does asking a local Ollama LLM to "correct" Tesseract's
raw OCR text actually improve measured accuracy, or does it (like the Qari
hybrid experiment) look promising but measure worse?

Same methodology as measure_ocr_accuracy.py / the hybrid-engine comparison:
native PDF text layer as ground truth, CER/WER/word-recognition-only on the
LLM-corrected text vs. Tesseract's own raw text, same reference for both.

Uses urllib (stdlib) rather than requests to avoid a new dependency, and
writes the JSON payload as UTF-8 bytes directly to the socket rather than
ever passing Arabic text through a shell argument (a real bug hit earlier:
Windows/Git-Bash shell-argument passing corrupted Arabic text before it
reached curl, producing a false "the model can't read Arabic" result).

--show-text is OFF by default deliberately: this script must be safe to run
against ODOKAN_UMA_8T10.pdf under the same narrow-engagement rule
measure_ocr_accuracy.py's docstring documents (aggregate numbers only,
never page content) — pass --show-text only for the general-purpose
document, never for that one.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pymupdf

from smart_text_extractor.ocr.engine import OcrEngine
from smart_text_extractor.ocr.locate import find_tessdata_dir, find_tesseract_cmd
from smart_text_extractor.ocr.native_pdf_text import extract_native_text_result

from measure_ocr_accuracy import _measure_page

OLLAMA_URL = "http://localhost:11434/api/generate"
DPI = 300

_CONSERVATIVE_PROMPT = """أنت مصحّح نصوص OCR عربية. النص التالي مستخرَج آلياً وقد يحتوي أخطاء تعرّف حرفية (حرف بدل آخر، مسافات زائدة أو ناقصة). صحّح فقط الأخطاء الإملائية/الحرفية الواضحة. لا تُعِد صياغة الجمل، لا تُضِف أي كلمة أو معلومة غير موجودة، لا تحذف أي جزء، حافظ على نفس بنية الأسطر والترتيب تماماً. إن لم تكن متأكداً من كلمة، اتركها كما هي دون تغيير. أجب بالنص المُصحَّح فقط دون أي شرح أو مقدمة.

النص:
{text}"""

_ASSERTIVE_PROMPT = """أنت مصحّح نصوص OCR عربية محترف. النص التالي مُستخرَج بواسطة OCR من مستند عربي حقيقي وقد يحتوي أخطاء تعرّف حرفية بسبب جودة المسح: حرف مستبدل بآخر يشبهه شكلاً، حرف ناقص أو زائد، أو كلمة مشوّهة بالكامل.

اقرأ النص كاملاً لتفهم سياقه ومعناه العام، ثم صحّح **كل** كلمة تبدو غريبة أو لا تتناسب مع سياقها لتُصبح الكلمة الصحيحة التي يقتضيها المعنى والسياق المحيط — استخدم فهمك للسياق لحسم الكلمة الصحيحة حتى لو كان الخطأ الأصلي غير واضح لأول وهلة، لا فقط الأخطاء الفادحة الواضحة.

قيود صارمة (لا استثناء):
- لا تُضِف أي جملة أو معلومة أو تفصيل غير موجود أصلاً في النص.
- لا تحذف أي كلمة أو جزء من النص مهما بدت غريبة أو غير مفهومة — إن لم تستطع تحديد الكلمة الصحيحة بثقة، أبقِ الكلمة الأصلية كما هي حرفياً بدل حذفها. الحذف ممنوع تماماً؛ الإبقاء على كلمة غير مؤكَّدة أفضل دائماً من حذفها.
- لا تُعِد صياغة الجمل أسلوبياً.
- حافظ على نفس عدد الأسطر وترتيبها وبنيتها تماماً (بما فيها الفواصل | إن وُجدت).

أجب بالنص المُصحَّح فقط دون أي شرح أو مقدمة أو تعليق.

النص:
{text}"""

_PROMPTS = {"conservative": _CONSERVATIVE_PROMPT, "assertive": _ASSERTIVE_PROMPT}


def ask_ollama(model: str, prompt: str, think: bool = False) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False, "think": think}).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))
    response_text = result["response"].strip()
    # qwen3 has a habit of appending a trailing markdown-style double-space
    # to every line — not a real character error, just formatting noise
    # that would otherwise inflate CER against a reference with no such
    # habit. Normalized here, not on Tesseract's own output, so Tesseract's
    # numbers stay exactly as measured everywhere else this session.
    return "\n".join(line.rstrip() for line in response_text.split("\n"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--prompts", default="assertive", help="comma-separated: conservative,assertive")
    parser.add_argument(
        "--show-text", action="store_true",
        help="print raw/corrected text snippets (off by default — never pass this for ODOKAN_UMA_8T10.pdf)",
    )
    args = parser.parse_args()
    active_prompts = {name: _PROMPTS[name] for name in args.prompts.split(",")}

    tesseract_engine = OcrEngine(tesseract_cmd=find_tesseract_cmd(), tessdata_dir=find_tessdata_dir())
    zoom = DPI / 72
    matrix = pymupdf.Matrix(zoom, zoom)

    print(f"=== testing model: {args.model} on {args.pages} page(s) of {args.pdf.name} ===")

    totals: dict[str, dict[str, float]] = {
        "tesseract": {"char_errors": 0, "chars": 0, "word_errors": 0, "words": 0, "bow_errors": 0}
    }
    for name in active_prompts:
        totals[name] = {"char_errors": 0, "chars": 0, "word_errors": 0, "words": 0, "bow_errors": 0}

    def _accumulate(bucket: dict[str, float], acc) -> None:
        bucket["char_errors"] += round(acc.cer * acc.reference_char_count)
        bucket["chars"] += acc.reference_char_count
        bucket["word_errors"] += round(acc.wer * acc.reference_word_count)
        bucket["words"] += acc.reference_word_count
        bucket["bow_errors"] += round(acc.bag_of_words_wer * acc.reference_word_count)

    with pymupdf.open(str(args.pdf)) as document:
        for page_index in range(min(args.pages, len(document))):
            page = document.load_page(page_index)
            native_result = extract_native_text_result(page, render_dpi=DPI)
            if native_result is None:
                print(f"page {page_index+1}: no usable native text layer, skipped")
                continue
            reference_text = native_result.raw_text

            pixmap = page.get_pixmap(matrix=matrix)
            from PIL import Image

            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            tess_result = tesseract_engine.run(image)
            tess_acc = _measure_page(reference_text, tess_result.raw_text, page_index + 1)
            _accumulate(totals["tesseract"], tess_acc)

            print(f"--- page {page_index+1} ---")
            print(
                f"  TESSERACT      CER={tess_acc.cer*100:5.1f}% WER={tess_acc.wer*100:5.1f}% "
                f"word-rec={100-tess_acc.bag_of_words_wer*100:5.1f}%"
            )

            corrected_by_prompt: dict[str, str] = {}
            for prompt_name, prompt_template in active_prompts.items():
                corrected_text = ask_ollama(args.model, prompt_template.format(text=tess_result.raw_text))
                corrected_by_prompt[prompt_name] = corrected_text
                acc = _measure_page(reference_text, corrected_text, page_index + 1)
                _accumulate(totals[prompt_name], acc)
                print(
                    f"  {prompt_name.upper():14s} CER={acc.cer*100:5.1f}% WER={acc.wer*100:5.1f}% "
                    f"word-rec={100-acc.bag_of_words_wer*100:5.1f}%"
                )

            if args.show_text:
                print(f"  --- tesseract raw ---\n  {tess_result.raw_text[:400]!r}")
                for prompt_name, corrected_text in corrected_by_prompt.items():
                    print(f"  --- {prompt_name} corrected ---\n  {corrected_text[:400]!r}")
            print()

    print("=== overall (weighted across measured pages) ===")
    for name, bucket in totals.items():
        if not bucket["chars"]:
            continue
        cer = bucket["char_errors"] / bucket["chars"]
        wer = bucket["word_errors"] / bucket["words"]
        bow = bucket["bow_errors"] / bucket["words"]
        print(f"  {name.upper():14s} CER={cer*100:5.1f}%  WER={wer*100:5.1f}%  word-recognition-only={100-bow*100:5.1f}%")


if __name__ == "__main__":
    main()
