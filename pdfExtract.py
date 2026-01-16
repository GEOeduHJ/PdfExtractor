#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core PDF extraction and translation helpers used by the Streamlit UI.

This module focuses on page-based extraction (PyMuPDF preferred) and
post-processing helpers used by `app_streamlit.py`.
"""
import io
import os
import re
import statistics

# PyMuPDF is the primary extractor for this repo (preferred for layout-aware extraction)

# python-docx 지원 여부
try:
    from docx import Document
    HAVE_DOCX = True
except Exception:
    HAVE_DOCX = False

# This module exposes extraction and translation helpers for the Streamlit UI.





def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """PDF 바이너리에서 텍스트를 추출해 문자열로 반환합니다.
    pdfplumber 우선, 실패하면 PyPDF2로 폴백합니다. 텍스트가 없으면 빈 문자열을 반환합니다.
    """
    # Use the page-based extractor (PyMuPDF-preferred). Returns combined page texts.
    pages = extract_pages_from_pdf_bytes(pdf_bytes)
    if pages:
        parts = [p.get('text', '') for p in pages]
        return '\n\n'.join(parts)
    return ''


# PyMuPDF (better structured extraction)
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except Exception:
    HAVE_PYMUPDF = False


def reflow_from_word_tuples(word_tuples, line_tol=3.0, space_multiplier=0.4):
    """주어진 단어 튜플/딕셔너리 리스트에서 좌표를 이용해 적절한 공백을 넣어 리플로우한 텍스트를 반환합니다.

    - word_tuples: PyMuPDF의 get_text('words') 결과(튜플 리스트) 또는 pdfplumber의 extract_words() 결과(딕트 리스트)
    - line_tol: 같은 라인으로 간주할 y 좌표의 허용 오차(포인트)
    - space_multiplier: 평균 문자폭 대비 gap이 클 경우 공백으로 판단하는 임계치 배수

    추가 규칙:
    - 하이픈으로 끝나는 단어(prev)가 있으면 다음 단어와 바로 이어 붙여 하이픈 제거
    - 마침표/쉼표 등의 닫는 구두점은 앞단어에 붙여서 출력
    - 따옴표(열림 따옴표)가 단독 토큰으로 분리되면 다음 단어 앞에 붙여줌(pending_open_quote)
    """
    norm = []
    for w in word_tuples:
        try:
            if isinstance(w, dict):
                x0 = float(w.get('x0', w.get('x', 0)))
                x1 = float(w.get('x1', w.get('x1', x0)))
                y0 = float(w.get('top', w.get('y0', 0)))
                text = str(w.get('text', ''))
            elif isinstance(w, (list, tuple)) and len(w) >= 5:
                # PyMuPDF word tuple format: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
                x0 = float(w[0]); y0 = float(w[1]); x1 = float(w[2]); text = str(w[4])
            else:
                continue
            norm.append({'x0': x0, 'x1': x1, 'y0': y0, 'text': text})
        except Exception:
            continue

    if not norm:
        return ''

    # 정렬: y0(작은값 위쪽) 기준으로 그룹화한 뒤 x0 기준 정렬
    norm.sort(key=lambda n: (round(n['y0'], 1), n['x0']))

    # 라인 그룹화
    lines = []
    current_line = [norm[0]]
    current_y = norm[0]['y0']
    for item in norm[1:]:
        if abs(item['y0'] - current_y) <= line_tol:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
            current_y = item['y0']
    lines.append(current_line)

    out_lines = []
    opening_quotes = {'"', "\u201c", "\u2018", "'", '«', '‹'}
    closing_punct = {',', '.', ':', ';', '!', '?', ')', ']', '}', '%'}

    for line in lines:
        # 각 단어의 평균 문자 폭 추정
        char_widths = []
        for w in line:
            length = max(len(w['text']), 1)
            char_w = (w['x1'] - w['x0']) / length
            if char_w > 0:
                char_widths.append(char_w)
        if char_widths:
            avg_char_w = statistics.median(char_widths)
        else:
            avg_char_w = 5.0

        # 줄을 하나의 문자열로 구성
        line_str = ''
        prev = None
        prev_item = None
        pending_open_quote = None

        for w in line:
            cur_text = w['text']

            # 만약 이전에 열림 따옴표가 남아있으면 현재 단어 앞에 붙임
            if pending_open_quote:
                cur_text = pending_open_quote + cur_text
                pending_open_quote = None

            if prev_item is None:
                # 첫 단어는 그대로 추가
                line_str = cur_text
                prev_item = w
                prev = {'x1': w['x1'], 'text': cur_text}
                # 짧은 단일 열림 따옴표 토큰(예: " 또는 “)이면 보류
                if cur_text in opening_quotes and len(cur_text.strip()) == 1:
                    pending_open_quote = cur_text
                    # remove the lone quote from line_str (it will be prefixed to next)
                    line_str = ''
                    prev_item = None
                    prev = None
                continue

            # gap 계산
            gap = w['x0'] - prev_item['x1']
            gap = max(0.0, gap)

            # 하이픈으로 끝난 이전 단어 처리: 항상 붙여서 연결 (하이픈 제거)
            if prev and prev.get('text', '').endswith('-'):
                # remove trailing hyphen from current line_str
                line_str = line_str.rstrip('-')
                line_str += cur_text
                prev_item = w
                prev = {'x1': w['x1'], 'text': cur_text}
                continue

            # 닫는 구두점(콤마, 마침표 등)은 앞단어에 붙여서 출력
            if cur_text in closing_punct:
                # no space before punctuation
                line_str += cur_text
                prev_item = w
                prev = {'x1': w['x1'], 'text': cur_text}
                continue

            # 열림 따옴표만 있는 토큰이면 다음 단어로 이월
            if cur_text in opening_quotes and len(cur_text.strip()) == 1:
                pending_open_quote = cur_text
                # do not add anything now
                prev_item = w
                prev = {'x1': w['x1'], 'text': cur_text}
                continue

            # 일반 공백 판단: gap이 평균 문자폭 * space_multiplier 보다 크면 공백 추가
            if gap > (avg_char_w * space_multiplier):
                line_str += ' ' + cur_text
            else:
                # 작은 간격이면 공백 없이 연속
                line_str += cur_text

            prev_item = w
            prev = {'x1': w['x1'], 'text': cur_text}

        # strip leading/trailing spaces
        out_lines.append(line_str.strip())

    return '\n'.join(out_lines)


def extract_pages_from_pdf_bytes(pdf_bytes: bytes) -> list:
    """PDF 바이너리에서 페이지별 텍스트와 메타(이미지 여부)를 반환합니다.
    반환값: [{'text': str, 'has_images': bool}, ...]

    우선 PyMuPDF(fitz)를 시도하여 블록/단락 단위로 더 구조화된 텍스트를 얻고
    이미지 존재 여부는 page.get_images()로 확인합니다. 실패하면 pdfplumber와
    PyPDF2 순으로 폴백합니다.
    """
    pages = []

    # 1) PyMuPDF 우선
    if HAVE_PYMUPDF:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype='pdf')
            for p in doc:
                page_text = ''
                # 우선 word-level 정보를 얻어 레이아웃 기반 리플로우 시도
                try:
                    words = p.get_text('words')  # list of tuples
                except Exception:
                    words = None

                if words:
                    try:
                        page_text = reflow_from_word_tuples(words)
                    except Exception:
                        page_text = ''

                # words가 없거나 결과가 비어있으면 블록 단위로 폴백
                if not page_text:
                    try:
                        blocks = p.get_text('blocks')  # (x0, y0, x1, y1, text, block_no)
                    except Exception:
                        blocks = []
                    blocks = sorted(blocks, key=lambda b: (round(b[1]), round(b[0])))
                    line_texts = []
                    for b in blocks:
                        text = b[4].strip()
                        if text:
                            line_texts.append(text)
                    page_text = '\n\n'.join(line_texts)

                try:
                    images = p.get_images(full=True)
                    has_images = bool(images)
                except Exception:
                    has_images = False
                pages.append({'text': page_text, 'has_images': has_images})
            if pages:
                return pages
        except Exception:
            pass

    return pages


def extract_structured_from_pdf_bytes(pdf_bytes: bytes, sample_pages: int = 8) -> list:
    """논문/레이아웃 문서용 구조화 추출(프로토타입).

    반환값: [{'page': int, 'body_text': str, 'headers': [str], 'footers': [str],
             'page_number': int|None, 'footnotes': [str], 'has_images': bool}, ...]

    동작:
    - PyMuPDF(`fitz`) 기반으로 페이지의 spans/words를 수집
    - 문서 전반에 걸쳐 반복되는 상단/하단 텍스트를 헤더/푸터로 간주하여 본문에서 제거
    - 하단의 작은 폰트 텍스트를 푸트노트(footnote)로 수집
    - 본문은 `get_text('words')` 결과에서 헤더/푸터/푸트노트 영역을 제외한 단어들로 재조합

    주: 휴리스틱 기반 프로토타입이며 모든 저널 레이아웃에서 완벽하지 않습니다.
    """
    if not HAVE_PYMUPDF:
        # fallback: return simple structure using existing extractor
        pages = extract_pages_from_pdf_bytes(pdf_bytes)
        out = []
        for i, p in enumerate(pages):
            out.append({
                'page': i + 1,
                'body_text': p.get('text', ''),
                'headers': [],
                'footers': [],
                'page_number': None,
                'footnotes': [],
                'has_images': p.get('has_images', False),
            })
        return out

    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    except Exception:
        return []

    page_count = doc.page_count
    sample_n = min(sample_pages, page_count)

    # Collect repeated top/bottom spans as header/footer candidates
    header_candidates = {}
    footer_candidates = {}
    for i in range(sample_n):
        p = doc[i]
        ph = p.rect.height
        try:
            d = p.get_text('dict')
        except Exception:
            d = {}
        for block in d.get('blocks', []):
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    text = span.get('text', '').strip()
                    if not text:
                        continue
                    bbox = span.get('bbox', [])
                    y0 = bbox[1] if len(bbox) >= 4 else 0
                    if y0 < ph * 0.12:
                        header_candidates[text] = header_candidates.get(text, 0) + 1
                    if y0 > ph * 0.88:
                        footer_candidates[text] = footer_candidates.get(text, 0) + 1

    thr = max(1, int(0.6 * sample_n))
    headers_common = {t for t, c in header_candidates.items() if c >= thr}
    footers_common = {t for t, c in footer_candidates.items() if c >= thr}

    structured = []
    for i in range(page_count):
        p = doc[i]
        ph = p.rect.height

        # spans (for sizes and bbox) and words (for precise coordinates)
        try:
            d = p.get_text('dict')
        except Exception:
            d = {}
        spans = []
        sizes = []
        for block in d.get('blocks', []):
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    spans.append(span)
                    sz = span.get('size', None)
                    if sz:
                        sizes.append(sz)

        median_size = statistics.median(sizes) if sizes else None

        # detect footnote-like spans: smaller-than-median fonts located near bottom
        footnote_texts = []
        footnote_bbox_ys = []
        for span in spans:
            text = span.get('text', '').strip()
            if not text:
                continue
            bbox = span.get('bbox', [])
            if len(bbox) < 4:
                continue
            y0 = bbox[1]
            y1 = bbox[3]
            size = span.get('size', 0) or 0
            if median_size and size and size < median_size * 0.85 and y0 > ph * 0.75:
                footnote_texts.append(text)
                footnote_bbox_ys.append((y0, y1))

        # identify headers/footers on this page and page number
        headers = []
        footers = []
        page_number = None
        for span in spans:
            text = span.get('text', '').strip()
            if not text:
                continue
            bbox = span.get('bbox', [])
            if len(bbox) < 4:
                continue
            y0 = bbox[1]
            x0 = bbox[0]
            x1 = bbox[2]
            # header/footer matching
            if text in headers_common and y0 < ph * 0.12:
                headers.append(text)
            if text in footers_common and y0 > ph * 0.88:
                footers.append(text)
            # page number heuristic: small integer near bottom center
            if re.fullmatch(r'\d{1,4}', text):
                cx = (x0 + x1) / 2.0
                pw = p.rect.width
                if y0 > ph * 0.78 and abs(cx - pw / 2.0) < pw * 0.25:
                    try:
                        page_number = int(text)
                    except Exception:
                        page_number = None

        # build exclusion ranges (y ranges) for headers/footers/footnotes
        exclude_y_ranges = []
        for span in spans:
            text = span.get('text', '').strip()
            bbox = span.get('bbox', [])
            if len(bbox) < 4:
                continue
            y0 = bbox[1]
            y1 = bbox[3]
            if text in headers_common and y0 < ph * 0.12:
                exclude_y_ranges.append((0, y1 + 1))
            if text in footers_common and y0 > ph * 0.88:
                exclude_y_ranges.append((y0 - 1, ph))
        for y0, y1 in footnote_bbox_ys:
            exclude_y_ranges.append((max(0, y0 - 1), min(ph, y1 + 1)))

        def in_excluded(y: float) -> bool:
            for a, b in exclude_y_ranges:
                if y >= a and y <= b:
                    return True
            return False

        # filter words using excluded y ranges
        try:
            words = p.get_text('words')  # list of tuples
        except Exception:
            words = []

        filtered_words = []
        for w in words:
            # PyMuPDF word tuple: (x0, y0, x1, y1, 'word', block_no, line_no, word_no)
            if len(w) < 5:
                continue
            y0 = float(w[1])
            if in_excluded(y0):
                continue
            filtered_words.append(w)

        body_text = reflow_from_word_tuples(filtered_words)

        structured.append({
            'page': i + 1,
            'body_text': body_text,
            'headers': headers,
            'footers': footers,
            'page_number': page_number,
            'footnotes': footnote_texts,
            'has_images': bool(p.get_images(full=True)),
        })

    return structured


def normalize_extracted_text(text: str) -> str:
    """문단 단위로 처리하여 PDF의 소프트 줄바꿈(라인 래핑)을 제거하고 하이픈 분할도 복구합니다.
    - 연속된 빈 줄(\n\n 이상)은 문단 구분으로 유지
    - 문단 내부의 단일 줄바꿈은 공백으로 교체
    - 하이픈으로 이어진 단어(exam-\nple)는 이어 붙임
    - 문장부호 주변의 공백 정리
    """
    if not text:
        return ''

    # 문단 단위로 분리(빈줄 2개 이상을 문단 경계로 간주)
    paras = re.split(r'\n{2,}', text)
    cleaned = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        # 하이픈으로 분할된 단어 복구: 'exam-\nple' -> 'example'
        p = re.sub(r'-\n\s*', '', p)
        # 나머지 줄바꿈은 공백으로 치환하여 문장이 이어지도록 함
        p = re.sub(r'\n+', ' ', p)
        # 문장부호 앞의 불필요한 공백 제거
        p = re.sub(r'\s+([,.:;!?])', r'\1', p)
        # 문장부호 뒤에 공백이 없으면 추가
        p = re.sub(r'([,.:;!?])(\S)', r'\1 \2', p)
        # 여러 공백을 하나로
        p = re.sub(r'\s{2,}', ' ', p)
        cleaned.append(p.strip())

    return '\n\n'.join(cleaned)


# language tools: langdetect, spaCy, kss
try:
    from langdetect import detect as _lang_detect
    HAVE_LANGDETECT = True
except Exception:
    HAVE_LANGDETECT = False

try:
    import spacy
    HAVE_SPACY = True
    # try to load small English model; if not available, we'll use blank English and mark model missing
    try:
        _nlp = spacy.load('en_core_web_sm')
        HAVE_SPACY_MODEL = True
    except Exception:
        try:
            _nlp = spacy.blank('en')
            HAVE_SPACY_MODEL = False
        except Exception:
            _nlp = None
            HAVE_SPACY_MODEL = False
except Exception:
    HAVE_SPACY = False
    HAVE_SPACY_MODEL = False
    _nlp = None

try:
    import kss
    HAVE_KSS = True
except Exception:
    HAVE_KSS = False


def detect_language(text: str) -> str:
    """간단한 언어 감지: langdetect 사용 가능하면 사용, 아니면 영어로 간주."""
    if not text or len(text.strip()) < 20:
        return 'en'
    if HAVE_LANGDETECT:
        try:
            return _lang_detect(text)
        except Exception:
            return 'en'
    return 'en'


def reflow_and_segment(text: str) -> str:
    """리플로우 + 문장분할
    - 먼저 기존 normalize_extracted_text로 소프트 줄바꿈을 제거
    - 언어 감지 후 (ko/en) 각각 kss 또는 spaCy로 문장분할
    - 문장 단위로 재조합하여 문단 구조를 유지
    """
    if not text:
        return ''

    # 기본적인 정규화로 소프트 줄바꿈 제거
    cleaned = normalize_extracted_text(text)

    # 문단 단위로 분리
    paras = re.split(r'\n{2,}', cleaned)
    lang = detect_language(cleaned)

    out_paras = []
    for para in paras:
        p = para.strip()
        if not p:
            continue
        sentences = None
        # Korean
        if lang and lang.startswith('ko') and HAVE_KSS:
            try:
                sentences = kss.split_sentences(p)
            except Exception:
                sentences = None
        # English (spaCy)
        if sentences is None and HAVE_SPACY and _nlp is not None:
            try:
                doc = _nlp(p)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except Exception:
                sentences = None
        # Fallback: simple split on sentence enders
        if sentences is None:
            sentences = re.split(r'(?<=[.!?])\s+', p)

        # join sentences with single space to form paragraph
        para_out = ' '.join(s.strip() for s in sentences if s.strip())
        out_paras.append(para_out)

    return '\n\n'.join(out_paras)





def fix_urls_and_dois(text: str) -> str:
    """간단한 후처리: URL/DOI와 같은 링크 주변의 잘못된 공백을 정리합니다.
    예: 'https: //doi. org/10. 1080/03054985. 2022. 2087618' -> 'https://doi.org/10.1080/03054985.2022.2087618'
    이 함수는 문서의 일반 문장부호에는 영향을 적게 주도록 제한된 정규식을 사용합니다.
    중요: 단락 경계(\n)는 그대로 유지합니다.
    """
    if not text:
        return text

    t = text
    # 정규화: 프로토콜에서 생긴 공백 제거 (http(s)://)
    t = re.sub(r"https?[ \t]*:[ \t]*/[ \t]*/", "https://", t, flags=re.I)
    t = re.sub(r"http[ \t]*:[ \t]*/[ \t]*/", "http://", t, flags=re.I)
    # www. 주변 공백 제거 (공백에는 개행 제외)
    t = re.sub(r"www[ \t]*\.[ \t]*", "www.", t, flags=re.I)
    # doi.org 같은 도메인 주변 공백 제거
    t = re.sub(r"doi[ \t]*\.[ \t]*org", "doi.org", t, flags=re.I)
    # 슬래시 뒤의 공백 제거 (경로에서의 잘못된 공백) — 개행은 보존
    t = re.sub(r"/[ \t]+", "/", t)
    # 도메인 및 확장자 사이의 공백 제거 (e.g., 'doi. org' 또는 'example. com') — 개행은 보존
    t = re.sub(r"\.[ \t]+", ".", t)
    # 숫자.숫자 형태(예: DOI 접두부 10. 1080)를 합침 (개행 제외)
    t = re.sub(r"(\d)[ \t]*\.[ \t]*(\d)", r"\1.\2", t)
    # 여러 공백(개행 제외)을 하나로
    t = re.sub(r"[ \t]{2,}", " ", t)

    return t


# Optional translation support (googletrans fallback)
try:
    from googletrans import Translator as _GoogleTranslator
    HAVE_TRANSFORMERS = True
except Exception:
    HAVE_TRANSFORMERS = False

# expose flag for external UIs (Streamlit/frontends may read this flag)
# NOTE: Flask app was removed; templates are no longer used here.

# simple translator cache
_TRANSLATOR_CACHE = {}

def get_translator(src: str = 'en', tgt: str = 'ko'):
    key = f"{src}-{tgt}"
    if key in _TRANSLATOR_CACHE:
        return _TRANSLATOR_CACHE[key]
    if not HAVE_TRANSFORMERS:
        raise RuntimeError('googletrans 라이브러리가 설치되어 있지 않습니다. pip install googletrans==4.0.0-rc1')
    # Initialize Translator: prefer default constructor, fallback to service_urls if needed.
    try:
        g = _GoogleTranslator()
    except Exception:
        try:
            g = _GoogleTranslator(service_urls=['translate.googleapis.com'])
        except Exception as e:
            # If we cannot instantiate the translator, provide a no-op translator that returns original text
            def _noop(text, max_length=1024):
                return [{'translation_text': text}]

            _TRANSLATOR_CACHE[key] = _noop
            return _noop

    def _translate_fn(text, max_length=1024):
        """Translate `text` using googletrans and return a list-like result compatible with the previous pipeline output.
        Always returns a list like: [{'translation_text': 'translated string'}]. On internal errors, returns the original text.
        """
        try:
            # If caller requested automatic source detection, omit the `src` arg
            if src == 'auto' or src is None:
                res = g.translate(text, dest=tgt)
            else:
                res = g.translate(text, src=src, dest=tgt)
            # googletrans may return None or an object with .text; handle both
            if res is None:
                return [{'translation_text': text}]
            translated = getattr(res, 'text', None)
            if translated is None:
                # sometimes googletrans returns a dict-like object
                try:
                    # attempt to coerce to string
                    translated = str(res)
                except Exception:
                    translated = text
            return [{'translation_text': translated}]
        except Exception:
            # On any translation error (including JSON parsing issues inside googletrans),
            # return the original text so callers can continue safely.
            return [{'translation_text': text}]

    _TRANSLATOR_CACHE[key] = _translate_fn
    return _translate_fn


def safe_translate(translator_fn, text: str, max_chunk: int = 800) -> str:
    """Translate `text` in safe-sized chunks using `translator_fn`.
    Splits on sentence boundaries when possible to keep chunks under `max_chunk` characters.
    Returns the translated string (original text on failures).
    """
    if not text:
        return text

    # If short enough, translate in one call
    if len(text) <= max_chunk:
        try:
            out = translator_fn(text)
            if isinstance(out, list) and out:
                return out[0].get('translation_text', str(out[0]))
            if isinstance(out, dict):
                return out.get('translation_text', str(out))
            return str(out)
        except Exception:
            return text

    # Otherwise split into sentence-like pieces
    pieces = re.split(r'(?<=[\.!?])\s+', text)
    translated_pieces = []
    cur = ''
    for piece in pieces:
        if not piece:
            continue
        if len(cur) + len(piece) + 1 <= max_chunk:
            cur = (cur + ' ' + piece).strip()
            continue
        # translate current chunk
        try:
            out = translator_fn(cur)
            if isinstance(out, list) and out:
                translated_pieces.append(out[0].get('translation_text', str(out[0])))
            elif isinstance(out, dict):
                translated_pieces.append(out.get('translation_text', str(out)))
            else:
                translated_pieces.append(str(out))
        except Exception:
            translated_pieces.append(cur)
        cur = piece

    # last chunk
    if cur:
        try:
            out = translator_fn(cur)
            if isinstance(out, list) and out:
                translated_pieces.append(out[0].get('translation_text', str(out[0])))
            elif isinstance(out, dict):
                translated_pieces.append(out.get('translation_text', str(out)))
            else:
                translated_pieces.append(str(out))
        except Exception:
            translated_pieces.append(cur)

    return ' '.join(p.strip() for p in translated_pieces if p is not None)


# Translation and web server routes removed — use the Streamlit front-end.


if __name__ == '__main__':
    print('pdfExtract.py provides extraction helpers. Run app_streamlit.py to start the Streamlit UI.')
