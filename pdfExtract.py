#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
간단한 Flask 웹앱: PDF 파일을 업로드하면 내부 텍스트를 추출합니다.
- 텍스트 추출: pdfplumber 사용 (없으면 PyPDF2로 폴백)
- HTML 폼에서 업로드 후 추출된 텍스트를 화면에 표시하고 .txt로 다운로드 가능
- API 엔드포인트: POST /api/extract (multipart/form-data, key: file)

주의: 이미지 기반(스캔) PDF는 OCR(예: Tesseract)이 필요합니다.
"""

from flask import Flask, request, render_template, jsonify, send_file
import io
import base64
import os
import re
from werkzeug.utils import secure_filename
import statistics
import uuid
import time

# 라이브러리 로드 (있으면 사용)
try:
    import pdfplumber
    HAVE_PDFPLUMBER = True
except Exception:
    HAVE_PDFPLUMBER = False

try:
    from PyPDF2 import PdfReader
    HAVE_PYPDF2 = True
except Exception:
    HAVE_PYPDF2 = False

# python-docx 지원 여부
try:
    from docx import Document
    HAVE_DOCX = True
except Exception:
    HAVE_DOCX = False

# ASGI 래퍼 사용 가능 여부 (uvicorn으로 실행할 때 필요)
try:
    from asgiref.wsgi import WsgiToAsgi
    HAVE_ASGIREF = True
except Exception:
    HAVE_ASGIREF = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 최대 업로드 50MB
ALLOWED_EXTENSIONS = {'pdf'}

# ASGI 앱 (asgiref가 있으면 uvicorn으로 실행 가능)
if HAVE_ASGIREF:
    asgi_app = WsgiToAsgi(app)
else:
    asgi_app = None


def allowed_file(filename: str) -> bool:
    return bool(filename and '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """PDF 바이너리에서 텍스트를 추출해 문자열로 반환합니다.
    pdfplumber 우선, 실패하면 PyPDF2로 폴백합니다. 텍스트가 없으면 빈 문자열을 반환합니다.
    """
    text_parts = []

    # pdfplumber 사용 시도
    if HAVE_PDFPLUMBER:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            if text_parts:
                return '\n\n'.join(text_parts)
        except Exception:
            # pdfplumber 처리 중 오류 발생하면 폴백 시도
            pass

    # PyPDF2 폴백
    if HAVE_PYPDF2:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                except Exception:
                    page_text = None
                if page_text:
                    text_parts.append(page_text)
            if text_parts:
                return '\n\n'.join(text_parts)
        except Exception:
            pass

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

    # 2) pdfplumber 폴백
    if HAVE_PDFPLUMBER:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    try:
                        # pdfplumber에서 words를 얻어서 reflow 시도
                        try:
                            words = page.extract_words()
                        except Exception:
                            words = None

                        if words:
                            try:
                                text = reflow_from_word_tuples(words)
                            except Exception:
                                text = page.extract_text() or ''
                        else:
                            text = page.extract_text() or ''
                    except Exception:
                        text = ''
                    try:
                        has_images = bool(getattr(page, 'images', None))
                    except Exception:
                        has_images = False
                    pages.append({'text': text, 'has_images': has_images})
            if pages:
                return pages
        except Exception:
            pass

    # 3) PyPDF2 폴백 (이미지 검출은 복잡하므로 False 처리)
    if HAVE_PYPDF2:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(reader.pages)
            for i in range(num_pages):
                try:
                    page = reader.pages[i]
                    text = page.extract_text() or ''
                except Exception:
                    text = ''
                pages.append({'text': text, 'has_images': False})
            if pages:
                return pages
        except Exception:
            pass

    return pages


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


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/extract', methods=['POST'])
def extract():
    # 파일 존재 확인
    if 'file' not in request.files:
        return render_template('index.html', error='파일이 업로드되지 않았습니다.')

    file = request.files['file']
    raw_filename = file.filename or ''

    if raw_filename == '':
        return render_template('index.html', error='파일이 선택되지 않았습니다.')

    if not allowed_file(raw_filename):
        return render_template('index.html', error='허용되는 파일 형식은 PDF(.pdf)만 가능합니다.')

    filename = secure_filename(raw_filename)

    try:
        pdf_bytes = file.read()
        # 페이지별로 추출 (각 페이지는 dict: {'text', 'has_images'})
        pages = extract_pages_from_pdf_bytes(pdf_bytes)

        # OCR 폴백: 텍스트가 전혀 없으면 OCR 시도
        if not pages or all((p.get('text','').strip() == '' for p in pages)):
            ocr_pages = ocr_pdf_bytes(pdf_bytes)
            if ocr_pages:
                pages = ocr_pages

        # 페이지 리스트에서 텍스트만 추출하여 리플로우+문장분할 적용
        normalized_pages = [reflow_and_segment(p.get('text', '')) for p in pages]

        # apply URL/DOI fixes to each page
        normalized_pages = [fix_urls_and_dois(p) for p in normalized_pages]

        # 전체 합친 텍스트 (다운로드용) -- 페이지 구분 및 이미지 마커 포함
        merged_parts = []
        for i, p in enumerate(pages):
            page_text = normalized_pages[i] if i < len(normalized_pages) else ''
            merged_parts.append(f"=== 페이지 {i+1} ===")
            if page_text:
                merged_parts.append(page_text)
            if p.get('has_images'):
                # 이미지/표 마커와 위아래 공백 한 줄 추가
                merged_parts.append('\n[이미지 또는 표가 포함되어 있습니다]\n')
        normalized_text = '\n\n'.join(merged_parts)

        # apply URL/DOI fixes to full text as well
        normalized_text = fix_urls_and_dois(normalized_text)

        if not any([tp for tp in normalized_pages if tp.strip() != '']):
            msg = 'PDF에서 텍스트를 추출하지 못했습니다. 이 PDF가 이미지 기반(스캔)인 경우 OCR이 필요합니다.'
            return render_template('index.html', error=msg)

        # 웹에서 사용하기 위해 base64로 인코딩 (다운로드 링크로 사용)
        b64 = base64.b64encode(normalized_text.encode('utf-8')).decode('ascii')
        download_filename = os.path.splitext(filename)[0] + '.txt'
        download_docx_filename = os.path.splitext(filename)[0] + '.docx'

        # 템플릿에는 페이지별 텍스트(정규화된)와 각 페이지의 이미지 여부를 전달
        extracted_pages_text = normalized_pages
        extracted_pages_meta = [p.get('has_images', False) for p in pages]

        # Create in-memory outputs for download (only full TXT). Remove DOCX and per-page artifacts per request.
        download_id = uuid.uuid4().hex
        store_entry = {'created': time.time(), 'filename_base': os.path.splitext(filename)[0]}
        # full text bytes (only this will be stored)
        txt_bytes = normalized_text.encode('utf-8')
        store_entry['txt'] = txt_bytes

        # Do not generate DOCX or per-page files anymore
        store_entry['docx'] = None
        store_entry['pages'] = None

        DOWNLOAD_STORE[download_id] = store_entry

        return render_template('index.html', extracted_text=normalized_text, extracted_pages=extracted_pages_text, extracted_pages_has_images=extracted_pages_meta, download_id=download_id, download_filename=download_filename)

    except Exception as e:
        return render_template('index.html', error=f'텍스트 추출 중 오류가 발생했습니다: {e}')


@app.route('/api/extract', methods=['POST'])
def api_extract():
    """API 엔드포인트: multipart/form-data로 PDF 파일을 전송하면 JSON으로 추출 결과를 반환합니다."""
    if 'file' not in request.files:
        return jsonify(success=False, error='file not provided'), 400

    file = request.files['file']
    raw_filename = file.filename or ''

    if raw_filename == '':
        return jsonify(success=False, error='no filename'), 400

    if not allowed_file(raw_filename):
        return jsonify(success=False, error='file extension not allowed'), 400

    try:
        pdf_bytes = file.read()
        pages = extract_pages_from_pdf_bytes(pdf_bytes)
        # OCR 폴백: 텍스트가 전혀 없으면 OCR 시도
        if not pages or all((p.get('text','').strip() == '' for p in pages)):
            ocr_pages = ocr_pdf_bytes(pdf_bytes)
            if ocr_pages:
                pages = ocr_pages

        normalized_pages = [reflow_and_segment(p.get('text','')) for p in pages]
        # include has_images in response
        pages_info = [{'text': normalized_pages[i] if i < len(normalized_pages) else '', 'has_images': pages[i].get('has_images', False)} for i in range(len(pages))]
        normalized_text = '\n\n'.join([f"=== 페이지 {i+1} ===\n{pages_info[i]['text']}" + ("\n\n[이미지 또는 표가 포함되어 있습니다]" if pages_info[i]['has_images'] else '') for i in range(len(pages_info))])

        return jsonify(success=True, pages=pages_info, text=normalized_text), 200
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@app.route('/download_docx', methods=['POST'])
def download_docx():
    """POST 폼으로 전달된 텍스트를 docx로 만들어 다운로드합니다.
    폼 인자:
      - text: 전체 텍스트(또는 단일 페이지 텍스트)
      - filename: 출력 파일명
      - has_images (선택): '1' 또는 '0' (단일 페이지 다운로드용)
    """
    if not HAVE_DOCX:
        return render_template('index.html', error='python-docx 라이브러리가 설치되어 있지 않습니다. requirements.txt에 python-docx를 추가하고 설치하세요.')

    text = request.form.get('text', '')
    filename = request.form.get('filename') or 'extracted.docx'
    filename = secure_filename(filename)
    has_images_flag = request.form.get('has_images')

    if not text:
        return render_template('index.html', error='DOCX로 변환할 텍스트가 없습니다.')

    doc = Document()

    # 문단 단위로 분리하여 docx에 추가
    # 페이지 헤더(=== 페이지 N ===)는 굵게 표시
    for para in re.split(r'\n{2,}', text):
        p = para.strip()
        if not p:
            continue
        # 내부의 단일/다중 줄바꿈은 공백으로 바꿔서 문단 내 줄바꿈이 실제 단락으로 들어가지 않게 함
        p = re.sub(r'\n+', ' ', p)
        # 페이지 헤더 감지
        m = re.match(r'^===\s*페이지\s*(\d+)\s*===', p)
        if m:
            hdr = doc.add_paragraph()
            run = hdr.add_run(p)
            run.bold = True
            continue
        if '[이미지 또는 표가 포함되어 있습니다]' in p:
            # 위아래 한 줄 공백 추가 대신 단일 이탤릭 문단으로 추가
            img_p = doc.add_paragraph()
            img_run = img_p.add_run('[이미지 또는 표가 포함되어 있습니다]')
            img_run.italic = True
            continue
        # 일반 문단
        doc.add_paragraph(p)

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)

    return send_file(
        bio,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )


@app.errorhandler(413)
def request_entity_too_large(e):
    return render_template('index.html', error='파일 크기가 너무 큽니다 (최대 50MB).'), 413


# OCR support (pytesseract + pdf2image)
try:
    import pytesseract
    HAVE_PYTESSERACT = True
except Exception:
    HAVE_PYTESSERACT = False

try:
    from pdf2image import convert_from_bytes
    HAVE_PDF2IMAGE = True
except Exception:
    HAVE_PDF2IMAGE = False

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


def ocr_pdf_bytes(pdf_bytes: bytes) -> list:
    """PDF 바이트를 이미지로 변환한 뒤 pytesseract로 페이지별 텍스트를 추출합니다.
    반환: [{'text': str, 'has_images': True}, ...]
    주의: 시스템에 Tesseract 바이너리 설치가 필요합니다.
    """
    pages = []
    if not (HAVE_PYTESSERACT and HAVE_PDF2IMAGE and HAVE_PIL):
        return pages

    try:
        images = convert_from_bytes(pdf_bytes)
        for img in images:
            try:
                text = pytesseract.image_to_string(img)
            except Exception:
                text = ''
            pages.append({'text': text or '', 'has_images': True})
    except Exception:
        return []

    return pages


# In-memory store for generated download artifacts (bytes). Keys are UUID hex strings.
# Note: this is process-local and ephemeral. For production use a persistent store (Redis, S3, filesystem).
DOWNLOAD_STORE = {}
# Optional expiry (seconds) for entries — not enforced automatically here but available for future cleanup logic.
DOWNLOAD_EXPIRY_SECONDS = 3600


@app.route('/download_file/<download_id>', methods=['GET'])
def download_file(download_id):
    """Serve stored generated TXT output by id. Other formats removed per request."""
    ent = DOWNLOAD_STORE.get(download_id)
    if not ent:
        return render_template('index.html', error='요청하신 파일을 찾을 수 없습니다 (만료되었거나 잘못된 ID).'), 404

    # Only txt is supported now
    bio = io.BytesIO(ent.get('txt', b''))
    bio.seek(0)
    return send_file(bio, as_attachment=True, download_name=ent.get('filename_base', 'extracted') + '.txt', mimetype='text/plain; charset=utf-8')


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

# expose flag to templates
try:
    app.jinja_env.globals['have_transformers'] = HAVE_TRANSFORMERS
except Exception:
    pass

# simple translator cache
_TRANSLATOR_CACHE = {}

def get_translator(src: str = 'en', tgt: str = 'ko'):
    key = f"{src}-{tgt}"
    if key in _TRANSLATOR_CACHE:
        return _TRANSLATOR_CACHE[key]
    if not HAVE_TRANSFORMERS:
        raise RuntimeError('googletrans 라이브러리가 설치되어 있지 않습니다. pip install googletrans==4.0.0-rc1')
    g = _GoogleTranslator()

    def _translate_fn(text, max_length=1024):
        """Translate `text` using googletrans and return a list-like result compatible with the previous pipeline output.
        Returns: [{'translation_text': 'translated string'}]
        """
        try:
            res = g.translate(text, src=src, dest=tgt)
            translated = getattr(res, 'text', str(res))
            return [{'translation_text': translated}]
        except Exception as e:
            # re-raise so caller can show an error and fallback behavior remains unchanged
            raise

    _TRANSLATOR_CACHE[key] = _translate_fn
    return _translate_fn


@app.route('/translate', methods=['POST'])
def translate():
    """Translate stored extracted TXT to target language and return as download.
    Expects form fields: download_id, target (e.g. 'ko').
    """
    if not HAVE_TRANSFORMERS:
        return render_template('index.html', error='번역 기능을 사용하려면 googletrans를 설치하세요: pip install "googletrans==4.0.0-rc1"')

    download_id = request.form.get('download_id') or request.form.get('id')
    target = request.form.get('target', 'ko')
    if not download_id:
        return render_template('index.html', error='download_id가 제공되지 않았습니다.')

    ent = DOWNLOAD_STORE.get(download_id)
    if not ent:
        return render_template('index.html', error='요청하신 파일을 찾을 수 없습니다 (만료되었거나 잘못된 ID).'), 404

    txt_bytes = ent.get('txt', b'')
    try:
        src_text = txt_bytes.decode('utf-8')
    except Exception:
        src_text = txt_bytes.decode('utf-8', errors='replace')

    # split by paragraph to avoid tokenizer length issues
    paras = re.split(r'\n{2,}', src_text)
    try:
        translator = get_translator('en', target)
    except Exception as e:
        return render_template('index.html', error=f'번역기 로드 실패: {e}')

    translated_paras = []
    for p in paras:
        if not p.strip():
            translated_paras.append('')
            continue
        try:
            out = translator(p, max_length=1024)
            if isinstance(out, list) and out:
                translated_paras.append(out[0].get('translation_text', str(out[0])))
            elif isinstance(out, dict):
                translated_paras.append(out.get('translation_text', str(out)))
            else:
                translated_paras.append(str(out))
        except Exception:
            # on failure, append original paragraph
            translated_paras.append(p)

    translated_text = '\n\n'.join(translated_paras)
    bio = io.BytesIO(translated_text.encode('utf-8'))
    bio.seek(0)
    out_name = ent.get('filename_base', 'extracted') + f'.{target}.txt'
    return send_file(bio, as_attachment=True, download_name=out_name, mimetype='text/plain; charset=utf-8')


if __name__ == '__main__':
    # 개발용 간편 실행: 환경변수 USE_UVICORN=1 설정 시 uvicorn으로 실행
    use_uv = os.environ.get('USE_UVICORN') == '1'
    if use_uv:
        try:
            import uvicorn
            if asgi_app is None:
                print('asgiref 패키지가 없어 uvicorn을 사용할 수 없습니다. Flask 내장 서버로 실행합니다.')
                app.run(host='0.0.0.0', port=5000, debug=True)
            else:
                uvicorn.run(asgi_app, host='0.0.0.0', port=int(os.environ.get('PORT', '8000')), log_level='info')
        except Exception as e:
            print(f'uvicorn 실행 실패 ({e}). Flask 내장 서버로 실행합니다.')
            app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)
