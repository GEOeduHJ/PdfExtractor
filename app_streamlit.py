import streamlit as st
import re
import io


def secure_filename(filename: str) -> str:
    """Lightweight replacement for werkzeug.utils.secure_filename.
    Keeps ASCII alphanumerics, dots, dashes and underscores; replaces spaces with underscores.
    """
    if not filename:
        return ''
    # strip path
    fname = filename.split('/')[-1].split('\\')[-1]
    # replace spaces
    fname = fname.replace(' ', '_')
    # remove unsafe chars
    fname = re.sub(r'[^A-Za-z0-9._-]', '', fname)
    return fname

from pdfExtract import (
    extract_text_from_pdf_bytes,
    extract_pages_from_pdf_bytes,
    get_translator,
    HAVE_TRANSFORMERS,
    reflow_and_segment,
    fix_urls_and_dois,
    safe_translate,
)

st.set_page_config(page_title="PDF 텍스트 추출기 (Streamlit)", layout="centered")

st.title("PDF 텍스트 추출기")
st.markdown("PDF 파일을 업로드하면 내부 텍스트를 추출하고 다운로드하거나 번역할 수 있습니다.")

uploaded = st.file_uploader("PDF 파일 업로드", type=["pdf"])

def translate_text(src_text: str, target: str):
    try:
        translator = get_translator('auto', target)
    except Exception as e:
        st.error(f"번역기 로드 실패: {e}")
        return None

    # Split into paragraphs (keep existing paragraph boundaries)
    paras = re.split(r'\n{2,}', src_text)
    out = []
    for p in paras:
        if not p.strip():
            out.append('')
            continue

        # preserve page header markers
        if p.lstrip().startswith('=== 페이지'):
            parts = p.split('\n', 1)
            header = parts[0]
            body = parts[1] if len(parts) > 1 else ''
            if not body.strip():
                out.append(header)
                continue
            # Sentence-split by period and translate each sentence
            sents = [s for s in body.split('.')]
            trans_sents = []
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                chunk = s + '.'
                trans = safe_translate(translator, chunk)
                trans_sents.append(trans)
            body_trans = ' '.join(trans_sents).strip()
            out.append(header + '\n' + body_trans)
            continue

        # normal paragraph: split by period and translate sentence-by-sentence
        sents = [s for s in p.split('.')]
        trans_sents = []
        for s in sents:
            s = s.strip()
            if not s:
                continue
            chunk = s + '.'
            trans = safe_translate(translator, chunk)
            trans_sents.append(trans)
        para_trans = ' '.join(trans_sents).strip()
        out.append(para_trans)

    return '\n\n'.join(out)

if uploaded is not None:
    pdf_bytes = uploaded.read()
    st.info(f"업로드됨: {uploaded.name} ({len(pdf_bytes)} bytes)")

    with st.spinner('텍스트를 추출하는 중...'):
        pages = extract_pages_from_pdf_bytes(pdf_bytes)
        # fallback to raw extraction when page extraction yields nothing
        if not pages or all((p.get('text', '').strip() == '' for p in pages)):
            raw = extract_text_from_pdf_bytes(pdf_bytes)
            pages = [{'text': raw, 'has_images': False}]

        # normalize each page (reflow + sentence segmentation) and fix URLs/DOIs
        normalized_pages = [reflow_and_segment(p.get('text', '')) for p in pages]
        normalized_pages = [fix_urls_and_dois(p) for p in normalized_pages]

        # build full text with page headers and image markers (blank line above/below markers)
        merged_parts = []
        for i, p in enumerate(pages):
            page_text = normalized_pages[i] if i < len(normalized_pages) else ''
            merged_parts.append(f"=== 페이지 {i+1} ===")
            if page_text:
                merged_parts.append(page_text)
            if p.get('has_images'):
                merged_parts.append('\n[이미지 또는 표가 포함되어 있습니다]\n')
        full_text = '\n\n'.join(merged_parts)

    st.subheader('추출된 텍스트 (전체)')
    st.text_area('전체 텍스트', value=full_text, height=300)

    # 다운로드
    txt_bytes = full_text.encode('utf-8')
    default_name = secure_filename((uploaded.name or 'extracted').rsplit('.', 1)[0]) + '.txt'
    st.download_button('텍스트 파일로 다운로드', data=txt_bytes, file_name=default_name, mime='text/plain')

    # per-page view removed — merged full text shown above

    # Translation UI
    st.subheader('번역')
    if HAVE_TRANSFORMERS:
        target = st.selectbox('대상 언어', options=['ko', 'ja', 'zh'], index=0, format_func=lambda x: {'ko':'한국어 (ko)','ja':'일본어 (ja)','zh':'중국어 (zh)'}[x])
        if st.button('번역 후 다운로드'):
            with st.spinner('번역 중...'):
                translated = translate_text(full_text, target)
            if translated is not None:
                st.success('번역 완료')
                st.text_area('번역 결과', value=translated, height=300)
                tname = secure_filename((uploaded.name or 'translated').rsplit('.', 1)[0]) + f'.{target}.txt'
                st.download_button('번역된 텍스트 다운로드', data=translated.encode('utf-8'), file_name=tname, mime='text/plain')
    else:
        st.info('번역 기능을 사용하려면 서버에 googletrans 패키지를 설치하세요 (권장: googletrans==4.0.0-rc1).')
