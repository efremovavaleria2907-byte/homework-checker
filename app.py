import io
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
from pytesseract import Output

st.set_page_config(page_title="Проверяйка", page_icon="✓", layout="centered")

@dataclass
class Task:
    number: int
    page: int
    top: int
    bottom: int
    image: Image.Image
    text: str


def normalize_answer(value: str) -> str:
    value = (value or "").strip().lower().replace("ё", "е")
    value = re.sub(r"[\u00a0\t\r\n]+", " ", value)
    value = re.sub(r"[\s,;.:]+", "", value)
    value = value.replace("—", "-").replace("–", "-")
    return value


def ocr_page(page, scale=2.0):
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    data = pytesseract.image_to_data(img, lang="rus+eng", config="--psm 6", output_type=Output.DICT)
    return img, data


def grouped_lines(data):
    groups = {}
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        groups.setdefault(key, []).append(i)
    lines = []
    for inds in groups.values():
        inds.sort(key=lambda i: data["left"][i])
        text = " ".join(data["text"][i].strip() for i in inds if data["text"][i].strip())
        top = min(data["top"][i] for i in inds)
        bottom = max(data["top"][i] + data["height"][i] for i in inds)
        lines.append((top, bottom, text))
    return sorted(lines, key=lambda x: x[0])


def find_task_headings(lines):
    found = []
    pattern = re.compile(r"задани[ея]\s*№?\s*(\d{1,3})", re.I)
    for top, bottom, text in lines:
        m = pattern.search(text.replace("#", "№"))
        if m:
            found.append((int(m.group(1)), top, bottom))
    # Avoid duplicate OCR detections of the same heading.
    result = []
    for item in found:
        if result and item[0] == result[-1][0] and abs(item[1] - result[-1][1]) < 80:
            continue
        result.append(item)
    return result


def split_by_dotted_lines(img: Image.Image, min_y=100):
    """Fallback for pages where OCR misses 'Задание №...'.
    Looks for rows containing many small dark components spread horizontally.
    """
    arr = np.asarray(img.convert("L"))
    dark = arr < 205
    counts = dark.sum(axis=1)
    width = arr.shape[1]
    candidates = []
    for y in range(min_y, len(counts) - 1):
        # Dotted separators tend to be long but much sparser than text rows.
        ratio = counts[y] / max(width, 1)
        if 0.03 < ratio < 0.22:
            candidates.append(y)
    clusters = []
    for y in candidates:
        if not clusters or y - clusters[-1][-1] > 3:
            clusters.append([y])
        else:
            clusters[-1].append(y)
    return [int(sum(c) / len(c)) for c in clusters if len(c) >= 1]


def extract_tasks(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    tasks = []
    for pno, page in enumerate(doc):
        img, data = ocr_page(page)
        lines = grouped_lines(data)
        heads = find_task_headings(lines)
        if heads:
            for idx, (num, top, _) in enumerate(heads):
                bottom = heads[idx + 1][1] if idx + 1 < len(heads) else img.height
                # Keep the next task heading out of the previous card.
                crop_top = max(0, top - 20)
                crop_bottom = min(img.height, bottom - 10)
                crop = img.crop((0, crop_top, img.width, crop_bottom))
                text = "\n".join(t for t, _, _ in [] )
                # Build text from OCR lines in this vertical region.
                selected = [line[2] for line in lines if line[0] >= top and line[0] < bottom]
                tasks.append(Task(num, pno + 1, crop_top, crop_bottom, crop, "\n".join(selected)))
        else:
            # OCR fallback: split by dotted horizontal separators. This keeps the app
            # usable for scanned PDFs where the heading itself is hard to recognize.
            separators = split_by_dotted_lines(img)
            bounds = [0] + separators + [img.height]
            for a, b in zip(bounds, bounds[1:]):
                if b - a < 120:
                    continue
                crop = img.crop((0, a, img.width, b))
                txt = pytesseract.image_to_string(crop, lang="rus+eng", config="--psm 6")
                m = re.search(r"задани[ея]\s*№?\s*(\d{1,3})", txt, re.I)
                if m:
                    tasks.append(Task(int(m.group(1)), pno + 1, a, b, crop, txt))
    # Keep first occurrence of each number, then sort naturally.
    unique = {}
    for t in tasks:
        unique.setdefault(t.number, t)
    return [unique[k] for k in sorted(unique)]


def extract_answers(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    answers = {}
    for page in doc:
        img, data = ocr_page(page)
        lines = grouped_lines(data)
        heads = find_task_headings(lines)
        if not heads:
            continue
        for idx, (num, top, _) in enumerate(heads):
            bottom = heads[idx + 1][1] if idx + 1 < len(heads) else img.height
            block = "\n".join(t for a, b, t in lines if a >= top and a < bottom)
            ans = answer_from_block(block)
            if ans is not None:
                answers[num] = ans
    return answers


def answer_from_block(block: str) -> Optional[str]:
    clean_lines = [re.sub(r"\s+", " ", x).strip() for x in block.splitlines() if x.strip()]
    joined = " ".join(clean_lines)
    # Most answer PDFs use one of these labels.
    patterns = [
        r"(?:правильный ответ|верные варианты)\s*[:\-]?\s*([0-9][0-9\s,.;-]{0,24})",
        r"ответ\s*[:\-]?\s*([0-9][0-9\s,.;-]{0,24})",
    ]
    for pat in patterns:
        m = re.search(pat, joined, re.I)
        if m:
            raw = m.group(1)
            # Stop at a sentence boundary if OCR swallowed the next sentence.
            raw = re.split(r"(?<=[.;])\s+[А-ЯA-ZЁ]", raw)[0]
            raw = re.sub(r"[^0-9\s,;.-]", "", raw).strip(" .;,-")
            if raw:
                return raw
    # Fallback: look at the first short numeric-only line after an answer label.
    label_seen = False
    for line in clean_lines:
        if re.search(r"(?:правильный ответ|верные варианты|ответ)\b", line, re.I):
            label_seen = True
            continue
        if label_seen and re.fullmatch(r"[0-9][0-9\s,;.-]{0,24}", line):
            return line.strip(" .;,-")
    return None


def check(user: str, correct: str) -> bool:
    return normalize_answer(user) == normalize_answer(correct)

st.title("Проверяйка")
st.write("Загрузи PDF с заданиями и PDF с правильными ответами.")
st.info("Приложение рассчитано и на сканы: задания ищутся по надписи «Задание №…», а при необходимости — по разделительным пунктирным линиям.")

homework = st.file_uploader("PDF с домашним заданием", type=["pdf"], key="hw")
answers_file = st.file_uploader("PDF с ответами", type=["pdf"], key="ans")

if homework and answers_file:
    if st.button("Найти задания", type="primary"):
        with st.spinner("Распознаю страницы и разделяю задания…"):
            try:
                tasks = extract_tasks(homework.getvalue())
                answers = extract_answers(answers_file.getvalue())
                st.session_state["tasks"] = tasks
                st.session_state["answers"] = answers
            except Exception as e:
                st.error("Не удалось обработать PDF. Проверь, что это обычный PDF со страницами/сканами.")
                st.exception(e)

if "tasks" in st.session_state:
    tasks = st.session_state["tasks"]
    answers = st.session_state.get("answers", {})
    if not tasks:
        st.error("Я не нашёл ни одного задания. Для этого PDF попробуй повысить качество скана или пришли его — можно будет добавить отдельное правило распознавания.")
    else:
        st.success(f"Найдено заданий: {len(tasks)}")
        if not answers:
            st.warning("Задания найдены, но правильные ответы не распознаны. Проверь PDF с ответами.")
        for task in tasks:
            st.markdown(f"### Задание №{task.number}")
            st.image(task.image, use_container_width=True)
            correct = answers.get(task.number)
            if correct is None:
                st.warning("Правильный ответ для этого задания в PDF с ответами не найден.")
                st.text_input("Твой ответ", key=f"ans_{task.number}")
                continue
            user_answer = st.text_input("Твой ответ", key=f"ans_{task.number}", placeholder="Введи ответ")
            if st.button("Проверить", key=f"check_{task.number}"):
                if not user_answer.strip():
                    st.warning("Сначала введи ответ.")
                elif check(user_answer, correct):
                    st.success("✓ Правильно!")
                else:
                    st.error("✗ Неправильно.")
            with st.expander("Техническая информация"):
                st.caption(f"Распознанный эталон: {correct}")
