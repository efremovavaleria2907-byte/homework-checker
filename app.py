import re
from dataclasses import dataclass
from typing import Optional

import fitz
import numpy as np
import streamlit as st
from PIL import Image
import pytesseract


st.set_page_config(
    page_title="Проверяйка",
    page_icon="✓",
    layout="centered"
)


@dataclass
class Task:
    number: int
    page: int
    image: Image.Image
    text: str


def normalize_answer(value: str) -> str:
    value = (value or "").strip().lower().replace("ё", "е")
    value = re.sub(r"[\u00a0\t\r\n]+", " ", value)
    value = re.sub(r"[\s,;.:]+", "", value)
    value = value.replace("—", "-").replace("–", "-")
    return value


def render_page(page, scale=1.5):
    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False
    )

    return Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples
    )


def page_text(page) -> str:
    try:
        return page.get_text("text") or ""
    except Exception:
        return ""


def find_task_number(text: str) -> Optional[int]:
    if not text:
        return None

    text = text.replace("№", " № ")

    patterns = [
        r"задани[ея]\s*(?:№\s*)?(\d{1,3})\b",
        r"задани[ея]\s+(\d{1,3})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.I)

        if match:
            return int(match.group(1))

    return None


def merge_close(values, distance=6):
    result = []

    for value in sorted(values):
        if not result:
            result.append(value)
        elif value - result[-1] > distance:
            result.append(value)
        else:
            result[-1] = (result[-1] + value) / 2

    return result


def horizontal_vector_lines(page):
    """
    Ищет длинные горизонтальные линии,
    которые являются настоящими объектами PDF.
    """

    ys = []

    try:
        drawings = page.get_drawings()
    except Exception:
        return ys

    page_width = page.rect.width

    for drawing in drawings:
        for item in drawing.get("items", []):

            if not item:
                continue

            if item[0] != "l":
                continue

            p1, p2 = item[1], item[2]

            dx = abs(p2.x - p1.x)
            dy = abs(p2.y - p1.y)

            if dy <= 2.5 and dx >= page_width * 0.35:
                ys.append((p1.y + p2.y) / 2)

    return sorted(ys)


def raster_separator_lines(img: Image.Image):
    """
    Ищет горизонтальные разделители на сканированной странице.
    """

    gray = np.asarray(img.convert("L"))

    dark = gray < 205

    counts = dark.sum(axis=1)

    width = gray.shape[1]

    candidates = []

    for y, count in enumerate(counts):

        ratio = count / max(width, 1)

        if 0.18 <= ratio <= 0.98:
            candidates.append(y)

    clusters = []

    for y in candidates:

        if not clusters:
            clusters.append([y])

        elif y - clusters[-1][-1] > 2:
            clusters.append([y])

        else:
            clusters[-1].append(y)

    lines = []

    for cluster in clusters:

        y = int(sum(cluster) / len(cluster))

        left = max(0, y - 2)
        right = min(len(counts), y + 3)

        peak = max(counts[left:right]) / max(width, 1)

        if peak >= 0.18:
            lines.append(y)

    return merge_close(lines, distance=8)


def separator_lines(page, img):
    """
    Объединяет линии из PDF и линии, найденные на изображении.
    """

    vector = horizontal_vector_lines(page)

    scale = img.width / page.rect.width

    vector_px = [
        int(y * scale)
        for y in vector
    ]

    raster = raster_separator_lines(img)

    combined = merge_close(
        vector_px + raster,
        distance=10
    )

    return combined


def ocr_text(img):
    try:
        return pytesseract.image_to_string(
            img,
            lang="rus+eng",
            config="--psm 6"
        )
    except Exception:
        return ""


def crop_text_from_pdf(page, top, bottom):
    """
    Быстрый способ получить текст,
    если PDF уже содержит текстовый слой.
    """

    try:
        blocks = page.get_text("blocks")
    except Exception:
        return ""

    parts = []

    for block in blocks:

        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]

        if y1 >= top and y0 < bottom and text.strip():
            parts.append(
                (y0, text.strip())
            )

    parts.sort(key=lambda x: x[0])

    return "\n".join(
        text
        for _, text in parts
    )


def split_page_into_blocks(page, img):
    """
    Делит страницу на блоки по горизонтальным линиям.

    Надпись "Задание №..." для разделения
    больше не обязательна.
    """

    lines = separator_lines(page, img)

    margin = max(
        12,
        int(img.height * 0.015)
    )

    lines = [
        y
        for y in lines
        if margin < y < img.height - margin
    ]

    # Если нашли слишком много линий,
    # оставляем только настоящие векторные линии.

    if len(lines) > 25:

        vector = horizontal_vector_lines(page)

        scale = img.width / page.rect.width

        lines = [
            int(y * scale)
            for y in merge_close(vector, 6)
        ]

        lines = [
            y
            for y in lines
            if margin < y < img.height - margin
        ]

    boundaries = [0] + lines + [img.height]

    blocks = []

    for a, b in zip(
        boundaries,
        boundaries[1:]
    ):

        if b - a < max(
            120,
            int(img.height * 0.06)
        ):
            continue

        padding = 8

        top_px = max(
            0,
            a + padding
        )

        bottom_px = min(
            img.height,
            b - padding
        )

        crop = img.crop(
            (
                0,
                top_px,
                img.width,
                bottom_px
            )
        )

        scale = img.width / page.rect.width

        pdf_top = top_px / scale
        pdf_bottom = bottom_px / scale

        text = crop_text_from_pdf(
            page,
            pdf_top,
            pdf_bottom
        )

        # OCR используется только если
        # текстового слоя нет.

        if not text.strip():
            text = ocr_text(crop)

        blocks.append(
            (
                crop,
                text.strip()
            )
        )

    return blocks


def extract_tasks(pdf_bytes: bytes):

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    tasks = []

    next_number = 1

    for page_number, page in enumerate(doc):

        # Страница рендерится один раз.

        img = render_page(
            page,
            scale=1.5
        )

        blocks = split_page_into_blocks(
            page,
            img
        )

        if not blocks:

            text = page_text(page)

            if not text.strip():
                text = ocr_text(img)

            blocks = [
                (
                    img,
                    text.strip()
                )
            ]

        for crop, text in blocks:

            number = find_task_number(text)

            if number is None:
                number = next_number

            tasks.append(
                Task(
                    number=number,
                    page=page_number + 1,
                    image=crop,
                    text=text
                )
            )

            next_number = max(
                next_number,
                number + 1
            )

    # Убираем случайные дубли номеров.

    result = []

    used_numbers = set()

    for task in tasks:

        if task.number in used_numbers:
            continue

        used_numbers.add(task.number)

        result.append(task)

    # Если OCR дал одинаковые номера,
    # физический порядок блоков важнее OCR.

    if len(result) != len(tasks):

        result = [
            Task(
                number=index + 1,
                page=task.page,
                image=task.image,
                text=task.text
            )
            for index, task in enumerate(tasks)
        ]

    return result


def clean_answer(value: str) -> Optional[str]:

    if not value:
        return None

    value = value.strip()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    value = re.sub(
        r"^(?:ответ|правильный ответ|верный ответ|верные варианты)\s*[:\-]?\s*",
        "",
        value,
        flags=re.I
    )

    value = value.strip(
        " .,:;-"
    )

    if not value:
        return None

    return value


def answer_from_text(text: str):

    if not text:
        return None

    lines = [
        re.sub(
            r"\s+",
            " ",
            line
        ).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    joined = " ".join(lines)

    patterns = [

        r"(?:правильный\s+ответ|верный\s+ответ|верные\s+варианты|ответ)\s*[:\-]?\s*([0-9][0-9\s,.;-]{0,30})",

        r"(?:правильный\s+ответ|верный\s+ответ|верные\s+варианты|ответ)\s*[:\-]?\s*([А-ЯA-Z][А-ЯA-Z0-9\s,.;-]{0,20})",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            joined,
            re.I
        )

        if match:

            answer = clean_answer(
                match.group(1)
            )

            if answer:
                return answer

    for index, line in enumerate(lines):

        if re.search(
            r"(?:правильный\s+ответ|верные\s+варианты|ответ)\b",
            line,
            re.I
        ):

            rest = re.sub(
                r".*?(?:правильный\s+ответ|верные\s+варианты|ответ)\s*[:\-]?\s*",
                "",
                line,
                flags=re.I
            ).strip()

            if rest:

                answer = clean_answer(rest)

                if answer:
                    return answer

            for following in lines[
                index + 1:index + 3
            ]:

                if re.fullmatch(
                    r"[0-9А-ЯA-Zа-яa-z][0-9\s,;.\-А-ЯA-Zа-яa-z]{0,30}",
                    following
                ):

                    answer = clean_answer(
                        following
                    )

                    if answer:
                        return answer

    return None


def extract_answers(pdf_bytes: bytes):

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    answers = {}

    for page in doc:

        text = page_text(page)

        if not text.strip():

            img = render_page(
                page,
                scale=1.5
            )

            text = ocr_text(img)

        matches = list(
            re.finditer(
                r"задани[ея]\s*№?\s*(\d{1,3})\b",
                text,
                re.I
            )
        )

        if matches:

            for index, match in enumerate(matches):

                number = int(
                    match.group(1)
                )

                if index + 1 < len(matches):
                    end = matches[
                        index + 1
                    ].start()
                else:
                    end = len(text)

                block = text[
                    match.start():end
                ]

                answer = answer_from_text(
                    block
                )

                if answer is not None:
                    answers[number] = answer

        else:

            labels = re.findall(
                r"\b(?:ответ|правильный ответ|верные варианты)\b",
                text,
                re.I
            )

            if len(labels) == 1:

                answer = answer_from_text(
                    text
                )

                if answer is not None:

                    answers[
                        len(answers) + 1
                    ] = answer

    return answers


def check_answer(
    user: str,
    correct: str
):

    return (
        normalize_answer(user)
        ==
        normalize_answer(correct)
    )


st.title("Проверяйка")

st.write(
    "Загрузи PDF с заданиями и PDF с правильными ответами."
)

st.info(
    "Задания разделяются по горизонтальным линиям — "
    "сплошным или пунктирным. "
    "Номер задания распознаётся отдельно."
)

homework = st.file_uploader(
    "PDF с домашним заданием",
    type=["pdf"],
    key="homework_pdf"
)

answers_file = st.file_uploader(
    "PDF с ответами",
    type=["pdf"],
    key="answers_pdf"
)


if homework and answers_file:

    if st.button(
        "Найти задания",
        type="primary"
    ):

        with st.spinner(
            "Обрабатываю PDF…"
        ):

            try:

                tasks = extract_tasks(
                    homework.getvalue()
                )

                correct_answers = extract_answers(
                    answers_file.getvalue()
                )

                st.session_state[
                    "tasks"
                ] = tasks

                st.session_state[
                    "correct_answers"
                ] = correct_answers

            except Exception as exc:

                st.error(
                    "Не удалось обработать PDF."
                )

                st.exception(exc)


if "tasks" in st.session_state:

    tasks = st.session_state[
        "tasks"
    ]

    correct_answers = st.session_state.get(
        "correct_answers",
        {}
    )

    if not tasks:

        st.error(
            "Не удалось найти задания."
        )

    else:

        st.success(
            f"Найдено блоков заданий: {len(tasks)}"
        )

        if not correct_answers:

            st.warning(
                "Блоки заданий найдены, "
                "но правильные ответы из второго PDF "
                "не распознаны."
            )

        for task in tasks:

            st.markdown(
                f"### Задание №{task.number}"
            )

            st.image(
                task.image,
                use_container_width=True
            )

            correct = correct_answers.get(
                task.number
            )

            if correct is None:

                st.warning(
                    "Эталонный ответ для этого задания "
                    "не найден."
                )

                st.text_input(
                    "Твой ответ",
                    key=f"user_answer_{task.number}",
                    placeholder="Введи ответ"
                )

                continue

            user_answer = st.text_input(
                "Твой ответ",
                key=f"user_answer_{task.number}",
                placeholder="Введи ответ"
            )

            if st.button(
                "Проверить",
                key=f"check_button_{task.number}"
            ):

                if not user_answer.strip():

                    st.warning(
                        "Сначала введи ответ."
                    )

                elif check_answer(
                    user_answer,
                    correct
                ):

                    st.success(
                        "✓ Правильно!"
                    )

                else:

                    st.error(
                        "✗ Неправильно."
                    )

            with st.expander(
                "Техническая информация"
            ):

                st.caption(
                    f"Эталон: {correct}"
                )
