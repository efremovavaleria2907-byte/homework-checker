import io
import re
from dataclasses import dataclass
from typing import Optional

import fitz
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
from pytesseract import Output


st.set_page_config(
    page_title="Проверяйка",
    page_icon="✓",
    layout="centered"
)


@dataclass
class Task:
    number: int
    page: int
    top: int
    bottom: int
    image: Image.Image
    text: str


# ---------------------------------------------------------
# Нормализация ответов
# ---------------------------------------------------------

def normalize_answer(value: str) -> str:
    value = (value or "").strip().lower().replace("ё", "е")

    value = value.replace("—", "-")
    value = value.replace("–", "-")

    value = re.sub(r"[\u00a0\t\r\n]+", " ", value)
    value = re.sub(r"[\s,;.:]+", "", value)

    return value


def check(user: str, correct: str) -> bool:
    return normalize_answer(user) == normalize_answer(correct)


# ---------------------------------------------------------
# OCR страницы
# ---------------------------------------------------------

def render_page(page, scale=2.5):
    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False
    )

    return Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )


def ocr_page(page, scale=2.5):
    img = render_page(page, scale)

    data = pytesseract.image_to_data(
        img,
        lang="rus+eng",
        config="--psm 6",
        output_type=Output.DICT
    )

    return img, data


# ---------------------------------------------------------
# Группировка OCR в строки
# ---------------------------------------------------------

def grouped_lines(data):
    groups = {}

    n = len(data["text"])

    for i in range(n):
        text = (data["text"][i] or "").strip()

        if not text:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i]
        )

        groups.setdefault(key, []).append(i)

    lines = []

    for indexes in groups.values():
        indexes.sort(key=lambda i: data["left"][i])

        text = " ".join(
            data["text"][i].strip()
            for i in indexes
            if data["text"][i].strip()
        )

        top = min(data["top"][i] for i in indexes)

        bottom = max(
            data["top"][i] + data["height"][i]
            for i in indexes
        )

        left = min(data["left"][i] for i in indexes)

        right = max(
            data["left"][i] + data["width"][i]
            for i in indexes
        )

        lines.append(
            (top, bottom, left, right, text)
        )

    return sorted(lines, key=lambda x: x[0])


# ---------------------------------------------------------
# Поиск номеров заданий
# ---------------------------------------------------------

def find_task_numbers(text):
    """
    Ищет варианты:

    Задание №1
    Задание 1
    Задания №1
    ЗАДАНИЕ 1
    №1
    1.
    """

    text = text.replace("№", " № ")

    patterns = [
        r"\bзадани[ея]\s*№?\s*(\d{1,3})\b",
        r"\bзадани[ея]\s+(\d{1,3})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            number = int(match.group(1))

            if 1 <= number <= 200:
                return number

    return None


def find_task_headings(lines):
    found = []

    for top, bottom, left, right, text in lines:
        number = find_task_numbers(text)

        if number is not None:
            found.append(
                (number, top, bottom)
            )

    # Убираем повторные OCR-распознавания
    result = []

    for item in found:
        if result:
            previous = result[-1]

            if (
                item[0] == previous[0]
                and abs(item[1] - previous[1]) < 100
            ):
                continue

        result.append(item)

    return result


# ---------------------------------------------------------
# Поиск горизонтальных линий
# ---------------------------------------------------------

def vector_separators(page, scale):
    """
    Если PDF содержит настоящие векторные линии,
    PyMuPDF может увидеть их напрямую.
    """

    separators = []

    try:
        drawings = page.get_drawings()
    except Exception:
        return separators

    for drawing in drawings:
        for item in drawing.get("items", []):

            if not item:
                continue

            kind = item[0]

            if kind != "l":
                continue

            p1 = item[1]
            p2 = item[2]

            x1 = p1.x
            y1 = p1.y
            x2 = p2.x
            y2 = p2.y

            # Нас интересуют почти горизонтальные линии
            if abs(y2 - y1) > 2.5:
                continue

            length = abs(x2 - x1)

            # Не считаем маленькие линии
            if length < 100:
                continue

            y = ((y1 + y2) / 2) * scale

            separators.append(int(y))

    return separators


def raster_separators(img: Image.Image):
    """
    Ищет горизонтальные разделители на изображении.

    Поддерживает:
    - сплошные линии
    - пунктирные линии
    """

    gray = np.asarray(
        img.convert("L"),
        dtype=np.uint8
    )

    height, width = gray.shape

    # Чёрные/тёмные пиксели
    dark = gray < 180

    row_counts = dark.sum(axis=1)

    candidates = []

    for y in range(5, height - 5):

        count = int(row_counts[y])

        if count == 0:
            continue

        ratio = count / width

        # -------------------------------------------------
        # СПЛОШНАЯ линия
        # -------------------------------------------------

        if ratio >= 0.35:
            candidates.append(y)
            continue

        # -------------------------------------------------
        # ПУНКТИРНАЯ линия
        # -------------------------------------------------

        row = dark[y]

        # Ищем последовательности тёмных пикселей
        changes = np.diff(
            np.concatenate(
                [[False], row, [False]]
            ).astype(np.int8)
        )

        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        if len(starts) < 8:
            continue

        run_lengths = ends - starts

        # Пунктир состоит из большого количества
        # относительно коротких горизонтальных отрезков
        short_runs = np.sum(
            (run_lengths >= 2)
            & (run_lengths <= 30)
        )

        span = ends[-1] - starts[0]

        if (
            short_runs >= 8
            and span >= width * 0.45
            and ratio >= 0.015
            and ratio <= 0.30
        ):
            candidates.append(y)

    # Объединяем соседние строки одной линии
    clusters = []

    for y in candidates:

        if not clusters:
            clusters.append([y])
            continue

        if y - clusters[-1][-1] <= 4:
            clusters[-1].append(y)
        else:
            clusters.append([y])

    result = []

    for cluster in clusters:
        center = int(round(sum(cluster) / len(cluster)))

        # Отбрасываем слишком близкие линии
        if not result or center - result[-1] > 20:
            result.append(center)

    return result


def find_separators(page, img, scale):
    """
    Объединяет поиск линий в самом PDF и на картинке.
    """

    result = []

    result.extend(
        vector_separators(page, scale)
    )

    result.extend(
        raster_separators(img)
    )

    result.sort()

    # Объединяем почти одинаковые линии
    merged = []

    for y in result:

        if not merged:
            merged.append(y)
            continue

        if abs(y - merged[-1]) <= 12:
            merged[-1] = int(
                (merged[-1] + y) / 2
            )
        else:
            merged.append(y)

    return merged


# ---------------------------------------------------------
# Разделение страницы на блоки
# ---------------------------------------------------------

def build_blocks(img, separators):
    """
    Превращает:

        верх страницы
        ------
        задание
        ------
        задание
        ------
        низ страницы

    в отдельные блоки.
    """

    height = img.height

    useful = []

    for y in separators:

        if 30 < y < height - 30:
            useful.append(y)

    boundaries = [0]

    for y in useful:
        if y - boundaries[-1] >= 80:
            boundaries.append(y)

    if height - boundaries[-1] >= 80:
        boundaries.append(height)

    blocks = []

    for a, b in zip(
        boundaries,
        boundaries[1:]
    ):

        # Небольшой отступ от самой линии
        top = a + 8
        bottom = b - 8

        if bottom - top < 100:
            continue

        blocks.append(
            (top, bottom)
        )

    return blocks


# ---------------------------------------------------------
# OCR блока
# ---------------------------------------------------------

def ocr_block(crop):
    text = pytesseract.image_to_string(
        crop,
        lang="rus+eng",
        config="--psm 6"
    )

    return text.strip()


# ---------------------------------------------------------
# Определение номера задания в блоке
# ---------------------------------------------------------

def number_from_block(text):
    number = find_task_numbers(text)

    if number is not None:
        return number

    # Иногда OCR распознаёт только "№ 5"
    match = re.search(
        r"№\s*(\d{1,3})\b",
        text
    )

    if match:
        number = int(match.group(1))

        if 1 <= number <= 200:
            return number

    # Иногда заголовок выглядит как:
    # 1. Текст задания
    lines = [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]

    for line in lines[:4]:

        match = re.match(
            r"^(\d{1,3})[.)]\s+",
            line
        )

        if match:
            number = int(match.group(1))

            if 1 <= number <= 200:
                return number

    return None


# ---------------------------------------------------------
# Извлечение заданий
# ---------------------------------------------------------

def extract_tasks(pdf_bytes: bytes):

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    tasks = []

    # Масштаб должен совпадать с render_page
    scale = 2.5

    for page_index, page in enumerate(doc):

        img = render_page(
            page,
            scale=scale
        )

        separators = find_separators(
            page,
            img,
            scale
        )

        blocks = build_blocks(
            img,
            separators
        )

        # ---------------------------------------------
        # Сначала пытаемся использовать разделители
        # ---------------------------------------------

        page_tasks = []

        for top, bottom in blocks:

            crop = img.crop(
                (
                    0,
                    top,
                    img.width,
                    bottom
                )
            )

            text = ocr_block(crop)

            number = number_from_block(text)

            if number is None:
                continue

            page_tasks.append(
                Task(
                    number=number,
                    page=page_index + 1,
                    top=top,
                    bottom=bottom,
                    image=crop,
                    text=text
                )
            )

        # ---------------------------------------------
        # Если линии не помогли — используем заголовки
        # ---------------------------------------------

        if not page_tasks:

            _, data = ocr_page(
                page,
                scale=scale
            )

            lines = grouped_lines(data)
            headings = find_task_headings(lines)

            for index, (
                number,
                top,
                bottom
            ) in enumerate(headings):

                if index + 1 < len(headings):
                    next_top = headings[index + 1][1]
                else:
                    next_top = img.height

                crop_top = max(
                    0,
                    top - 25
                )

                crop_bottom = min(
                    img.height,
                    next_top - 10
                )

                if crop_bottom <= crop_top:
                    continue

                crop = img.crop(
                    (
                        0,
                        crop_top,
                        img.width,
                        crop_bottom
                    )
                )

                selected = [
                    line[4]
                    for line in lines
                    if line[0] >= top
                    and line[0] < next_top
                ]

                text = "\n".join(selected)

                page_tasks.append(
                    Task(
                        number=number,
                        page=page_index + 1,
                        top=crop_top,
                        bottom=crop_bottom,
                        image=crop,
                        text=text
                    )
                )

        tasks.extend(page_tasks)

    # ---------------------------------------------
    # Убираем дубли
    # ---------------------------------------------

    unique = {}

    for task in tasks:

        # Если номер уже есть, оставляем
        # первый найденный вариант.
        if task.number not in unique:
            unique[task.number] = task

    return [
        unique[number]
        for number in sorted(unique)
    ]


# ---------------------------------------------------------
# Поиск ответов
# ---------------------------------------------------------

def answer_from_block(block: str) -> Optional[str]:

    clean_lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in block.splitlines()
        if line.strip()
    ]

    joined = " ".join(clean_lines)

    patterns = [

        r"(?:правильный ответ|верные варианты)"
        r"\s*[:\-]?\s*"
        r"([0-9][0-9\s,.;-]{0,30})",

        r"\bответ"
        r"\s*[:\-]?\s*"
        r"([0-9][0-9\s,.;-]{0,30})",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            joined,
            re.IGNORECASE
        )

        if not match:
            continue

        raw = match.group(1)

        raw = re.sub(
            r"[^0-9\s,;.-]",
            "",
            raw
        )

        raw = raw.strip(
            " .;,-"
        )

        if raw:
            return raw

    # Второй вариант:
    # ищем строку после слова "ответ"

    label_seen = False

    for line in clean_lines:

        if re.search(
            r"(?:правильный ответ|верные варианты|ответ)\b",
            line,
            re.IGNORECASE
        ):
            label_seen = True
            continue

        if (
            label_seen
            and re.fullmatch(
                r"[0-9][0-9\s,;.-]{0,30}",
                line
            )
        ):
            return line.strip(
                " .;,-"
            )

    return None


def extract_answers(pdf_bytes: bytes):

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    answers = {}

    scale = 2.5

    for page in doc:

        img = render_page(
            page,
            scale=scale
        )

        separators = find_separators(
            page,
            img,
            scale
        )

        blocks = build_blocks(
            img,
            separators
        )

        # Сначала разбираем блоки,
        # разделённые линиями.

        for top, bottom in blocks:

            crop = img.crop(
                (
                    0,
                    top,
                    img.width,
                    bottom
                )
            )

            text = ocr_block(crop)

            number = number_from_block(text)

            if number is None:
                continue

            answer = answer_from_block(text)

            if answer is not None:
                answers[number] = answer

        # Если ничего не нашли — обычный OCR
        if not answers:

            _, data = ocr_page(
                page,
                scale=scale
            )

            lines = grouped_lines(data)
            headings = find_task_headings(lines)

            for index, (
                number,
                top,
                bottom
            ) in enumerate(headings):

                if index + 1 < len(headings):
                    next_top = headings[index + 1][1]
                else:
                    next_top = img.height

                block = "\n".join(
                    line[4]
                    for line in lines
                    if line[0] >= top
                    and line[0] < next_top
                )

                answer = answer_from_block(
                    block
                )

                if answer is not None:
                    answers[number] = answer

    return answers


# ---------------------------------------------------------
# Интерфейс
# ---------------------------------------------------------

st.title("Проверяйка")

st.write(
    "Загрузи PDF с заданиями и PDF с правильными ответами."
)

st.info(
    "Задания разделяются по сплошным и пунктирным "
    "горизонтальным линиям. Если линии не распознаются, "
    "программа дополнительно ищет номера заданий через OCR."
)


homework = st.file_uploader(
    "PDF с домашним заданием",
    type=["pdf"],
    key="homework"
)

answers_file = st.file_uploader(
    "PDF с ответами",
    type=["pdf"],
    key="answers"
)


if homework and answers_file:

    if st.button(
        "Найти задания",
        type="primary"
    ):

        with st.spinner(
            "Распознаю PDF и разделяю задания..."
        ):

            try:

                tasks = extract_tasks(
                    homework.getvalue()
                )

                answers = extract_answers(
                    answers_file.getvalue()
                )

                st.session_state["tasks"] = tasks
                st.session_state["answers"] = answers

            except Exception as error:

                st.error(
                    "Не удалось обработать PDF."
                )

                st.exception(error)


# ---------------------------------------------------------
# Результат
# ---------------------------------------------------------

if "tasks" in st.session_state:

    tasks = st.session_state["tasks"]
    answers = st.session_state.get(
        "answers",
        {}
    )

    if not tasks:

        st.error(
            "Я не нашёл ни одного задания."
        )

        st.write(
            "Проверь, что задания действительно разделены "
            "горизонтальными линиями и что PDF содержит "
            "изображения страниц."
        )

    else:

        st.success(
            f"Найдено заданий: {len(tasks)}"
        )

        if not answers:

            st.warning(
                "Задания найдены, но правильные ответы "
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

            correct = answers.get(
                task.number
            )

            if correct is None:

                st.warning(
                    "Правильный ответ для этого задания "
                    "не найден."
                )

                st.text_input(
                    "Твой ответ",
                    key=f"ans_{task.number}"
                )

                continue

            user_answer = st.text_input(
                "Твой ответ",
                key=f"ans_{task.number}",
                placeholder="Введи ответ"
            )

            if st.button(
                "Проверить",
                key=f"check_{task.number}"
            ):

                if not user_answer.strip():

                    st.warning(
                        "Сначала введи ответ."
                    )

                elif check(
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
                    f"Страница PDF: {task.page}"
                )

                st.caption(
                    f"Распознанный эталон: {correct}"
                )
