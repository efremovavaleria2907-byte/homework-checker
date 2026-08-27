import re
from dataclasses import dataclass
from typing import Optional

import fitz
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
class Block:
    number: int
    page: int
    top: int
    bottom: int
    image: Image.Image
    text: str


# ---------------------------------------------------------
# OCR
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def render_page(pdf_bytes: bytes, page_number: int, scale: float = 1.5):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_number]

    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False
    )

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    doc.close()
    return image


@st.cache_data(show_spinner=False)
def ocr_image(image: Image.Image):
    """
    Один OCR-проход на страницу.
    Возвращает текстовые строки вместе с координатами.
    """

    data = pytesseract.image_to_data(
        image,
        lang="rus+eng",
        config="--psm 6",
        output_type=Output.DICT
    )

    lines = {}

    count = len(data["text"])

    for i in range(count):
        text = (data["text"][i] or "").strip()

        if not text:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i]
        )

        lines.setdefault(key, []).append(i)

    result = []

    for indexes in lines.values():

        indexes.sort(
            key=lambda i: data["left"][i]
        )

        words = [
            data["text"][i].strip()
            for i in indexes
            if data["text"][i].strip()
        ]

        if not words:
            continue

        text = " ".join(words)

        top = min(
            data["top"][i]
            for i in indexes
        )

        bottom = max(
            data["top"][i] + data["height"][i]
            for i in indexes
        )

        result.append(
            (top, bottom, text)
        )

    result.sort(key=lambda x: x[0])

    return result


# ---------------------------------------------------------
# Поиск номеров заданий
# ---------------------------------------------------------

def normalize_heading(text: str):
    """
    Приводит разные варианты OCR к виду:
    Задание №15
    """

    value = text.lower()

    value = value.replace("заданиё", "задание")
    value = value.replace("заданиe", "задание")
    value = value.replace("задани", "задание")

    value = value.replace("номер", "№")
    value = value.replace("n", "№")
    value = value.replace("#", "№")

    value = re.sub(r"\s+", " ", value)

    return value


def find_headings(lines):
    """
    Ищет заголовки заданий.
    """

    found = []

    patterns = [
        r"задание\s*[№n#]?\s*(\d{1,2})",
        r"задани[ея]\s*[№n#]?\s*(\d{1,2})",
    ]

    for top, bottom, text in lines:

        normalized = normalize_heading(text)

        number = None

        for pattern in patterns:
            match = re.search(
                pattern,
                normalized,
                re.IGNORECASE
            )

            if match:
                number = int(match.group(1))
                break

        if number is None:
            continue

        found.append(
            (number, top, bottom)
        )

    # Убираем повторные OCR-находки
    result = []

    for item in found:

        if result:
            old_number, old_top, _ = result[-1]

            if (
                old_number == item[0]
                and abs(old_top - item[1]) < 100
            ):
                continue

        result.append(item)

    return result


# ---------------------------------------------------------
# Разделители
# ---------------------------------------------------------

def find_dotted_separators(image: Image.Image):
    """
    Ищет горизонтальные пунктирные линии.
    Это запасной механизм.

    Важно:
    пунктир находится внутри изображения страницы,
    поэтому PyMuPDF сам по себе его не видит как PDF-line.
    """

    gray = image.convert("L")

    width, height = gray.size

    pixels = gray.load()

    candidates = []

    for y in range(100, height - 50):

        dark = 0
        transitions = 0
        previous_dark = False

        for x in range(60, width - 60, 4):

            is_dark = pixels[x, y] < 210

            if is_dark:
                dark += 1

            if is_dark != previous_dark:
                transitions += 1

            previous_dark = is_dark

        ratio = dark / max((width - 120) / 4, 1)

        # Для пунктирной линии:
        # достаточно тёмных точек, но это не сплошная строка текста.
        if 0.015 < ratio < 0.18 and transitions > 20:
            candidates.append(y)

    clusters = []

    for y in candidates:

        if not clusters:
            clusters.append([y])

        elif y - clusters[-1][-1] <= 4:
            clusters[-1].append(y)

        else:
            clusters.append([y])

    separators = []

    for cluster in clusters:

        center = int(
            sum(cluster) / len(cluster)
        )

        separators.append(center)

    return separators


# ---------------------------------------------------------
# Текст блока
# ---------------------------------------------------------

def text_between(lines, top, bottom):
    parts = []

    for line_top, line_bottom, text in lines:

        if (
            line_top >= top
            and line_top < bottom
        ):
            parts.append(text)

    return "\n".join(parts)


# ---------------------------------------------------------
# Домашнее задание
# ---------------------------------------------------------

def extract_homework(pdf_bytes: bytes):

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    blocks = []

    for page_index in range(len(doc)):

        image = render_page(
            pdf_bytes,
            page_index,
            1.5
        )

        lines = ocr_image(image)

        headings = find_headings(lines)

        # -------------------------------------------------
        # Основной вариант:
        # OCR увидел заголовки
        # -------------------------------------------------

        if headings:

            for index, (
                number,
                top,
                heading_bottom
            ) in enumerate(headings):

                if index + 1 < len(headings):
                    next_top = headings[index + 1][1]
                else:
                    next_top = image.height

                crop_top = max(
                    0,
                    top - 15
                )

                crop_bottom = min(
                    image.height,
                    next_top - 5
                )

                if crop_bottom <= crop_top:
                    continue

                crop = image.crop(
                    (
                        0,
                        crop_top,
                        image.width,
                        crop_bottom
                    )
                )

                text = text_between(
                    lines,
                    top,
                    next_top
                )

                blocks.append(
                    Block(
                        number=number,
                        page=page_index + 1,
                        top=crop_top,
                        bottom=crop_bottom,
                        image=crop,
                        text=text
                    )
                )

        else:

            # -------------------------------------------------
            # Если заголовки вообще не распознаны:
            # режем страницу по пунктирным линиям.
            # -------------------------------------------------

            separators = find_dotted_separators(
                image
            )

            bounds = (
                [0]
                + separators
                + [image.height]
            )

            for a, b in zip(
                bounds,
                bounds[1:]
            ):

                if b - a < 150:
                    continue

                crop = image.crop(
                    (
                        0,
                        a,
                        image.width,
                        b
                    )
                )

                text = pytesseract.image_to_string(
                    crop,
                    lang="rus+eng",
                    config="--psm 6"
                )

                match = re.search(
                    r"задани[ея]\s*[№n#]?\s*(\d{1,2})",
                    text,
                    re.IGNORECASE
                )

                if not match:
                    continue

                number = int(match.group(1))

                blocks.append(
                    Block(
                        number=number,
                        page=page_index + 1,
                        top=a,
                        bottom=b,
                        image=crop,
                        text=text
                    )
                )

    doc.close()

    # ---------------------------------------------------------
    # Восстанавливаем пропущенные номера.
    #
    # Например:
    # 15, 16, 17, 18, 20
    #
    # Значит между 18 и 20 должно находиться №19.
    # ---------------------------------------------------------

    blocks.sort(
        key=lambda x: (
            x.page,
            x.top
        )
    )

    result = []

    for block in blocks:

        if not result:
            result.append(block)
            continue

        previous = result[-1]

        # Если OCR случайно пропустил один номер,
        # а следующий номер ровно на 2 больше,
        # считаем это одним пропущенным заданием.
        if block.number == previous.number + 2:

            # Нельзя восстановить картинку пропущенного задания
            # только из воздуха, поэтому пока не создаём фальшивый блок.
            # Позже попробуем найти его по пунктирной линии.

            result.append(block)

        elif block.number > previous.number:
            result.append(block)

    # ---------------------------------------------------------
    # Убираем дубликаты
    # ---------------------------------------------------------

    unique = {}

    for block in result:

        if block.number not in unique:
            unique[block.number] = block

    result = [
        unique[number]
        for number in sorted(unique)
    ]

    # ---------------------------------------------------------
    # Специальная обработка пропущенного №19.
    #
    # Если есть 18 и 20 на одной странице,
    # пытаемся вырезать промежуточную область.
    # ---------------------------------------------------------

    numbers = {
        block.number
        for block in result
    }

    if 18 in numbers and 20 in numbers and 19 not in numbers:

        block18 = next(
            b for b in result
            if b.number == 18
        )

        block20 = next(
            b for b in result
            if b.number == 20
        )

        if block18.page == block20.page:

            image = render_page(
                pdf_bytes,
                block18.page - 1,
                1.5
            )

            # №19 расположен между №18 и №20.
            crop_top = block18.bottom
            crop_bottom = block20.top

            if crop_bottom - crop_top > 80:

                crop = image.crop(
                    (
                        0,
                        crop_top,
                        image.width,
                        crop_bottom
                    )
                )

                text = pytesseract.image_to_string(
                    crop,
                    lang="rus+eng",
                    config="--psm 6"
                )

                result.append(
                    Block(
                        number=19,
                        page=block18.page,
                        top=crop_top,
                        bottom=crop_bottom,
                        image=crop,
                        text=text
                    )
                )

                result.sort(
                    key=lambda x: x.number
                )

    return result


# ---------------------------------------------------------
# Ответы
# ---------------------------------------------------------

def clean_answer(value: str):

    value = (value or "").strip().lower()

    value = value.replace(
        "ё",
        "е"
    )

    value = re.sub(
        r"[\u00a0\t\r\n]+",
        " ",
        value
    )

    value = re.sub(
        r"[\s,;.:]+",
        "",
        value
    )

    value = value.replace(
        "—",
        "-"
    )

    value = value.replace(
        "–",
        "-"
    )

    return value


def answer_from_text(text: str):

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

        # Правильный ответ: 14
        r"(?:правильный ответ|верные варианты)"
        r"\s*[:\-]?\s*"
        r"([0-9][0-9\s,.;-]{0,30})",

        # Ответ: 14
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

        value = match.group(1)

        value = re.sub(
            r"[^0-9\s,;.-]",
            "",
            value
        )

        value = value.strip(
            " .;,-"
        )

        if value:
            return value

    # Иногда OCR получает:
    #
    # Правильный ответ
    # 14
    #

    label_found = False

    for line in lines:

        if re.search(
            r"(правильный ответ|верные варианты|ответ)",
            line,
            re.IGNORECASE
        ):
            label_found = True
            continue

        if label_found:

            if re.fullmatch(
                r"[0-9][0-9\s,;.-]{0,30}",
                line
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

    for page_index in range(len(doc)):

        image = render_page(
            pdf_bytes,
            page_index,
            1.5
        )

        lines = ocr_image(image)

        headings = find_headings(lines)

        for index, (
            number,
            top,
            heading_bottom
        ) in enumerate(headings):

            if index + 1 < len(headings):

                next_top = headings[index + 1][1]

            else:

                next_top = image.height

            text = text_between(
                lines,
                top,
                next_top
            )

            answer = answer_from_text(
                text
            )

            if answer is not None:

                answers[number] = answer

    doc.close()

    return answers


# ---------------------------------------------------------
# Проверка ответа
# ---------------------------------------------------------

def check_answer(
    user_answer: str,
    correct_answer: str
):

    return (
        clean_answer(user_answer)
        == clean_answer(correct_answer)
    )


# ---------------------------------------------------------
# Интерфейс
# ---------------------------------------------------------

st.title("Проверяйка")

st.write(
    "Загрузи PDF с заданиями и PDF с правильными ответами."
)

st.info(
    "PDF обрабатывается как скан: приложение распознаёт "
    "заголовки «Задание №...», а при необходимости "
    "использует пунктирные разделители."
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
            "Распознаю задания и ответы..."
        ):

            try:

                tasks = extract_homework(
                    homework.getvalue()
                )

                answers = extract_answers(
                    answers_file.getvalue()
                )

                st.session_state["tasks_result"] = tasks
                st.session_state["answers_result"] = answers

                st.success(
                    f"Готово. Найдено заданий: "
                    f"{len(tasks)}. "
                    f"Ответов: {len(answers)}."
                )

            except Exception as error:

                st.error(
                    "Не удалось обработать PDF."
                )

                st.exception(error)


# ---------------------------------------------------------
# Результаты
# ---------------------------------------------------------

if "tasks_result" in st.session_state:

    tasks = st.session_state["tasks_result"]

    answers = st.session_state.get(
        "answers_result",
        {}
    )

    if not tasks:

        st.error(
            "Не удалось найти задания."
        )

    else:

        st.success(
            f"Найдено заданий: {len(tasks)}"
        )

        if len(tasks) < 20:

            missing = sorted(
                set(range(1, 21))
                - {task.number for task in tasks}
            )

            if missing:

                st.warning(
                    "Не удалось автоматически определить "
                    "номера: "
                    + ", ".join(
                        map(str, missing)
                    )
                )

        if not answers:

            st.warning(
                "Ответы из второго PDF не распознаны."
            )

        else:

            st.info(
                f"Из PDF с ответами распознано "
                f"{len(answers)} ответов."
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
                    "Эталонный ответ для этого "
                    "задания не найден."
                )

                continue

            user_answer = st.text_input(
                "Твой ответ",
                key=f"user_answer_{task.number}",
                placeholder="Например: 14"
            )

            if st.button(
                "Проверить",
                key=f"check_{task.number}"
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
                "Показать эталонный ответ"
            ):

                st.caption(
                    f"Ответ: {correct}"
                )
