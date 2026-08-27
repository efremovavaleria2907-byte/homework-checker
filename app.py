import re
import fitz
import streamlit as st
from dataclasses import dataclass
from PIL import Image


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


# ---------------------------------------------------------
# Нормализация ответов
# ---------------------------------------------------------

def normalize_answer(value):
    if value is None:
        return ""

    value = str(value).strip().lower()
    value = value.replace("ё", "е")
    value = value.replace("—", "-")
    value = value.replace("–", "-")

    # Убираем пробелы и знаки препинания.
    value = re.sub(r"[\s,;.:]+", "", value)

    return value


def answers_equal(user, correct):
    return normalize_answer(user) == normalize_answer(correct)


# ---------------------------------------------------------
# Получение текста PDF
# ---------------------------------------------------------

def get_page_text(page):
    """
    Получает весь текст страницы.
    Работает значительно быстрее OCR.
    """
    return page.get_text("text")


# ---------------------------------------------------------
# Поиск номера задания
# ---------------------------------------------------------

def find_task_number(text):
    if not text:
        return None

    patterns = [
        r"задание\s*№?\s*(\d{1,3})",
        r"задани[ея]\s*№?\s*(\d{1,3})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

    return None


# ---------------------------------------------------------
# Поиск горизонтальных линий непосредственно в PDF
# ---------------------------------------------------------

def find_horizontal_lines(page):
    """
    Ищет горизонтальные линии средствами PyMuPDF.

    Это намного надежнее, чем искать пунктир
    по пикселям изображения.
    """

    page_width = page.rect.width
    lines = []

    try:
        drawings = page.get_drawings()
    except Exception:
        return []

    for drawing in drawings:
        for item in drawing.get("items", []):

            # item вида:
            # ('l', point1, point2)
            if not item:
                continue

            if item[0] != "l":
                continue

            p1 = item[1]
            p2 = item[2]

            y1 = p1.y
            y2 = p2.y

            x1 = p1.x
            x2 = p2.x

            # Горизонтальность
            if abs(y1 - y2) > 2:
                continue

            length = abs(x2 - x1)

            # Берём только достаточно длинные линии.
            # Маленькие линии внутри задания игнорируем.
            if length < page_width * 0.35:
                continue

            y = (y1 + y2) / 2

            # Не берём линии совсем у края страницы.
            if y < 20 or y > page.rect.height - 20:
                continue

            lines.append(y)

    # Объединяем линии, находящиеся практически на одном уровне.
    lines.sort()

    result = []

    for y in lines:
        if not result or abs(y - result[-1]) > 5:
            result.append(y)

    return result


# ---------------------------------------------------------
# Текст внутри прямоугольной области страницы
# ---------------------------------------------------------

def text_in_rect(page, rect):
    blocks = page.get_text("blocks")

    pieces = []

    for block in blocks:
        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]

        block_rect = fitz.Rect(x0, y0, x1, y1)

        # Если блок текста пересекается с нашим прямоугольником.
        if block_rect.intersects(rect):
            if text.strip():
                pieces.append(text.strip())

    return "\n".join(pieces)


# ---------------------------------------------------------
# Рендер участка страницы
# ---------------------------------------------------------

def render_rect(page, rect, scale=1.7):
    matrix = fitz.Matrix(scale, scale)

    pix = page.get_pixmap(
        matrix=matrix,
        clip=rect,
        alpha=False
    )

    return Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )


# ---------------------------------------------------------
# Поиск заданий на странице
# ---------------------------------------------------------

def extract_tasks_from_page(page, page_number):
    page_height = page.rect.height
    page_width = page.rect.width

    # -----------------------------------------------------
    # Сначала пытаемся найти заголовки "Задание №..."
    # -----------------------------------------------------

    blocks = page.get_text("blocks")

    headings = []

    for block in blocks:
        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]

        number = find_task_number(text)

        if number is not None:
            headings.append(
                (number, y0, y1)
            )

    headings.sort(key=lambda x: x[1])

    # Убираем повторные срабатывания.
    clean_headings = []

    for item in headings:
        number, top, bottom = item

        duplicate = False

        for old_number, old_top, old_bottom in clean_headings:
            if (
                number == old_number
                and abs(top - old_top) < 20
            ):
                duplicate = True
                break

        if not duplicate:
            clean_headings.append(item)

    # Если нашли хотя бы два заголовка,
    # это самый надежный способ.
    if len(clean_headings) >= 2:

        tasks = []

        for i, (number, top, bottom) in enumerate(clean_headings):

            if i + 1 < len(clean_headings):
                next_top = clean_headings[i + 1][1]
            else:
                next_top = page_height

            rect = fitz.Rect(
                0,
                max(0, top - 8),
                page_width,
                min(page_height, next_top)
            )

            text = text_in_rect(page, rect)

            image = render_rect(page, rect)

            tasks.append(
                Task(
                    number=number,
                    page=page_number,
                    image=image,
                    text=text
                )
            )

        return tasks

    # -----------------------------------------------------
    # Если заголовки не распознаны —
    # режем страницу по настоящим линиям PDF.
    # -----------------------------------------------------

    lines = find_horizontal_lines(page)

    # Добавляем начало и конец страницы.
    boundaries = [0]

    for y in lines:
        if 40 < y < page_height - 40:
            boundaries.append(y)

    boundaries.append(page_height)

    boundaries = sorted(set(boundaries))

    blocks_found = []

    for a, b in zip(boundaries, boundaries[1:]):

        # Слишком маленькие куски пропускаем.
        if b - a < 80:
            continue

        rect = fitz.Rect(
            0,
            a,
            page_width,
            b
        )

        text = text_in_rect(page, rect)

        if len(text.strip()) < 5:
            continue

        number = find_task_number(text)

        if number is None:
            continue

        image = render_rect(page, rect)

        blocks_found.append(
            Task(
                number=number,
                page=page_number,
                image=image,
                text=text
            )
        )

    return blocks_found


# ---------------------------------------------------------
# Извлечение всех заданий
# ---------------------------------------------------------

def extract_tasks(pdf_bytes):
    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    tasks = []

    for page_index, page in enumerate(doc):

        page_tasks = extract_tasks_from_page(
            page,
            page_index + 1
        )

        tasks.extend(page_tasks)

    doc.close()

    # -----------------------------------------------------
    # Убираем дубли.
    # -----------------------------------------------------

    unique = {}

    for task in tasks:

        if task.number not in unique:
            unique[task.number] = task

    result = [
        unique[number]
        for number in sorted(unique)
    ]

    return result


# ---------------------------------------------------------
# Поиск ответов
# ---------------------------------------------------------

def extract_answers(pdf_bytes):
    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    answers = {}

    current_number = None

    for page in doc:

        text = get_page_text(page)

        if not text:
            continue

        # Разбиваем на строки.
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        for i, line in enumerate(lines):

            number = find_task_number(line)

            if number is not None:
                current_number = number

            # Ищем конструкции:
            # Ответ: 123
            # Правильный ответ: 123
            # Верные варианты: 123
            match = re.search(
                r"(?:правильный\s+ответ|верные\s+варианты|ответ)"
                r"\s*[:\-]?\s*"
                r"([0-9][0-9\s,;.\-]*)",
                line,
                re.IGNORECASE
            )

            if match and current_number is not None:

                answer = match.group(1).strip()

                answer = re.sub(
                    r"[^0-9\s,;.\-]",
                    "",
                    answer
                ).strip(" .,;-")

                if answer:
                    answers[current_number] = answer
                    continue

            # Иногда ответ находится на следующей строке.
            if re.search(
                r"(?:правильный\s+ответ|верные\s+варианты|ответ)\s*[:\-]?$",
                line,
                re.IGNORECASE
            ):

                if i + 1 < len(lines):

                    next_line = lines[i + 1]

                    if re.fullmatch(
                        r"[0-9][0-9\s,;.\-]*",
                        next_line
                    ):

                        if current_number is not None:
                            answers[current_number] = (
                                next_line.strip(" .,;-")
                            )

    doc.close()

    return answers


# ---------------------------------------------------------
# Интерфейс
# ---------------------------------------------------------

st.title("Проверяйка")

st.write(
    "Загрузи PDF с заданиями и PDF с правильными ответами."
)

st.info(
    "Задания разделяются по заголовкам «Задание №...» "
    "или по настоящим горизонтальным линиям PDF. "
    "OCR не используется, поэтому обработка происходит значительно быстрее."
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
            "Разбираю PDF…"
        ):

            try:

                tasks = extract_tasks(
                    homework.getvalue()
                )

                answers = extract_answers(
                    answers_file.getvalue()
                )

                # ВАЖНО:
                # записываем результаты только сюда.
                # Никаких виджетов с такими же key нет.
                st.session_state["found_tasks"] = tasks
                st.session_state["found_answers"] = answers

                st.success(
                    f"Обработка завершена. "
                    f"Найдено заданий: {len(tasks)}. "
                    f"Ответов: {len(answers)}."
                )

            except Exception as e:

                st.error(
                    "Не удалось обработать PDF."
                )

                st.exception(e)


# ---------------------------------------------------------
# Результаты
# ---------------------------------------------------------

if "found_tasks" in st.session_state:

    tasks = st.session_state["found_tasks"]
    answers = st.session_state.get(
        "found_answers",
        {}
    )

    if not tasks:

        st.error(
            "Не удалось найти задания в PDF."
        )

    else:

        st.success(
            f"Найдено заданий: {len(tasks)}"
        )

        if len(tasks) < 20:

            st.warning(
                f"В PDF, судя по всему, должно быть 20 заданий, "
                f"но программа нашла только {len(tasks)}. "
                f"Если это так, пришли сам PDF — тогда можно "
                f"точно подстроить правило разделения."
            )

        if not answers:

            st.warning(
                "Правильные ответы из второго PDF "
                "не распознаны."
            )

        else:

            st.info(
                f"Распознано правильных ответов: "
                f"{len(answers)}"
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
                    "Эталонный ответ для этого задания "
                    "не найден."
                )

                st.text_input(
                    "Твой ответ",
                    key=f"user_answer_{task.number}"
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

                elif answers_equal(
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
                    f"Правильный ответ: {correct}"
                )
