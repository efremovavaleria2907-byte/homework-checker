import re
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
import streamlit as st


st.set_page_config(
    page_title="Проверяйка",
    page_icon="✓",
    layout="centered",
)


@dataclass
class Task:
    number: int
    page: int
    top: float
    bottom: float
    image_bytes: bytes
    text: str


def normalize_answer(value: str) -> str:
    value = (value or "").strip().lower().replace("ё", "е")
    value = value.replace("—", "-").replace("–", "-")
    # Для ответов ЕГЭ/тестов пробелы и знаки препинания не важны.
    return re.sub(r"[\s,;.:]+", "", value)


def page_text(page: fitz.Page) -> str:
    return page.get_text("text", sort=True) or ""


def text_blocks(page: fitz.Page):
    """
    Возвращает текстовые блоки вместе с их координатами.
    Это работает с обычными PDF намного быстрее и надёжнее OCR.
    """
    blocks = []
    for block in page.get_text("blocks", sort=True):
        x0, y0, x1, y1, text = block[:5]
        text = (text or "").strip()
        if text:
            blocks.append((float(x0), float(y0), float(x1), float(y1), text))
    return blocks


def find_task_headings(page: fitz.Page):
    """
    Ищет «Задание №1», «Задание 1», а также небольшие OCR/шрифтовые
    варианты вроде «Задание N1».
    """
    result = []
    pattern = re.compile(
        r"задани[ея]\s*(?:№|N|No)?\s*(\d{1,3})",
        re.IGNORECASE,
    )

    for x0, y0, x1, y1, text in text_blocks(page):
        for match in pattern.finditer(text.replace("\n", " ")):
            result.append((int(match.group(1)), y0, y1))

    # Если PDF разбивает «Задание №» и номер на разные блоки.
    if not result:
        blocks = text_blocks(page)
        for i, (_, y0, _, y1, text) in enumerate(blocks):
            if re.search(r"задани[ея]\s*(?:№|N|No)?\s*$", text, re.I):
                for _, ny0, _, ny1, ntext in blocks[i + 1:i + 4]:
                    m = re.fullmatch(r"\s*(\d{1,3})\s*", ntext)
                    if m and abs(ny0 - y1) < 80:
                        result.append((int(m.group(1)), y0, ny1))
                        break

    # Убираем повторы.
    unique = []
    for item in sorted(result, key=lambda x: x[1]):
        if unique and item[0] == unique[-1][0] and abs(item[1] - unique[-1][1]) < 20:
            continue
        unique.append(item)
    return unique


def render_crop(page: fitz.Page, top: float, bottom: float, zoom: float = 1.35) -> bytes:
    rect = fitz.Rect(0, max(0, top), page.rect.width, min(page.rect.height, bottom))
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
    return pix.tobytes("png")


def extract_tasks(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    tasks = []

    for pno, page in enumerate(doc):
        heads = find_task_headings(page)

        if not heads:
            # Если на странице ровно одно задание, но заголовок слегка
            # отличается, попробуем найти номер в полном тексте страницы.
            text = page_text(page)
            matches = list(re.finditer(
                r"задани[ея]\s*(?:№|N|No)?\s*(\d{1,3})",
                text,
                re.I,
            ))
            if len(matches) == 1:
                heads = [(int(matches[0].group(1)), 0, 0)]

        if not heads:
            continue

        for i, (number, top, _) in enumerate(heads):
            bottom = heads[i + 1][1] if i + 1 < len(heads) else page.rect.height

            # Небольшие поля вокруг задания.
            crop_top = max(0, top - 12)
            crop_bottom = min(page.rect.height, bottom - 4)

            # Текст только этого блока.
            selected = []
            for _, y0, _, y1, text in text_blocks(page):
                if y1 >= crop_top and y0 < crop_bottom:
                    selected.append(text)

            text = "\n".join(selected)
            image_bytes = render_crop(page, crop_top, crop_bottom)

            tasks.append(
                Task(
                    number=number,
                    page=pno + 1,
                    top=crop_top,
                    bottom=crop_bottom,
                    image_bytes=image_bytes,
                    text=text,
                )
            )

    doc.close()

    # Сохраняем все найденные задания, но не дублируем один и тот же номер.
    unique = {}
    for task in tasks:
        unique.setdefault(task.number, task)

    return [unique[n] for n in sorted(unique)]


def split_answer_blocks(text: str):
    """
    Делит PDF с ответами по «Задание №...».
    В отличие от старой версии, не требует, чтобы ответ находился
    на той же строке, что и заголовок.
    """
    pattern = re.compile(
        r"(?im)^\s*задани[ея]\s*(?:№|N|No)?\s*(\d{1,3})\b"
    )
    matches = list(pattern.finditer(text))

    blocks = []
    for i, match in enumerate(matches):
        number = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append((number, text[start:end]))

    return blocks


def answer_from_block(block: str) -> Optional[str]:
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in block.splitlines()
        if line.strip()
    ]
    joined = " ".join(lines)

    # Сначала ищем явную подпись ответа.
    patterns = [
        r"(?:правильн(?:ый|ого)\s+ответ|верные?\s+варианты|ответ)\s*[:\-]?\s*([0-9][0-9\s,.;-]{0,40})",
        r"(?:ответ)\s*(?:№|N)?\s*[:\-]?\s*([0-9][0-9\s,.;-]{0,40})",
    ]

    for pattern in patterns:
        match = re.search(pattern, joined, re.I)
        if match:
            raw = match.group(1)
            raw = re.split(r"[.!?]\s+", raw)[0]
            raw = re.sub(r"[^0-9,\s;.-]", "", raw).strip(" .;,-")
            if raw:
                return raw

    # Если подпись и значение стоят на соседних строках.
    for i, line in enumerate(lines):
        if re.search(
            r"(?:правильн(?:ый|ого)\s+ответ|верные?\s+варианты|ответ)\b",
            line,
            re.I,
        ):
            same_line = re.sub(
                r"^.*?(?:правильн(?:ый|ого)\s+ответ|верные?\s+варианты|ответ)\s*[:\-]?\s*",
                "",
                line,
                flags=re.I,
            ).strip()
            if re.fullmatch(r"[0-9][0-9\s,;.-]{0,40}", same_line):
                return same_line.strip(" .;,-")

            for next_line in lines[i + 1:i + 4]:
                if re.fullmatch(r"[0-9][0-9\s,;.-]{0,40}", next_line):
                    return next_line.strip(" .;,-")

    # Последний мягкий fallback: короткая строка только с цифрами.
    for line in reversed(lines):
        if re.fullmatch(r"[0-9][0-9\s,;.-]{0,20}", line):
            return line.strip(" .;,-")

    return None


def extract_answers(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = []

    for page in doc:
        all_text.append(page_text(page))

    doc.close()

    full_text = "\n".join(all_text)
    answers = {}

    for number, block in split_answer_blocks(full_text):
        answer = answer_from_block(block)
        if answer is not None:
            answers[number] = answer

    return answers


def check(user: str, correct: str) -> bool:
    return normalize_answer(user) == normalize_answer(correct)


st.title("Проверяйка")
st.write("Загрузи PDF с заданиями и PDF с правильными ответами.")
st.info(
    "PDF читается напрямую, без Tesseract/OCR. "
    "Программа ищет заголовки «Задание №...», поэтому обработка намного быстрее."
)

homework = st.file_uploader(
    "PDF с домашним заданием",
    type=["pdf"],
    key="homework_pdf",
)

answers_file = st.file_uploader(
    "PDF с ответами",
    type=["pdf"],
    key="answers_pdf",
)

if homework and answers_file:
    if st.button("Найти задания", type="primary"):
        # Не трогаем значения, связанные с widget key.
        # Результаты храним под отдельными ключами session_state.
        with st.spinner("Быстро разбираю PDF…"):
            try:
                tasks = extract_tasks(homework.getvalue())
                answers = extract_answers(answers_file.getvalue())

                st.session_state["parsed_tasks"] = tasks
                st.session_state["parsed_answers"] = answers
                st.session_state["parsed_ok"] = True

            except Exception as exc:
                st.session_state["parsed_ok"] = False
                st.error("Не удалось обработать PDF.")
                st.exception(exc)

if st.session_state.get("parsed_ok"):
    tasks = st.session_state.get("parsed_tasks", [])
    answers = st.session_state.get("parsed_answers", {})

    st.success(
        f"Готово. Найдено заданий: {len(tasks)}. "
        f"Найдено ответов: {len(answers)}."
    )

    if not tasks:
        st.error(
            "В PDF не найдено «Задание №...». "
            "Если пришлёшь этот PDF отдельно, можно подстроить правило точно под его разметку."
        )

    elif not answers:
        st.warning(
            "Задания найдены, но PDF с ответами не распознан. "
            "Проверь, что второй файл действительно содержит номера заданий и ответы."
        )

    for task in tasks:
        st.markdown(f"### Задание №{task.number}")
        st.image(task.image_bytes, use_container_width=True)

        correct = answers.get(task.number)

        if correct is None:
            st.warning("Эталонный ответ для этого задания не найден.")
        else:
            user_answer = st.text_input(
                "Твой ответ",
                key=f"user_answer_{task.number}",
                placeholder="Введи ответ",
            )

            if st.button("Проверить", key=f"check_{task.number}"):
                if not user_answer.strip():
                    st.warning("Сначала введи ответ.")
                elif check(user_answer, correct):
                    st.success("✓ Правильно!")
                else:
                    st.error("✗ Неправильно.")

            with st.expander("Техническая информация"):
                st.caption(f"Эталонный ответ: {correct}")
                st.caption(f"Страница PDF: {task.page}")
