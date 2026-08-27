import re
import streamlit as st
import fitz


st.set_page_config(
    page_title="Проверяйка",
    page_icon="✓",
    layout="centered",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 760px;
        padding: 1rem 1rem 3rem;
    }

    .task-card {
        padding: 18px;
        border: 1px solid #dddddd;
        border-radius: 16px;
        margin: 14px 0 8px 0;
        background: rgba(128, 128, 128, 0.06);
    }

    .task-number {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    @media (max-width: 600px) {
        .block-container {
            padding: 0.8rem 0.7rem 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_pdf(uploaded_file):
    """Извлекает текст из PDF."""
    document = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    pages = []

    for page in document:
        pages.append(page.get_text())

    document.close()

    return "\n".join(pages)


def normalize(text):
    """
    Приводит ответы к единому виду.
    Например:
    '  Москва  ' == 'москва'
    'Ё' == 'Е'
    """
    text = str(text).lower()
    text = text.replace("ё", "е")

    # Убираем лишние пробелы
    text = re.sub(r"\s+", " ", text)

    # Убираем пробелы вокруг некоторых знаков
    text = re.sub(r"\s*([,.;:])\s*", r"\1", text)

    return text.strip()


def split_into_blocks(text):
    """
    Делит документ на задания.

    Поддерживаются разделители:
    ----------
    ----------
    ----------------
    ————————
    ...............
    ...............
    _________
    """

    separator_pattern = (
        r"(?:"
        r"\n\s*[-–—_]{3,}\s*\n"
        r"|"
        r"\n\s*[.·•]{4,}\s*\n"
        r"|"
        r"\n\s*={3,}\s*\n"
        r")"
    )

    blocks = re.split(separator_pattern, text)

    return [
        block.strip()
        for block in blocks
        if block.strip()
    ]


def remove_task_number(text):
    """Убирает '1.', '2)', 'Задание 3:' и подобные обозначения."""

    pattern = (
        r"^\s*"
        r"(?:задание\s*)?"
        r"\d+"
        r"\s*"
        r"(?:[.)\-:—]|$)"
        r"\s*"
    )

    return re.sub(
        pattern,
        "",
        text,
        count=1,
        flags=re.IGNORECASE
    ).strip()


def parse_document(text):
    """Получает список заданий или ответов."""

    blocks = split_into_blocks(text)

    result = []

    for block in blocks:
        cleaned = remove_task_number(block)

        if cleaned:
            result.append(cleaned)

    return result


def answers_are_equal(student_answer, correct_answer):
    """
    Сейчас проверка точная:
    ответ ученика должен совпадать с ответом
    из файла ответов после нормализации.
    """

    student = normalize(student_answer)
    correct = normalize(correct_answer)

    return bool(student) and student == correct


# -------------------------
# Состояние приложения
# -------------------------

if "screen" not in st.session_state:
    st.session_state.screen = "upload"

if "tasks" not in st.session_state:
    st.session_state.tasks = []

if "correct_answers" not in st.session_state:
    st.session_state.correct_answers = []

if "results" not in st.session_state:
    st.session_state.results = {}


# -------------------------
# Экран загрузки
# -------------------------

if st.session_state.screen == "upload":

    st.title("Проверяйка")

    st.write(
        "Загрузи PDF с заданиями и PDF с правильными ответами."
    )

    st.info(
        "Каждое задание должно быть отделено от следующего "
        "пунктирной или сплошной линией."
    )

    homework_file = st.file_uploader(
        "PDF с домашним заданием",
        type=["pdf"],
        key="homework"
    )

    answers_file = st.file_uploader(
        "PDF с ответами",
        type=["pdf"],
        key="answers"
    )

    if homework_file and answers_file:

        if st.button(
            "Начать решать",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Читаю документы..."):

                homework_text = read_pdf(homework_file)
                answers_text = read_pdf(answers_file)

                tasks = parse_document(homework_text)
                correct_answers = parse_document(answers_text)

            if not tasks:

                st.error(
                    "Не удалось найти задания в PDF. "
                    "Проверь, что между заданиями есть разделители."
                )

            elif not correct_answers:

                st.error(
                    "Не удалось найти ответы в PDF."
                )

            elif len(tasks) != len(correct_answers):

                st.error(
                    f"Количество заданий и ответов не совпадает.\n\n"
                    f"Заданий найдено: {len(tasks)}\n"
                    f"Ответов найдено: {len(correct_answers)}"
                )

            else:

                st.session_state.tasks = tasks
                st.session_state.correct_answers = correct_answers
                st.session_state.results = {}
                st.session_state.screen = "solve"

                st.rerun()


# -------------------------
# Экран решения
# -------------------------

elif st.session_state.screen == "solve":

    tasks = st.session_state.tasks
    correct_answers = st.session_state.correct_answers
    results = st.session_state.results

    total = len(tasks)
    checked = len(results)
    correct_count = sum(results.values())

    st.title("Домашнее задание")

    progress = checked / total if total else 0

    st.progress(progress)

    st.caption(
        f"Проверено: {checked} из {total}  •  "
        f"Правильных: {correct_count}"
    )

    for index, (task, correct_answer) in enumerate(
        zip(tasks, correct_answers),
        start=1
    ):

        st.markdown(
            f"""
            <div class="task-card">
                <div class="task-number">
                    Задание {index}
                </div>
                <div>
                    {task.replace(chr(10), "<br>")}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        answer = st.text_input(
            "Твой ответ",
            key=f"answer_{index}",
            placeholder="Введи ответ..."
        )

        if st.button(
            "Проверить",
            key=f"check_{index}",
            use_container_width=True
        ):

            results[index] = answers_are_equal(
                answer,
                correct_answer
            )

            st.session_state.results = results

            st.rerun()

        if index in results:

            if results[index]:

                st.success("✓ Правильно!")

            else:

                st.error("✗ Неправильно")


    # -------------------------
    # Итог
    # -------------------------

    if checked == total:

        st.divider()

        percentage = round(
            correct_count / total * 100
        )

        if correct_count == total:

            st.success(
                f"Отлично! Все {total} заданий выполнены правильно."
            )

        else:

            st.warning(
                f"Результат: {correct_count} из {total} "
                f"правильных ({percentage}%)."
            )

        st.write(
            "Можно ещё раз проверить задания, "
            "в которых был неправильный ответ."
        )


    st.divider()

    if st.button(
        "Загрузить другую работу",
        use_container_width=True
    ):

        st.session_state.clear()

        st.rerun()
