"""
app.py
Streamlit дашборд для сравнения open source LLM через Ollama
Запуск: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from comparator import compare_models

st.set_page_config(page_title="LLM Comparison Dashboard", layout="wide")

st.title(" Model Comparison Dashboard")
st.caption("Сравнение открытых LLM по скорости ответов через Ollama")

# сайдбар настройки
with st.sidebar:
    st.header("Настройки")

    default_models = ["llama3.2:1b", "phi3:mini", "gemma2:2b"]
    models_input = st.text_area(
        "Модели (по одной на строку, имена как в `ollama list`)",
        value="\n".join(default_models),
        height=100,
    )
    models = [m.strip() for m in models_input.splitlines() if m.strip()]

# история сравнений
if "history" not in st.session_state:
    st.session_state.history = []

# присет промпты
presets = {
    "— свой промпт —": "",
    "Объяснение простыми словами": "Объясни рекурсию простыми словами, как для новичка.",
    "Генерация SQL": "Напиши SQL-запрос: выбрать топ-5 клиентов по сумме заказов за 2025 год.",
    "Перевод": "Переведи на английский: 'Модель показывает хорошее качество на простых задачах, но проседает на сложном reasoning.'",
    "Логика / reasoning": "У Маши было 3 яблока, она отдала половину Пете, а потом купила ещё 4. Сколько яблок у Маши?",
    "Код-ревью": "Найди баг в этом Python-коде: def add(a, b): return a - b",
}

preset_choice = st.selectbox("Быстрый выбор промпта", list(presets.keys()))
prompt = st.text_area(
    "Промпт",
    value=presets[preset_choice],
    height=100,
    placeholder="Введи промпт для сравнения моделей...",
)

col_run, col_stream = st.columns([1, 3])
with col_run:
    run_clicked = st.button("Сравнить", type="primary", use_container_width=True)
with col_stream:
    stream_mode = st.toggle("Замерять time-to-first-token (stream)", value=True)

# запуск сравнения
if run_clicked:
    if not prompt.strip():
        st.warning("Введи промпт.")
    elif not models:
        st.warning("Укажи хотя бы одну модель.")
    else:
        results = []
        progress = st.progress(0, text="Запускаю модели...")
        raw_results = compare_models(prompt, models, stream=stream_mode)

        for i, r in enumerate(raw_results):
            progress.progress((i + 1) / len(raw_results), text=f"Готово: {r.model}")
            results.append(r)
        progress.empty()

        st.session_state.history.append({"prompt": prompt, "results": results})

# вывод последнего сравнения
if st.session_state.history:
    last = st.session_state.history[-1]
    st.subheader("Результаты последнего сравнения")

    table_rows = []
    for r in last["results"]:
        table_rows.append({
            "Модель": r.model,
            "Статус": "ошибка" if r.error else "ок",
            "Время (сек)": round(r.total_time_sec, 2) if not r.error else "—",
            "TTFT (сек)": round(r.time_to_first_token_sec, 2) if r.time_to_first_token_sec else "—",
            "Токенов": r.tokens_generated if not r.error else "—",
            "Токенов/сек": round(r.tokens_per_sec, 1) if r.tokens_per_sec else "—",
        })

    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # график скорости
    ok_results = [r for r in last["results"] if not r.error]
    if ok_results:
        chart_df = pd.DataFrame({
            "Модель": [r.model for r in ok_results],
            "Токенов/сек": [r.tokens_per_sec for r in ok_results],
        }).set_index("Модель")
        st.bar_chart(chart_df)

    # развёрнутые ответы
    st.subheader("Ответы моделей")
    for r in last["results"]:
        with st.expander(f"{r.model}" + (" — ошибка" if r.error else "")):
            if r.error:
                st.error(r.error)
            else:
                st.write(r.response)

    # агрегированная статистика по истории 
    if len(st.session_state.history) > 1:
        st.subheader("Агрегированная статистика по всем прогонам")
        agg_rows = []
        for entry in st.session_state.history:
            for r in entry["results"]:
                if not r.error:
                    agg_rows.append({"Модель": r.model, "Токенов/сек": r.tokens_per_sec, "Время (сек)": r.total_time_sec})
        if agg_rows:
            agg_df = pd.DataFrame(agg_rows).groupby("Модель").mean().round(2)
            st.dataframe(agg_df, use_container_width=True)

else:
    st.info("Введи промпт, выбери модели и нажми «Сравнить».")
