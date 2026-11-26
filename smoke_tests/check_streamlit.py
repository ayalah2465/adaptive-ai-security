import streamlit as st

st.set_page_config(
    page_title="Adaptive AI Security Agent",
    page_icon="🛡️",
    layout="wide"
)

# Translation dictionary
TRANSLATIONS = {
    "en": {
        "title": "Adaptive AI Security Agent",
        "subtitle": "UI Environment Smoke Test",
        "success": "Interface is running successfully",
        "counter": "Action counter",
        "direction": "Text direction",
        "run": "Run test action",
        "reset": "Reset counter"
    },
    "he": {
        "title": "סוכן אבטחת AI אדפטיבי",
        "subtitle": "בדיקת תקינות סביבת הממשק",
        "success": "הממשק פועל בהצלחה",
        "counter": "מונה פעולות",
        "direction": "כיוון תצוגה",
        "run": "ביצוע פעולת בדיקה",
        "reset": "איפוס מונה"
    }
}

# Initialize session state
if "language" not in st.session_state:
    st.session_state.language = "en"

if "counter" not in st.session_state:
    st.session_state.counter = 0

# Language selector
st.selectbox(
    "Language / שפה",
    options=["en", "he"],
    format_func=lambda value: {
        "en": "English",
        "he": "עברית"
    }[value],
    key="language"
)

language = st.session_state.language
translations = TRANSLATIONS[language]

direction = "rtl" if language == "he" else "ltr"
alignment = "right" if language == "he" else "left"

# Apply language direction
st.markdown(
    f"""
    <style>
        .stApp {{
            direction: {direction};
            text-align: {alignment};
        }}

        [data-testid="stMarkdownContainer"] {{
            direction: {direction};
            text-align: {alignment};
        }}

        [data-testid="stMetric"] {{
            direction: {direction};
            text-align: {alignment};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Button callbacks
def increment_counter():
    st.session_state.counter += 1


def reset_counter():
    st.session_state.counter = 0


# Main interface
st.title(translations["title"])
st.subheader(translations["subtitle"])

st.success(translations["success"])

col1, col2 = st.columns(2)

with col1:
    st.metric(
        translations["counter"],
        st.session_state.counter
    )

with col2:
    st.metric(
        translations["direction"],
        direction.upper()
    )

st.button(
    translations["run"],
    key="run_test",
    on_click=increment_counter
)

st.button(
    translations["reset"],
    key="reset_test",
    on_click=reset_counter
)