from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import pandas as pd; import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#--------------------------------------------------------------
st.set_page_config(page_title='NLP!', page_icon="📋", initial_sidebar_state="expanded", layout="centered")
current_directory = os.getcwd()
logo = Image.open(r"D:\NewDataScience\++\Translation\images\logo.png")
#--------------------------------------------------------------

col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image(logo, width=60)

with col2:
    main_menu = option_menu(None, ["Tranlator"],
                            icons=["translate"],
                            menu_icon="cast", default_index=0, orientation="horizontal")

def local_css(file_name):
    """ Method for reading styles.css and applying necessary changes to HTML"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# تنظیم فونت و عنوان
st.markdown("""
<style>
    /* Define font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
    /* Apply font */
    body {
        font-family: 'Roboto', sans-serif;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #13db3e;'>Translation with AI!</h3>", unsafe_allow_html=True)

if main_menu == "Translator":
    # بارگذاری مدل و توکنایزر در ابتدای برنامه
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    local_css("css/styles.css")

    st.write('')

    col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
    with col1:
        # Create selectbox
        type_language_selection_i = st.selectbox("Input Language:", ("English", "Persian", "Turkish"))  
    with col2:
        st.empty()
    with col3:
        type_language_selection_o = st.selectbox("Output Language:", ("Persian", "English", "Turkish"))

    # دیکشنری تطابق زبان ورودی/خروجی
    language_mapping = {
        "English": "en",
        "Persian": "fa",
        "Turkish": "tr"
    }

    type_lang_i = language_mapping.get(type_language_selection_i)
    type_lang_o = language_mapping.get(type_language_selection_o)

    input_text = st.text_area("Input Your Text:", height=150)

    if st.button("Translate"):
        st.write("Please Wait!")

        tokenizer.src_lang = type_lang_i

        encoded_persian = tokenizer(input_text, return_tensors="pt")
        generated_tokens = model.generate(**encoded_persian, forced_bos_token_id=tokenizer.get_lang_id(type_lang_o))

        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        st.write('')

        # عنوان ترجمه بر اساس زبان خروجی
        text_alignment = "right" if type_lang_o == 'fa' else "left"
        st.markdown(f"<h6 style='text-align: {text_alignment}; color: #ea2323;'>{translated_text[0]}</h6>", unsafe_allow_html=True)
