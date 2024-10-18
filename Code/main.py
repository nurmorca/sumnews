import streamlit as st
import pandas as pd
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer, PegasusForConditionalGeneration, MT5ForConditionalGeneration
import nltk
from scraping import scrape_trt_news, scrape_cnn_articles


def load_model(model_name, model_class, tokenizer_class, token):
    print(f"Loading model {model_name}...")
    tokenizer = tokenizer_class.from_pretrained(model_name, use_auth_token=token)
    model = model_class.from_pretrained(model_name, use_auth_token=token)
    nltk.download('punkt')
    print(f"Model {model_name} loaded!")
    return tokenizer, model

# Function to preprocess text
def preprocess_text(text):
    text = text.replace("?", '"').replace("?", "'")
    text = text.strip().replace("\n", " ")

    if not text.endswith("."):
        text = text + "."

    return text

# Function to generate summary
def generate_summary(text, tokenizer, model, prefix=""):
    text = preprocess_text(text)
    inputs = tokenizer(prefix + text, max_length=1024, truncation=True, return_tensors="pt")

    outputs = model.generate(inputs['input_ids'],
                             num_beams=4,
                             no_repeat_ngram_size=3,
                             min_length=60,
                             max_length=128,
                             length_penalty=2.0,
                             temperature=0.5)
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    summary = nltk.sent_tokenize(decoded_output.strip())
    summary = '\n'.join(summary)
    return summary

def display_text(text, text_type):
    with st.expander(f"{text_type}"):
        st.markdown(text)

def news(key, new, lang):
    st.subheader(str(key + 1) + ". " + new[0])
    if lang == 'english':
        if st.button(f"See full text of the story {key + 1}", key=key):
            display_text(new[2], 'Full text')
        if st.button(f"Generate summary for story {key + 1} with T5"):
            summary = generate_summary(new[2], tokenizer_t5, model_t5, prefix="summarize: ")
            display_text(summary, 'Summary')
        if st.button(f"Generate summary for story {key + 1} with Pegasus"):
            summary = generate_summary(new[2], tokenizer_pegasus, model_pegasus)
            display_text(summary, 'Summary')
    elif lang == 'turkish':
        if st.button(f"Haber {key + 1}'in tamamını görmek için tıklayın.", key=key):
            display_text(new[2], 'Haberin Tamamı')
        if st.button(f"mT5 ile haber {key + 1} özeti için tıklayın."):
            summary = generate_summary(new[2], tokenizer_mt5, model_mt5, prefix="summarize: ")
            display_text(summary, 'Özet')


token = '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'
model_name_t5 = 'morca/t5-ft'
model_name_pegasus = 'morca/pegasus-l-ft'
model_name_mt5_tr = 'morca/mt5-tr-ft'


st.sidebar.title("Language Selection / Dil Seçimi")
language = st.sidebar.radio("Select the language for news summarization / Dil seçimi yapın:", ('English', 'Türkçe'))


if language == 'English':
    st.header('News Summarizer')
    st_model_load = st.text("Models are loading...")
    tokenizer_t5, model_t5 = load_model(model_name_t5, AutoModelForSeq2SeqLM, T5Tokenizer, token)
    tokenizer_pegasus, model_pegasus = load_model(model_name_pegasus, PegasusForConditionalGeneration, PegasusTokenizer,
                                                  token)
    st.success('Models loaded!')
    st_model_load.text("")
    df = scrape_cnn_articles()
    for i in range(len(df)):
        new = df.loc[i, :].values.tolist()
        news(i, new, 'english')
elif language == 'Türkçe':
    st.header('Haber Özetleyici')
    st_model_load = st.text("Model yükleniyor...")
    tokenizer_mt5, model_mt5 = load_model(model_name_mt5_tr, MT5ForConditionalGeneration, T5Tokenizer, token)
    st.success('Model yüklendi!')
    st_model_load.text("")
    df = scrape_trt_news()
    for i in range(len(df)):
        new = df.loc[i, :].values.tolist()
        news(i, new, 'turkish')
