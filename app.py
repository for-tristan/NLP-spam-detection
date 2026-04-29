import streamlit as st
import re
import pickle
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore

st.set_page_config(page_title="NLP Spam Detection", layout="centered")

st.markdown("""
    <style>
        h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stAlert, div[data-testid="stTextInput"] label p {
            text-align: center;
        }
        div.stButton > button, div[data-testid="stTextInput"] input {
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

if 'result' not in st.session_state:
    st.session_state.result = ""

def clear():
    st.session_state.result = ""

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.title("Spam Detection", anchor="center", )

    algorithm_choice = st.selectbox(
        "Select Algorithm",
        [
            "Algorithm 1: Naive Bayes",
            "Algorithm 2: SVM",
            "Algorithm 3: Logistic Regression",
            "Algorithm 4: Random Forest",
            "Algorithm 5: CNN"
        ],
        on_change=clear
    )

    input_text = st.text_area("Enter your message:", height=150)
    process_clicked = st.button("Process", use_container_width=True)
    st.text_input("Result:", value=st.session_state.result, disabled=True)



@st.cache_resource
def load_model_by_name(choice):
    if "Naive Bayes" in choice:
        with open('nb_model.pkl', 'rb') as f: model = pickle.load(f)
        with open('nb_vectorizer.pkl', 'rb') as f: vec = pickle.load(f)
        return model, vec, "traditional"
    
    elif "SVM" in choice:
        with open('svm_model.pkl', 'rb') as f: model = pickle.load(f)
        with open('svm_vectorizer.pkl', 'rb') as f: vec = pickle.load(f)
        return model, vec, "traditional"
    
    elif "Logistic" in choice:
        with open('lr_model.pkl', 'rb') as f: model = pickle.load(f)
        with open('lr_vectorizer.pkl', 'rb') as f: vec = pickle.load(f)
        return model, vec, "traditional"
    
    elif "Random" in choice:
        with open('rf_spam_model.pkl', 'rb') as f: model = pickle.load(f)
        with open('rf_vectorizer.pickle', 'rb') as f: vec = pickle.load(f)
        return model, vec, "traditional"
    
    elif "CNN" in choice:
        cnn_model = load_model('cnn_model.keras') 
        with open('cnn_tokenizer.pkl', 'rb') as f: tokenizer = pickle.load(f)
        return cnn_model, tokenizer, "deep_learning"



def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

if process_clicked:
    if input_text.strip() == "":
        st.warning("Please enter a message to process.")
    else:
        with st.spinner(f'Processing'):
            cleaned = clean_text(input_text)
            model, processor, model_type = load_model_by_name(algorithm_choice)
            
            if model_type == "traditional":
                vectorized_text = processor.transform([cleaned])
                prediction = model.predict(vectorized_text)[0]
                st.session_state.result = "Spam" if prediction == 1 else "Not Spam"
            
            elif model_type == "deep_learning":
                seq = processor.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=150, padding='post', truncating='post')
                prob = model.predict(padded, verbose=0)[0][0]
                st.session_state.result = "Spam" if prob >= 0.5 else "Not Spam"
            
            st.rerun()