import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✍️",
    layout="centered"
)


# --------------------------------------------------
# LOAD THE LSTM MODEL
# --------------------------------------------------

model = load_model('next_word_lstm.h5')


# --------------------------------------------------
# LOAD THE TOKENIZER
# --------------------------------------------------

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# --------------------------------------------------
# FUNCTION TO PREDICT THE NEXT WORD
# --------------------------------------------------

def predict_next_word(model, tokenizer, text, max_sequence_len):

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding='pre'
    )

    predicted = model.predict(token_list, verbose=0)

    predicted_word_index = np.argmax(predicted, axis=1)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None


# --------------------------------------------------
# FRONTEND
# --------------------------------------------------

st.title("✍️ Next Word Predictor")

st.subheader("LSTM Based Text Prediction")

st.write(
    "Enter a sequence of words and let the LSTM model predict "
    "the most likely next word."
)

st.divider()


# Input

input_text = st.text_input(
    "Enter your sentence",
    value="To be or not to",
    placeholder="Type your sentence here..."
)

st.caption("💡 Try entering at least 3–4 words for better predictions.")


# Prediction button

if st.button("🔮 Predict Next Word", use_container_width=True):

    if input_text.strip():

        max_sequence_len = model.input_shape[1] + 1

        next_word = predict_next_word(
            model,
            tokenizer,
            input_text,
            max_sequence_len
        )

        st.divider()

        st.success("Prediction Complete!")

        st.metric(
            label="Predicted Next Word",
            value=next_word
        )

    else:

        st.warning("Please enter some text first.")


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("**Model**\n\nLSTM")

with col2:
    st.info("**Architecture**\n\nDeep RNN")

with col3:
    st.info("**Task**\n\nNext Word")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption("Built with TensorFlow • Keras • Streamlit")