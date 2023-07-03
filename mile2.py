import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import pipeline

# Set up the Streamlit app
st.title("Sentiment Analysis App")
st.write('Welcome to my Sentiment Analysis app!')

#subtitle
st.markdown("Sentiment Analysis App using 'streamlit' hosted on hugging spaces ")
st.markdown("")

user_input = st.text_area("Enter your text", value="")
form = st.form(key='sentiment-form')
submit = form.form_submit_button('Submit')

classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I've been waiting for HuggingFAcecourse my whole life.")

classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english") 
result = classifier(user_input)[0]    
label = result['label']    
score = result['score']

if submit:
    classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")
    result = classifier(user_input)[0]
    label = result['label']
    score = result['score']
if label == 'POSITIVE':
        st.success(f'{label} sentiment (score: {score})')
else:
    st.error(f'{label} sentiment (score: {score})')

# Load the sentiment analysis model and tokenizer
model_name = "textattack/bert-base-uncased-SST-2"
model2 = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Model selection
model_options = {
    "BERT-base-uncased-SST-2": "textattack/bert-base-uncased-SST-2",
    "BERT-base-cased-finetuned-mrpc": "bert-base-cased-finetuned-mrpc"
}
model_name = st.selectbox("Select a pretrained model", list(model_options.keys()))
model_path = model_options[model_name]

# Sentiment analysis
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize input text
        inputs = tokenizer.encode_plus(user_input, return_tensors="pt", padding=True, truncation=True)

        # Perform sentiment analysis
        with torch.no_grad():
            outputs = model2(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()

        sentiment = "Positive" if predicted_label == 1 else "Negative"
        st.success(f"The sentiment of the text is: {sentiment}")

