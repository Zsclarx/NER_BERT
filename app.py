import streamlit as st
from transformers import BertTokenizerFast, TFBertForTokenClassification
import tensorflow as tf

# Load the fine-tuned model and tokenizer
model = TFBertForTokenClassification.from_pretrained("bert-ner-conll")
tokenizer = BertTokenizerFast.from_pretrained("bert-ner-conll")

label_list = [
    'O', 'B-PER', 'I-PER', 'B-ORG', 'B-MISC', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC'
]

def predict(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=128)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    outputs = model(input_ids)
    logits = outputs.logits
    
    predictions = tf.argmax(logits, axis=-1).numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids.numpy()[0])
    
    results = []
    for token, pred, mask in zip(tokens, predictions, attention_mask[0]):
        if mask == 1:  # Exclude padding tokens
            results.append((token, label_list[pred]))
    
    return results

st.title("NER with BERT")
st.write("Enter text to perform Named Entity Recognition:")

input_text = st.text_area("Input Text")

if st.button("Predict"):
    if input_text:
        predictions = predict(input_text)
        st.write("### Predictions:")
        for token, label in predictions:
            if label != "O":
                st.write(f"`{token}` - **{label}**")
            else:
                st.write(f"`{token}` - {label}")
    else:
        st.write("Please enter some text to predict.")
