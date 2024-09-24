import streamlit as st
import joblib
import re

# Load the model and vectorizer
model = joblib.load('model/rf.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer1.pkl')

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):  
        return ''  
    text = re.sub(r'[^\w\s]', ' ', text)  
    text = re.sub(r'\d+', ' ', text)      
    text = re.sub(r'\s+', ' ', text)      
    text = text.strip().lower()            
    return text

# Emotion dictionary
emotion_dict = {
    'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 
    'neutral': 4, 'sadness': 5, 'shame': 6, 'surprise': 7
}

emotions_emoji = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", 
    "happy": "🤗", "joy": "😂", "neutral": "😐", 
    "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Sample sentences
sample_sentences = [
    "I am feeling really angry about the situation.",
    "That movie was disgusting and made me feel sick.",
    "I fear the worst will happen.",
    "I am so happy to see my friends again!",
    "The concert tickets went on sale today."
]

# Streamlit App
st.title('Emotion Detection App')
st.markdown("## Enter your text below to detect emotions!")

# Sidebar for sample sentences
st.sidebar.header("Sample Sentences")
for sentence in sample_sentences:
    if st.sidebar.button(sentence):
        st.session_state.user_input = sentence  # Set the selected sentence in session state

# Text input area
user_input = st.text_area('Text Input:', value=st.session_state.get('user_input', ''), height=150)

# Button for prediction
if st.button('Predict'):
    if user_input:
        clean_input = clean_text(user_input)
        input_features = vectorizer.transform([clean_input]).toarray()
        prediction = model.predict(input_features)
        predicted_emotion = list(emotion_dict.keys())[list(emotion_dict.values()).index(prediction[0])]
        emoji = emotions_emoji.get(predicted_emotion.lower(), '')

        # Display results
        st.markdown("### Prediction Results")
        col1, col2 = st.columns(2)

        with col1:
            st.success("Predicted Emotion")
            st.write(f"**{predicted_emotion}** {emoji}")

        with col2:
            st.success("Cleaned Input")
            st.write(f"**{clean_input}**")

    else:
        st.warning("Please enter some text to get predictions.")
