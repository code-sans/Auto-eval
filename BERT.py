import nltk
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

# Load Models
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("summarization")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

# Summarization
def summarize_text(text):
    summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
    return summary[0]['summary_text']

# Extract Keywords using TF-IDF
def extract_keywords(text, top_n=5):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    return feature_array[tfidf_sorting][:top_n]

# POS Tagging & Lemmatization
def get_final_keywords(text):
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = nltk.WordNetLemmatizer()
    final_keywords = [lemmatizer.lemmatize(word.lower()) for word, tag in pos_tags if tag.startswith('N') or tag.startswith('V')]
    return final_keywords

# Compute BERT and SBERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_sbert_embeddings(text):
    return sbert_model.encode(text)

def calculate_similarity(ref_text, student_text):
    ref_embedding_bert = get_bert_embeddings(ref_text)
    ans_embedding_bert = get_bert_embeddings(student_text)
    bert_similarity = cosine_similarity([ref_embedding_bert], [ans_embedding_bert])[0][0]
    
    ref_embedding_sbert = get_sbert_embeddings(ref_text)
    ans_embedding_sbert = get_sbert_embeddings(student_text)
    sbert_similarity = cosine_similarity([ref_embedding_sbert], [ans_embedding_sbert])[0][0]
    
    return bert_similarity, sbert_similarity

def evaluate_answer(reference_answer, student_answer):
    # Summarize answers
    ref_summary = summarize_text(reference_answer)
    student_summary = summarize_text(student_answer)
    
    # Extract keywords
    ref_keywords = extract_keywords(ref_summary)
    student_keywords = extract_keywords(student_summary)
    
    # POS tagging & lemmatization
    ref_final_keywords = get_final_keywords(reference_answer)
    student_final_keywords = get_final_keywords(student_answer)
    
    # Compute Similarities
    bert_similarity, sbert_similarity = calculate_similarity(ref_summary, student_summary)
    
    print("🔹 Reference Keywords:", ref_keywords)
    print("🔹 Student Keywords:", student_keywords)
    print("🔹 Final Keywords (Ref):", ref_final_keywords)
    print("🔹 Final Keywords (Student):", student_final_keywords)
    print(f"🔹 BERT Similarity: {bert_similarity:.2f}")
    print(f"🔹 SBERT Similarity: {sbert_similarity:.2f}")
    
    keyword_match = len(set(ref_final_keywords).intersection(set(student_final_keywords))) / len(set(ref_final_keywords))
    print(f"🔹 Keyword Match Score: {keyword_match:.2f}")
    
    if sbert_similarity >= 0.7 and keyword_match >= 0.5:
        print("✅ The student's answer is correct.")
    else:
        print("❌ The student's answer needs improvement.")

# Example Usage
reference_answer =  """PEAS is a type of model
When we define an AI agent or rational agent, we can categorize its properties using the PEAS representation model. It consists of four components:

P: Performance Measure
E: Environment
A: Actuators
S: Sensors
PEAS for a Self-Driving Car

Performance: Safety, efficiency, legal compliance, passenger comfort.
Environment: Roads, traffic, road signs, pedestrians.
Actuators: Steering, accelerator, brakes, indicators, horn.
Sensors: Cameras, GPS, speedometer, odometer, accelerometer, LiDAR. """

student_answer = """
Performance: Safety, time, legal drive, comfort.
Environment: Roads, other vehicles, road signs, pedestrians

"""

evaluate_answer(reference_answer, student_answer)
