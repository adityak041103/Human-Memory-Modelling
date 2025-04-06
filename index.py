from transformers import pipeline
from nltk.corpus import wordnet as wn
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime, timedelta
import numpy as np
import math
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Parsing - Emotion, Superset/Hypernyms, Best hypernyms, set/object/array, data fetch from brain of code, compare memory supersets with input created superset, matching? count++, time update, naya wala bhi dalega, else -> naya daal do

nltk.download('punkt')
nltk.download('wordnet')

# Load models
emotion_analyzer = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# File to store memory
DATA_FILE = "./memory_store.json"

def load_data():
    """Load stored knowledge from JSON."""
    try:
        with open(DATA_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"supersets": []}

def save_data(data):
    """Save memory data to JSON."""
    with open(DATA_FILE, "w") as file:
        json.dump(data, file, indent=4)

# # Function to get hypernyms
# def get_hypernyms(word, pos=None):
#     """Retrieve hypernyms for a word, filtered by POS."""
#     word = wn.morphy(word) or word  
#     synsets = wn.synsets(word, pos=pos)
#     hypernyms = set()

#     for synset in synsets:
#         for hypernym in synset.hypernyms():
#             hypernyms.add(hypernym.name().split('.')[0])

#     return list(hypernyms)

# # Function to get the best hypernym contextually
# def get_best_hypernym(word, context_words, pos=None):
#     """Find the best hypernym based on word embeddings and context similarity."""
#     hypernyms = get_hypernyms(word, pos)
#     if not hypernyms:
#         return None

#     word_embedding = embedding_model.encode([word])
#     hypernym_embeddings = embedding_model.encode(hypernyms)
#     context_embedding = embedding_model.encode([" ".join(context_words)])

#     similarities = cosine_similarity(word_embedding, hypernym_embeddings).flatten()
#     context_similarities = cosine_similarity(context_embedding, hypernym_embeddings).flatten()

#     # Weighted similarity: 70% word similarity + 30% context similarity
#     weighted_scores = 0.7 * similarities + 0.3 * context_similarities

#     best_index = np.argmax(weighted_scores)
#     best_hypernym = hypernyms[best_index]

#     print(f"Word: {word}")
#     print(f"Hypernyms: {hypernyms}")
#     print(f"Weighted Scores: {weighted_scores}")
#     print(f"Best Hypernym: {best_hypernym}\n")

#     return best_hypernym if max(weighted_scores) >= 0.45 else None


def get_hypernyms(word):
    """Retrieve hypernyms for a given word using WordNet."""
    word = wn.morphy(word) or word  
    synsets = wn.synsets(word)
    hypernyms = set()

    for synset in synsets:
        for hypernym in synset.hypernyms():
            hypernyms.add(hypernym.name().split('.')[0])

    return list(hypernyms)

def get_best_hypernym(word):
    """Find the best hypernym using word embeddings."""
    hypernyms = get_hypernyms(word)
    if not hypernyms:
        return None

    word_embedding = embedding_model.encode([word])
    hypernym_embeddings = embedding_model.encode(hypernyms)
    similarities = cosine_similarity(word_embedding, hypernym_embeddings).flatten()

    print(f"Word: {word}")
    print(f"Hypernyms: {hypernyms}")
    print(f"Similarities: {similarities}")
    print(f"Max Similarity Value: {max(similarities)}")
    print(f"Max Similarity Word: {hypernyms[similarities.argmax()]}")
    print()

    if max(similarities) >= 0.50:  # Similarity threshold
        return hypernyms[similarities.argmax()]
    return None

def process_sentence(sentence):
    """Extract entities, emotions, and generate supersets."""
    ner_results = ner_model(sentence)
    ner_entities = {entity['word'] for entity in ner_results}
    words = nltk.word_tokenize(sentence)

    # Detect emotions
    emotion = emotion_analyzer(sentence)[0]
    emotion["score"] = round(emotion["score"], 3)

    result = []
    superset = []

    for word in words:
        if word in ner_entities:
            superset.append("human")
            result.append(f"{word} - Entity Detected")
        else:
            best_hypernym = get_best_hypernym(word)
            if best_hypernym:
                superset.append(best_hypernym)
                result.append(f"{word} - Best Hypernym: {best_hypernym}")
            else:
                result.append(f"{word} - No suitable hypernyms")

    return result, tuple(superset), emotion

def update_or_create_superset(superset, superset_data):
    """Update or create a superset with a dynamic threshold for similarity."""
    if not superset or len(superset) < 2:
        return

    current_time = datetime.now()
    
    # Check if a similar superset already exists
    for s in superset_data["supersets"]:
        shared = sum(1 for i in s["concepts"] if i in superset)
        similarity = shared / min(len(s["concepts"]), len(superset))

        if similarity >= 0.66:  # If 66% of elements match, update existing
            s["count"] += 1
            s["last_updated"] = current_time.isoformat()
            return

    # Otherwise, add a new superset
    superset_data["supersets"].append({
        "concepts": list(superset),
        "count": 1,
        "last_updated": current_time.isoformat(),
        "first_seen": current_time.isoformat(),
    })

def reciprocal_decay_function(count):
    """Reciprocal-based decay factor, similar to y = 1/x."""
    return 1 / (math.log(count + math.e))  # log for smooth decay

def apply_temporal_decay(superset_data):
    """Apply a smooth reciprocal decay to memory counts."""
    today = datetime.now()
    
    for s in superset_data["supersets"]:
        days_elapsed = (today - datetime.fromisoformat(s["last_updated"])).days
        if days_elapsed >= 1:
            decay_factor = reciprocal_decay_function(s["count"])
            s["count"] *= (1 - (decay_factor * days_elapsed))
            s["last_updated"] = today.isoformat()

    # Remove memories that decayed too much
    superset_data["supersets"] = [s for s in superset_data["supersets"] if s["count"] > 0.2]

# def apply_temporal_decay(superset_data):
#     """Apply decay based on frequency (common memories decay slower)."""
#     today = datetime.now()
#     for s in superset_data["supersets"]:
#         days_elapsed = (today - datetime.fromisoformat(s["last_updated"])).days
        
#         val = s["count"]
#         t1 = val + val*0.5
#         t2 = val - val*0.5

#         if days_elapsed > 0:
#             if s["count"] > t1:
#                 decay_factor = 0.95  
#             elif s["count"] < t2:
#                 decay_factor = 0.75  
#             else:
#                 decay_factor = 0.85  

#             s["count"] *= (decay_factor ** days_elapsed)
#             s["last_updated"] = today.isoformat()

#     # Remove memories that decayed too much
#     superset_data["supersets"] = [s for s in superset_data["supersets"] if s["count"] > 0.20]

def calculate_dynamic_threshold(first_seen, base_threshold=5, scaling_factor=2):
    """Dynamically calculate how many repetitions convert memory to knowledge."""
    days_elapsed = (datetime.now() - datetime.fromisoformat(first_seen)).days
    return base_threshold + scaling_factor * math.log(days_elapsed + 1)

def get_adaptive_threshold(superset_data):
    """Determine a self-adjusting threshold based on memory distribution."""
    counts = [s["count"] for s in superset_data["supersets"]]
    
    if len(counts) == 0:
        return 5  # Default threshold if no data exists

    # Scale threshold dynamically
    base_threshold = max(5, int(np.percentile(counts, 60)))  # Use 60th percentile for adaptability
    return min(10, base_threshold)  # Limit upper bound

def check_for_knowledge(superset_data):
    """Identify and mark memories that have converted into knowledge."""
    global_threshold = get_adaptive_threshold(superset_data)
    
    for s in superset_data["supersets"]:
        individual_threshold = calculate_dynamic_threshold(s["first_seen"], base_threshold=global_threshold)
        
        if s["count"] >= individual_threshold:
            s["is_knowledge"] = True  # Promote to knowledge
    
    # Print Knowledge Base
    knowledge = [s for s in superset_data["supersets"] if s.get("is_knowledge", False)]
    
    print("\n📌 Knowledge Base:")
    for k in knowledge:
        print(f"{k['concepts']}, Count: {k['count']} (Knowledge!)")

    return knowledge

# Main Execution
if __name__ == "__main__":
    sentence = input("Enter your Sentence:- ")
    print(f"\n🔹 Processing: {sentence}\n")

    result, superset, emotion = process_sentence(sentence)

    # Load memory
    superset_data = load_data()

    # Apply decay before updating
    apply_temporal_decay(superset_data)
    # set interval chala do for every day 12 am

    # Print word analysis
    for res in result:
        print(res)

    # Print detected emotion
    print("\n🔹 Detected Emotion:", emotion)

    # Update memory
    if superset:
        update_or_create_superset(superset, superset_data)

    # Save updated memory
    save_data(superset_data)

    # Check for knowledge formation
    check_for_knowledge(superset_data)