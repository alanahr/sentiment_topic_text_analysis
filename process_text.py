import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from bertopic.representation import LLM

# 1. Load the two-speaker transcription file
with open("transcript.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Assuming a simple line-by-line transcript format
# E.g. "Speaker 1: Hello, how are you?"
docs = [line.strip() for line in lines if line.strip()]

# 2. Setup Open-Source LLM for Summaries via Ollama
llm_model = OllamaLLM(model="llama3.2")
prompt = """
You are an AI assistant analyzing a conversation between two speakers. 
Given the following conversation segment, provide a concise summary of the key points discussed.

Conversation segment:
[DOCUMENTS]

Summary:
"""
representation_model = LLM(llm=llm_model, prompt=prompt)

# 3. Initialize BERTopic with LLM representation
# We use all-MiniLM-L6-v2 for fast, lightweight embedding generation
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(
    embedding_model=embedding_model,
    representation_model=representation_model,
    min_topic_size=5  # Adjust depending on how granular/frequent topics change
)

# 4. Fit the model and get topic assignments for each utterance
topics, probabilities = topic_model.fit_transform(docs)

# 5. Process the file to inject topics and summaries
output_lines = []
current_topic = -999

for i, line in enumerate(lines):
    if not line.strip():
        continue
    
    assigned_topic = topics[i]
    
    # If the topic changes, insert the new topic name and LLM-generated summary
    if assigned_topic != current_topic:
        current_topic = assigned_topic
        topic_info = topic_model.get_topic(assigned_topic)
        
        # Convert top topic keywords into a readable string
        topic_words = ", ".join([word for word, score in topic_info[:3]]) if topic_info else "General"
        
        # Request an LLM-generated summary for the overall theme using [Ollama](https://ollama.com)
        # Or alternatively fallback to a local huggingface pipeline
        try:
            summary = llm_model.invoke(f"Summarize this topic: {topic_words}")
        except Exception:
            summary = "Summary generation unavailable."

        output_lines.append(f"\n--- NEW TOPIC: {topic_words} ---\n")
        output_lines.append(f"--- SUMMARY: {summary.strip()} ---\n\n")

    output_lines.append(line)

# 6. Save the enriched transcript
with open("enriched_transcript.txt", "w", encoding="utf-8") as f:
    f.writelines(output_lines)

print("Processing complete! Enriched transcript saved as 'enriched_transcript.txt'.")
