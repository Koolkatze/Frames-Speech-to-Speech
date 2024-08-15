import re
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine


# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
        
# Set your OpenAI API key
openai_api_key = "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"
# Initialize the OpenAI client with your API key
client = OpenAI(api_key=openai_api_key)

# Function to split the text into conversations
# Function to split the text into conversations
def split_conversations(file_path):
    # Open and read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split by "Conversation ended at:" and keep the delimiter
    chunks = re.split('(Conversation ended at: [0-9- :]+)', content)
    
    # Reassemble the chunks correctly
    conversations = []
    for i in range(0, len(chunks)-1, 2): # Pair elements
        conversation_with_timestamp = chunks[i].strip() + "\n" + chunks[i+1].strip()
        conversations.append(conversation_with_timestamp)
    
    return conversations

def get_embeddings_for_conversations(conversations):
    embeddings = []
    for conversation in conversations:
        response = client.embeddings.create(
            model="text-embedding-ada-002",  # Choose an appropriate model for your needs
            input=conversation,
            encoding_format="float"  # Choose "float" or "base64" based on your needs
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

def write_embeddings_to_file(embeddings, file_path="embeddings.txt"):
    with open(file_path, 'w', encoding='utf-8') as file:
        for i, embedding in enumerate(embeddings):
            # Convert the embedding list to a string representation
            embedding_str = np.array(embedding).tolist()
            file.write(f"Conversation {i+1} Embedding:\n{embedding_str}\n\n")

# Replace 'your_file_path.txt' with the actual path to your text file
file_path = 'C:/Users/kris_/Python/low-latency-sts/chatlog.txt'
# Call the function to split conversations and store them in the 'conversations' variable
conversations = split_conversations(file_path)

# Now proceed to get embeddings and write them to a file
embeddings = get_embeddings_for_conversations(conversations)
write_embeddings_to_file(embeddings, "conversation_embeddings.txt")

#print("Embeddings have been written to conversation_embeddings.txt")

# Function to load embeddings from a file
def load_embeddings(file_path):
    embeddings = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("Conversation"):
                embedding = eval(next(file))  # Read the next line which contains the embedding
                embeddings.append(np.array(embedding))
    return embeddings

# Function to compute the cosine similarity between two embeddings
def cosine_similarity(vec_a, vec_b):
    return 1 - cosine(vec_a, vec_b)

# Function to find the most similar conversation
def find_most_similar_conversation(query, embeddings):
    # Assume client is already initialized and query is a string
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query,
        encoding_format="float"
    )
    query_embedding = np.array(response.data[0].embedding)
    
    # Compute similarities
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    
    # Find the index of the highest similarity
    most_similar_index = np.argmax(similarities)
    return most_similar_index, similarities[most_similar_index]

# Load embeddings
embeddings_file_path = "conversation_embeddings.txt"  # Adjust the path as necessary
embeddings = load_embeddings(embeddings_file_path)

# Example query
query = "Did we talk about AI?"

# Assuming `client` is your OpenAI client initialized with your API key
most_similar_index, similarity = find_most_similar_conversation(query, embeddings)
print(f"The most similar conversation is at index {most_similar_index} with a similarity of {similarity:.4f}") 