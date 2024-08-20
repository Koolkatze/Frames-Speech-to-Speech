import os
import torch
import argparse
import pyaudio
import torchaudio
import numpy as np
import wave
import openai
from openai import OpenAI
import speech_recognition as sr
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer, util
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Set up the faster-whisper model
model_size = "medium.en"
whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Initialize the OpenAI client with the API key
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Function to play audio using PyAudio
def play_audio(file_path):
    # Open the audio file
    wf = wave.open(file_path, 'rb')
    # Create a PyAudio instance
    p = pyaudio.PyAudio()
    # Open a stream to play audio
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    # Read and play audio data
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    # Stop and close the stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# Model and device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load XTTS configuration
xtts_config = XttsConfig()
xtts_config.load_json("C:/Users/kris_/Python/fsts2/XTTS-v2/config.json")

# Initialize XTTS model
xtts_model = Xtts.init_from_config(xtts_config)
xtts_model.load_checkpoint(xtts_config, checkpoint_dir="C:/Users/kris_/Python/fsts2/XTTS-v2/", eval=True)
xtts_model.cuda()  # Move the model to GPU if available

# Function to synthesize speech using XTTS
def process_and_play(prompt, audio_file_pth):
    tts_model = xtts_model
    try:

        # Use XTTS to synthesize speech
        outputs = tts_model.synthesize(
            prompt,  # Pass the prompt as a string directly
            xtts_config,
            speaker_wav=audio_file_pth,  # Pass the file path directly
            gpt_cond_len=24,
            temperature=0.6,
            language='en',
            speed=1.2  # Specify the desired language
        )

        # Get the synthesized audio tensor from the dictionary
        synthesized_audio = outputs['wav']

        # Save the synthesized audio to the output path
        src_path = f'{output_dir}/output.wav'
        sample_rate = xtts_config.audio.sample_rate
        sf.write(src_path, synthesized_audio, sample_rate)

        print("Audio generated successfully.")
        play_audio(src_path)
    except Exception as e:
        print(f"Error during audio generation: {e}")

def get_relevant_context(user_input, vault_embeddings, vault_content, model, top_k=6):
    """
    Retrieves the top-k most relevant context from the vault based on the user input.
    """
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the user input
    input_embedding = model.encode([user_input])
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0]
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def chatgpt_streamed(user_input, system_message, conversation_history, bot_name, vault_embeddings, vault_content, model):
    """
    Function to send a query to a Local model
    """
    # Get relevant context from the vault
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content, model)
    # Concatenate the relevant context with the user's input
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = "\n".join(relevant_context) + "\n\n" + user_input
    messages = [{"role": "system", "content": system_message}] + conversation_history + [{"role": "user", "content": user_input_with_context}]
    temperature = 1
    streamed_completion = client.chat.completions.create(
        model="local-model",
        messages=messages,
        stream=True
    )
    full_response = ""
    line_buffer = ""
    for chunk in streamed_completion:
        delta_content = chunk.choices[0].delta.content
        if delta_content is not None:
            line_buffer += delta_content
            if '\n' in line_buffer:
                lines = line_buffer.split('\n')
                for line in lines[:-1]:
                    print(NEON_GREEN + line + RESET_COLOR)
                    full_response += line + '\n'
                line_buffer = lines[-1]
    if line_buffer:
        print(NEON_GREEN + line_buffer + RESET_COLOR)
        full_response += line_buffer
    return full_response

# Function to transcribe the recorded audio using faster-whisper
def transcribe_with_whisper(audio_file):
    segments, info = whisper_model.transcribe(audio_file, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    return transcription.strip()

# Function to record audio from the microphone and save to a file
def record_audio(file_path):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []
    print("Recording...")
    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        pass
    print("Recording stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

# New function to handle a conversation with a user
def user_chatbot_conversation():
    conversation_history = []
    system_message = open_file("chatbot2.txt")
    # Load the sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Load the initial content from the vault.txt file
    vault_content = []
    if os.path.exists("vault.txt"):
        with open("vault.txt", "r", encoding="utf-8") as vault_file:
            vault_content = vault_file.readlines()
    # Create embeddings for the initial vault content
    vault_embeddings = model.encode(vault_content) if vault_content else []
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    while True:
        audio_file = "temp_recording.wav"
        record_audio(audio_file)
        user_input = transcribe_with_whisper(audio_file)
        os.remove(audio_file)  # Clean up the temporary audio file
        if user_input.lower() == "exit":  # Say 'exit' to end the conversation
            break
        elif user_input.lower().startswith(("print info", "Print info")):  # Print the contents of the vault.txt file
            print("Info contents:")
            if os.path.exists("vault.txt"):
                with open("vault.txt", "r", encoding="utf-8") as vault_file:
                    print(NEON_GREEN + vault_file.read() + RESET_COLOR)
            else:
                print("Info is empty.")
            continue
        elif user_input.lower().startswith(("delete info", "Delete info")):  # Delete the vault.txt file
            confirm = input("Are you sure? Say 'Yes' to confirm: ")
            if confirm.lower() == "yes":
                if os.path.exists("vault.txt"):
                    os.remove("vault.txt")
                    print("Info deleted.")
                    vault_content = []
                    vault_embeddings = []
                    vault_embeddings_tensor = torch.tensor(vault_embeddings)
                else:
                    print("Info is already empty.")
            else:
                print("Info deletion cancelled.")
            continue
        elif user_input.lower().startswith(("insert info", "Insert info")):
            print("Recording for info...")
            audio_file = "vault_recording.wav"
            record_audio(audio_file)
            vault_input = transcribe_with_whisper(audio_file)
            os.remove(audio_file)  # Clean up the temporary audio file
            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                vault_file.write(vault_input + "\n")
            print("Wrote to info.")
            # Update the vault content and embeddings
            vault_content = open("vault.txt", "r", encoding="utf-8").readlines()
            vault_embeddings = model.encode(vault_content)
            vault_embeddings_tensor = torch.tensor(vault_embeddings)
            continue
        print(CYAN + "You:", user_input + RESET_COLOR)
        conversation_history.append({"role": "user", "content": user_input})
        print(PINK + "Emma:" + RESET_COLOR)
        chatbot_response = chatgpt_streamed(user_input, system_message, conversation_history, "Chatbot", vault_embeddings_tensor, vault_content, model)
        conversation_history.append({"role": "assistant", "content": chatbot_response})
        prompt2 = chatbot_response
        audio_file_pth2 = "C:/Users/kris_/Python/fsts2/XTTS-v2/samples/emma2.wav"
        process_and_play(prompt2, audio_file_pth2)
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

user_chatbot_conversation()  # Start the conversation
