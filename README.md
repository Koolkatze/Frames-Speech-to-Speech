# Frames-Speech-to-Speech
This is an low latency, uncensored, local, portable, speech to speech with easy voice clonation and with streamable text for Frames AR Glasses from Brilliant Labs (WIP).

REQUIREMENTS:

1. CUDA 11.8 to 12.4
2. ffmpeg installed (https://phoenixnap.com/kb/ffmpeg-windows)
3. CUDA 11.8 to 12.4 https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
4. NVIDIA GPU (will prob work with only CPU too)
5. microphone
6. local LLM setup (default is LM studio but this could change to Ollama soon as it has a browser interface experience) Download any uncensored LLM of your preference
7. You might need Pytorch (https://pytorch.org/) Use your specific parameters to download your version.
8. will add more if needed here

HOW TO INSTALL:
git clone https://github.com/Koolkatze/Frames-Speech-to-Speech.git
cd dir Frames-Speech-to-Speech
pip install -r requirements.txt (if error check your CUDA toolkit version requirement satisfied, uninstall CUDA and install correct CUDA toolkit).
download https://nordnet.blob.core.windows.net/bilde/checkpoints.zip
extract checkpoints.zip to Frames-Speech-to-Speech folder
on https://huggingface.co/coqui/XTTS-v2 download model
place XTTS-v2 folder in Frames-Speech-to-Speech folder
git clone https://github.com/myshell-ai/OpenVoice.git
In talk3.py (openvoice version) set your reference voice PATH (your audio should be .mp3 extension)  on line 247
In xtalk.py (xtts version):
set PATH to config.json line 69
set PATH to XTTS-v2 folder line 73
set PATH to reference voice (your audio should be .wav extension) line 251
start LM studio server (or similar)
run talk3.py (low latency version)
run xtalk.py (quality voice version)

ROADMAP:

1. Implement Ollama Web UI experience
2. Print the string of text for the Ollama LLM answer.
3. Make voice command for changing voices on the go (including their own individual PROMPT or CHARACTER).
4. Make voice command for changing LLM Models on the go.
5. Use Frame AR Glasses sensors to send info (Camera, Movement/Gravity sensors, Tap buttons).
6. Make this info profitable by creating a Multimodal video-streaming/Speech-to-speech with OpenAI's ChatGPT 4o type of interaction with the world.
