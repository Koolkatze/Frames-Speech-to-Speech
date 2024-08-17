# Frames-Speech-to-Speech
REQUIREMENTS:

- Windows 10/11 
- Python 3.10 https://www.python.org/downloads/release/python-3100/
- CUDA Toolkit 12.1 https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64
- CUDNN Library https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip/
- ffmpeg installed https://phoenixnap.com/kb/ffmpeg-windows or pip install ffmpeg
- NVIDIA GPU (will prob work with only CPU too)
- microphone
- local LLM setup (default is LM studio but working on OLlama to use WEB UI)
- You might need Pytorch (https://pytorch.org/) Use this command: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
- If an ERROR like this occurs: "Could not load library cudnn_ops_infer64_8.dll. Error code 126
  Please make sure cudnn_ops_infer64_8.dll is in your library path!"
  go to https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip/ take all the /bin files inside (.dll) and move them to
  your PC's C:\Users\ "INSERT YOUR USER HERE"\AppData\Local\Programs\Python\Python310\Lib\site-packages\torch\lib

HOW TO INSTALL:

1. git clone https://github.com/Koolkatze/Frames-Speech-to-Speech.git
2. git clone https://github.com/mphilli/English-to-IPA.git
- cd English-to-IPA
- python -m pip install .
- pip install cn2an -U
4. cd dir Frames-Speech-to-Speech
  - git clone https://github.com/myshell-ai/OpenVoice.git
  - python -m pip install .
5. pip3 install whisper-timestamped
6. pip install -r requirements.txt
7. download https://nordnet.blob.core.windows.net/bilde/checkpoints.zip
8. extract checkpoints.zip to Frames-Speech-to-Speech folder
9. on https://huggingface.co/coqui/XTTS-v2 download model
10. place XTTS-v2 folder in Frames-Speech-to-Speech folder
11. pip install --upgrade --force-reinstall ctranslate2==4.0
12. In talk3.py set your reference voice PATH (use a .mp3 extension file) on line 247
13. In xtalk.py:
- set PATH to config.json line 69
- set PATH to XTTS-v2 folder line 73
- set PATH to reference voice (use a .wav extension file) line 251
14. start LM studio server (or similar)
- Edit chatbot2.txt to create a Chat's Character personality.
- Edit vault.txt to create Chats Knowledge about yourself (or user).
15. run: python talk3.py (low latency version)
16. run: python xtalk.py (quality voice version)

ROADMAP:

1. Run a docker in the program to try and use LMStudio instead of OLlama (possible alternative).
2. Change LMStudio for OLlama to use its Web UI.
3. Read OLlamas output or Chatbot's answer inside OLlama and stream the text string to Frames by Brilliant Labs by using Brilliant Labs NOA Assistant and OLlama Web UI sharing the string info.
4. Using all the sensors inside Frames by Brilliant Labs (Camera, Movement/Gravity, Tap Buttons) to control and share info with OLlama and enhance the chatting experience.
5. Implementing video stream through the glasses camera to the preferred LLM inside OLlama or LMStudio (with Docker) to make a ChatGPT type of chatting with any opensource model.

