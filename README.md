# Frames-Speech-to-Speech
REQUIREMENTS:

- Windows 10/11 
- Python 3.10 https://www.python.org/downloads/release/python-3100/
- CUDA Toolkit 11.8 https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64 Select windows version and .exe(local).
- ffmpeg installed https://phoenixnap.com/kb/ffmpeg-windows or pip install ffmpeg
- NVIDIA GPU (will prob work with only CPU too)
- microphone
- local LLM setup (default is LM studio but working on OLlama to use WEB UI)
- You might need Pytorch (https://pytorch.org/) (Included in HOW TO INSTALL)
- If an ERROR like this occurs: "Could not load library cudnn_ops_infer64_8.dll. Error code 126"
  Please make sure cudnn_ops_infer64_8.dll is in your library path!"
  go to https://github.com/Purfview/whisper-standalone-win/releases/tag/libs download "cuBLAS.and.cuDNN_CUDA11_win_v2.zip take all the files inside .zip (.dll) and move them to
  your PC's C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
- If an ERROR like this occurs: "Could not load library cublas64_12.dll.": Go to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin, take cublas64_11.dll make a copy of it and rename it cublas64_12.dll

HOW TO INSTALL:

Use Allways Windosws PowerShell Terminall

0. pip install ffmpeg
1. git clone https://github.com/Koolkatze/Frames-Speech-to-Speech.git
3. cd dir Frames-Speech-to-Speech
4. pip install -r requirements.txt
5. pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/test/cu118
6. pip install numpy==1.22.0
7. Coffe Break...
8. cd Frames-Speech-to-Speech
9. download https://nordnet.blob.core.windows.net/bilde/checkpoints.zip
10. extract checkpoints.zip to Frames-Speech-to-Speech folder
11. on https://huggingface.co/coqui/XTTS-v2 download model
12. place XTTS-v2 folder in Frames-Speech-to-Speech folder
13. In talk3.py set your reference voice PATH (use a .mp3 extension file) on line 247
14. In xtalk.py:
- set PATH to config.json line 69
- set PATH to XTTS-v2 folder line 73
- set PATH to reference voice (use a .wav extension file) line 251
14. start LM studio server (or similar)
- Edit chatbot2.txt to create a Chat's Character personality.
- Edit vault.txt to create Chats Knowledge about yourself (or user).
15. run: python talk3.py (low latency version)
16. run: python xtalk.py (quality voice version)

To clone a voice using OpenVoice just use cd Frames-Speech-to-Speech and then python openvoice_app.py
after that use the https:// Link provided by the app on your internet Browser and follow the instructions.

ROADMAP:

0. pip install ffmpeg
1. Run a docker in the program to try and use LMStudio instead of OLlama (possible alternative).
2. Change LMStudio for OLlama to use its Web UI.
3. Read OLlamas output or Chatbot's answer inside OLlama and stream the text string to Frames by Brilliant Labs by using Brilliant Labs NOA Assistant and OLlama Web UI sharing the string info.
4. Using all the sensors inside Frames by Brilliant Labs (Camera, Movement/Gravity, Tap Buttons) to control and share info with OLlama and enhance the chatting experience.
5. Implementing video stream through the glasses camera to the preferred LLM inside OLlama or LMStudio (with Docker) to make a ChatGPT type of chatting with any opensource model.

