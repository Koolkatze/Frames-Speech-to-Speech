# Frames-Speech-to-Speech
REQUIREMENTS:

- Windows 10/11 
- Python 3.11 https://www.python.org/downloads/release/python-3110/
- CUDA Toolkit 11.8 to 12.4 https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
- CUDNN Library https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
- ffmpeg installed (https://phoenixnap.com/kb/ffmpeg-windows)
- NVIDIA GPU (will prob work with only CPU too)
- microphone
- local LLM setup (default is LM studio but working on OLlama to use WEB UI)
- You might need Pytorch (https://pytorch.org/) Use your own parameters to download your setup file.
- If an ERROR like this occurs: "Could not load library cudnn_ops_infer64_8.dll. Error code 126
  Please make sure cudnn_ops_infer64_8.dll is in your library path!"
  go to https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=Agnostic&cuda_version=12 take all the /bin files inside (.dll) and move them to
  your PC's C:\Users\"INSERT YOUR USER HERE"\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib

HOW TO INSTALL:

1. git clone https://github.com/Koolkatze/Frames-Speech-to-Speech.git
2. cd dir Frames-Speech-to-Speech
3. pip3 install whisper-timestamped
4. git clone https://github.com/myshell-ai/OpenVoice.git
5. pip install -r requirements.txt
6. download https://nordnet.blob.core.windows.net/bilde/checkpoints.zip
7. extract checkpoints.zip to Frames-Speech-to-Speech folder
8. on https://huggingface.co/coqui/XTTS-v2 download model
9. place XTTS-v2 folder in Frames-Speech-to-Speech folder
10. In talk3.py (openvoice version) set your reference voice PATH (use a .mp3 extension file) on line 247
11. In xtalk.py (xtts version):
- set PATH to config.json line 69
- set PATH to XTTS-v2 folder line 73
- set PATH to reference voice (use a .wav extension file) line 251
12. start LM studio server (or similar)
- Edit chatbot2.txt to create a Chat's Character personality.
- Edit vault.txt to create Chats Knowledge about yourself (or user).
13. run talk3.py (low latency version)
14. run xtalk.py (quality voice version)

