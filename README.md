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
  go to Frames-Speech-to-Speech/cudnn-windows-x86_64-8.9.7.29_cuda12-archive/bin/ take all the files inside (.dll) and move them to
  your PC's C:\Users\"INSERT YOUR USER HERE"\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib

HOW TO INSTALL:

1. git clone https://github.com/Koolkatze/Frames-Speech-to-Speech.git
2. cd dir Frames-Speech-to-Speech
3. git clone https://github.com/linto-ai/whisper-timestamped.git
4. pip install -r requirements.txt
5. download https://nordnet.blob.core.windows.net/bilde/checkpoints.zip
6. extract checkpoints.zip to Frames-Speech-to-Speech folder
7. on https://huggingface.co/coqui/XTTS-v2 download model
8. place XTTS-v2 folder in Frames-Speech-to-Speech folder
9. git clone https://github.com/myshell-ai/OpenVoice.git
10. In talk3.py (openvoice version) set your reference voice PATH (use a .mp3 extension file) on line 247
11. In xtalk.py (xtts version):
- set PATH to config.json line 69
- set PATH to XTTS-v2 folder line 73
- set PATH to reference voice (use a .wav extension file) line 251
10. start LM studio server (or similar)
11. run talk3.py (low latency version)
12. run xtalk.py (quality voice version)

