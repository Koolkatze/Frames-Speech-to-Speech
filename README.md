# Frames-Speech-to-Speech
REQUIREMENTS:

- CUDA Toolkit 11.8 to 12.4 https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
- ffmpeg installed (https://phoenixnap.com/kb/ffmpeg-windows)
- NVIDIA GPU (will prob work with only CPU too)
- microphone
- local LLM setup (default is LM studio but working on OLlama to use WEB UI)
- You might need Pytorch (https://pytorch.org/) Use your own parameters to download your setup file.
- will add more if needed here

HOW TO INSTALL:

1. git clone https://github.com/Koolkatze/Frames-Speech-to-Speech.git
2. cd dir Frames-Speech-to-Speech
3. pip install -r requirements.txt
4. download https://nordnet.blob.core.windows.net/bilde/checkpoints.zip
5. extract checkpoints.zip to Frames-Speech-to-Speech folder
6. on https://huggingface.co/coqui/XTTS-v2 download model
7. place XTTS-v2 folder in Frames-Speech-to-Speech folder
8. git clone https://github.com/myshell-ai/OpenVoice.git
9. In talk3.py (openvoice version) set your reference voice PATH (use a .mp3 extension file) on line 247
10. In xtalk.py (xtts version):
- set PATH to config.json line 69
- set PATH to XTTS-v2 folder line 73
- set PATH to reference voice (use a .wav extension file) line 251
10. start LM studio server (or similar)
11. run talk3.py (low latency version)
12. run xtalk.py (quality voice version)

