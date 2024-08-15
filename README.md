# Speech to Speech with RAG

### YouTube Tutorial:
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/aNGUTBFP_Wg/0.jpg)](https://www.youtube.com/watch?v=aNGUTBFP_Wg)

### Requirements for this setup:
- ffmpeg installed (https://phoenixnap.com/kb/ffmpeg-windows)
- NVIDIA GPU (will prob work with only CPU too)
- microphone
- local LLM setup (default is LM studio)
- You might need Pytorch (https://pytorch.org/)
- will add more if needed here

### How to install and setup:

1. git clone https://github.com/All-About-AI-YouTube/speech-to-rag.git
2. cd dir speech-to-rag
3. pip install -r requirements.txt
4. download https://nordnet.blob.core.windows.net/bilde/checkpoints.zip
5. extract checkpoints.zip to speech-to-rag folder
6. on https://huggingface.co/coqui/XTTS-v2 download model
7. place XTTS-v2 folder in speech-to-rag folder
8. In talk3.py (openvoice version) set your reference voice PATH on line 247
9. In xtalk.py (xtts version):
- set PATH to config.json line 69
- set PATH to XTTS-v2 folder line 73
- set PATH to reference voice line 251
10. start LM studio server (or similar)
11. run talk3.py (low latency version)
12. run xtalk.py (quality voice version)

