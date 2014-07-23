# Automatic Child-Directed Speech Detector 


## Installation

You have to install Python 2.7 and pip to run the detector


```bash
# Install requirements
pip install -r requirements.txt
```

# Running the system

To detect child directed speech in an audio file, run the following commands:
```bash
cd scripts
python pipeline.py PATH_TO_AUDIO_FILE.wav
```

Note that the audio files has to be a mono 16-bit wave file with 16khz sample 
rate. You can use tools like [sox](http://sox.sourceforge.net/) to convert the
audio, if necessary. 
