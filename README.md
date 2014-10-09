# Automatic Child-Directed Speech Detector 


## Installation

You have to install Python 2.7 and pip to run the detector. 

Then run the following command to install the required python packages.

```bash
# Install requirements
pip install -r requirements.txt
```

You also need to install [sox](http://sox.sourceforge.net/) and 
[openSMILE](http://opensmile.sourceforge.net/) to run the pipeline.

## Running the system

To detect child directed speech in an audio file, run the following commands:
```bash
cd scripts
python pipeline.py PATH_TO_AUDIO_FILE.wav
```

Note that the audio files has to be a mono 16-bit wave file with 16khz sample 
rate. You can use tools like [sox](http://sox.sourceforge.net/) to convert the
audio, if necessary. 

The pipeline writes the annotations to the file AUDION_NAME_cds.txt.

## Citations

If you use the detector for research purposes, then we ask that you cite the following paper:

```latex
@inproceedings{cds2014,
 author = {Sebastian Schuster and Stephanie Pancoast and Milind Ganjoo and Michael C. Frank and Dan Jurafsky},
 title = {Speaker-Independent Detection of Child-Directed Speech},
 booktitle = {Proceedings of the IEEE Workshop on Spoken Language Technology},
 year = {2014}
}
```

