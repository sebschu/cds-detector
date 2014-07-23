import argparse
import sys
import os
import csv
import re
import shutil
import pickle
from tempfile import mkdtemp
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

sys.path.append("../src/python")

from pipeline_utils import extract_segments, extract_features, classify

CONF_DIR = "../config/"
MODEL_DIR = "../models"
T = 0.010         # Period per frame
SAMPLE_RATE = 16000

DEFAULT_OPTS = {
    "smooth_win": 150,  # Window size for smoothing (in 10ms)
    "threshold": 0.3,   # Threshold for "speech" detection
    "min_gap": 10,      # In frames
    "min_dur": 50      # In frames
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run pipeline.")
    parser.add_argument("audio_file", help="Audio file to run pipeline on.")
    config_help_msg = "Configuration to use (e.g. hw4, default: emobase2010)"
    parser.add_argument("config_name", help=config_help_msg,
                        default="emobase2010", nargs="?")
    parser.add_argument("--output_dir", help="Optional output directory, if not\
specified a temporary directory will be created.")
    
    args = parser.parse_args()

    if not os.path.isfile(args.audio_file):
        sys.exit("Audio file '{0}' not found\n".format(args.audio_file))

    if args.output_dir:
        if not os.path.isdir(args.output_dir):
            print ("Directory '{0}' does not exist, creating..."
                   .format(args.output_dir))
            os.makedirs(args.output_dir)
        work_dir = args.output_dir
    else:
        print "Creating temporary directory for output..."
        work_dir = mkdtemp()

    # Extract features for segment
    segment_list, numf, speech_prob = extract_segments(args.audio_file, work_dir,
                                          opts=DEFAULT_OPTS, conf_dir=CONF_DIR)



    print "Number of frames: {}".format(numf)

    noise_scaler = pickle.load(open(os.path.join(MODEL_DIR, "noise-scaler.p"), 'rb'))
    
    motherese_scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.p"), 'rb'))

    # Extract features for each segment. Collect results.
    pred_per_frame = np.zeros(numf/100)
    
    filtered_segments = []
    
    for seg in segment_list:
        feats = extract_features(seg["filename"], args.config_name,
                                 model_dir=MODEL_DIR, conf_dir=CONF_DIR)
        
        feats_scaled = noise_scaler.transform(feats)
        
        pred = classify(feats_scaled, "noise-rbf_svm", model_dir=MODEL_DIR)
        if pred:
            filtered_segments.append((seg, feats))
        else:
            print "\"{}\" is classified as noise.".format(seg["filename"])
        
    
    for seg, feats in filtered_segments:
        try:
            feats_scaled = motherese_scaler.transform(feats)
            pred = classify(feats_scaled, "rbf_nu_svm", model_dir=MODEL_DIR)
            if pred:
                for i in xrange(seg["startf"]/100, seg["endf"]/100 + 1):
                    pred_per_frame[i] = 1
        except:
            print "Unexpected error:", sys.exc_info()[0]
            print "Error extracting features; file may be too short. Skipping."

    audio_id = os.path.basename(os.path.splitext(args.audio_file)[0])
    
    if not args.output_dir:
        shutil.rmtree(work_dir)


   
    of = open(audio_id + "_cds.txt", "w")
    print >>of, "Time\tCDS"
    for f in range(numf/100):
        print >>of, "%d\t%d" % (f, pred_per_frame[f])
    
    