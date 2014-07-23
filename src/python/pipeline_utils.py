import os
import pickle
import numpy as np
import random
import csv
import shutil
from subprocess import call
from tempfile import mkdtemp

EXEC_NAME = "SMILExtract"
SOX = "sox"
CONF_NAME = "prosodyAcf"
T = 0.010         # Period per frame


def extract_features(wav_file, config_name, conf_dir, model_dir,
                     work_dir=None):

    # create temp dir
    if not work_dir:
        tmp_dir = mkdtemp()
    else:
        tmp_dir = work_dir

    # load scaler
    #scaler = pickle.load(open(os.path.join(model_dir, "scaler.p"), 'rb'))

    out_fbase = os.path.join(tmp_dir, "{}-{}".
                             format(random.randint(0, 10000), config_name))
    if os.path.isfile(out_fbase + ".csv"):
        os.remove(out_fbase + ".csv")

    print "Extracting features for " + wav_file + "..."
    call(EXEC_NAME + " -C {}/{}.conf".format(conf_dir, config_name)
         + " -O {}.csv".format(out_fbase)
         + " -I " + wav_file + " -noconsoleoutput", shell=True)

    feats = np.loadtxt(out_fbase + ".csv", skiprows=1, delimiter=',')[1:]
    #feats = scaler.transform(feats)

    # remove temp file
    if not work_dir:
        shutil.rmtree(tmp_dir)

    return [feats]


def classify(feats, clf_name, model_dir):
    y = batch_classify(feats, clf_name, model_dir)
    return y[0]

def batch_classify(feats, clf_name, model_dir):
    clf = pickle.load(open(os.path.join(model_dir, clf_name + ".p"), 'rb'))
    y = clf.predict(feats)
    return y

def extract_segments(wav_file, output_dir, opts, conf_dir, work_dir=None):
    base_name = os.path.splitext(os.path.basename(wav_file))[0]
    # Generate features
    if not work_dir:
        tmp_dir = mkdtemp()
    else:
        tmp_dir = work_dir

    feats_file = os.path.join(tmp_dir, "{}-{}.csv"
                              .format(base_name, CONF_NAME))
    print "Extracting speech features..."
    call(EXEC_NAME + " -C {}/{}.conf".format(conf_dir, CONF_NAME)
         + " -O " + feats_file
         + " -I " + wav_file + " -noconsoleoutput", shell=True)

    print "Extracting segments..."
    # Read features file and discard temporary directory
    with open(feats_file, 'r') as f:
        rdr = csv.DictReader(f, delimiter=',')
        speech_prob = np.array([float(r['voiceProb_sma']) for r in rdr])
    if not work_dir:
        shutil.rmtree(tmp_dir)

    # Detect speech
    win = np.ones(opts["smooth_win"],)  # window for smoothing
    speech_prob_smooth = np.convolve(speech_prob, win / opts["smooth_win"],
                                     'same')
    is_speech = speech_prob_smooth > opts["threshold"]

    # Find points of transition between speech and non-speech
    changepts = np.flatnonzero(is_speech[1:] - is_speech[:-1])
    num_segs = changepts.size / 2

    # Merge consecutive with small gaps (< 5 frames)
    gaps = (changepts[[i for i in range(1, num_segs * 2 - 1) if i % 2 == 0]]
            - changepts[[i for i in range(1, num_segs * 2 - 1) if i % 2 != 0]])
    merge_idx = np.where(gaps < opts["min_gap"])[0]
    for idx in merge_idx:
        st = changepts[2 * (idx + 1) - 1] - 1
        en = changepts[2 * (idx + 1)] + 1
        is_speech[st:en] = 1

    # Eliminate islands that are too small (< .3 seconds)
    while True:
        durations = (changepts[range(1, num_segs * 2, 2)]
                     - changepts[range(0, num_segs * 2, 2)])
        eliminate_idx = np.where(durations < opts["min_dur"])[0]
        if len(eliminate_idx) < 1:
            break
        for idx in eliminate_idx:
            st = changepts[2 * idx] - 1
            en = changepts[2 * idx + 1] + 1
            is_speech[st:en] = 0

        # Determine new points of transition
        changepts = np.flatnonzero(is_speech[1:] - is_speech[:-1])
        num_segs = changepts.size / 2

    # Format for CSV and create new file
    seg_file = os.path.join(output_dir, "{}_segments.csv".format(base_name))
    segment_list = []
    with open(seg_file, 'w') as fh:
        fh.write("Label,Start,End\n")
        for i in range(num_segs):
            st = changepts[2 * i]
            en = changepts[2 * i + 1]
            length = float(en - st)
            seg_count = max(1, int(length / 1000000))
            for j in range(seg_count):
                st_local = st + (j * length / seg_count)
                en_local = st + ((j + 1) * length / seg_count)
                print "Extracting segment {}: {} to {}".format(i, st_local * T, en_local * T)
                new_id = base_name + "_%d_%d" % (st_local, en_local)
                new_file = os.path.join(output_dir, "{}.wav".format(new_id))
                cmd = ("%s %s -c 1 -r 16000 %s trim =%.2f =%.2f"
                   % (SOX, wav_file, new_file, st_local * T, en_local * T))
                call(cmd, shell=True)
                fh.write("%s,%0.2f,%0.2f\n" % (new_id, st_local * T, en_local * T))
                segment_list.append({"filename": new_file,
                                 "startf": st, "endf": en})
    return (segment_list, len(speech_prob), speech_prob)
