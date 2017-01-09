"""
Classes for various input sequences
"""
import random, json, os, sys, numpy as np
base_dir = '/mnt/data3/r2d2'

# logging
import gflags, glog # GLOG logging (https://pypi.python.org/pypi/glog/0.1)
FLAGS = gflags.FLAGS
FLAGS(sys.argv)
glog.init()
glog.setLevel(glog.INFO)

#------------------------------------------------------------------------------------------------------
# class providing textual descriptions of actions in videos, taken from MSR VTT (http://ms-multimedia-challenge.com/dataset)
# to get this data, download
# train http://ms-multimedia-challenge.com/static/resource/train_val_annotation.zip
# validation http://ms-multimedia-challenge.com/static/resource/test_videodatainfo.json
#------------------------------------------------------------------------------------------------------
class msr_vtt_textual_descriptions(object):
    CHARS = '.abcdefghijklmnopqrstuvwxyz0123456789 U'
    EOS = CHARS[0]
    BOS = CHARS[0]
    char_map = {c: CHARS.index(c) for c in CHARS}
    DIM = len(msr_vtt_textual_descriptions.CHARS)

    def __init__(self, phase):
        if phase.lower() == 'train':
            in_file_name = ''
            self.description = 'msr_vtt_textual_descriptions train'
        elif phase.lower() == 'test':
            in_file_name = ''
            self.description = 'msr_vtt_textual_descriptions test'
        # read once
        with open(os.path.join(base_dir, in_file_name), 'r') as R:
            inp = json.loads(R.read())
        self.sentences = [s['caption'].lower() for s in inp['sentences']]
        random.shuffle(self.sentences)
        self.print_stats()
        self.si = 0

    def print_stats(self):
        sl = np.array([len(s) for s in self.sentences])
        glog.info(
            '\nDataset stats for {}: {} seq with [{}..{}] <{}>_2, <{}>_1 chars/seq\n'.format(self.description, sl.size,
                                                                                             sl.min(), sl.max(),
                                                                                             sl.mean(), np.median(sl)))
    def __iter__(self):
        return self

    def next(self):
        if self.si >= len(self.sentences):
            glog.info('[{}] done epoch. reshuffling'.format(self.description))
            random.shuffle(self.sentences)
            self.si = 0
        return self.sentences[self.si]

#------------------------------------------------------------------------------------------------------