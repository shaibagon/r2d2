"""
Classes for various input sequences
"""
import random, json, os, sys, traceback, numpy as np
from threading import Thread
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
class seq_globals_t(object):
    def __init__(self):
        self.EOS = None  # end of sequence (index to charmap)
        self.BOS = None  # beginning of sequence (index to charmap)
        self.UNK = None  # unknown char (index to charmap)
        self.DIM = None
        self.CHARS = None
        self.CHAR_MAP = None
        self.seqIterator = None
        self.singleSequenceProvider = None

    def __str__(self):
        str = ''
        for a in ['eos','bos','unk','dim','chars','char_map']:
            str += '{}={}\n'.format(a, self.__getattribute__(a.upper()))
        return str

SEQ_GLOBALS = seq_globals_t()

class msr_vtt_textual_descriptions(object):
    def __init__(self):
        global SEQ_GLOBALS
        # attributes
        SEQ_GLOBALS.CHARS = '.abcdefghijklmnopqrstuvwxyz0123456789 U'
        SEQ_GLOBALS.EOS = 0
        SEQ_GLOBALS.BOS = 0
        SEQ_GLOBALS.UNK = len(SEQ_GLOBALS.CHARS)-1
        SEQ_GLOBALS.DIM = len(SEQ_GLOBALS.CHARS)
        SEQ_GLOBALS.CHAR_MAP = {c:i for i,c in enumerate(SEQ_GLOBALS.CHARS)}
        SEQ_GLOBALS.seqIterator = msr_vtt_textual_descriptions.seqIterator
        SEQ_GLOBALS.singleSequenceProvider = msr_vtt_textual_descriptions.singleSequenceProvider

    def __str__(self):
        return 'GLOBALS\nchars={}\neos={}\nbos={}\nunk={}\ndim={}\nchar_map={}\n'.format(SEQ_GLOBALS.CHARS,
                                                                                         SEQ_GLOBALS.EOS,
                                                                                         SEQ_GLOBALS.BOS,
                                                                                         SEQ_GLOBALS.UNK,
                                                                                         SEQ_GLOBALS.DIM,
                                                                                         SEQ_GLOBALS.CHAR_MAP)

    class seqIterator(object):
        def __init__(self, phase):
            if phase.lower() == 'train':
                in_file_name = 'train_val_videodatainfo.json'
                self.description = 'msr_vtt_textual_descriptions train'
            elif phase.lower() == 'test':
                in_file_name = 'test_videodatainfo.json'
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
            s = self.sentences[self.si]
            self.si += 1
            return s

    class singleSequenceProvider(Thread):
        """
        get sequences from inQ and output chunks to sequenceQ
        """

        def __init__(self, inQ, outQ, seq_len):
            Thread.__init__(self)
            self.inQ = inQ
            self.outQ = outQ
            self.seq_len = seq_len
            self.daemon = True
            self.start()

        def run(self):
            to_data = []
            to_label = []
            while True:
                try:
                    while len(to_data) < self.seq_len:
                        s = self.inQ.get()
                        l = [SEQ_GLOBALS.CHAR_MAP.get(c, SEQ_GLOBALS.UNK) for c in s]
                        self.inQ.task_done()
                        to_data.append(SEQ_GLOBALS.BOS)
                        to_data += l
                        to_label += l
                        to_label.append(SEQ_GLOBALS.EOS)
                    # we have enough data to push to batch
                    while len(to_data) >= self.seq_len:
                        data = np.zeros((self.seq_len, 1, SEQ_GLOBALS.DIM), dtype='f4')
                        data[range(self.seq_len), 0, to_data[:self.seq_len]] = 1.0
                        assert data.sum() == self.seq_len, "wrong hot vectors {} ones for {} rows".format(data.sum(),
                                                                                                          self.seq_len)
                        label = np.array(to_label[:self.seq_len])[:, None]
                        self.outQ.put((data.copy(), label.copy()))
                        del to_data[:self.seq_len]
                        del to_label[:self.seq_len]
                except Exception as e:
                    glog.error('singleSequenceProvider: got error ({}): {}\n{}'.format(type(e).__name__, e,
                                                                                       traceback.format_exc()))

#------------------------------------------------------------------------------------------------------
def get_configuration(config='msr-vtt-v0'):
    if config == 'msr-vtt-vnl':
        msr_vtt_textual_descriptions()
        configuration = {
            'variant': 'vnl',
            'input_params': {'train': {'seq_len': 150, 'batch_size': 75},
                             'test': {'seq_len': 50, 'batch_size': 50}},
            'layer_dims': [100, 75, 75, 75, SEQ_GLOBALS.DIM],
            'base_dir': base_dir,
            'test_niter': 1000,  # number of test iterations
            'test_interval': 5000,  # when to snap and test
            'train_niter': 100000,  # number of train iterations
            'base_lr': 0.01,
            'debug': False
        }
    elif 'msr-vtt' in config:
        msr_vtt_textual_descriptions()
        if len(config.split('-')) > 1:
            variant = config.split('-')[-1]
        else:
            variant = 'v0'
        configuration = {
            'variant': variant,
            'input_params': {'train': {'seq_len': 150, 'batch_size': 75},
                             'test': {'seq_len': 50, 'batch_size': 50}},
            'layer_dims': [100, 100, 100, 100, SEQ_GLOBALS.DIM],
            'base_dir': base_dir,
            'test_niter': 1000,  # number of test iterations
            'test_interval': 5000,  # when to snap and test
            'train_niter': 100000,  # number of train iterations
            'base_lr': 0.01,
            'debug': False
        }
    elif 'debug' in config:  # 'debug-v0', 'debug-v1' ...
        if len(config.split('-'))>1:
            variant = config.split('-')[-1]
        else:
            variant = 'v0'
        # dummy source for creating small net for debug/draw
        msr_vtt_textual_descriptions()
        configuration = {
            'variant': variant,
            'input_params' : {'train': {'seq_len': 3, 'batch_size': 10}, # unroll 3 time steps...
                              'test':  {'seq_len': 3,  'batch_size': 10}},
            'layer_dims': [100, 100, 100, SEQ_GLOBALS.DIM], # only two recurrent unit
            'base_dir': base_dir,
            'test_niter': 1, # number of test iterations
            'test_interval': 5000, # when to snap and test
            'train_niter': 1, # number of train iterations
            'base_lr': 0.001,
            'debug': True
        }
    else:
        raise Exception('unknown config {}.'.format(config))
    return configuration