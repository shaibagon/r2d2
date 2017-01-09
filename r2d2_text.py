import sys, os, json, time, random, numpy as np
sys.path.insert(0, os.environ['CAFFE_ROOT']+'/python')
#os.environ['GLOG_minloglevel'] = '1'
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
from caffe import layers as L, params as P

from threading import Thread, Lock
from Queue import Queue
import traceback

# logging
import gflags, glog # GLOG logging
FLAGS = gflags.FLAGS
FLAGS(sys.argv)
glog.init()
glog.setLevel(glog.INFO)

base_dir = '/mnt/data3/r2d2'

char_map = {
    '.': 0, #EOS/BOS
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5,
    'f': 6,
    'g': 7,
    'h': 8,
    'i': 9,
    'j': 10,
    'k': 11,
    'l': 12,
    'm': 13,
    'n': 14,
    'o': 15,
    'p': 16,
    'q': 17,
    'r': 18,
    's': 19,
    't': 20,
    'u': 21,
    'v': 22,
    'w': 23,
    'x': 24,
    'y': 25,
    'z': 26,
    '0': 27,
    '1': 28,
    '2': 29,
    '3': 30,
    '4': 31,
    '5': 32,
    '6': 33,
    '7': 34,
    '8': 35,
    '9': 36,
    ' ': 37,
    'U': 38 } # unknown character
SORTED_CHARS = sorted(char_map.keys(), key=lambda k:char_map[k])
DIM = len(char_map)

#------------------------------------------------------------------------------------------------------
# threaded input functionality
#------------------------------------------------------------------------------------------------------
class dispatchThread(Thread):
    """
    read input file and output sentences to Q
    """
    def __init__(self, sentencesQ, in_file_name):
        Thread.__init__(self)
        self.daemon = True
        self.sentencesQ = sentencesQ
        self.in_file_name = in_file_name
        self.start()

    def run(self):
        # read once
        with open(os.path.join(base_dir, self.in_file_name), 'r') as R:
            inp = json.loads(R.read())
        sentences = [s['caption'] for s in inp['sentences']]
        self.print_stats(sentences)
        while True:
            try:
                random.shuffle(sentences)
                for s in sentences:
                    self.sentencesQ.put(s.lower())
                glog.info('dispatchThread: [{}] done epoch. reshuffling'.format(self.in_file_name))
            except Exception as e:
                glog.error('dispatchThread: got error ({}): {}\n{}'.format(type(e).__name__, e, traceback.format_exc()))

    def print_stats(self, sentences):
        sl = np.array([len(s) for s in sentences])
        glog.info('\nDataset stats for {}: {} seq with [{}..{}] <{}>_2, <{}>_1 chars/seq\n'.format(self.in_file_name, sl.size, sl.min(), sl.max(), sl.mean(), np.median(sl)))

class singleSequenceProvider(Thread):
    """
    get sequences from inQ and output chunks to sequenceQ
    """
    def __init__(self, inQ, outQ, seq_len, bi=None):
        Thread.__init__(self)
        self.inQ = inQ
        self.outQ = outQ
        self.seq_len = seq_len
        self.bi = bi
        self.daemon = True
        self.start()

    def run(self):
        to_data = []
        to_label = []
        while True:
            try:
                while len(to_data) < self.seq_len:
                    s = self.inQ.get()
                    l = [char_map.get(c, DIM-1) for c in s]
                    self.inQ.task_done()
                    to_data.append(char_map['.'])
                    to_data += l
                    to_label += l
                    to_label.append(char_map['.'])
                # we have enough data to push to batch
                while len(to_data) >= self.seq_len:
                    data = np.zeros((self.seq_len,1,DIM), dtype='f4')
                    data[range(self.seq_len),0,to_data[:self.seq_len]] = 1.0
                    assert data.sum()==self.seq_len, "wrong hot vectors {} ones for {} rows".format(data.sum(),self.seq_len)
                    label = np.array(to_label[:self.seq_len])[:,None]
                    self.outQ.put((data.copy(),label.copy()))
                    del to_data[:self.seq_len]
                    del to_label[:self.seq_len]
            except Exception as e:
                glog.error('singleSequenceProvider: got error ({}): {}\n{}'.format(type(e).__name__, e, traceback.format_exc()))

class batchProvider(Thread):
    def __init__(self, inQs, outQ):
        Thread.__init__(self)
        self.inQs = inQs
        self.outQ = outQ
        self.daemon = True
        self.start()

    def run(self):
        while True:
            try:
                data = []
                label = []
                for bi in xrange(len(self.inQs)):
                    d, l = self.inQs[bi].get()
                    data.append(d)
                    label.append(l)
                    self.inQs[bi].task_done()
                self.outQ.put({'data':  np.concatenate(data, axis=1),
                               'label': np.concatenate(label, axis=1)})
                # glog.info('batchProvider: provided batch {},{} {},{}'.format(len(data), data[0].shape,
                #                                                              len(label), label[0].shape))
            except Exception as e:
                glog.error('batchProvider: got error ({}): {}\n{}'.format(type(e).__name__, e, traceback.format_exc()))

class input_text_layer(caffe.Layer):
    def setup(self, bottom, top):
        # params
        self.params = json.loads(self.param_str)
        self.batch_size = int(self.params['batch_size'])
        self.seq_len = int(self.params['seq_len'])
        assert len(bottom) == 0, "input_text_layer: input layer no bottoms"
        assert len(top) == 2*self.seq_len+1, "input_text_layer: 2*{}+1 tops".format(self.seq_len)
        # set threads in motion
        # thread to read from file
        self.sentencesQ = Queue(5000)
        dispatchThread(self.sentencesQ, self.params['source'])
        # threads to write per parralele seq in minibatch
        self.singleQs = []
        self.batchQ = Queue(100)
        for bi in xrange(self.batch_size):
            self.singleQs.append( Queue(100) )
            singleSequenceProvider(self.sentencesQ, self.singleQs[-1], self.seq_len, bi)
        batchProvider(self.singleQs, self.batchQ)
        self.iter_count = 0

    def q_status(self, iter_):
        avg_qs = sum([q.qsize() for q in self.singleQs])/float(len(self.singleQs))
        glog.info('Q STATUS [{}] at iter={}: |sentencesQ|={} <|singleQs|>={} |batchQ|={}'.format(self.params['source'], iter_,
                                                                                            self.sentencesQ.qsize(),
                                                                                            avg_qs, self.batchQ.qsize()))

    def reshape(self, bottom, top):
        for t in xrange(self.seq_len):
            # reshape all X,
            top[t].reshape(self.batch_size, DIM)
            # reshape all reset layer
            top[t + self.seq_len].reshape(self.batch_size)
        # reshape all labels
        top[2*self.seq_len].reshape(self.seq_len, self.batch_size)

    def forward(self, bottom, top):
        if (self.iter_count % 100)==0:
            self.q_status(self.iter_count)
        self.iter_count+=1
        batch = self.batchQ.get(block=True, timeout=60) # do not wait longer than 60 sec for a batch...
        data = batch['data'].copy()
        label = batch['label'].copy()
        self.batchQ.task_done()
        for t in xrange(self.seq_len):
            # data
            top[t].data[...] = data[t,:,:]
            # reset
            top[t+self.seq_len].data[...] = 1.0-data[t,:,char_map['.']] # seq with s[t]==BOS reset is 0
        top[2*self.seq_len].data[...] = label

    def backward(self, top, propagate_down, bottom):
        # no back prop for input layer
        pass
#------------------------------------------------------------------------------------------------------
# handling recurrent for train/validation
#------------------------------------------------------------------------------------------------------
class reccurence_handler(caffe.Layer):
    def setup(self, bottom, top):
        params = json.loads(self.param_str)
        self.output_shapes = params['h_out_shapes']
        self.debug = params.get('debug', False)
        self.net = None
        assert len(top) == len(self.output_shapes), \
            "reccurence_handler: must output same number of h_shapes {} != {}".format(len(top), len(self.output_shapes))

    def reshape(self, bottom, top):
        for i, s in enumerate(self.output_shapes):
            top[i].reshape(*s)

    def forward(self, bottom, top):
        assert self.net is not None, "reccurence_handler: must call set_net before first forward"
        dbg_str = ''
        for i, nm in enumerate(self.output_names):
            d = self.net.blobs[nm].data.copy()
            if self.debug:
                df = self.net.blobs[nm].diff.copy()
                dbg_str += ' %s L1(%.2f, %.2f) L2(%.2f, %.2f)' % (nm, np.abs(d).sum(), np.abs(df).sum(),
                                                                  (d ** 2).sum(), (df ** 2).sum())
            top[i].data[...] = d
        if self.debug:
            glog.info('reccurence_handler.forward: {}'.format(dbg_str))

    def backward(self, top, propagate_down, bottom):
        pass # no backprop

    def reset_hiddens(self):
        for i, nm in enumerate(self.output_names):
            self.net.blobs[nm].data[...] = np.zeros(self.net.blobs[nm].data.shape, dtype='f4')

    def set_net(self, net, prev_t_outout_blob_names):
        self.output_names = prev_t_outout_blob_names
        assert len(self.output_names) == len(self.output_names), \
            "reccurence_handler: must output same number of h_names {} != {}".format(len(top), len(self.output_names))
        self.net = net
        self.reset_hiddens()

#------------------------------------------------------------------------------------------------------
# Building blocks for R2D2
#------------------------------------------------------------------------------------------------------
def residual_block(ns_, in_, name_prefix_, weight_prefix_, dim_):
    fc = L.InnerProduct(in_, name=name_prefix_+'_fc',
                   inner_product_param={'bias_term': True,
                                        'num_output': dim_,
                                        'axis': -1,
                                        'weight_filler': {'type':'gaussian', 'std':0.01},
                                        'bias_filler': {'type':'constant', 'value': 0}},
                   param=[{'lr_mult': 1, 'decay_mult': 1, 'name': weight_prefix_+'_fc_w'},
                          {'lr_mult': 2, 'decay_mult': 0, 'name': weight_prefix_+'_fc_b'}])
    ns_.__setattr__(name_prefix_+'_fc', fc)
    bn = L.BatchNorm(fc, name=name_prefix_+'_bn',
                     param=[{'name': weight_prefix_ + '_bn_mu', 'lr_mult':0, 'decay_mult': 0},
                            {'name': weight_prefix_ + '_bn_sig', 'lr_mult': 0, 'decay_mult': 0},
                            {'name': weight_prefix_ + '_bn_eps', 'lr_mult': 0, 'decay_mult': 0}])
    ns_.__setattr__(name_prefix_+'_bn', bn)
    prelu = L.PReLU(bn, name=name_prefix_+'_prelu',
                    prelu_param={'channel_shared': False, 'filler': {'type': 'constant', 'value': 0.01}}, # share channels to avoid mess
                    param={'lr_mult': 1, 'decay_mult': 50, 'name': weight_prefix_+'_prelu'})
    ns_.__setattr__(name_prefix_+'_prelu', prelu)
    return ns_, prelu, name_prefix_+'_prelu'

def one_time_step_v0(ns_, x_t, hidden_prev_t_list, prefix_):
    """
    v0 variant: top of unit_t-1 is added as residual to *bottom* of unit_t
    :param ns_:
    :param x_t:
    :param hidden_prev_t_list:
    :param prefix_:
    :return:
    """
    # pass through an initial res block
    ns_, o, _ = residual_block(ns_, x_t, prefix_+'_in', 'rin', dim_=100)
    outputs = []
    for li in xrange(len(hidden_prev_t_list)):
        # residual here
        e = L.Eltwise(o, hidden_prev_t_list[li][1], name=prefix_+'_res{}'.format(li),eltwise_param={'operation':P.Eltwise.SUM})
        ns_.__setattr__(prefix_+'_res{}'.format(li), e)
        ns_, o, h_t_name = residual_block(ns_, e, prefix_+'_r{}'.format(li), 'v0_r{}'.format(li), dim_=100)
        outputs.append((h_t_name, o))
    # prediction layer
    ns_, out, out_name = residual_block(ns_, outputs[-1][1], prefix_ + '_pred', 'pred', dim_=DIM)
    outputs.append((out_name, out))
    return ns_, outputs

def one_time_step_v1(ns_, x_t, hidden_prev_t_list, prefix_):
    """
    v1 variant: top of unit_t-1 (BEFORE residual update) is added as residual to *top* of unit_t
    :param ns_:
    :param x_t:
    :param hidden_prev_t_list:
    :param prefix_:
    :return:
    """
    # pass through an initial res block
    ns_, e, _ = residual_block(ns_, x_t, prefix_+'_in', 'rin', dim_=100)
    outputs = []
    for li in xrange(len(hidden_prev_t_list)):
        ns_, o, h_t_name = residual_block(ns_, e, prefix_+'_r{}'.format(li), 'v1_r{}'.format(li), dim_=100)
        outputs.append((h_t_name, o))
        # residual here
        e = L.Eltwise(o, hidden_prev_t_list[li][1], name=prefix_ + '_res{}'.format(li),
                      eltwise_param={'operation': P.Eltwise.SUM})
        ns_.__setattr__(prefix_ + '_res{}'.format(li), e)
    # prediction layer
    ns_, out, out_name = residual_block(ns_, e, prefix_ + '_pred', 'pred', dim_=DIM)
    outputs.append((out_name, out))
    return ns_, outputs

def one_time_step_v2(ns_, x_t, hidden_prev_t_list, prefix_):
    """
    v2 variant: top of unit_t-1 (AFTER residual update) is added as residual to *top* of unit_t
    :param ns_:
    :param x_t:
    :param hidden_prev_t_list:
    :param prefix_:
    :return:
    """
    # pass through an initial res block
    ns_, e, _ = residual_block(ns_, x_t, prefix_+'_in', 'rin', dim_=100)
    outputs = []
    for li in xrange(len(hidden_prev_t_list)):
        ns_, o, h_t_name = residual_block(ns_, e, prefix_+'_r{}'.format(li), 'v2_r{}'.format(li), dim_=100)
        # residual here
        e = L.Eltwise(o, hidden_prev_t_list[li][1], name=prefix_+'_res{}'.format(li),eltwise_param={'operation':P.Eltwise.SUM})
        ns_.__setattr__(prefix_+'_res{}'.format(li), e)
        outputs.append((prefix_+'_res{}'.format(li), e))
    # prediction layer
    ns_, out, out_name = residual_block(ns_, outputs[-1][1], prefix_ + '_pred', 'pred', dim_=DIM)
    outputs.append((out_name, out))
    return ns_, outputs

def one_time_step_vanilla(ns_, x_t, hidden_prev_t_list, prefix_):
    """
    "vanilla" variant: top of unit_t-1 is *concat* to *bottom* of unit_t
    TODO: number of hidden variables should be tuned to roughly maintain the same number of variables.
    :param ns_:
    :param x_t:
    :param hidden_prev_t_list:
    :param prefix_:
    :return:
    """
    # pass through an initial res block
    ns_, o, _ = residual_block(ns_, x_t, prefix_+'_in', 'rin', dim_=100)
    outputs = []
    for li in xrange(len(hidden_prev_t_list)):
        # concat instead of (+)
        e = L.Concat(o, hidden_prev_t_list[li][1], name=prefix_+'_concat{}'.format(li),
                     concat_param={'axis':-1})
        ns_.__setattr__(prefix_+'_concat{}'.format(li), e)
        ns_, o, h_t_name = residual_block(ns_, e, prefix_+'_r{}'.format(li), 'vnl_r{}'.format(li), dim_=100)
        outputs.append((h_t_name, o))
    # prediction layer
    ns_, out, out_name = residual_block(ns_, outputs[-1][1], prefix_ + '_pred', 'pred', dim_=DIM)
    outputs.append((out_name, out))
    return ns_, outputs

def time_unroll(ns_, all_x, all_reset, variant, hidden_prev_t_list, num_time_steps):
    pred = []
    for t in xrange(num_time_steps):
        # reset
        reset_hidden_prev_t = []
        for li in xrange(len(hidden_prev_t_list)):
            rh = L.Scalar(hidden_prev_t_list[li][1], all_reset[t], name='reset_h{}_t{}'.format(li,t),
                          scalar_param={'axis':0}, propagate_down=[True, False])
            ns_.__setattr__('reset_h{}_t{}'.format(li,t), rh)
            reset_hidden_prev_t.append(('reset_h{}_t{}'.format(li,t), rh))
        if variant == 'v0':
            ns_, outputs_t = one_time_step_v0(ns_, all_x[t], reset_hidden_prev_t, prefix_='v0_t{}'.format(t))
        elif variant == 'v1':
            ns_, outputs_t = one_time_step_v1(ns_, all_x[t], reset_hidden_prev_t, prefix_='v1_t{}'.format(t))
        elif variant == 'v2':
            ns_, outputs_t = one_time_step_v2(ns_, all_x[t], reset_hidden_prev_t, prefix_='v2_t{}'.format(t))
        elif variant in ('vnl', 'vanilla', 'rnn'):
            ns_, outputs_t = one_time_step_vanilla(ns_, all_x[t], reset_hidden_prev_t, prefix_='vnl_t{}'.format(t))
        else:
            raise Exception('unknown variant {}'.format(variant))
        hidden_prev_t_list = outputs_t[:-1] # last output is prediction
        pred.append(outputs_t[-1][1])
    return ns_, pred, [n[0] for n in outputs_t[:-1]]

def the_whole_shabang(phase, num_hidden_layers=3, variant='v0', seq_len=None, bs=None):
    ns = caffe.NetSpec()
    params = {}
    if phase == caffe.TRAIN:
        params['batch_size'] = bs if bs is not None else 75
        params['seq_len'] = seq_len if seq_len is not None else 200
        params['source'] = 'train_val_videodatainfo.json'
    else:
        params['batch_size'] = bs if bs is not None else 50
        params['seq_len'] = seq_len if seq_len is not None else  50
        params['source'] = 'test_videodatainfo.json'
    # inputs (all time steps and hidden states):
    all_inputs = L.Python(name='input', ntop=2 * params['seq_len'] + 1,
                          python_param={'module': 'r2d2_text', 'layer': 'input_text_layer',
                                        'param_str': json.dumps(params)},
                          include={'phase': phase})
    all_x = all_inputs[:params['seq_len']]
    all_h_reset = all_inputs[params['seq_len']:2*params['seq_len']]
    ns.label = all_inputs[2*params['seq_len']]
    for t in xrange(params['seq_len']):
        ns.__setattr__('x_t{}'.format(t), all_x[t])
        ns.__setattr__('reset_t{}'.format(t), all_h_reset[t])

    # hiddens
    rec_param = {'h_out_shapes': [(params['batch_size'], 100) for _ in xrange(num_hidden_layers)],
                 'debug': False}
    hiddens = L.Python(name='reccurence_handler', ntop=num_hidden_layers,
                                     python_param={'module': 'r2d2_text',
                                                   'layer': 'reccurence_handler',
                                                   'param_str': json.dumps(rec_param)})
    hidden_prev_t_list = []
    input_hidden_names = []
    for li in xrange(num_hidden_layers):
        ns.__setattr__('h{}_t0'.format(li), hiddens[li])
        hidden_prev_t_list.append(('h{}_t0'.format(li), hiddens[li]))
        input_hidden_names.append(hidden_prev_t_list[-1][0])
    # unrolling all the temporal layers
    ns, pred_layers_all_t, output_hidden_names = time_unroll(ns, all_x, all_h_reset, variant, hidden_prev_t_list, params['seq_len'])
    # reshape all prediction layers and concat them into a single layer
    rs = []
    for t in xrange(params['seq_len']):
        rs.append( L.Reshape(pred_layers_all_t[t], reshape_param={'shape':{'dim':1}, 'axis': 0, 'num_axes': 0}) )
        ns.__setattr__('reshape_t{}'.format(t), rs[-1])
    # concat
    ns.concat_all_pred = L.Concat(*rs, name='concat_all_pred', concat_param={'axis': 0}) # concat on the temporal dimension
    # loss layer
    ns.loss = L.SoftmaxWithLoss(ns.concat_all_pred, ns.label, name='loss', propagate_down=[True,False], loss_weight=1, softmax_param={'axis':-1})
    ns.accuracy = L.Accuracy(ns.concat_all_pred, ns.label,
                             name='accuracy', accuracy_param={'axis':-1}, propagate_down=[False,False])
    ns.acc3 = L.Accuracy(ns.concat_all_pred, ns.label, name='acc3',
                         accuracy_param={'axis': -1, 'top_k': 3}, propagate_down=[False,False])

    return str(ns.to_proto()), input_hidden_names, output_hidden_names
#------------------------------------------------------------------------------------------------------
def make_deploy_net(num_hidden_layers=3, variant='v0'):
    ns = caffe.NetSpec()
    ns.input_hot_vecotr = L.DummyData(name='input_hot_vecotr', dummy_data_param={'shape': {'dim':[1, DIM]}})

    # hiddens
    rec_param = {'h_out_shapes': [(1, 100) for _ in xrange(num_hidden_layers)]}
    hiddens = L.Python(name='reccurence_handler', ntop=num_hidden_layers,
                                     python_param={'module': 'r2d2_text',
                                                   'layer': 'reccurence_handler',
                                                   'param_str': json.dumps(rec_param)})
    # input hiddens
    hidden_in = []
    for li in xrange(num_hidden_layers):
        ns.__setattr__('h{}_t0'.format(li), hiddens[li])
        hidden_in.append(('h{}_t0'.format(li), hiddens[li]))

    if variant == 'v0':
        ns, outputs = one_time_step_v0(ns, ns.input_hot_vecotr, hidden_in, prefix_='v0')
    elif variant == 'v1':
        ns, outputs = one_time_step_v1(ns, ns.input_hot_vecotr, hidden_in, prefix_='v1')
    elif variant == 'v2':
        ns, outputs = one_time_step_v2(ns, ns.input_hot_vecotr, hidden_in, prefix_='v2')
    elif variant in ('vnl', 'vanilla', 'rnn'):
        ns, outputs = one_time_step_vanilla(ns, ns.input_hot_vecotr, hidden_in, prefix_='vnl')
    else:
        raise Exception('unknown variant {}'.format(variant))

    # add SoftmaxLayer
    ns.prob = L.Softmax(outputs[-1][1], name='prob', softmax_param={'axis':-1})

    with open('./deploy_'+variant+'.prototxt', 'w') as W:
        W.write(str(ns.to_proto()))
    net = caffe.Net('./deploy_'+variant+'.prototxt', caffe.TEST)
    net.layers[list(net._layer_names).index('reccurence_handler')].set_net(net, [h[0] for h in outputs[:-1]])
    return net, [h[0] for h in hidden_in], [h[0] for h in outputs[:-1]]
#------------------------------------------------------------------------------------------------------
# sampling sentences from trained net.
#------------------------------------------------------------------------------------------------------
def sample_sentence(net, temperatures):
    # for inn in net['h_in_names']:
    #     net['net'].blobs[inn].data[...] = np.zeros(net['net'].blobs[inn].data.shape, dtype='f4')
    sentences = []
    for T in temperatures:
        sentence = ''
        net.layers[list(net._layer_names).index('reccurence_handler')].reset_hiddens()  # fresh start
        vec = np.zeros((1, DIM), dtype='f4')
        cIdx = char_map['.']
        vec[0, cIdx] = 1  # send BOS
        while True:
            net.blobs['input_hot_vecotr'].data[...] = vec
            net.forward()
            # get output
            p = net.blobs['prob'].data
            # # set hidden states
            # for inn, outn in zip(net['h_in_names'],net['h_out_names']):
            #     net['net'].blobs[inn].data[...] = net['net'].blobs[outn].data[...]
            c = sample_char_from_p(p, T=T)
            sentence += c
            if c == '.' or len(sentence)>300:
                break
            vec[0,cIdx] = 0
            cIdx = char_map[c]
            vec[0,cIdx] = 1
        sentences.append(sentence)
    return sentences

def sample_char_from_p(p, T):
    try:
        w = np.power(p.flatten(), T).astype('f8')
        w = w/w.sum()
        idx = np.random.choice(w.size, size=None, replace=False, p=w)
    except ValueError:
        idx = char_map['U']
    return SORTED_CHARS[idx]

#------------------------------------------------------------------------------------------------------
# Testing utilities
#------------------------------------------------------------------------------------------------------
class testerThread(Thread):
    def __init__(self, inQ, val_net_dict, deploy_net_dict, niter):
        Thread.__init__(self)
        self.daemon = True
        self.inQ = inQ
        self.val_net = val_net_dict
        self.deploy_net = deploy_net_dict
        self.niter = niter
        self.start()

    def run(self):
        while True:
            try:
                weights_file_name, at_iter = self.inQ.get()
                testerThread.test_net(weights_file_name, at_iter, self.niter, self.val_net, self.deploy_net)
            except Exception as e:
                glog.error('testerThread: failed testing iter {} ({}): {}\n{}'.format(at_iter, type(e).__name__, e, traceback.format_exc()))
            finally:
                self.inQ.task_done()

    @staticmethod
    def test_net(weights_file_name, at_iter, niter, val_net, deploy_net):
        val_net['net'].copy_from(weights_file_name)
        deploy_net['net'].copy_from(weights_file_name)
        # reset hidden states
        # for k in val_net['h_in_names']:
        #     val_net['net'].blobs[k].data[...] = np.zeros(val_net['net'].blobs[k].data.shape,dtype='f4')
        val_net['net'].layers[list(val_net['net']._layer_names).index('reccurence_handler')].reset_hiddens()  # fresh start
        glog.info('Testing at iter {} for {} iterations'.format(at_iter, niter))
        Ts = [2,5,10]
        ssent = sample_sentence(deploy_net, temperatures=Ts)
        for i,T in enumerate(Ts):
            glog.info('Testing at iter {}. Sampling sentence for T={}: |{}|'.format(at_iter, T, ssent[i]))

        report = {k:0 for k in val_net['net'].outputs}
        for i in xrange(niter):
            out = val_net['net'].forward()
            # # take care of the hidden states
            # for inn, outn in zip(val_net['h_in_names'], val_net['h_out_names']):
            #     val_net['net'].blobs[inn].data[...] = val_net['net'].blobs[outn].data.copy()
            # record batch outputs
            for r in report.keys():
                report[r] += out[r] # val_net['net'].blobs[r].data.copy()
            if ((i+1)%100) == 0:
                glog.info('    testing at_iter {} iter {}/{}'.format(at_iter, i+1, niter))
        # report
        report_str = ''
        for k in val_net['net'].outputs:
            report_str += '  overall {} = {}'.format(k, report[k]/float(niter))
        glog.info('Testing at iter {}: {}\n'.format(at_iter, report_str))

def snap_and_test_r2d2_text_threaded(solver, prefix, testQ):
    solver.snapshot()
    iter_ = solver.iter
    weights_file_name = '{}_iter_{}.caffemodel'.format(prefix, iter_)
    testQ.put((weights_file_name, iter_))

# def snap_and_test(solver, prefix, niter, test_net_dict, deploy_net_dict):
#     solver.snapshot()
#     iter_ = solver.iter
#     weights_file_name = '{}_iter_{}.caffemodel'.format(prefix, iter_)
#     testerThread.test_net(weights_file_name, at_iter=iter_, niter=niter, val_net=test_net_dict, deploy_net=deploy_net_dict)

#------------------------------------------------------------------------------------------------------
def write_solver(prefix, niter, snap_iter, ntiter, train_net_str, val_net_str):
    train_file_name = './'+prefix+'_train.prototxt'
    with open(train_file_name, 'w') as W:
        W.write('name: "train_{}"\n'.format(prefix))
        # screws with accuracy layers... W.write('force_backward: true\n')
        W.write(train_net_str)
    val_file_name = './'+prefix+'_val.prototxt'
    with open(val_file_name, 'w') as W:
        W.write('name: "val_{}"\n'.format(prefix))
        W.write(val_net_str)
    solver_file_name = './'+prefix+'_solver.protoxt'
    with open(solver_file_name, 'w') as W:
        W.write('train_net: "{}"\n'.format(train_file_name))
        W.write('test_net: "{}"\n'.format(val_file_name))
        W.write('test_iter: {}\n'.format(ntiter)) # no automatic test, see http://stackoverflow.com/a/34387104/
        W.write('test_interval: {}\n'.format(snap_iter))
        W.write('snapshot: {}\n'.format(snap_iter))
        W.write('test_initialization: true\n')
        W.write('base_lr: 0.001\n')
        #W.write('lr_policy: "step"\n')
        #W.write('stepsize: {}\n'.format(int(niter/2)))
        #W.write('gamma: 0.1\n')
        #W.write('lr_policy: "fixed"\n')
        W.write('lr_policy: "poly"\n')
        W.write('power: 1.5\n')
        W.write('max_iter: {}\n'.format(niter)) # "poly" policy must have 'max_iter' defined.
        W.write('display: 50\naverage_loss: 50\n')
        W.write('momentum: 0.95\nmomentum2: 0.99\n')
        W.write('solver_mode: GPU\n')
        W.write('device_id: 0\n')
        W.write('snapshot_prefix: "{}"\n'.format(prefix))
        # regularization
        W.write('weight_decay: 0.00001\n')
        W.write('regularization_type: "L2"\n')
        #W.write('debug_info: true\n')
    return solver_file_name


def train_r2d2_text(variant='v0', niter=100000, ntiter=10000, snap_iter=5000, prefix='r2d2_text', start_with=None):
    prefix = prefix+'_'+variant
    num_hidden_layers = 3
    train_net_str, train_hidden_names, train_hidden_out = the_whole_shabang(caffe.TRAIN, num_hidden_layers=num_hidden_layers, variant=variant)
    test_net_str, test_hidden_names, test_hidden_outs = the_whole_shabang(caffe.TEST, num_hidden_layers=num_hidden_layers, variant=variant)
    solver_file_name = write_solver(prefix, niter, snap_iter, ntiter, train_net_str, test_net_str)
    solver = caffe.AdamSolver(solver_file_name)
    val_net_dict = {'net': solver.test_nets[0], 'h_in_names': test_hidden_names, 'h_out_names': test_hidden_outs}
    # "warm start" with weight (caffemodel) file
    if start_with is not None:
        glog.info("\n","-"*50)
        glog.info("warm start. loading {}".format(start_with))
        glog.info("-"*50,"\n")
        if start_with.endswith('solverstate'):
            solver.restore(start_with)
        else:
            solver.net.copy_from(start_with)

    # setup test thread
    deploy_net, deploy_hidden_in, deploy_hidden_out = make_deploy_net(num_hidden_layers=num_hidden_layers, variant=variant)
    # deploy_net_dict = {'net': deploy_net, 'h_in_names': deploy_hidden_in, 'h_out_names': deploy_hidden_out}
    # testQ = Queue(10) # do not let test "pile up"...
    # testerThread(testQ, val_net_dict, deploy_net_dict, ntiter)

    # # reset hidden input states
    # for inn in train_hidden_names:
    #     solver.net.blobs[inn].data[...] = np.zeros(solver.net.blobs[inn].data.shape, dtype='f4')
    solver.net.layers[list(solver.net._layer_names).index('reccurence_handler')].set_net(solver.net, train_hidden_out)
    solver.test_nets[0].layers[list(solver.test_nets[0]._layer_names).index('reccurence_handler')].set_net(solver.test_nets[0], test_hidden_outs)
    solver.solve()
    # for iter_ in xrange(int(niter/snap_iter)):
    #     # if (iter_%snap_iter)==0: # and iter_>0:
    #     snap_and_test_r2d2_text_threaded(solver, prefix, testQ)
    #     # set train hidden state
    #     solver.step(snap_iter)
    #     # get hidden state
    #     # for inn, outn in zip(train_hidden_names, train_hidden_out):
    #     #     solver.net.blobs[inn].data[...] = solver.net.blobs[outn].data.copy()
    # snap_and_test_r2d2_text(solver, prefix, testQ)
    solver.net.save('./{}_final_{}.caffemodel'.format(prefix, niter))
    # testQ.join() # wait for testing to complete

if __name__ == '__main__':
    # variant
    variant_ = sys.argv[1] if len(sys.argv) > 1 else 'v0'
    # "warm start"?
    start_with_ = sys.argv[2] if len(sys.argv) > 2 and os.path.isfile(sys.argv[1]) else None
    train_epoch = 1340 # iter_
    test_epoch = 1150 # iter_
    train_r2d2_text(variant=variant_, niter=train_epoch*100, ntiter=test_epoch, snap_iter=train_epoch, prefix='r2d2_text', start_with=start_with_)
