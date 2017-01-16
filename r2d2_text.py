import sys, os, json, numpy as np

sys.path.insert(0, os.environ['CAFFE_ROOT'] + '/python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
from caffe import layers as L, params as P

from seq_sources import glog, SEQ_GLOBALS, get_configuration

from threading import Thread, Event
from Queue import Queue, Empty
import traceback

phase_str = {caffe.TRAIN:'train', caffe.TEST:'test'}
# ------------------------------------------------------------------------------------------------------
# threaded input functionality
# ------------------------------------------------------------------------------------------------------
def empty_q(q):
    # empty a Queue
    while True:
        try:
            e = q.get(timeout=1)
        except Empty:
            return
        q.task_done()

class dispatchThread(Thread):
    """
    read input file and output sentences to Q
    """

    def __init__(self, sentencesQ, seq_source):
        Thread.__init__(self)
        self.daemon = True
        self.sentencesQ = sentencesQ
        self.source = seq_source
        self.start()

    def run(self):
        for s in self.source:
            try:
                self.sentencesQ.put(s)
            except Exception as e:
                glog.error('dispatchThread: got error ({}): {}\n{}'.format(type(e).__name__, e, traceback.format_exc()))

class singleSequenceProvider(Thread):
    """
    get sequences from inQ and output chunks to sequenceQ
    """

    def __init__(self, inQ, outQ, syncEvent, seq_len):
        Thread.__init__(self)
        self.inQ = inQ
        self.outQ = outQ
        self.seq_len = seq_len
        self.syncEvent = syncEvent
        self.daemon = True
        self.start()

    def run(self):
        to_data = []
        to_label = []
        while self.syncEvent.is_set():
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

class batchProvider(Thread):
    def __init__(self, inQs, outQ, syncEvent):
        Thread.__init__(self)
        self.inQs = inQs
        self.outQ = outQ
        self.syncEvent = syncEvent
        self.daemon = True
        self.start()

    def run(self):
        while self.syncEvent.is_set():
            try:
                data = []
                label = []
                for bi in xrange(len(self.inQs)):
                    d, l = self.inQs[bi].get()
                    data.append(d)
                    label.append(l)
                    self.inQs[bi].task_done()
                self.outQ.put({'data': np.concatenate(data, axis=1),
                               'label': np.concatenate(label, axis=1)})
            except Exception as e:
                glog.error('batchProvider: got error ({}): {}\n{}'.format(type(e).__name__, e, traceback.format_exc()))


# ------------------------------------------------------------------------------------------------------
class input_text_layer(caffe.Layer):
    def setup(self, bottom, top):
        # params
        params = json.loads(self.param_str)
        self.batch_size = int(params['batch_size'])
        self.seq_len = int(params['seq_len'])
        self.reset_every_n_iterations = params.get('reset_every', None)
        assert len(bottom) == 0, "input_text_layer: input layer no bottoms"
        assert len(top) == 2 * self.seq_len + 1, "input_text_layer: 2*{}+1 tops".format(self.seq_len)

        # set threads in motion
        self.sentencesQ = Queue(100)
        self.batchQ = Queue(20)
        self.singleQs = [Queue(10) for _ in xrange(self.batch_size)]

        # thread to read from file - this thread is not sync it always run
        dispatchThread(self.sentencesQ, SEQ_GLOBALS.seqIterator(phase_str[self.phase]))

        # threads to write per parralele seq in minibatch
        self.syncEvent = Event()  # init to False
        self.syncEvent.set() # make it True
        self.set_threads_in_motion()
        self.iter_count = 0

    def set_threads_in_motion(self):
        self.sync_threads = []
        for bi in xrange(self.batch_size):
            self.sync_threads.append( singleSequenceProvider(self.sentencesQ, self.singleQs[bi], self.syncEvent, self.seq_len) )
        self.sync_threads.append( batchProvider(self.singleQs, self.batchQ, self.syncEvent) )

    def q_status(self, iter_):
        avg_qs = sum([q.qsize() for q in self.singleQs]) / float(len(self.singleQs))
        glog.info(
            'Q STATUS [{}] at iter={}: |sentencesQ|={} <|singleQs|>={} |batchQ|={}'.format(phase_str[self.phase], iter_,
                                                                                           self.sentencesQ.qsize(),
                                                                                           avg_qs, self.batchQ.qsize()))
    def reset_all_queues(self):
        self.syncEvent.clear() # make all threads hang
        self.q_status(self.iter_count)
        for t in self.sync_threads:
            t.wait()
        # clean all queues
        empty_q(self.batchQ)
        for q in self.singleQs:
            empty_q(q)
        self.syncEvent.set() # clear the flag, let the threads run again
        self.set_threads_in_motion()
        glog.info('done reseting threads and queues')
        self.q_status(self.iter_count)

    def reshape(self, bottom, top):
        for t in xrange(self.seq_len):
            # reshape all X,
            top[t].reshape(self.batch_size, SEQ_GLOBALS.DIM)
            # reshape all reset layer
            top[t + self.seq_len].reshape(self.batch_size)
        # reshape all labels
        top[2 * self.seq_len].reshape(self.seq_len, self.batch_size)

    def forward(self, bottom, top):
        if (self.iter_count % 100) == 0:
            self.q_status(self.iter_count)
        self.iter_count += 1
        self.syncEvent.wait() # make sure reset done.
        batch = self.batchQ.get(block=True, timeout=60)  # do not wait longer than 60 sec for a batch...
        data = batch['data'].copy()
        label = batch['label'].copy()
        self.batchQ.task_done()
        for t in xrange(self.seq_len):
            # data
            top[t].data[...] = data[t, :, :]
            # reset
            top[t + self.seq_len].data[...] = 1.0 - data[t, :, SEQ_GLOBALS.BOS]  # seq with s[t]==BOS reset is 0
        top[2 * self.seq_len].data[...] = label
        # do we need to reset?
        if self.reset_every_n_iterations is not None and (self.iter_count%self.reset_every_n_iterations) == 0:
            glog.info('Done {} iterations, resetting!'.format(self.iter_count))
            Thread(target=self.reset_all_queues)


    def backward(self, top, propagate_down, bottom):
        # no back prop for input layer
        pass

#------------------------------------------------------------------------------------------------------
# handling recurrent for train/validation via paramter weight sharing
#------------------------------------------------------------------------------------------------------
class parameter_in(caffe.Layer):
    def setup(self, bottom, top):
        # keep as many params as "bottoms"
        for _ in xrange(len(bottom)):
            self.blobs.add_blob(1)

    def reshape(self, bottom, top):
        for bi in xrange(len(bottom)):
            self.blobs[bi].reshape(*bottom[bi].data.shape)
            self.blobs[bi].data[...] = 0 # reset

    def forward(self, bottom, top):
        # store the inputs in the parameter blobs
        for bi in xrange(len(bottom)):
            self.blobs[bi].data[...] = bottom[bi].data

    def backward(self, top, propagate_down, bottom):
        # nothing to do
        pass

    def reset_params(self):
        for pi in xrange(len(self.blobs)):
            self.blobs[pi].data[...] = 0
#------------------------------------------------------------------------------------------------------
def count_net_params(net):
    """
    count the number of trainable/tunable parameters in a net

    :param net:
    :return:
    """
    np = {'all':0, 't0': 0}
    for li in xrange(len(net.layers)):
        l = net.layers[li]
        for bi in xrange(len(l.blobs)):
            np['all'] += l.blobs[bi].data.size
            if 't0' in net._layer_names[li]:
                np['t0'] += l.blobs[bi].data.size
    return np
# ------------------------------------------------------------------------------------------------------
# Building blocks for R2D2
# ------------------------------------------------------------------------------------------------------
def residual_block(ns_, in_, name_prefix_, weight_prefix_, dim_, phase_):
    fc = L.InnerProduct(in_, name=name_prefix_ + '_fc',
                        inner_product_param={'bias_term': True,
                                             'num_output': dim_,
                                             'axis': -1,
                                             'weight_filler': {'type': 'gaussian', 'std': 0.001},
                                             'bias_filler': {'type': 'constant', 'value': 0}},
                        param=[{'lr_mult': 1, 'decay_mult': 1, 'name': weight_prefix_ + '_fc_w'},
                               {'lr_mult': 2, 'decay_mult': 0, 'name': weight_prefix_ + '_fc_b'}])
    ns_.__setattr__(name_prefix_ + '_fc', fc)
    bn = L.BatchNorm(fc, name=name_prefix_ + '_bn', batch_norm_param={'use_global_stats': phase_==caffe.TEST},
                     param=[{'name': weight_prefix_ + '_bn_mu', 'lr_mult': 0, 'decay_mult': 0},
                            {'name': weight_prefix_ + '_bn_sig', 'lr_mult': 0, 'decay_mult': 0},
                            {'name': weight_prefix_ + '_bn_eps', 'lr_mult': 0, 'decay_mult': 0}])
    ns_.__setattr__(name_prefix_ + '_bn', bn)
    prelu = L.PReLU(bn, name=name_prefix_ + '_prelu',
                    prelu_param={'channel_shared': False, 'filler': {'type': 'constant', 'value': 0.01}},
                    param={'lr_mult': 1, 'decay_mult': 20, 'name': weight_prefix_ + '_prelu'})
    ns_.__setattr__(name_prefix_ + '_prelu', prelu)
    return ns_, prelu, name_prefix_ + '_prelu'

def one_time_step_v0(ns_, x_t, hidden_prev_t_list, dims, phase, prefix_):
    """
    v0 variant: top of unit_t-1 is added as residual to *bottom* of unit_t
    """
    assert (len(dims) == len(hidden_prev_t_list)+2), \
        "must provide dims for input layer, all hiddens and prediction layer.\nGot {} hiddens and {} dims".format(len(hidden_prev_t_list), len(dims))
    assert all([dims[0]==d_ for d_ in dims[:-1]]), "for v0 type R2D2 all dims (except last) must be identicle"
    # pass through an initial res block
    ns_, o, _ = residual_block(ns_, x_t, prefix_ + '_in', 'rin', dim_=dims[0], phase_=phase)
    outputs = []
    for li in xrange(len(hidden_prev_t_list)):
        # residual here
        e = L.Eltwise(o, hidden_prev_t_list[li][1], name=prefix_ + '_res{}'.format(li),
                      eltwise_param={'operation': P.Eltwise.SUM})
        ns_.__setattr__(prefix_ + '_res{}'.format(li), e)
        ns_, o, h_t_name = residual_block(ns_, e, prefix_ + '_r{}'.format(li), 'v0_r{}'.format(li), dim_=dims[li+1], phase_=phase)
        outputs.append((h_t_name, o))
    # prediction layer
    ns_, out, out_name = residual_block(ns_, outputs[-1][1], prefix_ + '_pred', 'pred', dim_=dims[-1], phase_=phase)
    outputs.append((out_name, out))
    return ns_, outputs


def one_time_step_v1(ns_, x_t, hidden_prev_t_list, dims, phase, prefix_):
    """
    v1 variant: top of unit_t-1 (BEFORE residual update) is added as residual to *top* of unit_t
    """
    assert (len(dims) == len(hidden_prev_t_list)+2), \
        "must provide dims for input layer, all hiddens and prediction layer.\nGot {} hiddens and {} dims".format(len(hidden_prev_t_list), len(dims))
    # pass through an initial res block
    ns_, e, _ = residual_block(ns_, x_t, prefix_ + '_in', 'rin', dim_=dims[0], phase_=phase)
    outputs = []
    for li in xrange(len(hidden_prev_t_list)):
        ns_, o, h_t_name = residual_block(ns_, e, prefix_ + '_r{}'.format(li), 'v1_r{}'.format(li), dim_=dims[li+1], phase_=phase)
        outputs.append((h_t_name, o))
        # residual here
        e = L.Eltwise(o, hidden_prev_t_list[li][1], name=prefix_ + '_res{}'.format(li),
                      eltwise_param={'operation': P.Eltwise.SUM})
        ns_.__setattr__(prefix_ + '_res{}'.format(li), e)
    # prediction layer
    ns_, out, out_name = residual_block(ns_, e, prefix_ + '_pred', 'pred', dim_=dims[-1], phase_=phase)
    outputs.append((out_name, out))
    return ns_, outputs


def one_time_step_v2(ns_, x_t, hidden_prev_t_list, dims, phase, prefix_):
    """
    v2 variant: top of unit_t-1 (AFTER residual update) is added as residual to *top* of unit_t
    """
    assert (len(dims) == len(hidden_prev_t_list)+2), \
        "must provide dims for input layer, all hiddens and prediction layer.\nGot {} hiddens and {} dims".format(len(hidden_prev_t_list), len(dims))
    # pass through an initial res block
    ns_, e, _ = residual_block(ns_, x_t, prefix_ + '_in', 'rin', dim_=dims[0], phase_=phase)
    outputs = []
    for li in xrange(len(hidden_prev_t_list)):
        ns_, o, h_t_name = residual_block(ns_, e, prefix_ + '_r{}'.format(li), 'v2_r{}'.format(li), dim_=dims[li+1], phase_=phase)
        # residual here
        e = L.Eltwise(o, hidden_prev_t_list[li][1], name=prefix_ + '_res{}'.format(li),
                      eltwise_param={'operation': P.Eltwise.SUM})
        ns_.__setattr__(prefix_ + '_res{}'.format(li), e)
        outputs.append((prefix_ + '_res{}'.format(li), e))
    # prediction layer
    ns_, out, out_name = residual_block(ns_, outputs[-1][1], prefix_ + '_pred', 'pred', dim_=dims[-1], phase_=phase)
    outputs.append((out_name, out))
    return ns_, outputs


def one_time_step_vanilla(ns_, x_t, hidden_prev_t_list, dims, phase, prefix_):
    """
    "vanilla" variant: top of unit_t-1 is *concat* to *bottom* of unit_t
    TODO: number of hidden variables should be tuned to roughly maintain the same number of variables.
    """
    assert (len(dims) == len(hidden_prev_t_list)+2), \
        "must provide dims for input layer, all hiddens and prediction layer.\nGot {} hiddens and {} dims".format(len(hidden_prev_t_list), len(dims))
    # pass through an initial res block
    ns_, o, _ = residual_block(ns_, x_t, prefix_ + '_in', 'rin', dim_=dims[0], phase_=phase)
    outputs = []
    for li in xrange(len(hidden_prev_t_list)):
        # concat instead of (+)
        e = L.Concat(o, hidden_prev_t_list[li][1], name=prefix_ + '_concat{}'.format(li),
                     concat_param={'axis': -1})
        ns_.__setattr__(prefix_ + '_concat{}'.format(li), e)
        ns_, o, h_t_name = residual_block(ns_, e, prefix_ + '_r{}'.format(li), 'vnl_r{}'.format(li), dim_=dims[li+1], phase_=phase)
        outputs.append((h_t_name, o))
    # prediction layer
    ns_, out, out_name = residual_block(ns_, outputs[-1][1], prefix_ + '_pred', 'pred', dim_=dims[-1], phase_=phase)
    outputs.append((out_name, out))
    return ns_, outputs


def time_unroll(ns_, all_x, all_reset, variant, hidden_prev_t_list, dims, phase, num_time_steps):
    assert (len(dims) == len(hidden_prev_t_list)+2), \
        "must provide dims for input layer, all hiddens and prediction layer.\nGot {} hiddens and {} dims".format(len(hidden_prev_t_list), len(dims))
    pred = []
    for t in xrange(num_time_steps):
        # reset: set prev time step hiddens when new sequence come along for this batch.
        reset_hidden_prev_t = []
        for li in xrange(len(hidden_prev_t_list)):
            rh = L.Scale(hidden_prev_t_list[li][1], all_reset[t], name='reset_h{}_t{}'.format(li, t),
                          scale_param={'axis': 0}, propagate_down=[True, False])
            ns_.__setattr__('reset_h{}_t{}'.format(li, t), rh)
            reset_hidden_prev_t.append(('reset_h{}_t{}'.format(li, t), rh))
        if variant == 'v0':
            ns_, outputs_t = one_time_step_v0(ns_, all_x[t], reset_hidden_prev_t, dims,
                                              prefix_='v0_t{}'.format(t), phase=phase)
        elif variant == 'v1':
            ns_, outputs_t = one_time_step_v1(ns_, all_x[t], reset_hidden_prev_t, dims,
                                              prefix_='v1_t{}'.format(t), phase=phase)
        elif variant == 'v2':
            ns_, outputs_t = one_time_step_v2(ns_, all_x[t], reset_hidden_prev_t, dims,
                                              prefix_='v2_t{}'.format(t), phase=phase)
        elif variant in ('vnl', 'vanilla', 'rnn'):
            ns_, outputs_t = one_time_step_vanilla(ns_, all_x[t], reset_hidden_prev_t, dims,
                                                   prefix_='vnl_t{}'.format(t), phase=phase)
        else:
            raise Exception('unknown variant {}'.format(variant))
        hidden_prev_t_list = outputs_t[:-1]  # last output is prediction
        pred.append(outputs_t[-1][1])
    return ns_, pred, [n[1] for n in outputs_t[:-1]]


def the_whole_shabang(phase, configuration):

    num_hidden_layers = len(configuration['layer_dims'])-2
    variant = configuration.get('variant', 'v0')
    layers_dims = configuration['layer_dims']

    ns = caffe.NetSpec()

    # inputs (all time steps and hidden states):
    params = configuration['input_params'][phase_str[phase]]
    if phase == caffe.TEST:
        params['reset_every'] = configuration['test_niter']
    all_inputs = L.Python(name='input', ntop=2 * params['seq_len'] + 1,
                          python_param={'module': 'r2d2_text', 'layer': 'input_text_layer',
                                        'param_str': json.dumps(params)},
                          include={'phase': phase})
    all_x = all_inputs[:params['seq_len']]
    all_h_reset = all_inputs[params['seq_len']:2 * params['seq_len']]
    ns.label = all_inputs[2 * params['seq_len']]
    for t in xrange(params['seq_len']):
        ns.__setattr__('x_t{}'.format(t), all_x[t])
        ns.__setattr__('reset_t{}'.format(t), all_h_reset[t])

    # # hiddens
    # rec_param = {'debug': configuration.get('debug', False),
    #              'output_shapes': [(params['batch_size'] ,h_) for h_ in layers_dims[1:-1]]}
    # hiddens = L.Python(name='recurrence_handler', ntop=num_hidden_layers,
    #                    python_param={'module': 'r2d2_text',
    #                                  'layer': 'recurrence_handler',
    #                                  'param_str': json.dumps(rec_param)})
    hidden_prev_t_list = []
    for li in xrange(num_hidden_layers):
        hin = L.Parameter(name='h{}_t0_{}'.format(li, phase_str[phase]),
                          parameter_param={'shape':{'dim':[params['batch_size'], layers_dims[li+1]]}},
                          param={'lr_mult': 0, 'decay_mult': 0, 'name': 'h{}_share_{}'.format(li, phase_str[phase])})
        ns.__setattr__('h{}_t0'.format(li), hin)
        hidden_prev_t_list.append(('h{}_t0'.format(li), hin))
    # unrolling all the temporal layers
    ns, pred_layers_all_t, output_hiddens = time_unroll(ns, all_x, all_h_reset, variant, hidden_prev_t_list,
                                                             layers_dims, phase, params['seq_len'])
    # take care of hiddens of last time step...
    param = []
    inputs = []
    for li in xrange(num_hidden_layers):
        param.append({'lr_mult': 0, 'decay_mult': 0, 'name': 'h{}_share_{}'.format(li, phase_str[phase])})
        inputs.append(output_hiddens[li])
    ns.keeing_hidden_states = L.Python(*inputs, name='keeing_hidden_states_{}'.format(phase_str[phase]), ntop=0,
                                       python_param={'module': 'r2d2_text', 'layer': 'parameter_in'},
                                       param=param)

    # reshape all prediction layers and concat them into a single layer
    rs = []
    for t in xrange(params['seq_len']):
        rs.append(L.Reshape(pred_layers_all_t[t], reshape_param={'shape': {'dim': 1}, 'axis': 0, 'num_axes': 0}))
        ns.__setattr__('reshape_t{}'.format(t), rs[-1])
    # concat
    ns.concat_all_pred = L.Concat(*rs, name='concat_all_pred',
                                  concat_param={'axis': 0})  # concat on the temporal dimension
    # loss layer
    ns.loss = L.SoftmaxWithLoss(ns.concat_all_pred, ns.label, name='loss', propagate_down=[True, False], loss_weight=1,
                                softmax_param={'axis': -1})
    ns.accuracy = L.Accuracy(ns.concat_all_pred, ns.label,
                             name='accuracy', accuracy_param={'axis': -1}, propagate_down=[False, False])
    ns.acc3 = L.Accuracy(ns.concat_all_pred, ns.label, name='acc3',
                         accuracy_param={'axis': -1, 'top_k': 3}, propagate_down=[False, False])

    return str(ns.to_proto())

#------------------------------------------------------------------------------------------------------
def write_solver(prefix, configuration, train_net_str, val_net_str, init_test=False):
    train_file_name = os.path.join(configuration['base_dir'], prefix + '_train.prototxt')
    with open(train_file_name, 'w') as W:
        W.write('name: "train_{}"\n'.format(prefix))
        W.write(train_net_str)
    val_file_name = os.path.join(configuration['base_dir'], prefix + '_val.prototxt')
    with open(val_file_name, 'w') as W:
        W.write('name: "val_{}"\n'.format(prefix))
        W.write(val_net_str)
    solver_file_name = os.path.join(configuration['base_dir'], prefix + '_solver.protoxt')
    niter = configuration.get('train_niter', 100000)
    with open(solver_file_name, 'w') as W:
        W.write('train_net: "{}"\n'.format(train_file_name))
        W.write('test_net: "{}"\n'.format(val_file_name))
        W.write('test_iter: {}\n'.format(configuration.get('test_niter', 1000)))  # no automatic test, see http://stackoverflow.com/a/34387104/
        W.write('test_interval: {}\n'.format(configuration.get('test_interval', 5000)))
        W.write('snapshot: {}\n'.format(configuration.get('test_interval', 5000)))
        W.write('test_initialization: {}\n'.format('true' if init_test else 'false'))
        W.write('base_lr: {}\n'.format(configuration['base_lr']))
        #W.write('lr_policy: "step"\n')
        #W.write('stepsize: {}\n'.format(int(niter/2)))
        W.write('gamma: 0.1\n')
        W.write('lr_policy: "multistep"\n')
        W.write('stepvalue: {}\nstepvalue: {}\n'.format(int(niter/3),int(2*niter/3)))
        # W.write('lr_policy: "fixed"\n')

        # W.write('lr_policy: "poly"\n')
        # W.write('power: 1.5\n')

        W.write('max_iter: {}\n'.format(niter))  # "poly" policy must have 'max_iter' defined.
        W.write('display: 50\naverage_loss: 50\n')
        W.write('momentum: 0.95\nmomentum2: 0.999\n')
        W.write('solver_mode: GPU\n')
        W.write('device_id: 0\n')
        W.write('snapshot_prefix: "{}"\n'.format(prefix))
        # regularization
        W.write('weight_decay: 0.00001\n')
        W.write('regularization_type: "L1"\n')
        # W.write('debug_info: true\n')
    return solver_file_name


def train_r2d2_text(config='msr-vtt-v0', start_with=None):

    configuration = get_configuration(config)

    prefix = 'char_r2d2_' + config

    train_net_str = the_whole_shabang(caffe.TRAIN, configuration)
    test_net_str = the_whole_shabang(caffe.TEST, configuration)
    solver_file_name = write_solver(prefix, configuration, train_net_str, test_net_str,
                                    init_test=start_with_ is not None)

    solver = caffe.AdamSolver(solver_file_name)

    # "warm start" with weight (caffemodel) file
    if start_with is not None:
        glog.info("\n", "-" * 50)
        glog.info("warm start. loading {}".format(start_with))
        glog.info("-" * 50, "\n")
        if start_with.endswith('solverstate'):
            solver.restore(start_with)
        else:
            solver.net.copy_from(start_with)

    # set recurrence handler
    # solver.net.layers[list(solver.net._layer_names).index('recurrence_handler')].set_net(solver.net, train_hidden_out)
    # solver.test_nets[0].layers[list(solver.test_nets[0]._layer_names).index('recurrence_handler')].set_net(solver.test_nets[0], test_hidden_outs)

    # some statistics
    train_np = count_net_params(solver.net)
    test_np = count_net_params(solver.test_nets[0])
    glog.info('\nStarting training.\nTrain net has {} params\nValidationnet has {} params\n'.format(train_np, test_np))

    # run solver
    solver.solve()

    # save final result
    solver.net.save('./{}_final_{}.caffemodel'.format(prefix, solver.iter))


if __name__ == '__main__':
    # source
    config_ = sys.argv[1] if len(sys.argv) > 1 else 'msr-vtt-v0'
    # "warm start"?
    start_with_ = sys.argv[2] if len(sys.argv) > 2 and os.path.isfile(sys.argv[2]) else None
    glog.info('\n\nTraining with config={} {}\n\n'.format(config_, '' if start_with_ is None else 'starting with '+start_with_))
    train_r2d2_text(config=config_, start_with=start_with_)
