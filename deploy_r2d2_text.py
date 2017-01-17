import sys, os, json, numpy as np
sys.path.insert(0, os.environ['CAFFE_ROOT'] + '/python')
os.environ['GLOG_minloglevel'] = '2'
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
from caffe import layers as L

from seq_sources import glog, get_configuration, SEQ_GLOBALS
from r2d2_text import one_time_step_v0, one_time_step_v1, one_time_step_v2, one_time_step_vanilla

# ------------------------------------------------------------------------------------------------------
# sampling sentences from trained net.
# ------------------------------------------------------------------------------------------------------
def sample_sentence(net, temperatures):
    # for inn in net['h_in_names']:
    #     net['net'].blobs[inn].data[...] = np.zeros(net['net'].blobs[inn].data.shape, dtype='f4')
    sentences = []
    for T in temperatures:
        sentence = ''
        net.layers[list(net._layer_names).index('keeing_hidden_states')].reset_params()  # fresh start
        vec = np.zeros((1, SEQ_GLOBALS.DIM), dtype='f4')
        cIdx = SEQ_GLOBALS.BOS
        vec[0, cIdx] = 1  # send BOS
        while True:
            # net.blobs['input_hot_vecotr'].data[...] = vec
            out = net.forward(input_hot_vecotr=vec)
            # get output
            p = out['prob']
            vec[0, cIdx] = 0
            c, cIdx = sample_char_from_p(p, T=T)
            sentence += c
            if cIdx == SEQ_GLOBALS.EOS or len(sentence) > 300:
                break
            vec[0, cIdx] = 1
        sentences.append(sentence)
    return sentences

def sample_char_from_p(p, T):
    if T <= 0:
        # MAP
        idx = np.argmax(p)
    else:
        try:
            w = np.power(p.flatten(), T).astype('f8')
            w = w / w.sum()
            idx = np.random.choice(w.size, size=None, replace=False, p=w)
        except ValueError:
            idx = SEQ_GLOBALS.UNK
    return SEQ_GLOBALS.CHARS[idx], idx

#------------------------------------------------------------------------------------------------------
# construct a deploy net (single time step) with recurrence handler
#------------------------------------------------------------------------------------------------------
def make_deploy_net(config):
    configuration = get_configuration(config)
    variant = configuration.get('variant', 'v0')
    num_hidden_layers = len(configuration['layer_dims']) - 2
    layers_dims = configuration['layer_dims']

    ns = caffe.NetSpec()
    ns.input_hot_vecotr = L.Input(name='input_hot_vecotr', ntop=1, input_param={'shape': {'dim': [1, SEQ_GLOBALS.DIM]}})

    # # hiddens
    # rec_param = {'debug': False,
    #              'output_shapes': [(1, h_) for h_ in layers_dims[1:-1]]}
    # hiddens = L.Python(name='recurrence_handler', ntop=num_hidden_layers,
    #                    python_param={'module': 'r2d2_text',
    #                                  'layer': 'recurrence_handler',
    #                                  'param_str': json.dumps(rec_param)})
    # input hiddens
    hidden_in = []
    for li in xrange(num_hidden_layers):
        hin = L.Parameter(name='h{}_t0'.format(li),
                          parameter_param={'shape':{'dim':[1, layers_dims[li+1]]}},
                          param={'lr_mult': 0, 'decay_mult': 0, 'name': 'h{}_share'.format(li)})
        ns.__setattr__('h{}_t0'.format(li), hin)
        hidden_in.append(('h{}_t0'.format(li), hin))

    if variant == 'v0':
        ns, outputs = one_time_step_v0(ns, ns.input_hot_vecotr, hidden_in, layers_dims,
                                       prefix_='v0_t0', phase=caffe.TEST)
    elif variant == 'v1':
        ns, outputs = one_time_step_v1(ns, ns.input_hot_vecotr, hidden_in, layers_dims,
                                       prefix_='v1_t0', phase=caffe.TEST)
    elif variant == 'v2':
        ns, outputs = one_time_step_v2(ns, ns.input_hot_vecotr, hidden_in, layers_dims,
                                       prefix_='v2_t0', phase=caffe.TEST)
    elif variant in ('vnl', 'vanilla', 'rnn'):
        ns, outputs = one_time_step_vanilla(ns, ns.input_hot_vecotr, hidden_in, layers_dims,
                                            prefix_='vnl_t0', phase=caffe.TEST)
    else:
        raise Exception('unknown variant {}'.format(variant))

    # take care of hiddens of last time step...
    param = []
    inputs = []
    for li in xrange(num_hidden_layers):
        param.append({'lr_mult': 0, 'decay_mult': 0, 'name': 'h{}_share'.format(li)})
        inputs.append(outputs[li][1])
    ns.keeing_hidden_states = L.Python(*inputs, name='keeing_hidden_states', ntop=0,
                                       python_param={'module': 'r2d2_text', 'layer': 'parameter_in'},
                                       param=param)
    # add SoftmaxLayer
    ns.prob = L.Softmax(outputs[-1][1], name='prob', softmax_param={'axis': -1})

    deploy_file_name = os.path.join(configuration['base_dir'], config + '_deploy.prototxt')
    with open(deploy_file_name, 'w') as W:
        W.write(str(ns.to_proto()))
    net = caffe.Net(deploy_file_name, caffe.TEST)
    return net

#------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    config_ = sys.argv[1]
    weights_ = sys.argv[2]
    net = make_deploy_net(config_)
    # get the weights
    net.copy_from(weights_)
    T_ = [2,5,10, -1]
    for t,s in zip(T_, sample_sentence(net, T_)):
        print "t={}: |{}|".format(t, s)
    net.save(config_+'_deploy_weights.caffemodel')
    net.save_hdf5(config_+'_deploy_weights.cafeemodel.h5')
