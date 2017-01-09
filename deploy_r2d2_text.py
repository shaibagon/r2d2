import sys, os, json, numpy as np
sys.path.insert(0, os.environ['CAFFE_ROOT'] + '/python')
os.environ['GLOG_minloglevel'] = '2'
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
from caffe import layers as L

from seq_sources import get_configuration, SEQ_GLOBALS
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
        net.layers[list(net._layer_names).index('recurrence_handler')].reset_hiddens()  # fresh start
        vec = np.zeros((1, SEQ_GLOBALS.DIM), dtype='f4')
        cIdx = SEQ_GLOBALS.BOS
        vec[0, cIdx] = 1  # send BOS
        while True:
            net.blobs['input_hot_vecotr'].data[...] = vec
            net.forward()
            # get output
            p = net.blobs['prob'].data
            vec[0, cIdx] = 0
            c, cIdx = sample_char_from_p(p, T=T)
            sentence += c
            if cIdx == SEQ_GLOBALS.EOS or len(sentence) > 300:
                break
            vec[0, cIdx] = 1
        sentences.append(sentence)
    return sentences

def sample_char_from_p(p, T):
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
    ns.input_hot_vecotr = L.DummyData(name='input_hot_vecotr', dummy_data_param={'shape': {'dim': [1, SEQ_GLOBALS.DIM]}})

    # hiddens
    rec_param = {'debug':False,
                 'output_shapes': [(1, h_) for h_ in layers_dims[1:-1]]}
    hiddens = L.Python(name='recurrence_handler', ntop=num_hidden_layers,
                       python_param={'module': 'r2d2_text',
                                     'layer': 'recurrence_handler',
                                     'param_str': json.dumps(rec_param)})
    # input hiddens
    hidden_in = []
    for li in xrange(num_hidden_layers):
        ns.__setattr__('h{}_t0'.format(li), hiddens[li])
        hidden_in.append(('h{}_t0'.format(li), hiddens[li]))

    if variant == 'v0':
        ns, outputs = one_time_step_v0(ns, ns.input_hot_vecotr, hidden_in, layers_dims, prefix_='v0')
    elif variant == 'v1':
        ns, outputs = one_time_step_v1(ns, ns.input_hot_vecotr, hidden_in, layers_dims, prefix_='v1')
    elif variant == 'v2':
        ns, outputs = one_time_step_v2(ns, ns.input_hot_vecotr, hidden_in, layers_dims, prefix_='v2')
    elif variant in ('vnl', 'vanilla', 'rnn'):
        ns, outputs = one_time_step_vanilla(ns, ns.input_hot_vecotr, hidden_in, layers_dims, prefix_='vnl')
    else:
        raise Exception('unknown variant {}'.format(variant))

    # add SoftmaxLayer
    ns.prob = L.Softmax(outputs[-1][1], name='prob', softmax_param={'axis': -1})

    deploy_file_name = os.path.join(configuration['base_dir'], config + '_deploy.prototxt')
    with open(deploy_file_name, 'w') as W:
        W.write(str(ns.to_proto()))
    net = caffe.Net(deploy_file_name, caffe.TEST)
    net.layers[list(net._layer_names).index('recurrence_handler')].set_net(net, [h[0] for h in outputs[:-1]])
    return net

#------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    config_ = sys.argv[1]
    weights_ = sys.argv[2]
    net = make_deploy_net(config_)
    # get the weights
    net.copy_from(weights_)
    T = [2,5,10]
    s = sample_sentence(net, T)
    for t,s in zip(T, sample_sentence(net, T)):
        print "t={}: |{}|".format(t, s)
