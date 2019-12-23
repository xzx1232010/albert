import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants
from run_classifier import InputFeatures, InputExample
import tokenization


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(example, label_list, max_seq_length, tokenizer):

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    # 对输入文本a做一些预处理
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None

    # 对输入文本a做一些预处理
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    # 如果两个文本序列的长度加起来>最长的序列长度限制，会做处理，会将序列减少到限制的序列长度
    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    # 将序列转化为字典索引
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    # 对小于最高序列长度的序列，用0进行补全
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def main():
    max_seq_len = 64
    label_list = ['0', '1']
    sentences = ("您好.麻烦您截图全屏辛苦您了.", "麻烦您截图大一点辛苦您了.最好可以全屏.")
    guid = 'test-%d' % 1
    text_a = tokenization.convert_to_unicode(str(sentences[0]))
    text_b = tokenization.convert_to_unicode(str(sentences[1]))
    label = str(0)
    predict_examples = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
    tokenizer = tokenization.FullTokenizer(vocab_file='./albert_config/vocab.txt', do_lower_case=True)
    features = convert_single_example(predict_examples, label_list, max_seq_len, tokenizer)
    export_dir = './export/1576720765'
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir)
            tensor_input_ids = graph.get_tensor_by_name('input_ids_1:0')
            tensor_input_mask = graph.get_tensor_by_name('input_mask_1:0')
            tensor_label_ids = graph.get_tensor_by_name('label_ids_1:0')
            tensor_segment_ids = graph.get_tensor_by_name('segment_ids_1:0')
            tensor_outputs = graph.get_tensor_by_name('loss/Softmax:0')
            result = sess.run(tensor_outputs, feed_dict={
                tensor_input_ids: np.array(features.input_ids).reshape(-1, max_seq_len),
                tensor_input_mask: np.array(features.input_mask).reshape(-1, max_seq_len),
                tensor_label_ids: np.array([features.label_id]),
                tensor_segment_ids: np.array(features.segment_ids).reshape(-1, max_seq_len),
            })
            print(*(result[0]), sep='\t')


if __name__ == '__main__':
    main()
