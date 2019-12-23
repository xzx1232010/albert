import numpy as np
import tokenization
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time
from tensorflow.contrib import util


class InputExample(object):
    def __init__(self, text_a, text_b):
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(example, max_seq_length, tokenizer):
    # 对输入文本做预处理
    tokens_a = tokenizer.tokenize(example.text_a)
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

    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    # 将序列转化为字典索引
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 对有效的字赋为1，补全的字补0
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    # 对小于最高序列长度的序列，用0进行补全
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)

    return feature


def prcoess_data(sentences, tokenizer):
    max_seq_len = 64
    text_a = tokenization.convert_to_unicode(sentences[0])
    text_b = tokenization.convert_to_unicode(sentences[1])
    predict_examples = InputExample(text_a=text_a, text_b=text_b)
    features = convert_single_example(predict_examples, max_seq_len, tokenizer)
    return features


def main():

    tokenizer = tokenization.FullTokenizer(vocab_file='./albert_config/vocab.txt', do_lower_case=True)
    begin = time.time()
    sentences = []
    sentences_1 = ("您好.麻烦您截图全屏辛苦您了.", "麻烦您截图大一点辛苦您了.最好可以全屏.")
    for i in range(600):
        sentences.append(sentences_1)

    input_ids = []
    input_mask = []
    label_ids = []
    segment_ids = []
    for i in sentences:
        features = prcoess_data(i, tokenizer)
        input_ids.append(features.input_ids)
        input_mask.append(features.input_mask)
        label_ids.append([0])
        segment_ids.append(features.segment_ids)

    server = '0.0.0.0:9000'
    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'test'
    request.model_spec.signature_name = 'serving_default'

    input_ids = util.make_tensor_proto(input_ids, shape=[len(sentences), 64])
    input_mask = util.make_tensor_proto(input_mask, shape=[len(sentences), 64])
    label_ids = util.make_tensor_proto(label_ids, shape=[len(sentences), 1])
    segment_ids = util.make_tensor_proto(segment_ids, shape=[len(sentences), 64])

    request.inputs['input_ids'].CopyFrom(input_ids)
    request.inputs['input_mask'].CopyFrom(input_mask)
    request.inputs['label_ids'].CopyFrom(label_ids)
    request.inputs['segment_ids'].CopyFrom(segment_ids)

    result = stub.Predict(request, 10.0)  # 10 secs timeout
    end = time.time() - begin
    output = np.array(result.outputs['probabilities'].float_val)
    print(len(sentences)/2)
    print('time {}'.format(end))


if __name__ == '__main__':
    main()
