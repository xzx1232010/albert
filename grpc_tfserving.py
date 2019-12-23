import numpy as np
import tokenization
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.contrib import util as contrib_util


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


def prcoess_data():
    max_seq_len = 64
    sentences = ("您好.麻烦您截图全屏辛苦您了.", "麻烦您截图大一点辛苦您了.最好可以全屏.")
    text_a = tokenization.convert_to_unicode(sentences[0])
    text_b = tokenization.convert_to_unicode(sentences[1])
    predict_examples = InputExample(text_a=text_a, text_b=text_b)
    tokenizer = tokenization.FullTokenizer(vocab_file='./albert_config/vocab.txt', do_lower_case=True)
    features = convert_single_example(predict_examples, max_seq_len, tokenizer)
    return features


def construct_tensor(shapes, value):
    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=dim) for dim in shapes]
    tensor_shape = tensor_shape_pb2.TensorShapeProto(dim=dims)
    tensor = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_INT32,
        tensor_shape=tensor_shape,
        int_val=value)
    return tensor


def main():
    begin = time.time()
    features = prcoess_data()

    server = '0.0.0.0:9000'
    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'test'
    request.model_spec.signature_name = 'serving_default'

    # tensor_input_ids = construct_tensor([1, 64], features.input_ids)
    # tensor_input_mask = construct_tensor([1, 64], features.input_mask)
    # tensor_label_ids = construct_tensor([1, 1], [0])
    # tensor_segment_ids = construct_tensor([1, 64], features.segment_ids)
    #
    # request.inputs['input_ids'].CopyFrom(tensor_input_ids)
    # request.inputs['input_mask'].CopyFrom(tensor_input_mask)
    # request.inputs['label_ids'].CopyFrom(tensor_label_ids)
    # request.inputs['segment_ids'].CopyFrom(tensor_segment_ids)

    request.inputs['input_ids'].CopyFrom(contrib_util.make_tensor_proto(features.input_ids, shape=[1, 64]))
    request.inputs['input_mask'].CopyFrom(contrib_util.make_tensor_proto(features.input_mask, shape=[1, 64]))
    request.inputs['label_ids'].CopyFrom(contrib_util.make_tensor_proto([0], shape=[1, 1]))
    request.inputs['segment_ids'].CopyFrom(contrib_util.make_tensor_proto(features.segment_ids, shape=[1, 64]))

    result = stub.Predict(request, 10.0)  # 10 secs timeout
    end = time.time() - begin
    output = np.array(result.outputs['probabilities'].float_val)
    print('time {}'.format(end))
    print(output)


if __name__ == '__main__':
    main()
