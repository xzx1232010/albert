import tensorflow as tf
from nets import inception


image_size = inception.inception_v3.default_image_size
label = {0: 'animal', 1: 'flower', 2: 'guitar', 3: 'houses', 4: 'plane'}


def preprocess_image(img, height, width, scope=None):
    with tf.name_scope(scope, 'inference_image', [img, height, width]):
        if img.dtype != tf.float32:
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        if height and width:
            # Resize the image to the specified height and width.
            img = tf.expand_dims(img, 0)
            img = tf.image.resize_bilinear(img, [height, width], align_corners=False)  # 不对齐角落
            img = tf.squeeze(img, [0])
        img = tf.subtract(img, 0.5)
        img = tf.multiply(img, 2.0)
        return img


def create_graph(model_path):
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def view_op_names(view_model_path):
    with tf.gfile.FastGFile(view_model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        for tensor_name in tensor_name_list:
            print(tensor_name, '\n')


def predict_picture(model_path):
    # 加载模型
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    # 读取图片
    with tf.Session() as sess:
        image_string = tf.gfile.FastGFile('./tmp/data/test_image/flower.jpg', 'rb').read()
        image = tf.image.decode_jpeg(image_string, channels=3)
        processed_image = preprocess_image(image, image_size, image_size)
        processed_images = tf.expand_dims(processed_image, 0)
        img = sess.run(processed_images)
        sess.close()
    with tf.Session() as sess:
        input_image_tensor = sess.graph.get_tensor_by_name("input:0")
        output_tensor_name = sess.graph.get_tensor_by_name('InceptionV3/Predictions/Softmax:0')
        probabilities = sess.run(output_tensor_name, feed_dict={input_image_tensor: img})
        sess.close()
    probabilities = probabilities[0, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, label[index]))


def main():
    view_model_path = './tmp/inception_v3_inf_graph.pb'
    view_op_names(view_model_path)  # 查看节点名称


if __name__ == '__main__':
    main()

