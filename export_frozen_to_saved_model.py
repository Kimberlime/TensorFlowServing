import tensorflow as tf
import os
from tensorflow.python.saved_model import signature_constants


def export_to_saved_model(graph_path, export_path_base, version, return_elements):
    """
    Exports TensorFlow frozen graph to SavedModel file.
    :param graph_path: frozen graph path
    :param export_path_base: path to export graph to
    :param version: model version
    :param return_elements: input and output tensor name list
    """
    export_path = os.path.join(export_path_base, version)

    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)
    with tf.io.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(graph_def, name="")  # set name to an empty string to remove prefix
            g = tf.compat.v1.get_default_graph()

            input_tensor_info = tf.compat.v1.saved_model.build_tensor_info(g.get_tensor_by_name(return_elements[0]))
            output_tensor_info_s = tf.saved_model.utils.build_tensor_info(g.get_tensor_by_name(return_elements[1]))
            output_tensor_info_m = tf.compat.v1.saved_model.build_tensor_info(g.get_tensor_by_name(return_elements[2]))
            output_tensor_info_l = tf.compat.v1.saved_model.build_tensor_info(g.get_tensor_by_name(return_elements[3]))

            signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                inputs={return_elements[0]: input_tensor_info},
                outputs={return_elements[1]: output_tensor_info_s,
                         return_elements[2]: output_tensor_info_m,
                         return_elements[3]: output_tensor_info_l},
                method_name=tf.saved_model.PREDICT_METHOD_NAME)

            builder.add_meta_graph_and_variables(sess,
                                                 [tf.saved_model.SERVING],
                                                 signature_def_map={
                                                     signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})

        builder.save()
        print('Exported trained model to', export_path)


if __name__ == '__main__':
    return_elements = ['input/input_data:0',
                       'pred_sbbox/concat_2:0',
                       'pred_mbbox/concat_2:0',
                       'pred_lbbox/concat_2:0']
    pb_file = './detector_frozen.pb'
    export_path = './object_detection_model'
    version = 1

    export_to_saved_model(pb_file, export_path, str(version), return_elements)
