import tensorflow as tf
import nets.inception_v4  as inception_v4

from tensorflow.contrib import slim


class TLModel(object):
    def __init__(self):
        #variables to set before training
        self.variables_to_train = None
        self.input = None
        self.labels = None
        return
    def build_eval_graph(self):
        self.add_inference_node(is_training = False)
        self.add_evalmetrics_node()
        return
    def add_evalmetrics_node(self):
        predictions = tf.argmax(self.output, 1)
    
        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, self.labels)
        })
        for metric, value in names_to_values.items():
            value = tf.Print(value, [value], metric)
            tf.summary.scalar(metric, value)
        self.names_to_updates = list(names_to_updates.values())
        return 
    def build_train_graph(self):
        #before building the graph, we need to specify the input and labels, and variables_to_train for the models
        self.add_inference_node(is_training = True)
        self.add_loss_node()
        return
    def add_inference_node(self, is_training=True):
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            self.output, self.end_points = inception_v4.inception_v4(self.input, num_classes=5, 
                                                                  is_training=is_training, dropout_keep_prob=0.8,create_aux_logits=True)
        return
    def add_loss_node(self):
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.output)
        return