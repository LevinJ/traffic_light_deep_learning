import os
import sys
sys.path.append('../models/slim')
import tensorflow as tf
from datasets import flowers
from nets import inception
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim


class TLModel(object):
    def __init__(self):
        #variables to set before training
        self.variables_to_train = None
        self.input = None
        self.labels = None
        return
    def build_train_graph(self):
        #before building the graph, we need to specify the input and labels, and variables_to_train for the models
        self.add_inference_node(is_training = True)
        self.add_loss_node()
        self.add_optimizer_node()
        self.add_train_summaries()
        return
    def add_inference_node(self, is_training=True):
        with slim.arg_scope(inception.inception_v4_arg_scope()):
            self.output, self.end_points = inception.inception_v4(self.input, num_classes=5, 
                                                                  is_training=is_training, dropout_keep_prob=0.8,create_aux_logits=True)
        return
    def add_loss_node(self):
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.output)
        return
    def add_optimizer_node(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train_op = slim.learning.create_train_op(self.loss, optimizer, variables_to_train=self.variables_to_train)
        return
    def add_train_summaries(self):
        end_points = self.end_points
        for end_point in end_points:
            x = end_points[end_point]
            tf.summary.histogram('activations/' + end_point, x)
            tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x))
        # Add summaries for losses and extra losses.
        
        tf.summary.scalar('loss', self.oss)
        # Add summaries for variables.
        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)
        return