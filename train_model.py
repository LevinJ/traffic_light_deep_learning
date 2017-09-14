import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.training import saver as tf_saver
from tl_nets.tl_model import TLModel



class TrainModel(object):
    def __init__(self):
             
        self.max_number_of_steps = 30
        
        
        self.log_every_n_steps = 100
        self.save_summaries_secs= 60
        self.save_interval_secs = 60*60#one hour
        
        self.train_dir = './logs'
        self.checkpoint_path = '../data/trained_models/vgg16/vgg_16.ckpt'
        self.checkpoint_exclude_scopes = ["InceptionV4/Logits", "InceptionV4/AuxLogits"]
        self.trainable_scopes = ["InceptionV4/Logits", "InceptionV4/AuxLogits"]
        
        
        return
    def __get_variables_to_train(self):
        """Returns a list of variables to train.
    
        Returns:
            A list of variables to train by the optimizer.
        """
        if self.trainable_scopes is None:
            return tf.trainable_variables()
        else:
            scopes = [scope.strip() for scope in self.trainable_scopes.split(',')]
    
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train
    def __get_init_fn(self):
        """Returns a function run by the chief worker to warm-start the training.
    
        Note that the init_fn is only run when initializing the model during the very
        first global step.
    
        Returns:
            An init function run by the supervisor.
        """  
        
        if self.checkpoint_path is None:
            return None
    
        # Warn the user if a checkpoint exists in the train_dir. Then we'll be
        # ignoring the checkpoint anyway.
        
        
        if tf.train.latest_checkpoint(self.train_dir):
            tf.logging.info(
                    'Ignoring --checkpoint_path because a checkpoint already exists in %s'
                    % self.train_dir)
            return None
    
        exclusions = []
        if self.checkpoint_exclude_scopes:
            exclusions = [scope.strip()
                                        for scope in self.checkpoint_exclude_scopes.split(',')]
     
        # TODO(sguada) variables.filter_variables()
        variables_to_restore = []
        all_variables = slim.get_model_variables()
        if tf.gfile.IsDirectory(self.checkpoint_path):
            global_step = slim.get_or_create_global_step()
            all_variables.append(global_step)
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        else:
            checkpoint_path = self.checkpoint_path
            
        for var in all_variables:
            excluded = False
             
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
    
        tf.logging.info('Fine-tuning from %s' % checkpoint_path)
    
        return slim.assign_from_checkpoint_fn(
                checkpoint_path,
                variables_to_restore)
    
    def run(self):
        
        tf.logging.set_verbosity(tf.logging.INFO)
        
        net = TLModel()
        #preapre input, label,and variables to restore
        
        net.variables_to_train = self.__get_variables_to_train()
        net.build_train_graph()
        
        
        
        ###########################
        # Kicks off the training. #
        ###########################
       
        slim.learning.train(
                net.train_op,
                self.train_dir,
                saver=tf_saver.Saver(max_to_keep=500),
                init_fn=self.__get_init_fn(),
                number_of_steps=self.max_number_of_steps,
                log_every_n_steps=self.log_every_n_steps,
                save_summaries_secs=self.save_summaries_secs,
                save_interval_secs=self.save_interval_secs)
        return
    


if __name__ == "__main__":   
    obj= TrainModel()
    obj.run()