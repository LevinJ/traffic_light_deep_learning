import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.flower_model import TLModel
from preparedata import PrepareData
import math



class EvaluateModel(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)  
        
        return
    
    def run(self): 
       
        tf.logging.set_verbosity(tf.logging.INFO)
        net = TLModel()
        _ = slim.get_or_create_global_step()
        batch_size = 2
      
        
        split_name = "train"
        net.input, _ , net.labels = self.get_input(split_name, is_training=False,batch_size=batch_size)
        net.build_eval_graph()
        
        
        logdir = './logs/evals/' + split_name
        checkpoint_path = './logs'
        
        num_batches = math.ceil(self.dataset.num_samples / float(batch_size))
        # Standard evaluation loop.
        print("one time evaluate...")
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
        else:
            checkpoint_file = checkpoint_path
        tf.logging.info('Evaluating %s' % checkpoint_file)
       
       
       
        slim.evaluation.evaluate_once(
            master='',
            checkpoint_path=checkpoint_file,
            logdir=logdir,
            num_evals=num_batches,
            eval_op=net.names_to_updates ,
            variables_to_restore=slim.get_variables_to_restore())
       
                    
        
        
        return
    
    


if __name__ == "__main__":   
    obj= EvaluateModel()
    obj.run()