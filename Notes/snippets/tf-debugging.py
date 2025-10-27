
*** op debug

* 命名
attn_mask = tf.concat(..., name="concat_attention_mask")

* print shape
abc = tf.Print(abc_ph, 
             [abc_ph, tf.shape(abc_ph)], 
             message='RuitengShape of abc_ph,: ', 
             summarize=100)

* print tensor
def print_tensor(name, tensor):
    with tf.device("/cpu:0"):
        print_op = tf.print(name, tensor, summarize=-1)
    with tf.control_dependencies([print_op]):
        tensor = tf.identity(tensor)
    return tensor



*** diff tensor/grad

t = tf.get_default_graph().get_tensor_by_name("model_1/debug_tensor:0")
grad_seq = tf.gradients(dummy_loss, t)