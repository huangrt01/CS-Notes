from ray.util.sgd import TFTrainer 
trainer = TFTrainer(
    model_creator=model_creator,
    data_creator=data_creator,
    num_replicas=4,
    use_gpu=True,
    verbose=True,
    config={
        "fit_config": {
            "steps_per_epoch": num_train_steps,
        },
        "evaluate_config": {
            "steps": num_eval_steps,
        }
    })


'''
Ray Cover了以下代码：

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})
...
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
with strategy.scope():
  multi_worker_model = model_creator()

'''

# PyTorch支持
from ray.util.sgd import TorchTrainer 
trainer = TorchTrainer( 
    model_creator=model_creator, 
    data_creator=data_creator, 
    optimizer_creator=optimizer_creator, 
    loss_creator=nn.MSELoss, 
    scheduler_creator=scheduler_creator, 
    scheduler_step_freq="epoch", 
    initialization_hook=init_hook, # setup env
    config={"lr": 0.001, "batch_size": 64},
    num_workers = 100
    num_replicas = 400,
    use_fp16=True   # Mixded Precision FP16
)
trainer.train(max_retries=5, checkpoint="auto")
trainer.validate()
trainer.shutdown()

