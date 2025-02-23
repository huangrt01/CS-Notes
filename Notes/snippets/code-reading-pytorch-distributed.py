
### run.py

逻辑：elastic_launch

命令行启动：利用python的console script

entry_points = {
    "console_scripts": [
        "torchrun = torch.distributed.run:main",
    ],
    "torchrun.logs_specs": [
        "default = torch.distributed.elastic.multiprocessing:DefaultLogsSpecs",
    ],
}
