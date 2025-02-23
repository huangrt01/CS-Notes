utils/data

### datasets.py

class IterableDataset(Dataset[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __add__(self, other: Dataset[T_co]):
        return ChainDataset([self, other])


### _utils/worker.py


# worker_loop被DataLoader的多进程类使用
# 子进程fetcher封装的业务逻辑：dataset、collate_fn
def _worker_loop(
    dataset_kind,
    dataset,
    index_queue,
    data_queue,
    done_event,
    auto_collation,
    collate_fn,
    drop_last,
    base_seed,
    init_fn,
    worker_id,
    num_workers,
    persistent_workers,
    shared_seed,
):

	if init_fn is not None:  # 调用init_fn
	    init_fn(worker_id)

	# 生成一个fetcher
	fetcher = _DatasetKind.create_fetcher(
	    dataset_kind, dataset, auto_collation, collate_fn, drop_last)     # collate_fn：组batch；  auto_collation：dataset内做batch的时候告诉外层，设为True

	watchdog = ManagerWatchdog()

		class ManagerWatchdog:  # type: ignore[no-redef]
	        def __init__(self) -> None:
	            self.manager_pid = os.getppid()
	            self.manager_dead = False

	        def is_alive(self):
	            if not self.manager_dead:
	                self.manager_dead = os.getppid() != self.manager_pid
	            return not self.manager_dead


	while watchdog.is_alive():
	    try:
	        # 在一般情况下， r的数据类型是: tuple[int, list[Any]]
	        # 其中int是全局batch_index; list[Any]是dataset中样本的index
	        r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
	    except queue.Empty:
	        continue
	    # 先处理三种特殊情况： 1) 新epoch; 2) 通过r is None主动退出；3）通过done_event延迟退出
	    if isinstance(r, _ResumeIteration):
	        # 用于开启一个新的epoch, 返回None, 并重新构造一个fetcher
	        data_queue.put((r, None))
	        iteration_end = False
	        fetcher = _DatasetKind.create_fetcher(
	            dataset_kind, dataset, auto_collation, collate_fn, drop_last)
	        continue
	    elif r is None:  # None是退出标志，这里会check done_event
	        assert done_event.is_set() or iteration_end
	        break
	    elif done_event.is_set() or iteration_end:
	        continue
	    
	    # 读取数据正常，解出r, 其中index 是一个list, len(index) = batch_size, 对于IteratorDataset
	    # 基值都是None, 对于其它dataset, 这个index可以直接从dataset取值


	    # idx：第几个mini-batch,可能乱序（如果shuffle）
	    # index：mini-batch中包含哪些行
	    idx, index = r  
	    data: Union[_IterableDatasetStopIteration, ExceptionWrapper]
	    if init_exception is not None:
	        data = init_exception
	        init_exception = None
	    else:
	        try:
	            data = fetcher.fetch(index)  # 事实上返回的是一个mini-batch, 不是一行
	        except Exception as e:
	            if isinstance(e, StopIteration) and dataset_kind == _DatasetKind.Iterable:
	                data = _IterableDatasetStopIteration(worker_id)
	                iteration_end = True
	            else:
	                data = ExceptionWrapper(where=f"in DataLoader worker process {worker_id}")
	    data_queue.put((idx, data))


### shutdown逻辑

See NOTE [ Data Loader Multiprocessing Shutdown Logic ]

### _utils/fetch.py

class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, possibly_batched_index):
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    self.ended = True
                    break
            if len(data) == 0 or (
                self.drop_last and len(data) < len(possibly_batched_index)
            ):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)




### dataloader.py

# dataloader

torch dataloader的问题：
1）全同步训练场景，无法graceful shutdown；
--> 用all reduce时loss*0的方法
2）rank*worker_num，需要文件分发机制



Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.


if sampler is None:  # give default samplers
  if self._dataset_kind == _DatasetKind.Iterable:
      # See NOTE [ Custom Samplers and IterableDataset ]
      # IterableDataset 不支持用户自定义 batch_sampler 或sampler,发送的全是None
      sampler = _InfiniteConstantSampler()
  else:  # map-style
      if shuffle:
          sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
      else:
          sampler = SequentialSampler(dataset)  # type: ignore[arg-type]


# Sampler: sampler.py

batch_sampler也是一个迭代器， 它会从sampler中依次取出下标，关组成一个mini-batch, 一次返回一个batch_size长度的下标list
- 当指定了batch_size, 又没有指定batch_sampler时， 系统会给一个默认的batch_sampler


class RandomSampler
def __iter__:
	if self.replacement:
    for _ in range(self.num_samples // 32):
        yield from torch.randint(
            high=n, size=(32,), dtype=torch.int64, generator=generator
        ).tolist()
    yield from torch.randint(
        high=n,
        size=(self.num_samples % 32,),
        dtype=torch.int64,
        generator=generator,
    ).tolist()
else:
    for _ in range(self.num_samples // n):
        yield from torch.randperm(n, generator=generator).tolist()
    yield from torch.randperm(n, generator=generator).tolist()[
        : self.num_samples % n
    ]


SubsetRandomSampler
WeightedRandomSampler


### DataLoaderIter

_SingleProcessDataLoaderIter

class _BaseDataLoaderIter:
	def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

  def _next_data(self):
      raise NotImplementedError


  # 在 _sampler_iter 为 None 的情况下，会调用 _reset 方法来重置迭代器的状态
  def __next__(self) -> Any:
        with torch.autograd.profiler.record_function(self._profile_name):
            if self._sampler_iter is None:
                # TODO(https://github.com/pytorch/pytorch/issues/76750)
                self._reset()  # type: ignore[call-arg]
            data = self._next_data()
            self._num_yielded += 1
            ...
            return data


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
       	...
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self._dataset,
            self._auto_collation,
            self._collate_fn,
            self._drop_last,
        )

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data


# pin memory
if isinstance(data, torch.Tensor):
  return data.pin_memory(device)



class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):

	self._rcvd_idx 表示下一个要在 __next__ 方法中返回的任务的索引。它跟踪了主进程期望从工作进程接收的下一个数据批次的顺序。
	self._send_idx 表示下一个要发送给工作进程的任务的索引。它记录了主进程已经发送出去的任务的编号，用于跟踪任务的发送进度。
	作用
	- 任务调度：控制主进程向工作进程发送任务的顺序和数量。主进程根据 self._send_idx 来决定下一个要发送的任务，并将其分配给合适的工作进程。
	- 任务数量限制：结合 self._prefetch_factor 和 self._num_workers，self._send_idx 可以确保在任何时候，主进程发送出去的未完成任务数量不会超过预取因子和工作进程数量的乘积，从而避免内存溢出等问题。

	(_send_idx, index)进入index queue
	(_send_idx, data)进入data queue

	def _next_data(self):
		while True:
			while self._rcvd_idx < self._send_idx:
	        info = self._task_info[self._rcvd_idx]
	        worker_id = info[0]
	        if (
	            len(info) == 2 or self._workers_status[worker_id]
	        ):  # has data or is still active
	            break
	        del self._task_info[self._rcvd_idx]
	        self._rcvd_idx += 1
	    else:
	        # no valid `self._rcvd_idx` is found (i.e., didn't break)
	        if not self._persistent_workers:
	            self._shutdown_workers()
	        raise StopIteration

	def _try_put_index(self):
        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))  # type: ignore[possibly-undefined]
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1



# NOTE [ Data Loader Multiprocessing Shutdown Logic ]
    #
    # Preliminary:
    #
    # Our data model looks like this (queues are indicated with curly brackets):
    #
    #                main process                              ||
    #                     |                                    ||
    #               {index_queue}                              ||
    #                     |                                    ||
    #              worker processes                            ||     DATA
    #                     |                                    ||
    #            {worker_result_queue}                         ||     FLOW
    #                     |                                    ||
    #      pin_memory_thread of main process                   ||   DIRECTION
    #                     |                                    ||
    #               {data_queue}                               ||
    #                     |                                    ||
    #                data output                               \/
    #
    # P.S. `worker_result_queue` and `pin_memory_thread` part may be omitted if
    #      `pin_memory=False`.

设计目标：
1. The iterator gracefully exits the workers when its last reference is gone or it is depleted.
--> we implement the shutdown logic in `__del__` of DataLoaderIterator.
--> `workers_done_event`

2. The iterator exits the workers when the loader process and/or worker processes exits normally or with error.
--> We set all workers and `pin_memory_thread` to have `daemon=True`.

3. All processes exit if any of them die unexpectedly by fatal signals.
a. A process won't hang when getting from a queue.
b. A process won't hang when putting into a queue;
--> 

