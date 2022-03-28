# `torch.utils.data`

## Overview

`TODO`



## Quick Look

### `torch.utils.data.dataset`

#### `torch.utils.data.dataset.Dataset`

数据集基类。继承该类的子类必须实现以下方法——通过索引获取数据样本。

```python
class Dataset(Generic[T_co]):
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError
```

一个相对少用的方法：

```python
class Dataset(Generic[T_co]):
    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])
```

注意，当前版本该类并没有获取数据集大小的方法，但实际上很多时候需要这个类能够被`len`调用。

```python
# No `def __len__(self)` default?
# See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
# in pytorch/torch/utils/data/sampler.py
```

基本的数据集主要划分为两类，一类称为`map-style`（映射类型）数据集，一类称为`iterable-style`（迭代类型）数据集。用户常自定义映射类型的数据集——可以通过索引直接访问到数据样本的数据集。

与数据集匹配使用的是采样器。



### `torch.utils.data.sampler`

#### `torch.utils.data.sampler.Sampler`

采样器基类。

初始化时需要提供前面提到的`Dataset`实例。

```python
class Sampler(Generic[T_co]):
    def __init__(self, data_source: Optional[Sized]) -> None:
        pass
```

继承该类的子类必须实现以下方法——生成索引迭代器。采样器——尽管名为“采样”——核心功能并非采样，而是提供一个迭代器，可生成用于采样动作的索引，让主调函数以调用迭代器的方式获取这些索引。

```python
class Sampler(Generic[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError
```



#### `torch.utils.data.sampler.SequentialSampler`

顺序采样器，产生`0 ~ n-1`作为索引。

```python
class SequentialSampler(Sampler[int]):
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)
```



#### `torch.utils.data.sampler.RandomSampler`

随机采样器。初始化的时候除了数据集外，还可以提供其它参数。

```python
class RandomSampler(Sampler[int]):
    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))
```

在默认初始化下，属性`num_samples`即为数据集本身大小。

```python
class RandomSampler(Sampler[int]):
    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __len__(self) -> int:
        return self.num_samples
```

在默认初始化下，方法

```python
def __iter__(self) -> Iterator[int]:
    n = len(self.data_source)
    if self.generator is None:
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
    else:
        generator = self.generator
    if self.replacement:
        for _ in range(self.num_samples // 32):
            yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
        yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
    else:
        yield from torch.randperm(n, generator=generator).tolist()
```

执行分支

```python
n = len(self.data_source)
generator = torch.Generator()
generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
yield from torch.randperm(n, generator=generator).tolist()
```

使用`torch`的种子生成器与随机排列方法，对`0 ~ n-1`的某种排列以迭代器方式返回。

基于单个索引迭代器可以获得批量索引迭代器。



#### `torch.utils.data.sampler.BatchSampler`

批采样器继承于采样器，初始化需要用户提供单样本采样器（实为单索引采样器）实例，批大小，以及是否丢弃最后不足一个批大小的样本批。

```python
class BatchSampler(Sampler[List[int]]):
    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
```

方法

```python
class BatchSampler(Sampler[List[int]]):
    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
```

其中

```python
for idx in self.sampler
```

调用了单样本采样器的`__iter__`方法，获得了单索引迭代器，累计获得`batch_size`个索引后返回。使用`yield`使得自身返回的依然是一个（批索引）迭代器。

`Dataset`提供的根据索引获取样本的方法以及`BatchSampler`提供的批索引，两者协作实现了批样本的获取。



### `torch.utils.data._utils.fetch`

#### `torch.utils.data._utils.fetch._BaseDatasetFetcher`

样本抓取器基类。

```python
class _BaseDatasetFetcher(object):    
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()
```



#### `torch.utils.data._utils.fetch._MapDatasetFetcher`

配合映射类型数据集的抓取器。

```python
class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
```

其中`possibly_batched_index`自然是来自`BatchSampler`提供的批索引。



### `torch.utils.data.dataloader`

#### `torch.utils.data.dataloader._DatasetKind`

为`Dataloader`提供一个工厂方法，返回映射模型或者迭代模式数据集对应的捕获器实例。

```python
class _DatasetKind(object):
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
```



#### `torch.utils.data.dataloader.DataLoader`

用户在数据加载环节通常直接使用的对象是`DataLoader`实例，并且通常以迭代器方式来调用这个实例。`DataLoader`的`__iter__`方法根据单进程或多进程的选项分别返回单进程批样本迭代器与多进程批样本迭代器。其中被采样的数据集是`Dataset`实例（且通常为映射模式），批索引迭代器是`BatchSampler`实例（并且通常在内部使用的单索引迭代器是`RandomSampler`实例），样本抓取器则由上述的`_DatasetKind`类方法获取，通常是`_MapDatasetFetcher`实例。



伴随着`DataLoader`实例的`BatchSampler`实例只有一个（无论有多少个`epoch`）。

```python
class DataLoader(Generic[T_co]):
    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1,
                 shuffle: bool = False, sampler: Optional[Sampler[int]] = None,
                 batch_sampler: Optional[Sampler[Sequence[int]]] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: int = 2,
                 persistent_workers: bool = False):
        """
        some code
        """
        self.batch_sampler = batch_sampler
        """
        some code
        """
```



#### `torch.utils.data.dataloader._SingleProcessDataLoaderIter`

在单进程模式下，`DataLoader`实例（在每次遍历数据集，即对应每个`epoch`开始加载数据时）生成的迭代器是`_SingleProcessDataLoaderIter`对象。整个训练环节，对一个数据集，只有一个`DataLoader`对象——如果有训练集与验证集，自然就是有两个`DataLoader`对象——但在单进程模式下，每个`epoch`一开始时，`DataLoader`都会返回一个新的`_SingleProcessDataLoaderIter`对象。

```python
class DataLoader(Generic[T_co]):
    def __iter__(self) -> '_BaseDataLoaderIter':
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()
   
    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)
```

尽管只有一个批索引采样器实例，但是在不同的`epoch`中，随机采样模式下，采样索引序列并不一样。这是因为索引采样器（`BatchSampler`与`RandomSampler`）实例是可迭代对象，而非迭代器，每个`epoch`开始时，`_SingleProcessDataLoaderIter`是不同的实例，而其基类每次初始化时都重新调用了`BatchSampler`的`__iter__`方法，获得一个新的批索引迭代器，而这个迭代器，也是重新调用了其成员`RandomSampler`实例的`__iter__`方法获得了新的单索引迭代器。

```python
class DataLoader(Generic[T_co]):
    @property
    def _index_sampler(self):
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler
        
class _BaseDataLoaderIter(object):
    def __init__(self, loader: DataLoader) -> None:
        """
        some code
        """
        self._index_sampler = loader._index_sampler
        self._sampler_iter = iter(self._index_sampler)  "create new batch-index iterator at the beginning of each epoch"
        """
        some code
        """
 
class BatchSampler(Sampler[List[int]]):
    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:  "create new index iterator at the beginning of each epoch"
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
```

通过上述方式，单进程数据加载迭代器在每个`epoch`获得全新的索引序列，并结合抓取器实例完成自身对外的迭代器功能。

```python
class _BaseDataLoaderIter(object):
    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        with torch.autograd.profiler.record_function(self._profile_name):
            if self._sampler_iter is None:
                self._reset()
            data = self._next_data()
            self._num_yielded += 1
            if self._dataset_kind == _DatasetKind.Iterable and \
                    self._IterableDataset_len_called is not None and \
                    self._num_yielded > self._IterableDataset_len_called:
                warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                            "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                                  self._num_yielded)
                if self._num_workers > 0:
                    warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                                 "IterableDataset replica at each worker. Please see "
                                 "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
                warnings.warn(warn_msg)
            return data

    next = __next__  # Python 2 compatibility

class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data
```



#### `torch.utils.data.dataloader._MultiProcessDataLoaderIter`

更常用的是多进程数据加载模式，`DataLoader`实例（在每次遍历数据集，即对应每个`epoch`开始加载数据时）生成的迭代器是`_MultiProcessDataLoaderIter`对象。这个对象在数据加载时采用“生产者/消费者模式”（一种面向过程的设计模式），初始化时生成`_num_workers`个负责加载数据的子进程（下边简称为子进程）0并启用，为`DataLoader`实例所在的进程（下边称为主进程）生成结果缓存队列，为每一个子进程生成对应的信息缓存队列。

```python
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)
		"""
		some code
		"""
        self._worker_result_queue = multiprocessing_context.Queue()
        """
        some code
        """
        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            index_queue = multiprocessing_context.Queue()
            index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=_utils.worker._worker_loop,
                args=(self._dataset_kind, self._dataset, index_queue,
                      self._worker_result_queue, self._workers_done_event,
                      self._auto_collation, self._collate_fn, self._drop_last,
                      self._base_seed, self._worker_init_fn, i, self._num_workers,
                      self._persistent_workers))
            w.daemon = True
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)
        """
        some code
        """
```

`self._worker_result_queue`用于缓存全体子进程加载的数据及其它必要信息。

`self._workers`是子进程列表。

`self._index_queues`是队列列表，各队列是各子进程接受主进程信息的缓存区。

主进程在每一个`epoch`开始时会重置（而非重新生成）`_MultiProcessDataLoaderIter`实例的状态。另外要注意，多进程是体现在多个子进程根据批索引抓取批样本，但是批索引的生成是由主进程控制的，只有一个`BatchSampler`实例。

```python
class _BaseDataLoaderIter(object):                       
    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called

class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):    
    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}
        self._tasks_outstanding = 0  
        self._workers_status = [True for i in range(self._num_workers)]
        # We resume the prefetching in case it was enabled
        if not first_iter:
            for idx in range(self._num_workers):
                self._index_queues[idx].put(_utils.worker._ResumeIteration())
                resume_iteration_cnt = self._num_workers
                while resume_iteration_cnt > 0:
                    return_idx, return_data = self._get_data()
                    if isinstance(return_idx, _utils.worker._ResumeIteration):
                        assert return_data is None
                        resume_iteration_cnt -= 1
                        # prime the prefetch loop
                        for _ in range(self._prefetch_factor * self._num_workers):
                            self._try_put_index()
        # prime the prefetch loop
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()
```

游标`self._send_idx`（分发任务游标）负责记录即将分派给子进程的任务索引（在这个场合下就是批样本的索引，注意，不是某批样本在数据集中的索引列表，而是描述这是第几批/第几个`batch`的索引，通常情况下该游标从`0`增长至`# iterations per epoch - 1`）。

游标`self._rcvd_idx`（返还任务游标）负责记录即将返回给`DataLoader`实例的调用者的批样本的索引，即负责指示将要抓取的是第几个`batch`的批样本（通常情况下该游标从`0`增长至`# iterations per epoch - 1`）。

`self._workers_status`负责记录各子进程目前是否可以接受新的任务（即抓取数据）。

重置的流程最后进行任务预分配的动作（注意，进行任务预分配之前，分发任务游标与返还任务游标均为`0`）。

```python
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
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

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1
```

`index = self._next_index()`这一语句表明了批索引（批样本在数据集中的索引列表）的生成由主进程（而非各子进程）控制，由唯一的`BatchSampler`实例负责——当然，每个`epoch`开始时，这个可迭代对象会重新生成一个新的批索引迭代器，在单进程模式中，是因为`_SingleProcessDataLoaderIter`实例被重新创建，在多进程模式中是因为被调用`reset`方法，注意该基类方法中的`self._sampler_iter = iter(self._index_sampler)`。

`self._index_queues[worker_queue_idx].put((self._send_idx, index))`本语句即为任务分发。

`self._task_info[self._send_idx] = (worker_queue_idx,)`本语句更新了主进程的任务信息记录，表明索引为`self._send_idx`的任务分发给了索引为`worker_queue_idx`的子进程。

`self._send_idx += 1`分发任务游标更新。

```python
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def _reset(self, loader, first_iter=False):
        """
        some code
        """
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()
```

`_MultiProcessingDataLoaderIter`实例发生重置（发生在初始化以及每一个`epoch`开始之时），假定有4个子进程，预加载比例为1（以及取决于`self._worker_queue_idx_cycle`的循环方式实现以及`self._workers_status`的更新方式），重置后分发任务游标为4，返还任务游标依然为0，但注意批索引生成器已经被调用了4次（`index = self._next_index()`），每个子进程分别领取了一个加载数据的任务。假定`self._worker_queue_idx_cycle`的循环方式是不停地循环返回`0~3`，那么重置后子进程`i`领取了任务`i`，此时有

```python
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    self._task_info = {
        0: (0,),
        1: (1,),
        2: (2,),
        3: (3,),
    }
```

在运行时（`_MultiProcessingDataLoaderIter`实例被调用`__next__`方法）

```python
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        """some code"""
        self._data_queue = self._worker_result_queue
    
    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
       
    def _get_data(self):
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            """ some code """
            
    def _next_data(self):
        while True:
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            """ some code """
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            """ some code """
            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data)
```

简单地看，每个子进程在接收到任务后会把数据推到公共的`self._worker_result_queue`，主进程会从中获取加载后的数据。在这里很自然的事情就是任务`i`不一定在任务`i+1`之前被完成，也就是说，在队列中`batch i`不一定在`batch i+1`之前，但是返还任务游标`self._rcvd_idx`是自增的，肯定是需要先返还`batch i`之后再返还`batch i+1`。（在我们的场合通常都是使用随机抓取，所以实际上在一次`epoch`中先用哪批数据后用哪批数据没有太大的区别，毕竟都是`shuffle`，但是从模块本身的需求来看，是需要按照批索引给定的索引来返还，要不然外部程序会出现样本索引与样本本身对不上号的情况。）

我们把队列中`batch i`在`batch i+j`之后的情况称为`out-of-order`。解决这类问题的方案是主进程维护一个信息记录对象——在这个场合中这个对象即为字典`self._task_info`。

使用任务`0~3`分发给`worker_0~3`这个场景来解释。

在被询问要求返回`batch 0`数据时，如果4个`worker`均完成任务，对应数据也是按某种次序（不一定是`0~3`的顺序）缓存到`self._worker_result_queue`中，注意，这时候`self._task_info`还是任务分发后的状态，即

```python
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    self._task_info = {
        0: (0,),
        1: (1,),
        2: (2,),
        3: (3,),
    }
    
    def _next_data(self):
        while True:
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                """ some code """
```

这时候上述代码的`if`分支会因为`self._workers_status[worker_id]`被触发，`self._rcvd_idx`依然等于`0`。注意，此时加载的数据在队列`self._worker_result_queue`中。

```python
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def _next_data(self):
        idx, data = self._get_data()
        """ some code """

        if idx != self._rcvd_idx:
            # store out-of-order samples
            self._task_info[idx] += (data,)
        else:
            del self._task_info[idx]
            return self._process_data(data)
```

接下来主进程从队列弹出一份数据。如果这数据恰好就是`batch 0`，执行上述代码`else`分支，如果这数据不是`batch 0`,譬如是`batch 1`，则把该数据缓存到`self._task_info[1]`中，即执行上述代码`if`分支，并继续整个循环直至从队列中弹出的是`batch 0`。

在被询问要求返回`batch 1`数据时，如果之前`batch 1`的数据已经从队列弹出缓存到`self._task_info`的话，下边代码的第一个`if`分支会被执行（中断内部循环），然后第二个`if`分支也会被执行——因为需要返回的数据已经在`self._task_info`中了，直接获取即可。

```python
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    self._task_info = {
        1: (1, data_1),
        2: (2,),
        3: (3,),
    }
    
    def _next_data(self):
        while True:
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                """ some code """
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)
```

最后，`_process_data`除了更新返回任务游标外，还要再次分发任务。

```python
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data
```

简而言之，当主进程需要某数据时，如果该数据还在队列中(甚至还在加载中，还不在队列里)，主进程会不停地从队列中弹出数据直到获得需要的数据，而这个过程中暂时不需要的数据（也就是乱序的数据：任务索引排在所需数据任务索引之后，但在结果缓存队列中却靠前的数据）会被缓存到信息记录对象中；如果该数据已经在信息记录对象中，则直接从信息记录对象中获取即可。



### `torch.utils.data._utils.worker`

#### `torch.utils.data._utils.worker._worker_loop`

多进程加载数据时每个子进程的进程方法。下边仅仅列出核心步骤——生成样本抓取器，监控任务分发队列，抓取样本并返回，注意`idx`是该批次是第几批数据，而`index`则是这批样本在数据集中的索引列表（由`BatchSampler`实例迭代生成）。

```python
def _worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event,
                 auto_collation, collate_fn, drop_last, base_seed, init_fn, worker_id,
                 num_workers, persistent_workers):
    from torch.utils.data import _DatasetKind
    fetcher = _DatasetKind.create_fetcher(dataset_kind, dataset, auto_collation, collate_fn, drop_last)
    while watchdog.is_alive():
        try:
            r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue
        idx, index = r
        data = fetcher.fetch(index)
        data_queue.put((idx, data))
```



## More Details

`TODO`





