import numpy as np
import pytest as pt
from rdds.lib.process_pool import ProcessPool, DummyPool
from multiprocessing import SimpleQueue


def random_delay():
    from random import randint
    from time import sleep
    sleep(randint(0, 2))


# Processing functions must be accessible from separate process (i.e. not inside local scope in test method)
def f_return_value(value: int, q: SimpleQueue):
    random_delay()
    q.put(value)


def f_dynamic_data_size(size: int, q: SimpleQueue):
    random_delay()
    arr = np.random.random_sample(int(size))
    q.put(arr)


def f_erroneous(*args, **kwargs):
    raise RuntimeError('I\'m expected to break, so here I am!')


def test_processpool():
    """
    Test pool general processing
    :return:
    """
    # GIVEN a worker pool
    q: SimpleQueue = ProcessPool.get_context().SimpleQueue()
    # WHEN processing 100 items
    args = [(1, q)] * 100
    pool = ProcessPool(function=f_return_value, args=args)
    completed_tasks = pool.run()
    result_sum = 0
    # THEN expect all items to be processed
    for task in completed_tasks:
        result_sum += q.get()
        assert task.process.exitcode == 0
    assert result_sum == 100
    pool.close()
    del q


def test_processpool_async():
    """
    Test pool general processing async mode
    :return:
    """
    # GIVEN a worker pool
    q: SimpleQueue = ProcessPool.get_context().SimpleQueue()
    # WHEN processing 100 items
    args = [(1, q)] * 100
    pool = ProcessPool(function=f_return_value, args=args)
    task_queue = pool.run_async()
    result_sum = 0
    # THEN expect all items to be processed
    for _ in range(0, len(args)):
        result_sum += q.get()
    for _ in range(0, len(args)):
        task = task_queue.get(timeout=1)
        assert task.process.exitcode == 0
    assert result_sum == 100
    pool.close()


def test_workerpool_restart():
    """
    Test pool restart capability.
    """
    # GIVEN a workerpool
    test_processpool()
    # WHEN it has completed
    # THEN expect the next workerpool to complete successfully as well
    test_processpool()


def test_processpool_worker_data_sizes():
    """
    Test worker pool with various number of workers and Queue IO data sizes.

    This test checks capability of async processing and Queue to transmit
    up to 1000 floating point size array.

    In order for processes to complete successfully (not hang on put())
    because of the underlying pipe filling up,
    the queue.read() must happen asynchronously from producers, i.e.
    ProcessPool.run() cannot be used (processes will deadlock).

    Data sizes >= 1E4 hangs the producer process.
    """
    for worker_instances in [1, 5, 10]:  # Parallel workers generating data
        for data_size in [1, 1E2, 1E3]:  # Floating point array sizes
            # GIVEN a worker pool with n_worker instances and IO data of size byte_size
            # WHEN running a job in a worker pool
            q = ProcessPool.get_context().SimpleQueue()
            args = [(data_size, q)] * worker_instances
            pool = ProcessPool(function=f_dynamic_data_size,
                               args=args)
            #  THEN expect all jobs to be processed and all data to be transferred
            completed_tasks_queue = pool.run_async()
            processed_results: int = 0
            for _ in range(0, len(args)):
                assert completed_tasks_queue.get().process.exitcode == 0
                assert q.get().shape == (data_size, )
                processed_results += 1
            assert processed_results == len(args)
            pool.close()
            del q


def test_capture_worker_error():
    """
    Test capturing exceptions in worker processes
    :return:
    """
    # GIVEN a worker pool
    pool = ProcessPool(function=f_erroneous, args=[(None, )])
    # WHEN running a function that raises an exception
    completed_tasks = pool.run()
    # THEN expect it to propagate to main process
    for task in completed_tasks:
        assert task.process.exitcode == 1
    pool.close()


def test_dummypool():
    """
    Test in-main-thread processing
    :return:
    """
    # GIVEN a pool that's running in main thread'
    q = ProcessPool.get_context().SimpleQueue()
    pool = DummyPool(function=f_return_value, args=[(1, q)] * 10)
    # WHEN running computations
    completed_tasks = pool.run()
    # THEN expect behavior to be identical to real pool
    result_sum = 0
    for _ in completed_tasks:
        result_sum += q.get()
    assert result_sum == 10
    pool.close()
    del q


def test_dummypool_async():
    """
    Test in-main-thread processing
    :return:
    """
    # GIVEN a pool that's running in main thread'
    q = ProcessPool.get_context().SimpleQueue()
    args = [(1, q)] * 10
    pool = DummyPool(function=f_return_value, args=args)
    # WHEN running computations
    completed_tasks_queue = pool.run_async()
    # THEN expect behavior to be identical to real pool
    result_sum = 0
    for _ in range(0, len(args)):
        assert completed_tasks_queue.get(timeout=1).process.exitcode == 0
        result_sum += q.get()
    assert result_sum == 10
    pool.close()
    del q


def test_dummypool_errors():
    """
    Test in-main-thread processing errors
    :return:
    """
    # GIVEN a pool that's running in main thread'
    q = ProcessPool.get_context().SimpleQueue()
    pool = DummyPool(function=f_erroneous, args=[(1, q)] * 10)
    # WHEN running computations
    # THEN expect it to raise exception
    with pt.raises(RuntimeError):
        _ = pool.run()
    pool.close()
    del q
