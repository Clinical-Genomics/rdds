import logging
import threading
from queue import Queue as Queue
from threading import Thread, Lock, Event
from multiprocessing import Process, ProcessError, cpu_count, log_to_stderr, get_context
from multiprocessing.context import SpawnContext
from typing import *
from time import sleep
from dataclasses import dataclass

LOGGER = log_to_stderr()
LOGGER.setLevel(logging.INFO)


@dataclass
class DummyProcess:
    name: str
    exitcode: int


@dataclass
class Task:
    args: Iterable
    process: Union[Process, DummyProcess]


CompletedTaskQueue = Queue


class ProcessPool:

    """
    Implements a simple multiprocessing pool supporting synchronous and asynchronous processing mode.
    """

    def __init__(self,
                 fn: Callable,
                 args: Iterable,
                 process_names: List[str] = None,
                 workers: int = cpu_count() - 1  # Reserve a core for main process
                 ):
        """
        :param fn: The function to execute concurrently
        :param args: Function arguments
        :param process_names: List of names to assign to processes for traceability
        :param workers: Maximum concurrent workers running 'fn' simultaneously
        """
        self._fn = fn
        self._args = args
        self._n_expected_tasks: int = len(self._args)
        self._process_names = process_names
        if self._process_names is None:
            self._process_names: List[Any] = [None] * self._n_expected_tasks
        self._workers: int = workers
        self._running_tasks: List[Task] = []
        self._running_tasks_lock: Lock = Lock()
        self._completed_tasks: List[Task] = []
        self._dispatch_thread: Thread = None
        self._collect_stopped_tasks_thread: Thread = None
        self._completed_tasks_async_queue = CompletedTaskQueue()
        self._processing_complete: Event = Event()
        self._ctx = self.get_context()

    @staticmethod
    def get_context() -> SpawnContext:
        """
        Return context for processes, queues etc.
        Don't fork py process in pool, this reduces RAM consumption by not inheriting objects.
        Add top of file before running any multiprocessing-module related code (breaks otherwise).
        :return:
        """
        return get_context('spawn')

    @property
    def n_expected_tasks(self) -> int:
        """
        Return the number of expected tasks to be executed by the pool.
        :return:
        """
        return self._n_expected_tasks

    def _dispatch(self):
        """
        Method that dispatches processes. Thread exits after all tasks has been dispatched.
        """
        # Start processing
        for process_args, process_name in zip(self._args, self._process_names):
            # TODO: This thread will consume lots of CPU if spinning on IF condition
            while True:
                # If there's an available slot for a new process, break out
                self._running_tasks_lock.acquire()
                if len(self._running_tasks) < self._workers:
                    self._running_tasks_lock.release()
                    break
                self._running_tasks_lock.release()
            # Dispatch a new process
            self._running_tasks_lock.acquire()
            process: Process = self._ctx.Process(target=self._fn,
                                                 args=process_args,
                                                 name=process_name)
            process.start()
            task: Task = Task(args=process_args, process=process)
            LOGGER.info(f'Dispatched task {task}')
            self._running_tasks.append(task)
            self._running_tasks_lock.release()

    def _collect_stopped_tasks(self):
        """
        Collects stopped processes and monitors progress.
        Thread does not exit until all processes are complete.
        :return:
        """
        LOGGER.debug('Starting stop process check')
        while len(self._completed_tasks) < self._n_expected_tasks:
            self._running_tasks_lock.acquire()
            for task in self._running_tasks:
                if not task.process.is_alive():
                    LOGGER.info(f'Process complete: {task.process}')
                    self._running_tasks.remove(task)
                    self._completed_tasks.append(task)
                    self._completed_tasks_async_queue.put(task)
            self._running_tasks_lock.release()

        self._processing_complete.set()
        LOGGER.info('All tasks complete')

    @staticmethod
    def _threading_excepthook(except_hook_args: threading.ExceptHookArgs):
        LOGGER.fatal(except_hook_args)
        raise except_hook_args.exc_value

    def run_async(self) -> CompletedTaskQueue:
        """
        Run processing in asynchronous mode. Task order in CompletedTaskQueue is determined
        by process duration.
        :return: Queue of completed Task as processing is ongoing
        """
        threading.excepthook = self._threading_excepthook
        self._dispatch_thread = Thread(target=self._dispatch,
                                       name='dispatch')
        self._collect_stopped_tasks_thread = Thread(target=self._collect_stopped_tasks,
                                                    name='collect_stopped_processes')
        self._collect_stopped_tasks_thread.start()
        self._dispatch_thread.start()

        return self._completed_tasks_async_queue

    def run(self) -> List[Task]:
        """
        Run processing in blocking mode. Ordering of results is determined by process duration.
        :return: Collected list of all the completed tasks.
        """
        # Dispatch jobs
        async_results = self.run_async()

        LOGGER.debug('Waiting for job completion in run()')
        self._processing_complete.wait()

        results: List[Task] = []
        for i in range(0, self._n_expected_tasks):
            results.append(async_results.get(timeout=1))  # Throws EmptyException
        if len(results) != self._n_expected_tasks:
            raise ValueError(f'Missing Task results: {results}, expected {self._n_expected_tasks} results')
        return results

    def close(self):
        """
        Close all file handles to processes (Process.__attr__, process resources etc) .
        Throws exception if any currently running processes.
        :return:
        """
        LOGGER.info('Shutting down pool.')
        self._processing_complete.wait()
        for task in self._running_tasks:
            task.process.close()
        for task in self._completed_tasks:
            task.process.close()
        if self._dispatch_thread.is_alive():
            raise RuntimeError('Expected dispatch thread to have exited')
        if self._collect_stopped_tasks_thread.is_alive():
            raise RuntimeError('Expected collect_stopped_processes thread to have exited')

class DummyPool:
    """
    Mimics the ProcessPool API but runs all processing in caller thread.
    Good for debugging purposes.

    NOTE: If you're using Queues to large extent, be aware of the risk
    that Queue will fill up causing a deadlock since queue.read() is not
    called until processing completes!
    """
    def __init__(self,
                 fn: Callable,
                 args: Iterable,
                 process_names: List[str] = None,
                 workers: int = None
                 ):
        self._fn = fn
        self._args = args
        self._n_expected_tasks = len(self._args)
        self._process_names = process_names
        if self._process_names is None:
            self._process_names: List[Any] = [f'DummyProcess-{i}' for i in range(0, self._n_expected_tasks)]

    @property
    def n_expected_tasks(self):
        return self._n_expected_tasks

    def run_async(self) -> CompletedTaskQueue:
        queue: Queue = Queue()
        completed_tasks = self.run()
        for task in completed_tasks:
            queue.put(task)
        return queue

    def run(self) -> List[Task]:
        completed_tasks: List[Task] = []
        for arg, process_name in zip(self._args, self._process_names):
            LOGGER.info(f'[DummyPool] Dispatching task {process_name}')
            self._fn(*arg)
            # If function returned to callee, then it's successfully completed.
            task: Task = Task(args=arg, process=DummyProcess(name=process_name, exitcode=0))
            LOGGER.info(f'Task completed {task}')
            completed_tasks.append(task)
        return completed_tasks

    def close(self): pass
