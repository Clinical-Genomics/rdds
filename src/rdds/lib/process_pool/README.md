# Process Pool

## Note on multiprocessing.Queue in Multi-producer Setup

Be aware that Queues do have a finite size, which
is system dependant (depends on kernel page size, max 64kB).

This limit is separate to the Queue.maxsize property
(that acts on items of data objects).

Furthermore, a process that has performed put()
will wait to close until the underlying pipe write
call has completed, if the pipe is full.

Therefore, make sure to have the reader side
get() on the Queue to allow processes
to complete.

If this is not the case, a producer process will
eventually deadlock on the put() method.

This issue gets more pronounced the more producers
and data chunks there are.

A way to mitigate this is to create a Queue
for every process, but it's just hiding
the problem a bit longer.

The same applies to sockets.

Read more in:
https://bugs.python.org/issue35378
https://bugs.python.org/issue8426
https://bugs.python.org/issue28699
https://bugs.python.org/issue28696
https://bugs.python.org/issue35629
https://bugs.python.org/issue35378

Patch to supposedly fix above issues:
https://github.com/python/cpython/commit/3766f18c524c57784eea7c0001602017d2122156
