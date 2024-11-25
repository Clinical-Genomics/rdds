# HDF5 Helper Module

## HD5py vs Process Context
Be aware of process context in conjunction with HD5 library.

Experienced a very intermittent issue in HD5 lib when the process
context was 'fork' compared to 'spawn'.

HD5 file IO failed intermittently with calls to hd5py never returning.

This caused processes to fail/ hang and never completing.

The HD5 manual states that files are actually not closed until
ALL references are removed, so this probably caused a mess
in a process-fork situation.

# Queue vs Process Context
Make sure that Queues are also created with the same multiprocessing.context
as the process, otherwise -SIGSEV will occur in the process.
