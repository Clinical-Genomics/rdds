import resource
from os import getpid


_SCALE = {'kB': 1000.0, 'mB': 1000.0*1000.0, 'KB': 1024.0, 'MB': 1024.0*1024.0}


class ProcessResourceUsage:

    """
    Interfaces host 'getrusage' tool and proc-fs to acquire process runtime resource usage.
    Read more in 'man getrusage' and 'man proc'.
    """

    def __init__(self):
        self._pid = getpid()
        self._proc_status = '/proc/%d/status' % self._pid

    @staticmethod
    def get_max_rss() -> float:
        """
        On a Linux machine, return the maximum RAM (resident memory) consumed by this process (including threads).
        :return: Max ram consumed in Bytes
        """
        usage_ram_KiB: int = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        usage_ram_B: float = float(usage_ram_KiB * 1E3)
        return usage_ram_B

    @staticmethod
    def get_max_rss_children() -> float:
        """
        On a Linux machine, return the maximum RAM (resident memory) consumed by largest children (including threads).
        :return: Max ram consumed in Bytes
        """
        usage_ram_KiB: int = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        usage_ram_B: float = float(usage_ram_KiB * 1E3)
        return usage_ram_B

    def _read_proc_status_virtual_memory(self, key: str):
        """
        Read /proc/PID/status and return field 'key'
        :param key: Str
        :return: Content in Key
        """
        try:
            t = open(self._proc_status)
            v = t.read()
            t.close()
        except:
            return 0.0  # non-Linux?
        # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
        i = v.index(key)
        v = v[i:].split(None, 3)  # whitespace
        if len(v) < 3:
            return 0.0  # invalid format?
        # convert Vm value to bytes
        return float(v[1]) * _SCALE[v[2]]

    def get_current_rss(self) -> float:
        """
        On a Linux machine, return current RAM (resident memory) usage.
        :return: Current RAM usage in bytes
        """
        return self._read_proc_status_virtual_memory('VmRSS:')

    def get_current_vm(self) -> float:
        """
        On a Linux machine, return current virtual memory.
        :return:
        """
        return self._read_proc_status_virtual_memory('VmSize:')

    def get_max_vm(self) -> float:
        """
        On a Linux machine, return current virtual memory (swap + disk) usage.
        :return:
        """
        return self._read_proc_status_virtual_memory('VmPeak:')

    def get_stack_size_bytes(self) -> float:
        """
        On a Linux machine, return size of process stack.
        :return:
        """
        return self._read_proc_status_virtual_memory('VmStk:')

    def __str__(self) -> str:
        s = f'[PID:{self._pid}] '
        s += f'RSS:{(self.get_current_rss() / 1E9):.2f}gB({(self.get_max_rss() / 1E9):.2f}gB), '
        s += f'VM:{(self.get_current_vm() / 1E9):.2f}gB({(self.get_max_vm() / 1E9):.2f}gB), '
        s += f'Stack:{(self.get_stack_size_bytes() / 1E6):.2f}mB, '
        return s
