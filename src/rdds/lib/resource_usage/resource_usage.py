import resource
from typing import List
from os import getpid
from os.path import join
from enum import Enum


KILOBYTE = float(1E3)
MEGABYTE = float(1E6)
GIGABYTE = float(1E9)
KIBIBYTE = 1024.0
# Definitions to translate prefixes to Bytes.
_SI_ISO_SCALE = {'kB': KILOBYTE, 'mB': KILOBYTE**2, 'KB': KIBIBYTE, 'MB': KIBIBYTE**2}
_INVALID_VALUE = 0.0


class ProcFields(str, Enum):
    """
    Mapping of Proc file internal fields.
    """
    vm_rss = 'VmRSS:'
    vm_size = 'VmSize:'
    vm_peak = 'VmPeak:'
    vm_stk = 'VmStk:'

    def __str__(self) -> str:
        # Return internal value on calling the symbolic name
        return self.value


class ProcessResourceUsage:

    """
    Interfaces host 'getrusage' tool and proc-fs to acquire process runtime resource usage.
    Read more in 'man getrusage' and 'man proc'.
    """

    def __init__(self):
        self._pid = getpid()
        self._proc_status: str = join('/proc', '%s/status' % self._pid)

    @staticmethod
    def get_max_rss() -> float:
        """
        On a Linux machine, return the maximum RAM (resident memory) consumed by this process (including threads).
        :return: Max ram consumed in Bytes
        """
        usage_ram_KiB: int = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(usage_ram_KiB * KILOBYTE)

    @staticmethod
    def get_max_rss_children() -> float:
        """
        On a Linux machine, return the maximum RAM (resident memory) consumed by largest children (including threads).
        :return: Max ram consumed in Bytes
        """
        usage_ram_KiB: int = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        return float(usage_ram_KiB * KILOBYTE)

    @staticmethod
    def _get_parse_key_from_proc_file(proc_file_content: str,
                                      key) -> float:
        """

        :param proc_file_content: String contents of proc file
        :param key: Key to fetch
        :return: Parsed value, as float
        """
        key_index_start: int = proc_file_content.index(key)
        entry: List[str] = proc_file_content[key_index_start:].split(None, 3)  # whitespace removal
        if len(entry) < 3:
            return _INVALID_VALUE  # invalid format found
        return float(entry[1]) * _SI_ISO_SCALE[entry[2]]

    def _read_proc_status_virtual_memory(self, key: str) -> float:
        """
        Read /proc/PID/status and return field 'key'
        :param key: Str
        :return: Content in Key as float
        """
        try:
            proc_file = open(self._proc_status)
            proc_file_content: str = proc_file.read()
            proc_file.close()
        except:
            return _INVALID_VALUE  # Failed reading proc file
        return self._get_parse_key_from_proc_file(proc_file_content=proc_file_content,
                                                  key=key)

    def get_current_rss(self) -> float:
        """
        On a Linux machine, return current RAM (resident memory) usage.
        :return: Current RAM usage in bytes
        """
        return self._read_proc_status_virtual_memory(ProcFields.vm_rss)

    def get_current_vm(self) -> float:
        """
        On a Linux machine, return current virtual memory.
        :return:
        """
        return self._read_proc_status_virtual_memory(ProcFields.vm_size)

    def get_max_vm(self) -> float:
        """
        On a Linux machine, return current virtual memory (swap + disk) usage.
        :return:
        """
        return self._read_proc_status_virtual_memory(ProcFields.vm_peak)

    def get_stack_size_bytes(self) -> float:
        """
        On a Linux machine, return size of process stack.
        :return:
        """
        return self._read_proc_status_virtual_memory(ProcFields.vm_stk)

    def __str__(self) -> str:
        info_string = f'[PID:{self._pid}] '
        info_string += f'RSS:{(self.get_current_rss() / GIGABYTE):.2f}gB({(self.get_max_rss() / GIGABYTE):.2f}gB), '
        info_string += f'VM:{(self.get_current_vm() / GIGABYTE):.2f}gB({(self.get_max_vm() / GIGABYTE):.2f}gB), '
        info_string += f'Stack:{(self.get_stack_size_bytes() / MEGABYTE):.2f}mB, '
        return info_string
