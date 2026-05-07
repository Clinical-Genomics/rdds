from re import search
from os import environ


def override_openblas_core_type_on_intel_xeon_cpus():
    """
    Due to an OpenBLAS bug on Intel Xeon (Sapphire Rapids) CPUs affecting numerical reproducibility,
    override the OpenBLAS core type to use Haswell core type which affects OpenBlas dynamic init gotoblas_dynamic_init()
    https://github.com/OpenMathLib/OpenBLAS/blob/3da0ff7bc29243cf448ddf12c6766716f1530ea2/driver/others/dynamic.c#L1126

    For example, this manifests on AWS VMs:
    - m7i-flex.xlarge
    - m8i.xlarge

    This makes Numpy and Tensorflow to be numerically reproducible on Intel Xeon CPUs.

    Set OPENBLAS_VERBOSE=2 to have OpenBLAS print detected core type on stdout.

    https://github.com/tensorflow/tensorflow/issues/62096
    https://github.com/numpy/numpy/issues/24903
    https://blogs.ed.ac.uk/sopa-scientific-computing/2023/02/07/openblas-optimisations/ (performance degradation on AMD EPYC cores)
    https://www.openmathlib.org/OpenBLAS/
    """
    print('Checking OpenBLAS CPU architecture compatibility...')
    try:
        with open('/proc/cpuinfo', 'r') as file:
            cpuinfo = file.read()
            matches = search('.*model name.*Intel.*Xeon.*', cpuinfo)
            if matches is None:
                # No Xeon CPU, no change needed
                return
            environ['OPENBLAS_CORETYPE'] = 'Haswell'
            print('Detected Intel Xeon CPU, forcing OpenBLAS coretype: Haswell.')
    except Exception as e:
        print(f'WARNING: Failed to determine CPU core type: {e}.\nNumerical reproducibility not guaranteed on Intel Xeon CPUs!')