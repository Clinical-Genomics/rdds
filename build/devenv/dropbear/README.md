# Dropbear Patches
This directory contains patches to dropbear SSH server.
https://github.com/mkj/dropbear

Dropbear is a small, lightweight SSH server, compared to OpenSSH.

The main goal of these patches is to disable commands requiring root level access
executed by dropbear. I.e. these patches are modifying dropbear to run
without root privileges (inside a `--fakeroot`-ed singularity container).

* `chown` command (used for modifying /dev/pts)
* `setuid` command (changing UID while instantiating SSH terminal)

The reason for this requirement is that inside a SLURM singularity container,
there can be no calls to such commands, they're disallowed by the singularity
installation configuration on Hasta.

Tested and works with dropbear tag `DROPBEAR_2022.83`.
