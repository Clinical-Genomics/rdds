# README: Tensorboard

## Forwarding from inside development environment
```bash
tensorboard --logdir . --port 4000

From outside of container do:
ssh -F build/devenv/devenv.sshconfig -oUserKnownHostsFile=/dev/null -L 4000:localhost:4000 -N devenv
```
