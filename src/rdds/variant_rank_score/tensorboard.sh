#!/bin/bash

ssh -F build/devenv/devenv.sshconfig -oUserKnownHostsFile=/dev/null devenv ". /opt/pyenv/bin/activate && tensorboard --logdir /rdds/tmp/variant-rank-score/models --port 4000" &
sleep 5
ssh -F build/devenv/devenv.sshconfig -oUserKnownHostsFile=/dev/null -L 4000:localhost:4000 -N devenv
