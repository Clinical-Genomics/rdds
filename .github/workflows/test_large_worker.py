name: TestLargeWorker

on:
  push:

jobs:
  test-worker:
    timeout-minutes: 2
    runs-on: ubuntu-latest
    steps:

      - name: Hello world
        run: |
            echo hello world!
