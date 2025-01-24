# Jupter Lab Server Configuration File
# jupyter lab --config build/devenv/jupyter_lab_config.py

# Set ip to '*' to bind on all interfaces (ips) for the public server
c.NotebookApp.ip = '*'

# There's no browser in container
c.NotebookApp.open_browser = False

# Access on port 2160
c.NotebookApp.port = 2160

# Allow root, required by runtime in container
c.NotebookApp.allow_root = True
