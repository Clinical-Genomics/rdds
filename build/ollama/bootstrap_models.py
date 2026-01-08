import subprocess as sp
from time import sleep

serve_proc = sp.Popen(['/opt/ollama/bin/ollama', 'serve'])
sleep(4)
sp.run(['/opt/ollama/bin/ollama', 'pull', 'llama3.2'], check=True)

print('Bootstrapping complete.')