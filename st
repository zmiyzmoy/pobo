source ~/poker_env/bin/activate
 
rm -f t4.py
wget --no-cache https://raw.githubusercontent.com/zmiyzmoy/pobo/main/t4.py
python3 t4.py > training_output.log 2>&1 &
tail -f training_output.log

ps aux | grep python3
kill -9 <PID>
free -h #checkmemory
nvidia-smi #checkGPU

rm -f __pycache__/t4.cpython-*.pyc
rm -f t4.pyc
kill -9 3360  # Останови текущий процесс
