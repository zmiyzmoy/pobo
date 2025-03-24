 source ~/poker_env/bin/activate
 
rm t4.py
wget https://raw.githubusercontent.com/zmiyzmoy/pobo/main/t4.py
python3 t4.py > training_output.log 2>&1 &
tail -f training_output.log
