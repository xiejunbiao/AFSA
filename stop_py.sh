ps aux | grep main_start.py |grep -v grep|cut -c 9-15|xargs kill -9
