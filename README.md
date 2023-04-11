# Prediction of Process Performance

## Requirements:
1. This project requires both linux and MacOS environments to collect data respectively.
2. Python version >= 3.0; use pip3;
3. Linux enrionment dependencies:
	```sh
	$ sudo apt-get install build-essential
	```
4. Python dependencies:
	```sh
   $ pip3 install -U scikit-learn
   $ python3 -m pip install -U matplotlib
   $ pip3 install seaborn
	```

## How to use proclog kernel module to collect linux sample data:
1. cd into kernel_modules folder
2. Make the proclog kernel module by following command.
   ```sh
   $ make
   ```
3. Install the kernel module by following command. Then a file /proc/log_file can be written to by kernel module.
   ```sh
   $ make exec
   ```  
4. You can verify this by running the following code, which lists all of the modules currently in the kernel. Among them, you should see proclog. Note that the kernel replaces dashes in your module’s filename with underscores when it loads it. If you want to remove it, you can run the following command. 
   ```sh
   $ lsmod
   ```
   ```sh
   $ make stop
   ```
5. You can verify the virtual file by running the following code, which lists all of process tasks.
   ```sh
   $ cat /proc/log_file
   ```
6. In the source code, we can check some logging to let it be known our module loaded okay by running the following command. If the operation gets denied, set the restriction to 0.  
   ```sh
   $ dmesg 
   ```
   ```sh
   $ sudo sysctl kernel.dmesg_restrict=0
   ```
7. Currently, we need to temporarily use export_log_file.c to export log_file from virtual file to actual file.
   ```sh
   $ ./export_log_file 
   ```

## How to use test functions to ensure kernel module is working correctly:
1. cd into test folder
2. if you plan to test tasks, you need to run the following:
   ```sh
   $ make
   ```
   ```sh
   $ ./test_tasks
   ```
3. if you plan to test CPU usage, you need to run the following:
   ```sh
   $ bash ./test_cpu_usage.sh
   ``` 

## How to use python functions to collect mac processes data:
1. cd into data_collecting folder   
2. 
   ```sh
   $ sudo python3 index.py
   ```

## How to run python functions to predict processes performance:
1. 
 