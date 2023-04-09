#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <linux/proc_fs.h>

int main(){
	remove_proc_entry("/proc/timing_log", NULL);

	printf("t1 executes. \n");

	FILE *proc;
	int test = 321;

	proc = fopen("/proc/log_file", "w");
	if(proc == NULL){
		printf("Could not open log file\n");
		return 0;
	}

	fprintf(proc, "This is test output\n");
	fprintf(proc, "%d\n", test);

	fclose(proc);

	printf("t1 finishes. \n");
}
