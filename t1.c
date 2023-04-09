//CSE 2431
//Andrew Maloney, Alec Wilson, Jiaqian Huang

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>

int main(){
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
}
