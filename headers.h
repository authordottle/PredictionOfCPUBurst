/********* headers.c ***********/
#include <linux/version.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h> // seq_read, ...
#include <linux/uaccess.h>

/*
    struct Process
        -burst_time: given, time it takes for the process to complete
        -next:      generated, next process in queue
        -previous:  generated, process before self in queue
*/

struct Process
{
    double burst_time;
    pid_t proc_pid;
    struct Process *next;
    struct Process *previous;
};