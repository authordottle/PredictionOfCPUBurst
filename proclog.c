/********* proclog.c ***********/
// Logger that creates a proc file
// idea from tldp.org/LDP/lkmpg/2.6/html/index.html
#include "headers.h"
#include "process.c"

#ifndef __KERNEL__
#define __KERNEL__
#endif

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Kernel module to log process times");

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
#define HAVE_PROC_OPS
#endif

// NOT APPLICABLE IN THIS CASE
// #if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 18, 0)
// #define HAVE_PROC_CREATE_SINGLE
// #endif

static void *proc_seq_start(struct seq_file *s, loff_t *pos)
{
	printk("Hit proc_seq_start");

	static unsigned long counter = 0;

	/* beginning a new sequence ? */
	if (*pos == 0)
	{
		/* yes => return a non null value to begin the sequence */
		return &counter;
	}
	else
	{
		/* no => it's the end of the sequence, return end to stop reading */
		*pos = 0;
		return NULL;
	}
}

static void *proc_seq_next(struct seq_file *s, void *v, loff_t *pos)
{
	printk("Hit proc_seq_next");

	char *temp = (char *)v;
	temp++;
	printk("Temp increased.");
	(*pos)++;
	printk("Position increased.");
	printk("Position is %Ld\n", (*pos));
	return NULL;
}

static void proc_seq_stop(struct seq_file *s, void *v)
{
	printk("Hit proc_seq_stop");
}

static long get_process_cpu_usage(struct task_struct *task)
{
	// adapted from https://stackoverflow.com/questions/16726779/how-do-i-get-the-total-cpu-usage-of-an-application-from-proc-pid-stat
	// /proc/[PID]/stat
	// #14 utime - CPU time spent in user code, measured in clock ticks
	// #15 stime - CPU time spent in kernel code, measured in clock ticks
	// #16 cutime - Waited-for children's CPU time spent in user code (in clock ticks)
	// #17 cstime - Waited-for children's CPU time spent in kernel code (in clock ticks)
	// #22 starttime - Time when the process started, measured in clock ticks
	unsigned long long utime, stime, cutime, cstime, start_time;
	unsigned long long utime_sec, stime_sec, start_time_sec;
	unsigned long long utime_msec, stime_msec, start_time_msec;
	long long cpu_usage = 0;
	long long elapsed_nsec, usage_nsec;
	long long elapsed_sec, usage_sec;
	int clk_tck = 100; // constants
	int number_of_cpu = 2;

	if (task == NULL)
	{
		return -EINVAL;
	}

	utime = task->utime;
	stime = task->stime;
	cutime = task->cutime;
	cstime = task->cstime;
	start_time = task->start_time;

	total_time = utime + stime;
	if (cutime != 0) {
		total_time += cutime;
	}
	if (cstime != 0) {
		total_time += cstime;
	}

	start_time_sec = start_time / clk_tck;

	uptime = ktime_divns(ktime_get_coarse_boottime(), NSEC_PER_SEC);

	elapsed_sec = (long)uptime - start_time_sec;
	cpu_usage = total_time / clk_tck / elapsed_sec * 100;

	return elapsed_sec;
}

static int proc_seq_show(struct seq_file *s, void *v)
{
	printk("Hit proc_seq_show");

	loff_t *spos = (loff_t *)v;

	struct task_struct *task;

	seq_printf(s,
			   "PID\t NAME\t CPU_USAGE\t start_time\t stime\t utime\t\n");
	for_each_process(task)
	{
		printk(KERN_INFO "Process: %s (pid: %d)\n", task->comm, task->pid);

		/* Get CPU usage for the process */
		long cpu_usage = get_process_cpu_usage(task);

		seq_printf(s,
				   "%d\t %s\t %lld\t %d\t \n ",
				   task->pid,
				   task->comm,
				   cpu_usage);
		//    task->start_time,
		//    task->stime,
		//   task->utime);
	}

	seq_printf(s, "%Ld\n", *spos);

	// char *temp = (char *)v;
	// do
	// {
	// 	seq_putc(s, *temp);
	// 	temp++;
	// } while (*temp != '\n');
	// seq_putc(s, '\n');

	return 0;
}

static struct seq_operations proc_seq_ops = {
	.start = proc_seq_start,
	.next = proc_seq_next,
	.stop = proc_seq_stop,
	.show = proc_seq_show};

// static int procfile_open(struct inode *inode, struct file *file)
// {
// 	printk("Hit procfile_open");
//     return single_open(file, proc_seq_show, NULL);
// }

static int procfile_open(struct inode *inode, struct file *file)
{
	printk("Hit procfile_open");
	return seq_open(file, &proc_seq_ops);
}

static ssize_t procfile_write(struct file *file, const char *buffer, size_t count, loff_t *off)
{
	printk("Hit procfile_write");
	return 1;
}

static int procfile_show(struct seq_file *m, void *v)
{
	printk("Hit procfile_show");
	return 0;
}

#ifdef HAVE_PROC_OPS
static const struct proc_ops proc_file_fops = {
	.proc_open = procfile_open,
	.proc_write = procfile_write,
	.proc_read = seq_read,
	.proc_lseek = seq_lseek,
	.proc_release = seq_release};
#else
static const struct file_operations proc_file_fops = {
	.owner = THIS_MODULE,
	.open = procfile_open,
	.write = procfile_write,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = seq_release};
#endif

static int __init init_kernel_module(void)
{
	printk(KERN_INFO "Process logger module loaded\n");

	// initialize: 1. struct to hold info about proc file 2. other variables
	struct proc_dir_entry *log_file;
	endflag = 0;

// adapted from stackoverflow.com/questions/8516021/proc-create-example-for-kernel-module
// fixed the version issue from https://stackoverflow.com/questions/64931555/how-to-fix-error-passing-argument-4-of-proc-create-from-incompatible-pointer
#ifdef HAVE_PROC_CREATE_SINGLE
	proc_create_single("log_file", 0, NULL, procfile_show);
#else
	proc_create("log_file", 0, NULL, &proc_file_fops);
#endif

	return 0;
}

static void __exit exit_kernel_module(void)
{
	remove_proc_entry("log_file", NULL);
	printk(KERN_INFO "Process logger module unloaded\n");
}

module_init(init_kernel_module);
module_exit(exit_kernel_module);