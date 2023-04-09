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
	unsigned long long utime, stime, start_time;
	unsigned long long utime_sec, stime_sec, start_time_sec;
	unsigned long long utime_msec, stime_msec, start_time_msec;
	long long cpu_usage = 0;
	long long elapsed_nsec, usage_nsec;
	long long elapsed_sec, usage_sec;
	int clk_tck = 100;
	int number_of_cpu = 2;

	if (task == NULL)
	{
		return -EINVAL;
	}

	utime = task->utime;
	stime = task->stime;
	start_time = task->start_time;

	utime_sec = utime / clk_tck;
	stime_sec = stime / clk_tck;
	start_time_sec = start_time / clk_tck;

	// need to convert /proc/uptime into human readable format
 	struct file *f;
    char buf[128];
    int len;

    // Open the /proc/uptime file for reading
    f = filp_open("/proc/uptime", O_RDONLY, 0);
    if (!f) {
        printk(KERN_ERR "Error opening /proc/uptime\n");
        return 123;
    }

    // Read the contents of the file into a buffer
    len = kernel_read(f, buf, sizeof(buf), 0);
    if (len <= 0) {
        printk(KERN_ERR "Error reading /proc/uptime\n");
        return 234;
    }

	// Convert the uptime value from a string to a float
    float uptime;
    sscanf(buf, "%f", &uptime);

    // Calculate the number of days, hours, minutes, and seconds in the uptime value
    int days = uptime / 86400;
    int hours = (uptime / 3600) % 24;
    int mins = (uptime / 60) % 60;
    int secs = (int) uptime % 60;

    // Print the uptime in a human-readable format
    printk(KERN_INFO "Uptime: %d days, %02d:%02d:%02d\n", days, hours, mins, secs);

	return uptime;

	uptime = ktime_divns(ktime_get_coarse_boottime(), NSEC_PER_SEC);

	elapsed_sec = (long)uptime - start_time_sec;
	usage_sec = utime_sec + stime_sec;
	cpu_usage = usage_sec * 100 / elapsed_sec;

	return uptime;
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