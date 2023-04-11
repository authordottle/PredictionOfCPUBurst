/********* proclog.c ***********/
// Logger that creates a proc file
#include "headers.h"
#include "helper.c"

#ifndef __KERNEL__
#define __KERNEL__
#endif

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Kernel module to log process times");

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
#define HAVE_PROC_OPS
#endif

static void *proc_seq_start(struct seq_file *s, loff_t *pos)
{
	printk("Hit proc_seq_start");

	static unsigned long counter = 0;
	// beginning a new sequence ? 
	if (*pos == 0)
	{
		// yes => return a non null value to begin the sequence 
		return &counter;
	}
	else
	{
		// no => it's the end of the sequence, return end to stop reading 
		*pos = 0;
		return NULL;
	}
}

static void *proc_seq_next(struct seq_file *s, void *v, loff_t *pos)
{
	printk("Hit proc_seq_next");

	char *temp = (char *)v;
	temp++;
	(*pos)++;
	return NULL;
}

static void proc_seq_stop(struct seq_file *s, void *v)
{
	printk("Hit proc_seq_stop");
}

static long get_process_elapsed_time(struct task_struct *task)
{
	// /proc/[PID]/stat
	// #14 utime - CPU time spent in user code, measured in clock ticks
	// #15 stime - CPU time spent in kernel code, measured in clock ticks
	// #16 cutime - Waited-for children's CPU time spent in user code (in clock ticks)
	// #17 cstime - Waited-for children's CPU time spent in kernel code (in clock ticks)
	// #22 starttime - Time when the process started, measured in clock ticks
	// unsigned long long utime, stime, cutime, cstime;
	unsigned long long start_time;
	// unsigned long long utime_sec, stime_sec, start_time_sec;
	// unsigned long long utime_msec, stime_msec, start_time_msec;
	// unsigned long long total_time;
	// long long cpu_usage = 0;
	// long long elapsed_nsec;
	// long long usage_nsec;
	long long elapsed_sec;
	// long long usage_sec;

	if (task == NULL)
	{
		pr_err("An error occurred in task\n");
		return -EINVAL; // Return "Invalid argument" error
	}

	// The reason for this is that the utime value in the /proc/[pid]/stat file is measured in clock ticks,
	// whereas the utime field in the task_struct is measured in nanoseconds.
	start_time = task->start_time;

	// kernel system timer
	uptime = ktime_divns(ktime_get_coarse_boottime(), NSEC_PER_SEC);

	elapsed_sec = (long long)(uptime * 1000000000) - start_time;

	return elapsed_sec;
}

static int proc_seq_show(struct seq_file *s, void *v)
{
	printk("Hit proc_seq_show");

	// ktime_t current_time = ktime_get();
	// s64 current_time_ns = ktime_to_ns(current_time);
	// long current_time_s = current_time_ns / 1000000000;
	long duration_time_s = 0;
	do
	{
		// current_time = ktime_get();
		duration_time_s ++;
		// current_time_ns = ktime_to_ns(current_time);
		// current_time_s = current_time_ns / 1000000000;
		// duration_time_s = current_time_s - start_time_s;
		// printk(KERN_INFO "%lld\n", duration_time_s);
	} while (duration_time_s <= 100);
	// start_time_s = current_time;

	loff_t *spos = (loff_t *)v;
	unsigned long long utime, stime;
	// unsigned long long cutime, cstime, start_time;
	unsigned long long total_time;
	struct task_struct *task;

	seq_printf(s,
			   "PID\t NAME\t ELAPSED_TIME\t TOTAL_TIME\t utime\t stime\t start_time\t uptime\t\n");
	for_each_process(task)
	{
		// printk(KERN_INFO "Process: %s (pid: %d)\n", task->comm, task->pid);

		utime = task->utime;
		stime = task->stime;
		total_time = utime + stime;
		long elapsed_time = get_process_elapsed_time(task);

		seq_printf(s,
				   "%d\t %s\t %ld\t %lld\t %lld\t %lld\t %lld\t %lld\t\n ",
				   task->pid,
				   task->comm,
				   elapsed_time,
				   total_time,
				   task->utime,
				   task->stime,
				   task->start_time,
				   ktime_divns(ktime_get_coarse_boottime(), NSEC_PER_SEC));
	}

	seq_printf(s, "%Ld\n", *spos);

	return 0;
}

static struct seq_operations proc_seq_ops = {
	.start = proc_seq_start,
	.next = proc_seq_next,
	.stop = proc_seq_stop,
	.show = proc_seq_show};

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

	ktime_t start_time = ktime_get();
	s64 start_time_ns = ktime_to_ns(start_time);
	start_time_s = start_time_ns / 1000000000;

	struct proc_dir_entry *log_file;

	// printk(KERN_INFO "There are %d running processes.\n", proc_count());

	log_file = proc_create("log_file", 0, NULL, &proc_file_fops);

	return 0;
}

static void export_virtual_file_into_actual_file(void)
{
	// Allocate a buffer to read data from the virtual file
	char *buffer = (char *)kmalloc(PAGE_SIZE, GFP_KERNEL);
	if (!buffer)
	{
		pr_err("Failed to allocate memory for buffer\n");
	}

	// Create the actual file on disk
	actual_file = filp_open(ACTUAL_FILE_PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (IS_ERR(actual_file))
	{
		pr_err("Failed to create actual file\n");
	}


	if (buffer)
	{
		kfree(buffer);
	}
	if (virtual_file)
	{
		filp_close(virtual_file, NULL);
	}
	if (actual_file)
	{
		filp_close(actual_file, NULL);
	}
}

static void __exit exit_kernel_module(void)
{
	export_virtual_file_into_actual_file();
	remove_proc_entry("log_file", NULL);

	printk(KERN_INFO "Process logger module unloaded\n");
}

module_init(init_kernel_module);
module_exit(exit_kernel_module);