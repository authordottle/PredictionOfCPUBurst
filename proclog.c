// Logger that creates a proc file
// idea from tldp.org/LDP/lkmpg/2.6/html/index.html
#include <linux/version.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h> // seq_read, ...
#include <linux/uaccess.h>

#ifndef __KERNEL__
#define __KERNEL__
#endif

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Kernel module to log process times");

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
#define HAVE_PROC_OPS
#endif

// #if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 18, 0)
// #define HAVE_PROC_CREATE_SINGLE
// #endif

// size of buffer ~32Kb
#define PROCFS_MAX_SIZE 32768

// buffer to hold information from log
static char procfs_buffer[PROCFS_MAX_SIZE];

// size of buffer
static unsigned long procfs_buffer_size = 0;

// pointer for buffer location in read
static char *buff_ptr;

// struct to hold info about proc file
struct proc_dir_entry *log_file;

static int endflag;

static void *proc_seq_start(struct seq_file *s, loff_t *pos)
{
	printk("Hit proc_seq_start");

	// (*pos) = endflag;

	// buff_ptr = procfs_buffer + ((*pos) * sizeof(char));

	// // if pos is greater than or equal to buffer size then leave sequence read
	// if ((*pos) >= procfs_buffer_size - 1 || *buff_ptr == '\0')
	// {
	// 	printk("End sequence read\n");
	// 	return NULL;
	// }

	// printk("Place in buffer is: %Ld\n", (*pos));

	// return buff_ptr;
	return NULL;
}

static void *proc_seq_next(struct seq_file *s, void *v, loff_t *pos)
{
	printk("Hit proc_seq_next");
	// printk("Sequence Next!");
	// char *temp = (char *)v;
	// while ((*temp) != '\n')
	// {

	// 	(*pos)++;
	// 	printk("position increased");
	// 	if ((*pos) >= procfs_buffer_size)
	// 	{
	// 		return NULL;
	// 	}
	// 	temp++;
	// 	printk("temp increased");
	// }
	// temp++;
	// endflag = (*pos);
	// printk("position is %Ld\n", (*pos));
	// return temp;
	return NULL;
}

static void proc_seq_stop(struct seq_file *s, void *v)
{
	// buff_ptr = NULL;
	printk("Hit proc_seq_stop");
}

static int proc_seq_show(struct seq_file *s, void *v)
{
	struct task_struct *task;
    seq_printf(s, "PID\tNAME\n");
    for_each_process(task) {
        seq_printf(s, "%d\t%s\n", task->pid, task->comm);
    }
	// printk("Showing value");
	// char *temp = (char *)v;
	// do
	// {
	// 	seq_putc(s, *temp);
	// 	temp++;
	// } while (*temp != '\n');
	// seq_putc(s, '\n');
	printk("Hit proc_seq_show");
	return 0;
}

static struct seq_operations proc_seq_ops = {
	.start = proc_seq_start,
	.next = proc_seq_next,
	.stop = proc_seq_stop,
	.show = proc_seq_show};

// static int procfile_open(struct inode *inode, struct file *file)
// {
//     return single_open(file, uptime_proc_show, NULL);
// }

static int procfile_open(struct inode *inode, struct file *file)
{
	printk("Hit procfile_open");
	 return single_open(file, proc_seq_show, NULL);
	// return seq_open(file, &proc_seq_ops);
}

// function to write to proc file
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

static void log_processes(void)
{
	struct task_struct *task;
	for_each_process(task)
	{
		printk(KERN_INFO "Process: %s (pid: %d)\n", task->comm, task->pid);
	}
}

static int __init init_kernel_module(void)
{
	printk(KERN_INFO "Process logger module loaded\n");

	// initialize
	endflag = 0;
// adapted from stackoverflow.com/questions/8516021/proc-create-example-for-kernel-module
// fixed the version issue from https://stackoverflow.com/questions/64931555/how-to-fix-error-passing-argument-4-of-proc-create-from-incompatible-pointer
#ifdef HAVE_PROC_CREATE_SINGLE
	proc_create_single("log_file", 0, NULL, procfile_show);
#else
	proc_create("log_file", 0, NULL, &proc_file_fops);
	// proc_create_data("log_file", 0644, NULL, &proc_file_fops, NULL);
#endif

	// // loop processes
	// log_processes();

	return 0;
}

static void __exit exit_kernel_module(void)
{
	remove_proc_entry("log_file", NULL);
	printk(KERN_INFO "Process logger module unloaded\n");
}

module_init(init_kernel_module);
module_exit(exit_kernel_module);
