// Logger that creates a proc file
// idea from tldp.org/LDP/lkmpg/2.6/html/index.html
#include <linux/version.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/uaccess.h>
#include <linux/sched/signal.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("JH");
MODULE_DESCRIPTION("Kernel module to log records of process time");

#ifndef __KERNEL__
#define __KERNEL__
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
#define HAVE_PROC_OPS
#endif

// size of buffer ~32Kb
#define PROCFS_MAX_SIZE 32768

#define PROCFS_NAME "timing_log"

// buffer to hold information from log
static char procfs_buffer[PROCFS_MAX_SIZE];

// size of buffer
static unsigned long procfs_buffer_size = 0;

// pointer for buffer location in read
static char *buff_ptr;

// struct to hold info about proc file
static struct proc_dir_entry *log_file;

static int endflag;

// static void *proc_seq_start(struct seq_file *s, loff_t *pos)
// {

// 	printk("Start of sequence read!\n");

// 	(*pos) = endflag;

// 	buff_ptr = procfs_buffer + ((*pos) * sizeof(char));

// 	// if pos is greater than or equal to buffer size then leave sequence read
// 	if ((*pos) >= procfs_buffer_size - 1 || *buff_ptr == '\0')
// 	{
// 		printk("End sequence read\n");
// 		return NULL;
// 	}

// 	printk("Place in buffer is: %Ld\n", (*pos));

// 	return buff_ptr;
// }

// static void *proc_seq_next(struct seq_file *s, void *v, loff_t *pos)
// {
// 	printk("Sequence Next!");
// 	char *temp = (char *)v;
// 	while ((*temp) != '\n')
// 	{

// 		(*pos)++;
// 		printk("position increased");
// 		if ((*pos) >= procfs_buffer_size)
// 		{
// 			return NULL;
// 		}
// 		temp++;
// 		printk("temp increased");
// 	}
// 	temp++;
// 	endflag = (*pos);
// 	printk("position is %Ld\n", (*pos));
// 	return temp;
// }

// static void proc_seq_stop(struct seq_file *s, void *v)
// {
// 	printk("Sequence stop!");
// 	buff_ptr = NULL;
// 	printk("Sequence stop 2: electric bugaloo");
// }

// static int proc_seq_show(struct seq_file *s, void *v)
// {
// 	printk("Showing value");
// 	char *temp = (char *)v;
// 	do
// 	{
// 		seq_putc(s, *temp);
// 		temp++;
// 	} while (*temp != '\n');
// 	seq_putc(s, '\n');
// 	return 0;
// }

// static struct seq_operations proc_seq_ops = {
// 	.start = proc_seq_start,
// 	.next = proc_seq_next,
// 	.stop = proc_seq_stop,
// 	.show = proc_seq_show};

/** 
 * This function is called then the /proc file is read
 *
 */
static int procfile_read (char *page, char **start, off_t offset, int count, int *eof, void *data);

static int procfile_read(char *buffer,
	      char **buffer_location,
	      off_t offset, int buffer_length, int *eof, void *data)
{
	int ret;
	
	printk(KERN_INFO "procfile_read (/proc/%s) called\n", PROCFS_NAME);
	
	if (offset > 0) {
		// we have finished to read, return 0 
		ret  = 0;
	} else {
		// fill the buffer, return the buffer size 
		memcpy(buffer, procfs_buffer, procfs_buffer_size);
		ret = sprintf(page, "Hello world\n");
		// ret = procfs_buffer_size;
	}

	return ret;
}

// static int procfile_open(struct inode *inode, struct file *file)
// {
// 	printk("open procfile");
// 	return seq_open(file, &proc_seq_ops);
// }

// function to write to proc file
static ssize_t procfile_write(struct file *file, const char *buffer, size_t count, loff_t *off)
{
	// get buffer size 
	procfs_buffer_size = count;
	if (procfs_buffer_size > PROCFS_MAX_SIZE ) {
		procfs_buffer_size = PROCFS_MAX_SIZE;
		printk("Proc file buffer overflow");
	}
	
	// write data to the buffer 
	if ( copy_from_user(procfs_buffer, buffer, procfs_buffer_size) ) {
		return -EFAULT;
	}
	
	return (ssize_t)procfs_buffer_size;
}

// struct that holds what functions run for different aspects of log file
// #ifdef HAVE_PROC_OPS
// static const struct proc_ops log_file_fops = {
// 	.proc_open = procfile_open,
// 	.proc_write = procfile_write,
// 	.proc_read = seq_read,
// 	.proc_lseek = seq_lseek,
// 	.proc_release = seq_release
// 	};
// #else
// static const struct file_operations log_file_fops = {
// 	.owner = THIS_MODULE,
// 	.read = procfile_read,
// 	.write = procfile_write,
// 	};
// #endif

#ifdef HAVE_PROC_OPS
static struct proc_ops proc_file_fops = {
        .proc_read     = procfile_read,
        .proc_write    = procfile_write,
};
#else
static const struct file_operations proc_file_fops = {
 .owner = THIS_MODULE,
 .read  = procfile_read,
 .write  = procfile_write,
};
#endif



static void log_processes(void)
{
	struct task_struct *task;
	for_each_process(task)
	{
		printk(KERN_INFO "Process: %s (pid: %d)\n", task->comm, task->pid);
	}
}

static int __init init_MyKernelModule(void)
{	
	// adapted from stackoverflow.com/questions/8516021/proc-create-example-for-kernel-module
	// fixed the version issue from https://stackoverflow.com/questions/64931555/how-to-fix-error-passing-argument-4-of-proc-create-from-incompatible-pointer
	log_file = proc_create("timing_log", 0, NULL, &proc_file_fops);
	if (log_file == NULL)
	{
		return -ENOMEM;
	}
	
	// log_file->mode 	  = S_IFREG | S_IRUGO;
	// log_file->uid 	  = 0;
	// log_file->gid 	  = 0;
	// log_file->size 	  = 37;

	endflag = 0;

	printk(KERN_INFO "Process logger module loaded\n");

	// log_processes();

	return 0;
}

static void __exit exit_MyKernelModule(void)
{
	remove_proc_entry("timing_log", NULL);
	printk(KERN_INFO "Process logger module unloaded\n");
	return;
}

module_init(init_MyKernelModule);
module_exit(exit_MyKernelModule);
