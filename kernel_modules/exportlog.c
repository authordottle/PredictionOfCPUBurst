/********* exportlog.c ***********/
// Export logger into actual file
#include "headers.h"

#ifndef __KERNEL__
#define __KERNEL__
#endif

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Kernel module to export contents of virtual file in /proc to actual file on disk");

#define DEVICE_NAME "export_file"
#define VIRTUAL_FILE_NAME "virtual_file"
#define ACTUAL_FILE_PATH "/tmp/actual_file"
#define PROC_FILE_PATH "/proc/log_file"

static int major_num;
struct file *virtual_file = NULL;
struct file *actual_file = NULL;
static char buffer[256];
static int buffer_size;

static void *proc_seq_start(struct seq_file *s, loff_t *pos)
{
	printk("exportlog: Hit proc_seq_start");

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
	printk("exportlog: Hit proc_seq_next");

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
	printk("exportlog: Hit proc_seq_stop");
}

static int proc_seq_show(struct seq_file *s, void *v)
{
	printk("exportlog: Hit proc_seq_show");

	loff_t *spos = (loff_t *)v;

	seq_printf(s,
			   "PID\t NAME\t ELAPSED_TIME\t TOTAL_TIME\t utime\t stime\t start_time\t uptime\t\n");

	return 0;
}

static struct seq_operations proc_seq_ops = {
	.start = proc_seq_start,
	.next = proc_seq_next,
	.stop = proc_seq_stop,
	.show = proc_seq_show};

static ssize_t device_read(struct file *file, char *buffer, size_t length, loff_t *offset)
{
    printk(KERN_INFO "Hit device_read\n");
    printk(KERN_INFO "Read operation not supported\n");
    return -EINVAL;
}

static ssize_t device_write(struct file *file, const char *buffer, size_t length, loff_t *offset)
{
    printk(KERN_INFO "Hit device_write\n");
    printk(KERN_INFO "Write operation not supported\n");
    return -EINVAL;
}

static int device_open(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Hit device_open\n");
	return seq_open(file, &proc_seq_ops);
}

static int device_release(struct inode *inode, struct file *file)
{
    printk(KERN_INFO "Hit device_release\n");
    printk(KERN_INFO "Device closed\n");
    return 0;
}

//     int ret = 0;

//     // Copy the virtual file's contents to the buffer
//     ret = kernel_read(virtual_file, *offset, buffer, length);
//     if (ret < 0) {
//         pr_err("Failed to read from virtual file\n");
//         return -EINVAL; // Return "Invalid argument" error
//     }
//     buffer_size = ret;

// printk(KERN_INFO "buffer is %d\n", buffer_size);



// return buffer_size;






    // *buffer = NULL;
    // ssize_t bytes_read;

    // // Allocate a buffer to read data from the virtual file
    // buffer = kmalloc(PAGE_SIZE, GFP_KERNEL);
    // if (!buffer)
    // {
    //     pr_err("Failed to allocate memory for buffer\n");
    //     return -EINVAL; // Return "Invalid argument" error
    // }

    // // Read data from the virtual file and write it to the actual file on disk
    // while ((bytes_read = kernel_read(virtual_file, buffer, PAGE_SIZE, &virtual_file->f_pos)) > 0)
    // {
    //     ssize_t bytes_written = kernel_write(actual_file, buffer, bytes_read, &actual_file->f_pos);
    //     if (bytes_written != bytes_read)
    //     {
    //         pr_err("Failed to write data to %s\n", ACTUAL_FILE_PATH);
    //         return -EINVAL; // Return "Invalid argument" error
    //     }
    // }





    // // Write the buffer to the actual file
    // ret = kernel_write(actual_file, buffer, buffer_size, 0);
    // if (ret < 0) {
    //     pr_err("Failed to read from actual file\n");
    //     return -EINVAL; // Return "Invalid argument" error
    // }

    // *offset += buffer_size;
    // return buffer_size;

static const struct file_operations proc_file_fops = {
    .owner = THIS_MODULE,
    .read = device_read,
    .write = device_write,
    .open = device_open,
    .release = device_release,
};

static int __init init_kernel_module(void)
{
    printk(KERN_INFO "Export file module loaded\n");

    // Open the virtual file
    virtual_file = filp_open(PROC_FILE_PATH, O_RDONLY, 0);
    if (IS_ERR(virtual_file))
    {
        pr_err("Failed to open virtual file\n");
        return -EINVAL; // Return "Invalid argument" error
    }

    // Create the actual file on disk
    actual_file = filp_open(ACTUAL_FILE_PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (IS_ERR(actual_file))
    {
        pr_err("Failed to create actual file\n");
        return -EINVAL; // Return "Invalid argument" error
    }

    // Register the device
    major_num = register_chrdev(0, DEVICE_NAME, &proc_file_fops);
    if (major_num < 0)
    {
        pr_err("Failed to register device\n");
        return major_num;
    }

    return 0;
}

static void __exit exit_kernel_module(void)
{

    if (virtual_file)
    {
        filp_close(virtual_file, NULL);
    }
    if (actual_file)
    {
        filp_close(actual_file, NULL);
    }
    if (buffer)
    {
        kfree(buffer);
    }

    printk(KERN_INFO "Export file module unloaded\n");
}

module_init(init_kernel_module);
module_exit(exit_kernel_module);
