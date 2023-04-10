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
#define ACTUAL_FILE_NAME "/tmp/actual_file"
#define PROC_FILE_PATH "/proc/log_file"

struct file *virtual_file;
struct file *disk_file;

static int major_num;
static struct file* virtual_file;
static struct file* actual_file;
static char buffer[256];
static int buffer_size;

static int device_open(struct inode* inode, struct file* file) {
    printk(KERN_INFO "Device opened\n");
    return 0;
}

static int device_release(struct inode* inode, struct file* file) {
    printk(KERN_INFO "Device closed\n");
    return 0;
}

static ssize_t device_read(struct file* file, char* buffer, size_t length, loff_t* offset) {
    printk(KERN_INFO "Read operation not supported\n");
    return -EINVAL;
}

static ssize_t device_write(struct file* file, const char* buffer, size_t length, loff_t* offset) {
    printk(KERN_INFO "Write operation not supported\n");
    return -EINVAL;
}

static ssize_t device_export(struct file* file, const char __user *buf, size_t length, loff_t* offset) {
    int ret = 0;

    // Copy the virtual file's contents to the buffer
    ret = kernel_read(virtual_file, *offset, buffer, length);
    if (ret < 0) {
        printk(KERN_ERR "Failed to read from virtual file\n");
        return ret;
    }
    buffer_size = ret;

    // Open the actual file
    actual_file = filp_open(ACTUAL_FILE_NAME, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (IS_ERR(actual_file)) {
        printk(KERN_ERR "Failed to open actual file\n");
        return PTR_ERR(actual_file);
    }

    // Write the buffer to the actual file
    ret = kernel_write(actual_file, buffer, buffer_size, 0);
    if (ret < 0) {
        printk(KERN_ERR "Failed to write to actual file\n");
        return ret;
    }

    // Cleanup
    filp_close(actual_file, NULL);
    *offset += buffer_size;
    return buffer_size;
}

static const struct file_operations proc_file_fops = {
	  .read = device_read,
    .write = device_write,
    .open = device_open,
    .release = device_release,
    .owner = THIS_MODULE,
    //.read_iter = device_export, // Use write_iter to support large files
    };

static int __init mymodule_init(void)
{
    *virtual_file = NULL;
    *disk_file = NULL;
    
    // Open the virtual file
    virtual_file = filp_open(PROC_FILE_PATH, O_RDONLY, 0);
    if (IS_ERR(virtual_file)) {
        printk(KERN_ERR "Failed to open virtual file\n");
        return PTR_ERR(virtual_file);
    }

    // Create the actual file on disk
    disk_file = filp_open(ACTUAL_FILE_NAME, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (IS_ERR(disk_file))
    {
         printk(KERN_ERR "Failed to create actual file\n");
        return PTR_ERR(virtual_file);
    }

    // Register the device
    major_num = register_chrdev(0, DEVICE_NAME, &proc_file_fops);
    if (major_num < 0) {
        printk(KERN_ERR "Failed to register device\n");
        return major_num;
    }
    printk(KERN_INFO "Export file module loaded\n");

    return 0;
}

static void __exit mymodule_exit(void)
{

    if (virtual_file)
    {
        filp_close(virtual_file, NULL);
    }
    if (disk_file)
    {
        filp_close(disk_file, NULL);
    }
    // if (buffer)
    // {
    //     kfree(buffer);
    // }

    pr_info("Module unloaded successfully\n");
}

module_init(mymodule_init);
module_exit(mymodule_exit);


// static int copy_proc_file_to_disk(const char *proc_file_path, const char *disk_file_path)
// {
//     char *buffer = NULL;
//     ssize_t bytes_read;
//     int err = 0;

//     // Allocate a buffer to read data from the virtual file
//     buffer = kmalloc(PAGE_SIZE, GFP_KERNEL);
//     if (!buffer)
//     {
//         err = -ENOMEM;
//         pr_err("Failed to allocate memory for buffer\n");
//         goto exit;
//     }

//     // Read data from the virtual file and write it to the actual file on disk
//     while ((bytes_read = kernel_read(proc_file, buffer, PAGE_SIZE, &proc_file->f_pos)) > 0)
//     {
//         ssize_t bytes_written = kernel_write(disk_file, buffer, bytes_read, &disk_file->f_pos);
//         if (bytes_written != bytes_read)
//         {
//             err = -EIO;
//             pr_err("Failed to write data to %s\n", disk_file_path);
//             goto exit;
//         }
//     }

