/********* exportlog.c ***********/
// Export logger into actual file
#include "headers.h"

#ifndef __KERNEL__
#define __KERNEL__
#endif

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Kernel module to export contents of virtual file in /proc to actual file on disk");

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
#define HAVE_PROC_OPS
#endif


#define DEVICE_NAME "export_file"
#define VIRTUAL_FILE_NAME "virtual_file"
#define ACTUAL_FILE_NAME "/tmp/actual_file"

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

#ifdef HAVE_PROC_OPS
static const struct proc_ops proc_file_fops = {
	  .read = device_read,
    .write = device_write,
    .open = device_open,
    .release = device_release,
    .llseek = no_llseek,
    .write_iter = device_export};
#else
static const struct file_operations proc_file_fops = {
	  .read = device_read,
    .write = device_write,
    .open = device_open,
    .release = device_release,
    .llseek = no_llseek,
    .owner = THIS_MODULE,
    .write_iter = device_export, // Use write_iter to support large files
    };
#endif

static int __init mymodule_init(void)
{
    int ret = 0;
    struct path virtual_file_path;

    // Get the path to the virtual file
    ret = kern_path(VIRTUAL_FILE_NAME, 0, &virtual_file_path);
    if (ret < 0) {
        printk(KERN_ERR "Failed to get virtual file path\n");
        return ret;
    }

    // Open the virtual file
    virtual_file = file_open(virtual_file_path, O_RDONLY, 0);
    if (IS_ERR(virtual_file)) {
        printk(KERN_ERR "Failed to open virtual file\n");
        return PTR_ERR(virtual_file);
    }

    // Register the device
    major_num = register_chrdev(0, DEVICE_NAME, &fops);
    if (major_num < 0) {
        printk(KERN_ERR "Failed to register device\n");
        return major_num;
    }
    printk(KERN_INFO "Export file module loaded\n");

    return 0;
}

static void __exit mymodule_exit(void)
{
    // Remove the virtual file from the /proc filesystem
    proc_remove(proc_file_entry);

    pr_info("Module unloaded successfully\n");
}

module_init(mymodule_init);
module_exit(mymodule_exit);


// static int copy_proc_file_to_disk(const char *proc_file_path, const char *disk_file_path)
// {
//     struct file *proc_file = NULL;
//     struct file *disk_file = NULL;
//     char *buffer = NULL;
//     ssize_t bytes_read;
//     int err = 0;

//     // Open the virtual file in the /proc filesystem
//     proc_file = filp_open(proc_file_path, O_RDONLY, 0);
//     if (IS_ERR(proc_file))
//     {
//         err = PTR_ERR(proc_file);
//         pr_err("Failed to open %s: %d\n", proc_file_path, err);
//         goto exit;
//     }

//     // Create the actual file on disk
//     disk_file = filp_open(disk_file_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
//     if (IS_ERR(disk_file))
//     {
//         err = PTR_ERR(disk_file);
//         pr_err("Failed to create %s: %d\n", disk_file_path, err);
//         goto exit;
//     }

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

// exit:
//     if (proc_file)
//     {
//         filp_close(proc_file, NULL);
//     }
//     if (disk_file)
//     {
//         filp_close(disk_file, NULL);
//     }
//     if (buffer)
//     {
//         kfree(buffer);
//     }
//     return err;
// }