#include <headers.h>

#include <linux/fs.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>

#define PROC_FILE_NAME "myprocfile"
#define DISK_FILE_NAME "/tmp/mydiskfile"

static struct proc_dir_entry *proc_file_entry;

static ssize_t proc_file_read(struct file *filep, char *buffer, size_t len, loff_t *offset)
{
    ssize_t bytes_read = 0;
    char *proc_file_content = "This is the content of my virtual file in /proc\n";
    size_t content_len = strlen(proc_file_content);

    if (*offset >= content_len)
    {
        // End of file
        return 0;
    }

    if (*offset + len > content_len)
    {
        len = content_len - *offset;
    }

    if (copy_to_user(buffer, proc_file_content + *offset, len))
    {
        return -EFAULT;
    }

    bytes_read = len;
    *offset += len;
    return bytes_read;
}

static const struct file_operations proc_file_ops = {
    .owner = THIS_MODULE,
    .read = proc_file_read,
};

static int __init mymodule_init(void)
{
    int err = 0;

    // Create the virtual file in the /proc filesystem
    proc_file_entry = proc_create(PROC_FILE_NAME, 0, NULL, &proc_file_ops);
    if (!proc_file_entry)
    {
        pr_err("Failed to create /proc/%s\n", PROC_FILE_NAME);
        err = -ENOMEM;
        goto exit;
    }

    // Export the contents of the virtual file to the actual file on disk
    err = simple_vfs_copy(PROC_FILE_NAME, DISK_FILE_NAME);
    if (err)
    {
        pr_err("Failed to export /proc/%s to %s: %d\n", PROC_FILE_NAME, DISK_FILE_NAME, err);
        goto exit_remove_proc_file;
    }

    pr_info("Module loaded successfully\n");
    return 0;

exit_remove_proc_file:
    proc_remove(proc_file_entry);
exit:
    return err;
}

static void __exit mymodule_exit(void)
{
    // Remove the virtual file from the /proc filesystem
    proc_remove(proc_file_entry);

    pr_info("Module unloaded successfully\n");
}

module_init(mymodule_init);
module_exit(mymodule_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("Export contents of virtual file in /proc to actual file on disk");

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