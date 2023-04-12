/********* index.c ***********/
// Export virtual file into actual file
#include <stdio.h>

char buffer[1024];
size_t bytes_read;
FILE *fp, *outfp;

void output_log_file()
{
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), fp)) > 0)
    {
        fwrite(buffer, 1, bytes_read, outfp);
    }
}

int main()
{
    char filename[] = "/proc/log_file";
    char output_filename[] = "linux_log_file.txt";

    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        perror("fopen");
        return 1;
    }

    outfp = fopen(output_filename, "w");
    if (outfp == NULL)
    {
        perror("fopen");
        fclose(fp);
        return 1;
    }

    output_log_file();

    fclose(fp);
    fclose(outfp);

    return 0;
}