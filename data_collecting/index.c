/********* index.c ***********/
// Export virtual file into actual file
#include <stdio.h>
#include <string.h>

#define ACTUAL_FILE_PATH "linux_log_file.csv"
#define PROC_FILE_PATH "/proc/log_file"
#define BUFFER_SIZE 32768
#define WHITE_SPACE " "
#define COMMA ","

char buffer[BUFFER_SIZE];
size_t bytes_read;
FILE *fp, *outfp;

// Output data into csv file
// Each column separated by comma in cse file
void output_log_file()
{
    ssize_t read;
    char *line = NULL;
    char *new_line;
    size_t len = 0;

    while ((read = getline(&line, &len, fp)) != -1)
    {
        char *token;
        token = strtok(line, WHITE_SPACE);

        while (token != NULL)
        {
            strcat(new_line, token);
            token = strtok(NULL, WHITE_SPACE);
            if (token != NULL) {
                strcat(new_line, COMMA);
            }
                 printf("%s \n", token);
        }

        // fprintf(outfp, "%s \n", new_line);
        
        // printf("%s \n", new_line);
    }
}

int main()
{
    char filename[] = PROC_FILE_PATH;
    char output_filename[] = ACTUAL_FILE_PATH;

    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        perror("fopen");
        return 1;
    }

    outfp = fopen(output_filename, "w+");
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