/********* index.c ***********/
// Export virtual file into actual file
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define ACTUAL_FILE_PATH "linux_log_file.csv"
#define PROC_FILE_PATH "/proc/log_file"
#define BUFFER_SIZE 32768
#define WHITE_SPACE " "
#define COMMA ","
#define NEXT_LINE "\n"

char buffer[BUFFER_SIZE];
size_t bytes_read;
FILE *fp, *outfp;

// Note: This function returns a pointer to a substring of the original string.
// If the given string was allocated dynamically, the caller must not overwrite
// that pointer with the returned value, since the original pointer must be
// deallocated using the same allocator with which it was allocated.  The return
// value must NOT be deallocated using free() etc.
char *trim_white_space(char *str)
{
    char *end;

    // Trim leading space
    while (isspace((unsigned char)*str))
        str++;

    if (*str == 0) // All spaces?
        return str;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end))
        end--;

    // Write new null terminator character
    end[1] = '\0';

    return str;
}

// Output data into csv file
// Each column separated by comma in cse file
void output_log_file()
{
    ssize_t read;
    char *line = NULL;
    char *new_line;
    size_t len = 0;

    // Loop indefinitely to continuously read from the file
    while (1)
    {
             printf("in loop\n");
        // Read a line from the file
        if (fgets(buffer, BUFFER_SIZE, fp) != NULL)
        {
                 printf("fgets good\n");
            char *token;
             printf("buffer %s\n", buffer);
            token = strtok(buffer, WHITE_SPACE);

        printf("token %s\n", token);
            while (token != NULL)
            {
                printf("in token loop %s\n", trim_white_space(token));
                
                strcat(new_line, trim_white_space(token));

                token = strtok(NULL, WHITE_SPACE);
                if (token != NULL)
                {
                     printf("hit if\n");
                    strcat(new_line, COMMA);
                }
                else
                {
                     printf("hit else \n");
                    strcat(new_line, NEXT_LINE);
                    break;
                }
            }

            fprintf(outfp, "%s \n", new_line);
            printf("hit\n");
        }
        else
        {
            // No new data in the file yet
            continue;
        }
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