/***** headers.h ******/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <signal.h>
#include <unistd.h>

#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>

#include <linux/version.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h> // seq_read, ...
#include <linux/uaccess.h>

/*
    struct Process
        -burst_time: given, time it takes for the process to complete
        -next:      generated, next process in queue
        -previous:  generated, process before self in queue
*/

struct Process
{
    double burst_time;
    pid_t proc_pid;
    struct Process *next;
    struct Process *previous;
};

/*
    void check_alloc()
        -checks if the given pointer has been allocated successfully and exits if it did not
*/

void check_alloc(void* ptr)
{
    if (!ptr)
    {
        printf("Allocation error!\n");
        exit(1);
    }
}

/*
    struct Process* create_new_process(float process_burst_time, pid_t pid)
        -creates a pointer to a Process struct
*/
struct Process *create_new_process(float process_burst_time, pid_t pid)
{
    struct Process *new_process = NULL;

    new_process = (struct Process *)malloc(sizeof(struct Process));
    check_alloc(new_process);

    new_process->burst_time = process_burst_time;
    new_process->next = NULL;
    new_process->proc_pid = pid;
    new_process->previous = NULL;

    return new_process;
}

// // void generate_data(UserList *user_list,int *nrUsers, float *quantum)
// void generate_data()
// {
//     char *argv[] = {"dummy",NULL};

//     srand(time(NULL));
//     // *nrUsers = rand() % MAX_USERS + 1;

//     // // quantum in interval [10,100]
//     // float scale = rand()/(float)RAND_MAX;

//     // *quantum = scale*(MAX_QUANTUM -10) + 10;

//     // for(int i = 0 ; i< (*nrUsers);i++)
//     // {
//         // generate number between 0.1 and 1
//         float weight = ((float)rand()/(float)(RAND_MAX)) +0.1;

//         // //add user to UserList
//         // struct User* u = addUser(user_list,weight,i);

//         //no. of processes for user "i"

//         int nrProc = rand()%MAX_PROC +1;

//         //create nrProc processes with a rand. burst time
//         // add it to the current user

//         for(int j = 0;j<nrProc;j++)
// 		{
//             float burst_time = (float)rand()/(float)(RAND_MAX)*MAX_BURST +0.1;

//             pid_t pid = fork();

//             if( pid == 0) {
//                 execve("./dum",argv,NULL);
//             }

//             struct Process *p = createNewProcess(burst_time,pid);
//             // linkProcessToUser(p,u);
//         }

//     // }

// }