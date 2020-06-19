#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include "common_threads.h"

//
// Here, you have to write (almost) ALL the code. Oh no!
// How can you show that a thread does not starve
// when attempting to acquire this mutex you build?
//

typedef struct __ns_mutex_t{
    int room1;
    int room2;
    sem_t mutex;
    sem_t t1;  //room2有空
    sem_t t2;
} ns_mutex_t;

void ns_mutex_init(ns_mutex_t *m) {
    m->room1=m->room2=0;
    sem_init(&m->mutex,1);
    sem_init(&m->t1,1);
    sem_init(&m->t2,0);
}

void ns_mutex_acquire(ns_mutex_t *m) {
    sem_wait(&m->mutex);
    m->room1+=1;
    sem_post(&m->mutex);

    sem_wait(&m->t1);
    m->room2+=1;
    sem_wait(&m->mutex);
    m->room1-=1;

    if(m->room1==0){
        sem_post(&m->mutex);
        sem_post(&m->t2);
    }
    else{
        sem_post(&m->mutex);
        sem_post(&m->t1);
    }

    sem_wait(&m->t2);
    m->room2-=1;
}

void ns_mutex_release(ns_mutex_t *m) {
    if(m->room2==0)
        sem_post(&m->t1);
    else
        sem_post(&m->t2);
}


void *worker(void *arg) {
    return NULL;
}

int main(int argc, char *argv[]) {
    printf("parent: begin\n");
    printf("parent: end\n");
    return 0;
}

