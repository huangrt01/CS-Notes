#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
int main(void) {
// open and set up a bunch of sockets (not shown)
// main loop
	while (1) {
	// initialize the fd_set to all zero
	fd_set readFDs;
	FD_ZERO(&readFDs);

	// now set the bits for the descriptors
	// this server is interested in
	// (for simplicity, all of them from min to max)
	int fd;
	for (fd = minFD; fd < maxFD; fd++)
		FD_SET(fd, &readFDs);
		// do the select
	int rc = select(maxFD+1, &readFDs, NULL, NULL, NULL);
	// check which actually have data using FD_ISSET()
	int fd;
	for (fd = minFD; fd < maxFD; fd++)
		if (FD_ISSET(fd, &readFDs))
			processFD(fd);
	}
}

struct aiocb { //填写信息
	int aio_fildes;   // File descriptor
	off_t aio_offset;   // File offset
	volatile void *aio_buf;	// Location of buffer
	size_t aio_nbytes;   // Length of transfer
};

int aio_read(struct aiocb *aiocbp);

int aio_error(const struct aiocb*aiocbp);
//checks whether the request referred to by aiocbp has completed. 
//If it has, the routine returns success (indicated by a zero);
//if not, EINPROGRESS is returned