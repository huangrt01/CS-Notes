int CompareAndSwap(int *address, int expected, int new) {
	if (*address == expected) {
		*address = new;
		return 1; // success
	}
	return 0; // failure
}

void AtomicIncrement(int*value, int amount) {
	do {
		int old =*value;
	} while (CompareAndSwap(value, old, old + amount) == 0);
}

void insert(int value) {
	node_t *n = malloc(sizeof(node_t));
	assert(n != NULL);
	n->value = value;
	pthread_mutex_lock(listlock);   // begin critical section
	n->next  = head;
	head     = n;
	pthread_mutex_unlock(listlock); // end critical section
}

void insert(int value) { //会有问题，比如当另一个线程成功执行了while，这个线程会retry
	node_t *n = malloc(sizeof(node_t));
	assert(n != NULL);
	n->value = value;
	do {
		n->next = head;
	} while (CompareAndSwap(&head, n->next, n) == 0);
}
