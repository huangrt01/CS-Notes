
Welcome to multi.py, a rudimentary multi-CPU scheduling simulator. This
simulator has a number of features to play with, so pay attention! Or don't,
because you are lazy that way. But when that exam rolls around...

To run the simulator, all you have to do is type:

prompt> ./multi.py

This will then run the simulator with some random jobs. 

Before we get into any such details, let's first examine the basics of such a
simulation. 

In the default mode, there are one or more CPUs in the system (as specified
with the -n flag). Thus, to run with 4 CPUs in your simulation, type:

prompt> ./multi.py -n 4

Each CPU has a cache, which can hold important data from one or more running
processes. The size of each CPU cache is set by the -M flag. Thus, to make
each cache have a size of '100' on your 4-CPU system, run:

prompt> ./multi.py -n 4 -M 100

To run a simulation, you need some jobs to schedule. There are two ways to do
this. The first is to let the system create some jobs with random
characteristics for you (this is the default, i.e., if you specify nothing,
you get this); there are also some controls to control (somewhat) the nature
of randomly-generated jobs, described further below.  The second is to specify
a list of jobs for the system to schedule precisely; this is also described in
more detail below.

Each job has two characteristics. The first is its 'run time' (how many time
units it will run for). The second is its 'working set size' (how much cache
space it needs to run efficiently). If you are generating jobs randomly, you
can control the range of these values by using the -R (maximum run-time flag)
and -W (maximum working-set-size flag); the random generator will then
generate values that are not bigger than those. 

If you are specifying jobs by hand, you set each of these explicitly, using
the -L flag. For example, if you want to run two jobs, each with run-time of
100, but with different working-set-sizes of 50 and 150, respectively, you
would do something like this:

prompt> ./multi.py -n 4 -M 100 -L 1:100:50,2:100:150

Note that you assigned each job a name, '1' for the first job, and '2' for the
second. When jobs are auto-generated, names are assigned automatically (just
using numbers). 

How effectively a job runs on a particular CPU depends on whether the cache of
that CPU currently holds the working set of the given job. If it doesn't, the
job runs slowly, which means that only 1 tick of its runtime is subtracted
from its remaining time left per each tick of the clock. This is the mode
where the cache is 'cold' for that job (i.e., it does not yet contain the
job's working set). However, if the job has run on the CPU previously for
'long enough', that CPU cache is now 'warm', and the job will execute more
quickly. How much more quickly, you ask? Well, this depends on the value of
the -r flag, which is the 'warmup rate'. Here, it is something like 2x by
default, but you can change it as needed.

How long does it take for a cache to warm up, you ask? Well, that is also set
by a flag, in this case, the -w flag, which sets the 'warmup time'. By
default, it is something like 10 time units. Thus, if a job runs for 10 time
units, the cache on that CPU becomes warm, and then the job starts running
faster. All of this, of course, is a gross approximation of how a real system
works, but that's the beauty of simulation, right?

So now we have CPUs, each with caches, and a way to specify jobs. What's left?
Aha, you say, the scheduling policy! And you are right. Way to go, diligent
homework-doing person!

The first (default) policy is simple: a centralized scheduling queue, with a
round-robin assignment of jobs to idle CPUs. The second is a per-CPU
scheduling queue approach (turned on with -p), in which jobs are assigned to
one of N scheduling queues (one per CPU); in this approach, an idle CPU will
(on occasion) peek into some other CPU's queue and steal a job, to improve
load balancing. How often this is done is set by a 'peek' interval (set by the
-P flag).

With this basic understanding in place, you should now be able to do the
homework, and study different approaches to scheduling.  To see a full list of
options for this simulator, just type: 

prompt> ./multi.py -h
Usage: multi.py [options]

Options:
Options:
  -h, --help            show this help message and exit
  -s SEED, --seed=SEED  the random seed
  -j JOB_NUM, --job_num=JOB_NUM
                        number of jobs in the system
  -R MAX_RUN, --max_run=MAX_RUN
                        max run time of random-gen jobs
  -W MAX_WSET, --max_wset=MAX_WSET
                        max working set of random-gen jobs
  -L JOB_LIST, --job_list=JOB_LIST
                        provide a comma-separated list of
                        job_name:run_time:working_set_size (e.g.,
                        a:10:100,b:10:50 means 2 jobs with run-times of 10,
                        the first (a) with working set size=100, second (b)
                        with working set size=50)
  -p, --per_cpu_queues  per-CPU scheduling queues (not one)
  -A AFFINITY, --affinity=AFFINITY
                        a list of jobs and which CPUs they can run on (e.g.,
                        a:0.1.2,b:0.1 allows job a to run on CPUs 0,1,2 but b
                        only on CPUs 0 and 1
  -n NUM_CPUS, --num_cpus=NUM_CPUS
                        number of CPUs
  -q TIME_SLICE, --quantum=TIME_SLICE
                        length of time slice
  -P PEEK_INTERVAL, --peek_interval=PEEK_INTERVAL
                        for per-cpu scheduling, how often to peek at other
                        schedule queue; 0 turns this off
  -w WARMUP_TIME, --warmup_time=WARMUP_TIME
                        time it takes to warm cache
  -r WARM_RATE, --warm_rate=WARM_RATE
                        how much faster to run with warm cache
  -M CACHE_SIZE, --cache_size=CACHE_SIZE
                        cache size
  -o, --rand_order      has CPUs get jobs in random order
  -t, --trace           enable basic tracing (show which jobs got scheduled)
  -T, --trace_time_left
                        trace time left for each job
  -C, --trace_cache     trace cache status (warm/cold) too
  -S, --trace_sched     trace scheduler state
  -c, --compute         compute answers for me

Probably the best to learn about are some of the tracing options (like -t, -T,
-C, and -S). Play around with these to see better what the scheduler and
system are doing.








