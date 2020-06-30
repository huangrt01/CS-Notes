#! /usr/bin/env python

# each job has a working-set size
# if it runs "in cache", it runs at rate X
#            "out of cache", rate Y (slower than X)
# sched policies
# - centralized
#   one queue,
# - distributed
#   many queues

from __future__ import print_function
from collections import *
from optparse import OptionParser
import random

# helper print function for columnar output
def print_cpu(cpu, str):
    print((' ' * cpu * 35) + str)
    return

#
# Job struct: tracks everything about each job
#
Job = namedtuple('Job', ['name', 'run_time', 'working_set_size', 'affinity', 'time_left'])

#
# class cache
#
# key question: how does a cache get warmed?
# simple model here:
# - run for 'cache_warmup_time' on CPU
# - after that amount of time on CPU, cache is "warm" for you
# cache has limited size, so only so many jobs can be "warm" at a time
# 
class cache:
    def __init__(self, cpu_id, jobs, cache_size, cache_rate_cold, cache_rate_warm, cache_warmup_time):
        self.cpu_id = cpu_id
        self.jobs = jobs
        self.cache_size = cache_size
        self.cache_rate_cold = cache_rate_cold
        self.cache_rate_warm = cache_rate_warm
        self.cache_warmup_time = cache_warmup_time

        # cache_contents
        # - should track whose working sets are in the cache
        # - it's a list of job_names that 
        #   * is len>=1, and the SUM of working sets fits into the cache
        # OR
        #   * len=1 and whose working set may indeed be too big
        self.cache_contents = []

        # cache_warming(cpu)
        # - list of job_name that are trying to warm up this cache right now
        # cache_warming_counter(cpu, job)
        # - counter for each, showing how long until the cache is warm for that job
        self.cache_warming = []
        self.cache_warming_counter = {}
        return

    def new_job(self, job_name):
        if job_name not in self.cache_contents and job_name not in self.cache_warming:
            # print_cpu(self.cpu_id, '*new cache*')
            if self.cache_warmup_time == 0:
                # special case (alas): no warmup, just right into cache
                self.cache_contents.insert(0, job_name)
                self.adjust_size()
            else:
                self.cache_warming.append(job_name)
                self.cache_warming_counter[job_name] = cache_warmup_time
        return

    def total_working_set(self):
        cache_sum = 0
        for job_name in self.cache_contents:
            cache_sum += self.jobs[job_name].working_set_size
        return cache_sum

    def adjust_size(self):
        working_set_total = self.total_working_set()
        while working_set_total > self.cache_size:
            last_entry = len(self.cache_contents) - 1
            job_gone = self.cache_contents[last_entry]
            # print_cpu(self.cpu_id, 'kicking out %s' % job_gone)
            del self.cache_contents[last_entry]
            self.cache_warming.append(job_gone)
            self.cache_warming_counter[job_gone] = cache_warmup_time
            working_set_total -= self.jobs[job_gone].working_set_size
        return

    def get_cache_state(self, job_name):
        if job_name in self.cache_contents:
            return 'w'
        else:
            return ' '
        
    def get_rate(self, job_name):
        if job_name in self.cache_contents:
            return self.cache_rate_warm
        else:
            return self.cache_rate_cold

    def update_warming(self, job_name):
        if job_name in self.cache_warming:
            self.cache_warming_counter[job_name] -= 1
            if self.cache_warming_counter[job_name] <= 0:
                self.cache_warming.remove(job_name)
                self.cache_contents.insert(0, job_name)
                self.adjust_size()
                # print_cpu(self.cpu_id, '*warm cache*')
        return

#
# class scheduler
#
# imitates a multi-CPU scheduler
#
class scheduler:
    def __init__(self, job_list, per_cpu_queues, affinity, peek_interval,
                 job_num, max_run, max_wset,
                 num_cpus, time_slice, random_order,
                 cache_size, cache_rate_cold, cache_rate_warm, cache_warmup_time,
                 solve, trace, trace_time_left, trace_cache, trace_sched):

        if job_list == '':
            # this means randomly generate jobs
            for j in range(job_num):
                run_time = int((random.random() * max_run)/10.0) * 10
                working_set = int((random.random() * max_wset)/10.0) * 10
                if job_list == '':
                    job_list = '%s:%d:%d' % (str(j), run_time, working_set)
                else:
                    job_list += (',%s:%d:%d' % (str(j), run_time, working_set))
                
        # just the job names
        self.job_name_list = []

        # info about each job
        self.jobs = {}
        
        for entry in job_list.split(','):
            tmp = entry.split(':')
            if len(tmp) != 3:
                print('bad job description [%s]: needs triple of name:runtime:working_set_size' % entry)
                exit(1)
            job_name, run_time, working_set_size = tmp[0], int(tmp[1]), int(tmp[2])
            self.jobs[job_name] = Job(name=job_name, run_time=run_time, working_set_size=working_set_size, affinity=[], time_left=[run_time])
            print('Job name:%s run_time:%d working_set_size:%d' % (job_name, run_time, working_set_size))
            # self.sched_queue.append(job_name)
            if job_name in self.job_name_list:
                print('repeated job name %s' % job_name)
                exit(1)
            self.job_name_list.append(job_name)
        print('')

        # parse the affinity list
        if affinity != '':
            for entry in affinity.split(','):
                # form is 'job_name:cpu.cpu.cpu'
                # where job_name is the name of an existing job
                # and cpu is an ID of a particular CPU (0 ... max_cpus-1)
                tmp = entry.split(':')
                if len(tmp) != 2:
                    print('bad affinity spec %s' % affinity)
                    exit(1)
                job_name = tmp[0]
                if job_name not in self.job_name_list:
                    print('job name %s in affinity list does not exist' % job_name)
                    exit(1)
                for cpu in tmp[1].split('.'):
                    self.jobs[job_name].affinity.append(int(cpu))
                    if int(cpu) < 0 or int(cpu) >= num_cpus:
                        print('bad cpu %d specified in affinity %s' % (int(cpu), affinity))
                        exit(1)

        # now, assign jobs to either ALL the one queue, or to each of the queues in RR style
        # (as constrained by affinity specification)
        self.per_cpu_queues = per_cpu_queues

        self.per_cpu_sched_queue = {}

        if self.per_cpu_queues:
            for cpu in range(num_cpus):
                self.per_cpu_sched_queue[cpu] = []
            # now assign jobs to these queues 
            jobs_not_assigned = list(self.job_name_list)
            while len(jobs_not_assigned) > 0:
                for cpu in range(num_cpus):
                    assigned = False
                    for job_name in jobs_not_assigned:
                        if len(self.jobs[job_name].affinity) == 0 or cpu in self.jobs[job_name].affinity:
                            self.per_cpu_sched_queue[cpu].append(job_name)
                            jobs_not_assigned.remove(job_name)
                            assigned = True
                        if assigned:
                            break

            for cpu in range(num_cpus):
                print('Scheduler CPU %d queue: %s' % (cpu, self.per_cpu_sched_queue[cpu]))
            print('')
                            
        else:
            # assign them all to same single queue
            self.single_sched_queue = []
            for job_name in self.job_name_list:
                self.single_sched_queue.append(job_name)
            for cpu in range(num_cpus):
                self.per_cpu_sched_queue[cpu] = self.single_sched_queue

            print('Scheduler central queue: %s\n' % (self.single_sched_queue))

        self.num_jobs = len(self.job_name_list)


        self.peek_interval = peek_interval

        self.num_cpus = num_cpus
        self.time_slice = time_slice
        self.random_order = random_order

        self.solve = solve

        self.trace = trace
        self.trace_time_left = trace_time_left
        self.trace_cache = trace_cache
        self.trace_sched = trace_sched

        # tracking each CPU: is it idle or running a job?
        self.STATE_IDLE = 1
        self.STATE_RUNNING = 2

        # the scheduler state (RUNNING or IDLE) of each CPU
        self.sched_state = {}
        for cpu in range(self.num_cpus):
            self.sched_state[cpu] = self.STATE_IDLE

        # if a job is running on a CPU, which job is it
        self.sched_current = {}
        for cpu in range(self.num_cpus):
            self.sched_current[cpu] = ''

        # just some stats
        self.stats_ran = {}
        self.stats_ran_warm = {}
        for cpu in range(self.num_cpus):
            self.stats_ran[cpu] = 0
            self.stats_ran_warm[cpu] = 0

        # scheduler (because it runs the simulation) also instantiates and updates each cache
        self.caches = {}
        for cpu in range(self.num_cpus):
            self.caches[cpu] = cache(cpu, self.jobs, cache_size, cache_rate_cold, cache_rate_warm, cache_warmup_time)

        return

    def handle_one_interrupt(self, interrupt, cpu):
        # HANDLE: interrupts here, so jobs don't run an extra tick
        if interrupt and self.sched_state[cpu] == self.STATE_RUNNING:
            self.sched_state[cpu] = self.STATE_IDLE
            job_name = self.sched_current[cpu]
            self.sched_current[cpu] = ''
            # print_cpu(cpu, 'tick done for job %s' % job_name)
            self.per_cpu_sched_queue[cpu].append(job_name)
        return

    def handle_interrupts(self):
        if self.system_time % self.time_slice == 0 and self.system_time > 0:
            interrupt = True
            # num_to_print = time + per-cpu info + cache status for each job - last set of space
            num_to_print = 8 + (7 * self.num_cpus) - 5
            if self.trace_time_left:
                num_to_print += 6 * self.num_cpus
            if self.trace_cache:
                num_to_print += 8 * self.num_cpus + self.num_jobs * (self.num_cpus)
            if self.trace:
                print('-' * num_to_print)
        else:
            interrupt = False

        if self.trace:
            print(' %3d   ' % self.system_time, end='')

        # INTERRUPTS first: this might deschedule a job, putting it into a runqueue
        for cpu in range(self.num_cpus):
            self.handle_one_interrupt(interrupt, cpu)
        return

    def get_job(self, cpu, sched_queue):
        # get next job?
        for job_index in range(len(sched_queue)):
            job_name = sched_queue[job_index]
            # len(affinity) == 0 is special case, which means ANY cpu is fine
            if len(self.jobs[job_name].affinity) == 0 or cpu in self.jobs[job_name].affinity:
                # extract job from runqueue, put in CPU local structures
                sched_queue.pop(job_index)
                self.sched_state[cpu] = self.STATE_RUNNING
                self.sched_current[cpu] = job_name
                self.caches[cpu].new_job(job_name)
                # print('got job %s' % job_name)
                return
        return

    def assign_jobs(self):
        if self.random_order:
            cpu_list = list(range(self.num_cpus))
            random.shuffle(cpu_list)
        else:
            cpu_list = range(self.num_cpus)
        for cpu in cpu_list:
            if self.sched_state[cpu] == self.STATE_IDLE:
                self.get_job(cpu, self.per_cpu_sched_queue[cpu])

    def print_sched_queues(self):
        # PRINT queue information
        if not self.trace_sched:
            return
        if self.per_cpu_queues:
            for cpu in range(self.num_cpus):
                print('Q%d: ' % cpu, end='')
                for job_name in self.per_cpu_sched_queue[cpu]:
                    print('%s ' % job_name, end='')
                print('  ', end='')
            print('    ', end='')
        else:
            print('Q: ', end='')
            for job_name in self.single_sched_queue:
                print('%s ' % job_name, end='')
            print('    ', end='')
        return
        
    def steal_jobs(self):
        if not self.per_cpu_queues or self.peek_interval <= 0:
            return

        # if it is time to steal
        if self.system_time > 0 and self.system_time % self.peek_interval == 0:
            for cpu in range(self.num_cpus):
                if len(self.per_cpu_sched_queue[cpu]) == 0:
                    # find IDLE job in some other CPUs queue
                    other_cpu_list = list(range(self.num_cpus))
                    other_cpu_list.remove(cpu)
                    other_cpu = random.choice(other_cpu_list)
                    # print('cpu %d is idle' % cpu)
                    # print('-> look at %d' % other_cpu)

                    for job_name in self.per_cpu_sched_queue[other_cpu]:
                        # print('---> examine job %s' % job_name)
                        if len(self.jobs[job_name].affinity) == 0 or cpu in self.jobs[job_name]:
                           self.per_cpu_sched_queue[other_cpu].remove(job_name)
                           self.per_cpu_sched_queue[cpu].append(job_name)
                           # print('stole job %s from %d to %d' % (job_name, other_cpu, cpu))
                           break
        return

    def run_one_tick(self, cpu):
        job_name = self.sched_current[cpu]
        job = self.jobs[job_name]

        # USE cache_contents to determine if cache is cold or warm
        # (list usage w/ time_left field: a hack to deal with namedtuple and its lack of mutability)
        current_rate = self.caches[cpu].get_rate(job_name)
        self.stats_ran[cpu] += 1
        if current_rate > 1:
            self.stats_ran_warm[cpu] += 1
        time_left = job.time_left.pop() - current_rate
        if time_left < 0:
            time_left = 0
        job.time_left.append(time_left)

        if self.trace:
            print('%s ' % job.name, end='')
            if self.trace_time_left:
                print('[%3d] ' % job.time_left[0], end='')

        # UPDATE: cache warming
        self.caches[cpu].update_warming(job_name)

        if time_left <= 0:
            self.sched_state[cpu] = self.STATE_IDLE
            job_name = self.sched_current[cpu]
            self.sched_current[cpu] = ''
            # remember: it is time X now, but job ran through this tick, so finished at X + 1
            # print_cpu(cpu, 'finished %s at time %d' % (job_name, self.system_time + 1))
            self.jobs_finished += 1
        return

    def run_jobs(self):
        for cpu in range(self.num_cpus):
            if self.sched_state[cpu] == self.STATE_RUNNING:
                self.run_one_tick(cpu)
            elif self.trace:
                print('- ', end='')
                if self.trace_time_left:
                    print('[   ] ', end='')

            # PRINT: cache state
            cache_string = ''
            for job_name in self.job_name_list:
                # cache_string += '%s%s ' % (job_name, self.caches[cpu].get_cache_state(job_name))
                cache_string += '%s' % self.caches[cpu].get_cache_state(job_name)
            if self.trace:
                if self.trace_cache:
                    print('cache[%s]' % cache_string, end='')
                print('     ', end='')
        return

    #
    # MAIN SIMULATION
    #
    def run(self):
        # things to track
        self.system_time = 0
        self.jobs_finished = 0

        while self.jobs_finished < self.num_jobs:
            # interrupts: may cause end of a tick, thus making job schedulable elsewhere
            self.handle_interrupts()

            # if it's time, do some job stealing
            self.steal_jobs()
                
            # assign_jobsign news jobs to CPUs (this can happen every tick?)
            self.assign_jobs()

            # run each CPU for a time slice and handle POSSIBLE end of job
            self.run_jobs()

            self.print_sched_queues()

            # to add a newline after all the job updates
            if self.trace:
                print('')

            # the clock keeps ticking            
            self.system_time += 1

        if self.solve:
            print('\nFinished time %d\n' % self.system_time)
            print('Per-CPU stats')
            for cpu in range(self.num_cpus):
                print('  CPU %d  utilization %3.2f [ warm %3.2f ]' % (cpu, 100.0 * float(self.stats_ran[cpu])/float(self.system_time),
                                                                      100.0 * float(self.stats_ran_warm[cpu])/float(self.system_time)))
            print('')
        return

#
# MAIN PROGRAM
#
parser = OptionParser()
parser.add_option('-s', '--seed',        default=0,     help='the random seed',                        action='store', type='int', dest='seed')
parser.add_option('-j', '--job_num',     default=3,     help='number of jobs in the system',           action='store', type='int', dest='job_num')
parser.add_option('-R', '--max_run',     default=100,   help='max run time of random-gen jobs',        action='store', type='int', dest='max_run')
parser.add_option('-W', '--max_wset',    default=200,   help='max working set of random-gen jobs',     action='store', type='int', dest='max_wset')
parser.add_option('-L', '--job_list',    default='',    help='provide a comma-separated list of job_name:run_time:working_set_size (e.g., a:10:100,b:10:50 means 2 jobs with run-times of 10, the first (a) with working set size=100, second (b) with working set size=50)', action='store', type='string', dest='job_list')
parser.add_option('-p', '--per_cpu_queues', default=False, help='per-CPU scheduling queues (not one)', action='store_true',        dest='per_cpu_queues')
parser.add_option('-A', '--affinity',    default='',    help='a list of jobs and which CPUs they can run on (e.g., a:0.1.2,b:0.1 allows job a to run on CPUs 0,1,2 but b only on CPUs 0 and 1', action='store', type='string', dest='affinity')
parser.add_option('-n', '--num_cpus',    default=2,     help='number of CPUs',                         action='store', type='int', dest='num_cpus')
parser.add_option('-q', '--quantum',     default=10,    help='length of time slice',                   action='store', type='int', dest='time_slice')
parser.add_option('-P', '--peek_interval', default=30,  help='for per-cpu scheduling, how often to peek at other schedule queue; 0 turns this off', action='store', type='int', dest='peek_interval')
parser.add_option('-w', '--warmup_time', default=10,    help='time it takes to warm cache',            action='store', type='int', dest='warmup_time')
parser.add_option('-r', '--warm_rate', default=2,     help='how much faster to run with warm cache', action='store', type='int', dest='warm_rate')
parser.add_option('-M', '--cache_size',  default=100,   help='cache size',                             action='store', type='int', dest='cache_size')
parser.add_option('-o', '--rand_order',  default=False, help='has CPUs get jobs in random order',      action='store_true',        dest='random_order')
parser.add_option('-t', '--trace',       default=False, help='enable basic tracing (show which jobs got scheduled)',      action='store_true',        dest='trace')
parser.add_option('-T', '--trace_time_left', default=False, help='trace time left for each job',       action='store_true',        dest='trace_time_left')
parser.add_option('-C', '--trace_cache', default=False, help='trace cache status (warm/cold) too',     action='store_true',        dest='trace_cache')
parser.add_option('-S', '--trace_sched', default=False, help='trace scheduler state',                  action='store_true',        dest='trace_sched')
parser.add_option('-c', '--compute',     default=False, help='compute answers for me',                 action='store_true',        dest='solve')

(options, args) = parser.parse_args()

random.seed(options.seed)

print('ARG seed %s' % options.seed)
print('ARG job_num %s' % options.job_num)
print('ARG max_run %s' % options.max_run)
print('ARG max_wset %s' % options.max_wset)
print('ARG job_list %s' % options.job_list)
print('ARG affinity %s' % options.affinity)
print('ARG per_cpu_queues %s' % options.per_cpu_queues)
print('ARG num_cpus %s' % options.num_cpus)
print('ARG quantum %s' % options.time_slice)
print('ARG peek_interval %s' % options.peek_interval)
print('ARG warmup_time %s' % options.warmup_time)
print('ARG cache_size %s' % options.cache_size)
print('ARG random_order %s' % options.random_order)
print('ARG trace %s' % options.trace)
print('ARG trace_time %s' % options.trace_time_left)
print('ARG trace_cache %s' % options.trace_cache)
print('ARG trace_sched %s' % options.trace_sched)
print('ARG compute %s' % options.solve)
print('')

#
# JOBS
# 
job_list = options.job_list
job_num = int(options.job_num)
max_run = int(options.max_run)
max_wset = int(options.max_wset)

#
# MACHINE
#
num_cpus = int(options.num_cpus)
time_slice = int(options.time_slice)

#
# CACHES
#
cache_size = int(options.cache_size)
cache_rate_warm = int(options.warm_rate)
cache_warmup_time = int(options.warmup_time)

do_trace = options.trace
if options.trace_time_left or options.trace_cache or options.trace_sched:
    do_trace = True

#
# SCHEDULER (and simulator)
#
S = scheduler(job_list=job_list, affinity=options.affinity, per_cpu_queues=options.per_cpu_queues, peek_interval=options.peek_interval,
              job_num=job_num, max_run=max_run, max_wset=max_wset,
              num_cpus=num_cpus, time_slice=time_slice, random_order=options.random_order,
              cache_size=cache_size, cache_rate_cold=1, cache_rate_warm=cache_rate_warm,
              cache_warmup_time=cache_warmup_time, solve=options.solve,
              trace=do_trace, trace_time_left=options.trace_time_left, trace_cache=options.trace_cache,
              trace_sched=options.trace_sched)

# Finally, ...
S.run()

