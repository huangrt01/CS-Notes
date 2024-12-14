# gdb python
# https://stackoverflow.com/questions/1781935/get-stacktrace-from-stuck-python-process-that-does-not-accept-signals

set $gstate = ((int (*)())            PyGILState_Ensure ) ()
call          ((int (*)(const char*)) PyRun_SimpleString) ("import faulthandler; fh = open('/tmp/stacks.txt', 'w'); faulthandler.dump_traceback(file=fh); fh.close()")
call          ((void(*)(int))         PyGILState_Release) ($gstate)