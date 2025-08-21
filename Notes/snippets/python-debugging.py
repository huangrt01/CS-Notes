def trace_function_call(func):
    """
    A decorator that prints the full, multi-level call stack each time a function 
    is called. The output is formatted to resemble a standard Python traceback.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the current stack
        stack = inspect.stack()
        
        print(f"\n--- Call Trace: Executing '{func.__name__}' ---")
        
        # Print the stack trace, starting from the immediate caller and going up.
        # stack[0] is the frame for this 'wrapper' function, so we skip it.
        for frame_info in stack[1:]:
            print(f"  File \"{frame_info.filename}\", line {frame_info.lineno}, in {frame_info.function}")
            # frame_info.code_context is a list of lines from the source code.
            # The line that made the call is usually the first one.
            if frame_info.code_context:
                print(f"    {frame_info.code_context[0].strip()}")
                
        print(f"--------------------------------------\n")
        
        # Call the original function
        return func(*args, **kwargs)
    return wrapper