### asyncio


# loop_in_executor内部，不能执行async函数


### uvloop

import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())