## CSAPP-Labs

[我的CSAPP-Labs实现](https://github.com/huangrt01/CSAPP-Labs)

- [x] Data Lab
- [x] Shell Lab (plus [OSTEP shell lab](https://github.com/remzi-arpacidusseau/ostep-projects/tree/master/processes-shell))

### Data Lab

#### Integer

##### bitXor(x,y)
```c++
int bitXor(int x, int y) {
  // ~优先级大于&
  return ~(~(~x&y)&~(~y&x));
}
```

##### tmin()
```c++
int tmin(void) {
  return 1<<31;
}
```

##### isTmax(x)
关注到0x80000000的特殊性质，只有它和0满足本身加本身等于0
```c++
int isTmax(int x) {
  // !!表示不为0
  // 0x7FFFFFFF + 1 = 0x80000000, 本身加本身等于0
  return !((x+1)+(x+1)) & !!(x+1);
}
```

##### allOddBits(x)
```c++
int allOddBits(int x) {
  int allodd= 0xAA + (0xAA<<8)+(0xAA<<16)+(0xAA<<24);
  return !((x&allodd)^allodd);
}
```

##### negate(x)
```c++
int negate(int x) {
  return ~x+1;
}
```
##### isAsciiDigit(x)
```c++
int isAsciiDigit(int x) {
  // 0x 0011 100*  or 0x 0011 0***
  return !((x^0x38)& ~0x1) | !((x^0x30)& ~0x7);
}
```

##### conditional(x,y,z)
```c++
int conditional(int x, int y, int z) {
  int cond = !x+(~0);
  return (cond&y) | (~cond&z);
}
```

##### isLessOrEqual(x,y)
判断y-x的符号，然后讨论上溢出和下溢出

```c++
int isLessOrEqual(int x, int y) {
  // y-x
  int yminusx= y + (~x+1);
  int geq0 = !((yminusx>>31)&yminusx);

  // negative overflow: y<0, x>=0, y-x>=0
  int negative_overflow= (y>>31)&!(x>>31)&!(yminusx>>31);
  // positive overflow: x<0, y>=0, y-x<0
  int positive_overflow= (x>>31)&!(y>>31)& (yminusx>>31);
  
  // return (y>=x)
  //printf("%d  %d   %d  %d   \n",yminusx,geq0,negative_overflow,positive_overflow);
  return (geq0 | positive_overflow) & !negative_overflow;
}
```
##### logicalNeg(x)

* 标答方法：太巧了！利用了补码体系中0的特殊性
```c++
int logicalNeg(int x) {
  return ((x | (~x + 1)) >> 31) + 1;
}
  

}
```
* 我的方法：观察零和非零数字和-1相加的情形，并讨论首位
```c++
int logicalNeg(int x){
	//my second method:
  unsigned xminus1= x+(~0);
  unsigned unsignx=x;

  // flag=0代表首位为1
  int flag= (~unsignx)>>31;  

  // 返回0的条件：首位为1 或 首位为0且除了首位存在1（加0xFFFFFFFF后首位为0）
  return flag & (~flag | ~(~xminus1>>31));
}
```
##### howManyBits(x)
二分的方法讨论即可
```c++
int howManyBits(int x) {
  int isNeg=!!(x>>31);    
  int shift,log2;   
  isNeg= ~(isNeg+~0);
  x=(isNeg&(~x))|(~isNeg&x);
  
  shift=(!!(x>>16))<<4; 
  x=x>>shift;
  log2=shift;

  shift=(!!(x>>8))<<3;
  x=x>>shift;
  log2=log2+shift;

  shift=(!!(x>>4))<<2;
  x=x>>shift;
  log2=log2+shift;

  shift=(!!(x>>2))<<1;
  x=x>>shift;
  log2=log2+shift;

  shift=(x>>1);
  x=x>>shift;
  log2=log2+shift;
  log2=log2+x+1;  // howManyBits(0)=1, sign bit, so plus 1 additionally

  return log2;
}
```


#### Float

* [32位浮点数（很好的资料，认真复习）](https://www.runoob.com/w3cnote/32-float-storage.html)：S(1位) E(8位) M(23位)，exponent要减127做偏移，且需要讨论exponent全0或全1的情形
  * ieee754 标准下，内存全零是最小的float
  * 嵌入式上的compiler不一定符合标准

##### floatScale2(uf)
```c++
unsigned floatScale2(unsigned uf) {
  unsigned input=uf;
  unsigned sign;
  unsigned exponent;
  unsigned fraction = uf&0xffffff;
  unsigned result;

  uf=uf>>23;
  exponent=uf&0xff;
  sign=uf>>8;
  if(!exponent)
  { //exponent全为0, E=-127, M=0.XXX..XXX
    if(fraction>>22)
      exponent=1;
    fraction = fraction << 1;
  }
  else if(!(~exponent&0xff)){ //exponent全1
    return input;
  }
  else{
    exponent=exponent+1;
    if(!~exponent) fraction=0;
  }
  fraction = fraction & ~(1 << 23);
  result = (sign << 31) + (exponent << 23) + fraction;
  //printf("sign:%d  exp:%d  frac:%d \n", sign, exponent, fraction);
  return result;
}
```
##### floatFloat2Int(uf)
* 注意细节：直接截断，无需四舍五入
```c++
int floatFloat2Int(unsigned uf) {
  unsigned sign;
  unsigned exponent;
  unsigned fraction = uf & 0x7fffff;
  unsigned result=0;

  uf = uf >> 23;
  exponent = uf & 0xff;
  sign = uf >> 8;
  if(exponent<=125){ //(1)E<=-2;  (2)exponent==0时, E=-127, M=0.XXX..XXX
    return 0;
  }
  else if(exponent>=158){ //E>=31
    return 0x80000000u;
  }
  else if(exponent==126){//E=-1
    return 0;
  }
  else{
    result=fraction | 0x800000;
    if(exponent<=150){ 
      result=result>>(150-exponent);
    }
    else{
      result=result<<(exponent-150);
    }
  }
  if(sign){
    result = ~result + 1;
  }
  return result;
}
```

##### floatPower2(x)

* 这题的测试文件有问题，会超时，需要把`btest.c`的第35行改成`#define TIMEOUT_LIMIT 100`

* 网上大部分代码都是错的，包括lab的测试代码在内都没有考虑`./btest -f floatPower2 -1 -126`这类情形。具体来说，当指数为-127全零时，可以用小数部分进一步具体表示$2^{-149}$到$2^{-126}$：

  $floatPower2(2^{-149})= 0x1$

  $floatPower2(2^{-126})=0x800000$

  

最终代码如下：

```c++
unsigned floatPower2(int x)
{
  unsigned exponent = x + 127;
  unsigned fraction = 0;
  unsigned result;
  if (x >= 128)
    exponent = 0xff;
  else if (x < -125)
  { // E= -inf ~ -126
    exponent = 0x00;
    if (x >= -149)
    { // 2^(-149)= 浮点1    fraction=0x000001
      fraction = 1 << (x + 149);
    }
  }
  result = (exponent << 23) + fraction;
  return result;
}
```

### Shell Lab

shell
* 空格分割，不断产生prompt
* ampersand `&`结尾 => background运行

command分为两类
* built-in command: current process执行
* the pathname of an executable file: forks a child process
* **job**的概念，指代initial child process，job用PID或JID标识，JID由`%`开头

The parent needs to block the SIGCHLD signals in this way in order to avoid the **race condition** where the child is reaped by sigchld handler(and thus removed from the job list) before the parent calls addjob.

**BUG**：

有signal时，getc()可能会有很奇怪的bug，不会返回EOF，最终影响fgets的行为，太坑了！！！具体来说，注释掉eval(cmdline)这一行读取文件则不会出错，最终的解决方案是先把文件整体读入内存...... bug留存进了branch getc_error分支

**Bonus**:

current Built-in commands:

* exit/quit
* cd
* pwd
* jobs
* bg/fg
* path: The `path` command takes 0 or more arguments, with each argument separated by whitespace from the others. A typical usage would be like this: `wish> path /bin /usr/bin`, which would add `/bin` and `/usr/bin` to the search path of the shell. If the user sets path to be empty, then the shell should not be able to run any programs (except built-in commands). The `path` command always overwrites the old path with the newly specified path.

based on the needs of [OSTEP Projects](https://github.com/remzi-arpacidusseau/ostep-projects/tree/master/processes-shell), additionally implement the following features:

* Wrong input error
* Paths
* Redirection
* Parallel commands

