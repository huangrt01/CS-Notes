### x64 Cheat Sheet

* https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf

  * %rax: By convention, %rax is used to store a function’s return value, if it exists and is no more than 64 bits long. (Larger return types like structs are returned using the stack.) 
    * %eax, %ax, %al
  * %rip: [Understanding %rip register in intel assembly](https://stackoverflow.com/questions/42215105/understanding-rip-register-in-intel-assembly)

  * %retq: [What is the difference between retq and ret?](https://stackoverflow.com/questions/42653095/what-is-the-difference-between-retq-and-ret)





* DCL(DoubleCheckedLocking) 的例子

  * ```java
    singletons[i].reference = new Singleton();
    ```

    ```assembly
    # note that the Symantec JIT using a handle-based object allocation system
    0206106A   mov         eax,0F97E78h
    0206106F   call        01F6B210                  ; allocate space for
                                                     ; Singleton, return result in eax
    02061074   mov         dword ptr [ebp],eax       ; EBP is &singletons[i].reference 
                                                    ; store the unconstructed object here.
    02061077   mov         ecx,dword ptr [eax]       ; dereference the handle to
                                                     ; get the raw pointer
    02061079   mov         dword ptr [ecx],100h      ; Next 4 lines are
    0206107F   mov         dword ptr [ecx+4],200h    ; Singleton's inlined constructor
    02061086   mov         dword ptr [ecx+8],400h
    0206108D   mov         dword ptr [ecx+0Ch],0F84030h
    ```