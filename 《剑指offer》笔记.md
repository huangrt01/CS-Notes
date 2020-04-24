### 《剑指offer——名企面试官精讲典型编程题》，何海涛，电子工业出版社，2017

#### chpt1 面试的流程

* 电话面试：说细节，大胆pardon
* 远程桌面面试：编程习惯，调试能力
* 现场面试：准备几个问题
* 行为面试->技术面试->应聘者提问


* 技能：了解、熟悉、精通
* 常考点：链表、二叉树、快排
* 细节：空指针空字符串（nullptr）、错误处理、溢出
  * C语言的整型溢出问题，[很好的文章](https://coolshell.cn/articles/11466.html/comment-page-1#comments)



##### 36.[二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)
* 方法一：二叉搜索树特性，中序遍历的递归/非递归实现，用nonlocal last记录上一次遍历的末尾节点
* 方法二：用flag指示返回最左/最右节点，递归后序遍历操作
```c++
Node *treeToDoublyList(Node *root)
{
    root = treeToDoublyList(root, 0);
    if (root == NULL)
        return NULL;
    Node *p = root;
    while (p->right) p = p->right;
    p->right = root;
    root->left = p;
    return root;
}
Node *treeToDoublyList(Node *root, int flag)
{ //flag=0:left, flag=1:right
    if (root == NULL)
        return NULL;
    Node *l = treeToDoublyList(root->left, 1);
    Node *r = treeToDoublyList(root->right, 0);
    root->left = l;
    root->right = r;
    if (l)
        l->right = root;
    if (r)
        r->left = root;
    Node *p = root;
    if (!flag) while (p->left) p = p->left;
    else while (p->right) p = p->right;
    return p;
}

```





动态规划与分治的区别：前者自底向上，后者自顶向下

