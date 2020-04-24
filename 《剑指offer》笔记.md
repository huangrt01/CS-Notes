### 《剑指offer——名企面试官精讲典型编程题》，何海涛，电子工业出版社，2017

动态规划与分治的区别：前者自底向上，后者自顶向下

#### chpt1 面试的流程

* 电话面试：说细节，大胆pardon
* 远程桌面面试：编程习惯，调试能力
* 现场面试：准备几个问题
* 行为面试->技术面试->应聘者提问


* 技能：了解、熟悉、精通
* 常考点：链表、二叉树、快排
* 细节：空指针空字符串（nullptr）、错误处理、溢出
  * C语言的整型溢出问题，[很好的文章](https://coolshell.cn/articles/11466.html/comment-page-1#comments)

#### chpt2 面试需要的基础知识
* C++：面向对象的特性、构造函数、析构函数、动态绑定、内存管理
  * e.g. 空类1字节
* 软件工程：常见的设计模式、UML图

#### chpt4 解决面试题的思路
解决复杂问题的三种方法：画图、举例、分解

##### 33.[二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)
* 法1:递归子树，直观的思路
* 法2:参照leetcode84，利用单调栈，[一篇很好的文章](https://blog.csdn.net/lucky52529/article/details/89155694)（有小错，修正版见我的[leetcode题解](https://github.com/huangrt01/CS-Notes)）


##### 36.[二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)
* leetcode 426.
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


##### 37.[序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)
* leetcode 297.
* 思路上可以使用DFS或者BFS
* C++具体实现，利用stringstream
```c++
class Codec {
public:
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        if(root==NULL)return "";
        ostringstream ostr;
        queue<TreeNode*>q;
        TreeNode*temp;
        q.push(root);
        int curNum=1;
        while(!q.empty()){
            temp=q.front();
            q.pop();
            if(!temp){
                if(curNum) ostr<<"null,";
            }
            else {
                ostr<<temp->val<<",";
                curNum--;
                q.push(temp->left);
                if(temp->left)curNum++;
                q.push(temp->right);
                if(temp->right)curNum++;
            }
        }
        return ostr.str();
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        if(data=="")return NULL;
        istringstream istr(data);
        queue<TreeNode*>q;
        TreeNode* root=new TreeNode;
        TreeNode **number=new TreeNode*;
        if(ReadStream (istr,number)){
            root=number[0];
            if(!root)return NULL;
            q.push(root);
        }
        else return NULL;
        TreeNode *temp;
        while(!q.empty()){
            temp=q.front();
            q.pop();
            if(!temp)continue;
            if(ReadStream(istr,number)){
                temp->left=number[0];
                q.push(temp->left);
            }
            else break;
            if(ReadStream(istr,number)){
                temp->right=number[0];
                q.push(temp->right);
            }
            else break;
        }
        return root;
    }
    bool ReadStream(istringstream &istr,TreeNode **number){
        string s;
        if(getline(istr,s,',')){
            if(s=="null")number[0]=NULL;
            else number[0]=new TreeNode(stoi(s));
            return 1;
        }
        return 0;
    }
};
```














