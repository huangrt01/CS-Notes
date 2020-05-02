## leetcode题目简评

**言简意赅，持续更新，利于速览复习。有导航、有代码、有细节、有引申。**

已记录题目编号：1, 3, 5, 10, 15, 20, 21, 26, 53, 54, 56, 65, 72, 79, 84, 88, 101, 102, 103, 104, 105, 121, 122, 123, 125, 136, 137, 138, 145, 146, 153, 154, 155, 161, 167, 169, 170, 172, 190, 191, 198, 203, 206, 215, 217, 219, 220, 226, 229, 240, 295, 297, 343,426, 653, 946, 974, 1209

#### 0000.资料
[leetcode精选题详解](https://github.com/azl397985856/leetcode)

[代码速查表](https://github.com/OUCMachineLearning/OUCML/tree/master/%E4%BB%A3%E7%A0%81%E9%80%9F%E6%9F%A5%E8%A1%A8)

[图解leetcode](https://github.com/MisterBooo/LeetCodeAnimation)

[生成这篇文章的一键md文件转换小玩具，兼笔记管理](https://github.com/huangrt01/Markdown-Transformer-and-Uploader)

#### 0001.two-sum [两数之和](https://leetcode-cn.com/problems/two-sum) 
* one-pass hash table

#### 0003.longest-substring-without-repeating-characters [无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

* 滑窗：O(n)复杂度，可以用字典储存i跳转的位置，把计算量从2n降到n，类似KMP的思想。

#### 0005.longest-palindromic-substring [最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring)
* 法1:中心扩散
* 法2:动态规划
* 法3:[Manacher算法](https://www.cnblogs.com/cloudplankroader/p/10988844.html)

#### 0010.regular-expression-matching [正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching) 
* [《剑指offer》第19题](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)
* 和0072.Edit-Space类似

<img src="https://www.zhihu.com/equation?tex=d%5Bi%5D%5Bj%5D%3D%5Cbegin%7Bcases%7D%0Ad%5Bi-1%5D%5Bj-1%5D%26%20p%5Bj%5D%3D%27.%27%5C%5C%0As%5Bi%5D%20%3D%3D%20p%5Bj%5D%5Cquad%20%5C%26%5C%26%20%5Cquad%20d%5Bi%20-%201%5D%5Bj%20-%201%5D%20%26%20p%5Bj%5D%3Da%5C%5C%0Ad%5Bi%5D%5Bj%20-%202%5D%5Cquad%20%7C%7C%5Cquad%20d%5Bi-1%5D%5Bj%5D%20%26%20p%5Bj-1%3Aj%5D%3D%27.%2A%27%5C%5C%0Ad%5Bi%5D%5Bj%20-%202%5D%5Cquad%20%7C%7C%5Cquad%20%28d%5Bi-1%5D%5Bj%5D%5Cquad%20%5C%26%5C%26%5Cquad%20s%5Bi%5D%3D%3Dp%5Bj-1%5D%29%20%20%20%20%20%20%26%20p%5Bj-1%3Aj%5D%3D%27a%2A%27%0A%5Cend%7Bcases%7D%0A%5Cnotag%0A" alt="d[i][j]=\begin{cases}
d[i-1][j-1]& p[j]='.'\\
s[i] == p[j]\quad \&\& \quad d[i - 1][j - 1] & p[j]=a\\
d[i][j - 2]\quad ||\quad d[i-1][j] & p[j-1:j]='.*'\\
d[i][j - 2]\quad ||\quad (d[i-1][j]\quad \&\&\quad s[i]==p[j-1])      & p[j-1:j]='a*'
\end{cases}
\notag
" class="ee_img tr_noresize" eeimg="1">

#### 0015.3sum [三数之和](https://leetcode-cn.com/problems/3sum)
* 方法一：先排序，在排序的基础上，虽然也是O(n^2)复杂度，但可以利用双指针尽量提高效率
* 方法二：hash，先预处理排序，并把重复元素合并

#### 0020.valid-parentheses [有效的括号](https://leetcode-cn.com/problems/valid-parentheses) 
* 栈的使用；利用map定义括号和反括号的映射关系

#### 0021.merge-two-sorted-lists [合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists)
* [《剑指offer》第25题](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)，经典题，引入一个头节点
* 代码模版：
```c++
ListNode*head=new ListNode(0);
ListNode*p=head;
...
return head->next;
```

#### 0026.remove-duplicates-from-sorted-array [删除排序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array) 
* O(n)的解法，注意原地操作

#### 0053.maximum-sum-subarray [最大子序和](https://leetcode-cn.com/problems/maximum-subarray) 
* less is more，O(n)的简洁解法，也可用分治
```c++
int maxSubArray(vector<int>& nums) {
    if(nums.size()==0)return 0;
    int i=1;
    int maxsum=nums[0];
    int sum=maxsum;
    while(i<=nums.size()-1){
        if(sum<0){  
            sum=nums[i];
        }
        else
            sum+=nums[i];
        if(sum>=maxsum)
        {
            maxsum=sum;
        } 
        i++;
    }
    return maxsum;
}
```

#### 0054.spiral-matrix [螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix)
* [《剑指offer》第29题](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)
* 最简洁的写法
```c++
vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int>ret;
        int m=matrix.size();
        if(!m)return ret;
        int n=matrix[0].size();
        int b=0,t=m-1,l=0,r=n-1;
        while(1){
            for(int j=l;j<=r;j++)ret.push_back(matrix[b][j]);
            if(++b>t)break;
            for(int i=b;i<=t;i++)ret.push_back(matrix[i][r]);
            if(--r<l)break;
            for(int j=r;j>=l;j--)ret.push_back(matrix[t][j]);
            if(--t<b)break;
            for(int i=t;i>=b;i--)ret.push_back(matrix[i][l]);
            if(++l>r)break;
        }
        return ret;
}
```

#### 0056.merge-intervals [合并区间](https://leetcode-cn.com/problems/merge-intervals)  
* 先sort再遍历
* 复习sort的cmp函数定义（这题不需要cmp函数）

```c++
static bool cmp1(vector<int> &a, vector<int> &b){
		return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
}
```

#### 0065.valid-number [有效数字](https://leetcode-cn.com/problems/valid-number)
* [《剑指offer》第20题](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof)，书上的代码结构很简洁，值得学习
```c++
int pointer;
bool isNumber(string s) {
    if(s=="")return -1;
    scanSpace(s);
    bool numeric=scanInteger(s);
    if(s[pointer]=='.'){
        ++pointer;
        numeric=scanUnsignedInteger(s)||numeric;
        //用||因为整数、小数部分有一即可
    }
    if(s[pointer]=='e'||s[pointer]=='E'){
        ++pointer;
        numeric=numeric&&scanInteger(s);
    }
    scanSpace(s);
    return numeric&&s[pointer]=='\0';
}
```
* 也可以用有限状态机来做

#### 0072.edit-distance [编辑距离](https://leetcode-cn.com/problems/edit-distance) 

* 很漂亮的动态规划

<img src="https://www.zhihu.com/equation?tex=D%5Bi%5D%5Bj%5D%3D%5Cbegin%7Bcases%7DD%5Bi-1%5D%5Bj-1%5D%26A%5Bi%5D%3DB%5Bi%5D%5C%5Cmin%28D%5Bi-1%5D%5Bj-1%5D%2CD%5Bi-1%5D%5Bj%5D%2CD%5Bi%5D%5Bj-1%5D%29%2B1%26A%5Bi%5D%21%3DB%5Bi%5D%5C%5C%5Cend%7Bcases%7D%5Cnotag%0A" alt="D[i][j]=\begin{cases}D[i-1][j-1]&A[i]=B[i]\\min(D[i-1][j-1],D[i-1][j],D[i][j-1])+1&A[i]!=B[i]\\\end{cases}\notag
" class="ee_img tr_noresize" eeimg="1">

#### 0079.word-search [单词搜索](https://leetcode-cn.com/problems/word-search)  
* 经典回溯法

#### 0084.largest-rectangle-in-histogram [柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram) 
* 单调栈，[很好的文章](https://blog.csdn.net/lucky52529/article/details/89155694)，但有小错，纠正代码如下：

```c++
int largestRectangleArea(vector<int> &heights)
{
    stack<int> st;
    int top = 0;
    int maxarea = 0;
    heights.insert(heights.begin(), 0);
    heights.push_back(0);  //左右插0，利于边界条件
    for (int i = 0; i < heights.size(); i++)
    {
        top = i;
        while (!st.empty() && heights[i] < heights[st.top()])
        {
            top = st.top();
            st.pop();
            maxarea = max(maxarea, heights[top] * (i-1 - st.top()));
        }//求出栈处往左右延伸的最大矩形面积
        st.push(i);
    }
    return maxarea;
}
```
* 单调栈模版
```c++
stack<int> st;
//此处一般需要给数组最后添加结束标志符
for (遍历这个数组)
{
  while (栈不为空 && 栈顶元素小于当前元素)
  {
    栈顶元素出栈;
    更新结果;
  }
  当前数据入栈;
}
```

#### 0088.merge-sorted-array [合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array)  
* 注意原地操作

#### 0101.symmetric-tree [对称二叉树](https://leetcode-cn.com/problems/symmetric-tree)
* [《剑指offer》第28题](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)
* 递归
```c++
bool isSymmetric(TreeNode* root) {
    if(!root)return 1;
    else return isSymmetric1(root->left,root->right);
}
bool isSymmetric1(TreeNode* a,TreeNode* b) {
    if(!(a||b))return 1;
    else if(!(a&&b))return 0;
    else return a->val==b->val&&isSymmetric1(a->left,b->right)&&isSymmetric1(a->right,b->left);
}
```

#### 0102.binary-tree-level-order-traversal [二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal)  (medium)
* [《剑指offer》第32-II题](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)
* 队列，设变量curNum和nextNum分别保存本层和下层的数的个数
* 引申：析构vector的方法：
  * `vector<int>().swap(num);`
  * `{ vector<int> tmp = curLevel;   curLevel.swap(tmp);} `
  * 本题不需要swap：clear虽然不会deallocate释放空间，但是会destroy执行析构函数，所以可以用同一个空间construct构造节点，如果swap了就要重新allocate空间在contruct节点
  * 如果要每次都释放空间，可以用`res.emplace_back(std::move(curLevel))`，涉及[emplace_back](https://www.cnblogs.com/ChrisCoder/p/9919646.html), [std::move](https://blog.csdn.net/p942005405/article/details/84644069/), [左值、右值引用](https://blog.csdn.net/p942005405/article/details/84644101), [这是一篇有关类定义的总结](https://blog.csdn.net/zzhongcy/article/details/86747794)


#### 0103.binary-tree-zigzag-level-order-traversal [二叉树的锯齿形层次遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal) 
* [《剑指offer》第32-III题](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)
* 在0102的基础上保存层数的奇偶性

#### 0104.maximum-depth-of-binary-tree [二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/) 
* 方法一：递归
* 方法二：BFS，queue
* 方法三：DFS，stack，[利用c++的pair](https://blog.csdn.net/sevenjoin/article/details/81937695)，或者python的tuple

#### 0105. construct-binary-tree-from-preorder-and-inorder-traversal [从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) 
* [《剑指offer》第7题](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)
* 找到中间节点，递归
  
#### 0121. best-time-to-buy-and-sell-stock [买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/) 
* O(n)遍历，记录之前的数组最小值

#### 0122. best-time-to-buy-and-sell-stock-ii [买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/) 

#### 0123. best-time-to-buy-and-sell-stock-iii [买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/) 
* [超巧妙的方法](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/discuss/404387/Intuitive-Python-O(n)-Time-and-O(1)-Space)，本质上是贪心的思想，先记录maxp的位置，一定会取到股票的最大最小值high、low处，再做处理

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        def max_p(ps):
            if not ps or len(ps) == 1:
                return 0, 0, 0
            very_low = 0
            low = 0
            high = 0
            profit = 0

            for i, p in enumerate(ps):
                if p < ps[low]:
                    low = i
                elif p - ps[low] > profit:
                    high = i
                    very_low = low
                    profit = p - ps[low]
                    
            return very_low, high, profit
        
        low, high, profit = max_p(prices)
        _, _, profit_right = max_p(prices[0:low])
        _, _, profit_left = max_p(prices[high+1:])
        _, _, profit_middle = max_p(prices[low:high+1][::-1])
        
        return profit + max(profit_left, profit_middle, profit_right)
```

#### 0125.valid-palindrome [验证回文串](https://leetcode-cn.com/problems/valid-palindrome) 

``` python
def isPalindrome(self, s: str) -> bool:
		s = ''.join(i for i in s if i.isalnum()).lower()
		return s == s[::-1]
```

#### 0136.single-number [只出现一次的数字](https://leetcode-cn.com/problems/single-number) 

* 位运算，xor性质

#### 0137.single-number-II [只出现一次的数字 II](https://leetcode-cn.com/problems/single-number-ii) 

* 非常巧妙的方法，多设一个数记录状态，位运算与有限状态机的结合，本质上，位运算的意义在于将n位信息转化为O(1)

```python
def singleNumber(self, nums: List[int]) -> int:
    seen_once = seen_twice = 0
    for num in nums:
        # first appearance: 
        # add num to seen_once, don't add to seen_twice because of presence in seen_once

        # second appearance: 
        # remove num from seen_once, add num to seen_twice
    
        # third appearance: 
        # don't add to seen_once because of presence in seen_twice, remove num from seen_twice
        
        seen_once = ~seen_twice & (seen_once ^ num)
        seen_twice = ~seen_once & (seen_twice ^ num)
    return seen_once

```

#### 0138.copy-list-with-random-pointer [复制带随机指针的链表](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)
* [《剑指offer》第35题](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/)
* 思路值得学习，在原链表的主线上复制节点，进行删改操作。

#### 0145.binary-tree-postorder-traversal [二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal)

* 方法一：教科书，先一路向左，用tag记录节点的右子树是否遍历

```c++
struct WTreeNode{TreeNode* TNode;bool tag;};
vector<int> postorderTraversal(TreeNode* root) {
    vector<int> ret;
    stack<WTreeNode*>s;
    TreeNode* p=root;
    if(root==NULL) return ret;
    WTreeNode* l;
    while(!s.empty()||p){
        if (p!=NULL){ //左子树不断入栈
            l=new WTreeNode; 
            l->TNode=p;l->tag=0;
            s.push(l);
            p=p->left;
        }
        else{
            l=s.top();
            s.pop();
            if(l->tag==1){
                ret.push_back(l->TNode->val);
            }
            else{   //右子树没输出
                l->tag=1;
                s.push(l);
                p=l->TNode->right;
            }
        }  
    }
    return ret;
}
```

* 引申：二叉树非递归遍历的模版

```c++
while(!s.empty()||p){
	if (p!=NULL){
	}
	else{
	}
}
```

* 方法二：后序遍历是左右根，倒过来是根右左，相当于左右遍历顺序相反的DFS，用栈即可，得到结果再reverse

```c++
vector<int> postorderTraversal1(TreeNode* root) {
    vector<int> ret;
    stack<TreeNode*>s;
    vector<int>invert;
    s.push(root);
    while(!s.empty()){
        TreeNode *p=s.top();s.pop();
        if(p!=NULL){
            invert.push_back(p->val);
            s.push(p->left);
            s.push(p->right);
        }
    }
    reverse(invert.begin(),invert.end());
    return invert;
}
```

#### 0146.lru-cache [LRU缓存机制](https://leetcode-cn.com/problems/lru-cache) 
* 双向链表+Map，自己实现双向链表可以高效实现move to head操作，也可以用STL的list

#### 0153.find-minimum-in-rotated-sorted-array [寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array) 
* 二分法，注意相等的情况

#### 0154.find-minimum-in-rotated-sorted-array-ii [寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii)
* [《剑指offer》第11题](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)
* 如果有重复数字，则难以判断mid是在左边还是右边，r-=1是解决这一问题的关键代码
```c++
int findMin(vector<int>& numbers) {
    //if(numbers.size()==0)return -1;
    int l=0,r=numbers.size()-1;
    int mid;
    while(l<r){
        if(r-l==1)
            return (numbers[l]<=numbers[r])?numbers[l]:numbers[r];
        mid=(l+r)/2;
        if(numbers[mid]>numbers[r]) l=mid;
        else if(numbers[mid]<numbers[l])r=mid;
        else
            r-=1;
            //return minArraySeq(numbers,l,r);
    }
    return 1;
}
```

#### 0155.min-stack [最小栈](https://leetcode-cn.com/problems/min-stack) 
* [《剑指offer》第30题](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)
* 用另一个栈记录min的变化值

#### 0161.one-edit-distance [相隔为 1 的编辑距离](https://leetcode-cn.com/problems/one-edit-distance) 
* 遍历较短的数字，直到遇到第一个和长数组对应位置不等的元素，再做判断处理

#### 0167.two-sum-ii-input-array-is-sorted [两数之和 II - 输入有序数组](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted) 
* 先二分查找到中间，再往两边扩散，这样当N很大时时间复杂度近似 <img src="https://www.zhihu.com/equation?tex=O%28log_%7B2%7DN%29" alt="O(log_{2}N)" class="ee_img tr_noresize" eeimg="1"> 

#### 0169. majority-element [多数元素](https://leetcode-cn.com/problems/majority-element)

* 方法一：hashmap

* 方法二：随机算法，先取再**验证**

* 方法三：[Boyer-Moore Voting Algorithm](https://zhuanlan.zhihu.com/p/85474828)：核心是利用这个数据的前缀特性，用军队打仗理解；每个非众数都会和一个数配对 

```c++
int majorityElement(vector<int> nums) {
    int solider = nums[0];
    int count=0;
    for(int i=0;i<nums.length();i++){
        if(nums[i]==solider){       //队伍增加了同胞
            count++;
        }
        else{       //没人去抵抗了
            if(count==0){
                solider=nums[i];
                count++;
            }else {
                count--;        //抓住一个人
            }  
        }
    }
    return solider;   
}
```

#### 0170.two-sum-iii-data-structure-design [两数之和 III - 数据结构设计](https://leetcode-cn.com/problems/two-sum-iii-data-structure-design) 
与两数之和联系的数据结构
* 有序数组：因此可以用sort排序，或者用multiset维持数据结构的有序性
* Hash表

#### 0172.factorial-trailing-zeroes [阶乘后的零](https://leetcode-cn.com/problems/factorial-trailing-zeroes) 
*  <img src="https://www.zhihu.com/equation?tex=n/5%2Bn/5%5E2%2Bn/5%5E3%2B%E2%80%A6" alt="n/5+n/5^2+n/5^3+…" class="ee_img tr_noresize" eeimg="1">  递归即可

#### 0190.reverse-bits [颠倒二进制位](https://leetcode-cn.com/problems/reverse-bits) 
* 法一：位运算
```c++
uint32_t reverseBits(uint32_t n) { //注意定义uint32_t，如果是int要小心算术移位
    unsigned int t = 0; 
    int i = 0;
    while (n >= 1)
    {
        t = (t << 1) | (n & 1);
        i++;
        n = n >> 1;
    }
    if (i == 0)  //细节：左移不要越界
        return t;
    return t << (32 - i);
}
```
* 法二：按字节操作
```python
def reverseByte(byte):
    return (byte * 0x0202020202 & 0x010884422010) % 1023
```

#### 0191.number-of-1-bits [位1的个数](https://leetcode-cn.com/problems/number-of-1-bits) 
* [《剑指offer》第15题](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)
* n=n&(n-1);
* 易错点：`return n&1+hammingWeight(n>>=1);`
  * 位运算优先级很低，n&1应该打括号
* 复习[运算符优先级](https://baike.baidu.com/item/%E8%BF%90%E7%AE%97%E7%AC%A6%E4%BC%98%E5%85%88%E7%BA%A7/4752611?fr=aladdin#4)
* 基本的优先级需要记住：
  * 指针最优，单目运算优于双目运算，如正负号。
  * 先算术运算，后移位运算，最后位运算。1 << 3 + 2 & 7等价于 (1 << (3 + 2))&7，逻辑运算最后结合。

#### 0198.house-robber [打家劫舍](https://leetcode-cn.com/problems/house-robber) 

* 简单DP

#### 0203.remove-linked-list-elements [移除链表元素](https://leetcode-cn.com/problems/remove-linked-list-elements) 
* [《剑指offer》第18题](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)
* 直接遍历，也可以用sentinel node简化操作（在LRU cache也有应用）

```c++
class Solution {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        if(head==NULL)return NULL;
        if(head->val==val)return head->next;
        ListNode *p=head;
        while(p->next){
            if(p->next->val==val){
                p->next=p->next->next;
                break;
            }
            p=p->next;
        }
        return head;
    }
};
```

#### 0206.reverse-linked-list [反转链表](https://leetcode-cn.com/problems/reverse-linked-list) 
* [《剑指offer》第24题](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)
```c++
ListNode* reverseList(ListNode* head) {
    if(!head||!head->next) return head;
    ListNode *p=head,*q=head->next; 
  	p->next=NULL;
  	ListNode* temp;
    while(q!=NULL){
    	temp=q->next;
			q->next=p;
			p=q;
      q=temp;
    }
    return p;
}
```

#### 0215.kth-largest-element-in-an-array [数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array) 
* 方法1: 快排
* 方法2: 利用小顶堆，保证size不大于k; C++中用priority_queue, 配合unordered_map
* 方法3: 快排变体，[quick select](https://www.cnblogs.com/shawshawwan/p/9220818.html)

* 引申：堆的实现
```c++
#define ElemType pair<int,int>
class CMaxHeap {  //小顶堆
private:
	ElemType *heap;
	int heapSize, MaxHeapSize;
public:
    CMaxHeap(int size) {
        heapSize = 0;
        MaxHeapSize = size;
        heap = new ElemType[size + 1];
    }
    ~CMaxHeap() { delete[] heap; }
    void ClearHeap() { heapSize = 0; }
    bool IsEmpty() { return heapSize == 0; }
    bool IsFull() { return heapSize == MaxHeapSize; }
    int getLength() { return heapSize; }
    ElemType top() { return heap[0]; }
    void push(ElemType e);
	ElemType pop();	//去堆顶元素
	void FixUp(int k);
	void FixDown(int k);
};

void CMaxHeap::FixDown(int k) {
	int i;
	i = 2 * k + 1;
	while (i < heapSize) {
		if (i < heapSize - 1 && heap[i] > heap[i + 1])
			i++;			//取孩子结点中较小者
		if (heap[k] < heap[i])
			break;
		swap(heap[k], heap[i]);
		k = i;
		i = 2 * k + 1;
	}
}
void CMaxHeap::FixUp(int k) {
	int i;
	i = (k - 1) / 2;
	while (k > 0 && heap[i] > heap[k]) {
		swap(heap[k], heap[i]);
		k = i;
		i = (k - 1) / 2;
	}
}
void CMaxHeap::push(ElemType e) {
	heap[heapSize] = e;
	heapSize++;
	FixUp(heapSize - 1);
}
ElemType CMaxHeap::pop() { //去掉堆顶
	swap(heap[0], heap[heapSize - 1]);
	heapSize--;
	FixDown(0);
	return heap[heapSize];
}
void heap_sort(ElemType *a, int l, int r) {//
	int N = r - l;
	ElemType *p = a + l;
	for (int k = (N - 1) / 2; k >= 0; k--)
		FixDown(p, k, N);
	while (N > 0) {
		exch(p, p + N);
		N--;
		FixDown(p, 0, N);
	}
}
```

* 引申：快排代码
```c++
template<class T>
void quick_sort(T*a, int l, int r) {//递归实现
	if (r <= l)return;
	int i = partition(a, l, r);//划分操作
	quick_sort(a, l, i - 1);
	quick_sort(a, i + 1, r);
}
template<class T>
int partition(T*a, int l, int r) {//思想，从两边向中间扫描
	int i = l - 1,j = r;
	T e = a[r];//最右端元素为划分元素
	while (1) {
		while (a[++i] < e);
		while (e < a[--j])if (j == left)break;
		if (i >= j)break;
		exch(a + i, a + j);
	}
	exch(a + i, a + right);
	return i;
}

//改进：中间元素法和小序列处理
template<class T>
void quick_sort(T*a, int l, int r) {//递归实现
	if (r-l< 5)return;
	exch(a[(l + r) / 2], a[r - 1]);
	compExch(a[l], a[r - 1]);//取三个值的中间值
	compExch(a[l], a[r]);
	compExch(a[r], a[r - 1]);
	int i = partition(a, l, r);//划分操作
	quick_sort(a, l, i - 1);
	quick_sort(a, i + 1, r);
}
```



#### 0217.contains-duplicate [存在重复元素](https://leetcode-cn.com/problems/contains-duplicate) 
* 排序或者hash

#### 0219.contains-duplicate-ii [存在重复元素 II](https://leetcode-cn.com/problems/contains-duplicate-ii) 
* 本题关注点在于是否有邻近的重复，因此除了用hash，可以尝试利用数据的邻近特性，例如JAVA的treeset：self-balancing Binary Search Tree (BST)，C++的multiset

#### 0220.contains-duplicate-iii [存在重复元素 III](https://leetcode-cn.com/problems/contains-duplicate-iii) 

* 方法一：[multiset](https://blog.csdn.net/sodacoco/article/details/84798621)+滑窗法，利用[lower_bound](https://www.cnblogs.com/tocy/p/STL_lower_bound_intro.html)  
```c++
bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
    int size = nums.size(); if(size <= 1) return false;
    if(k==0) return t==0;
    multiset<long long> window;
    //construct the first window
    for(int i=0; i<min(size, k+1); i++) {
        bool tmp = updateWindow(window, nums[i], t); if(tmp) return true;
        window.insert(nums[i]);
    }
    //slide the wondow
    for(int i=1; i+k<size; i++) {
        auto itPrev = window.find(nums[i-1]); window.erase(itPrev);
        bool tmp = updateWindow(window, nums[i+k], t); if(tmp) return true;
        window.insert(nums[i+k]);
    }
    return false;
}
private:
bool updateWindow(multiset<long long>& window, int val, int t) {
    auto itlower = window.lower_bound(val);
    if(itlower != window.end() && (*itlower)-val <= t) return true;
    if(itlower != window.begin()) {
        --itlower;
        if(val - (*itlower) <= t) return true;
    }
    return false;
}
```

* 方法二：巧妙的方法，注意到数据结构特点，要求没有邻近数，因此可以用bucket数据结构
  * 引申：桶排序 =>[基数排序](https://blog.csdn.net/qq_41900081/article/details/86831408) 
```java
public class Solution {
    // Get the ID of the bucket from element value x and bucket width w
    // In Java, `-3 / 5 = 0` and but we need `-3 / 5 = -1`.
    private long getID(long x, long w) {
        return x < 0 ? (x + 1) / w - 1 : x / w;
    }

    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        if (t < 0) return false;
        Map<Long, Long> d = new HashMap<>();
        long w = (long)t + 1;
        for (int i = 0; i < nums.length; ++i) {
            long m = getID(nums[i], w);
            // check if bucket m is empty, each bucket may contain at most one element
            if (d.containsKey(m))
                return true;
            // check the neighbor buckets for almost duplicate
            if (d.containsKey(m - 1) && Math.abs(nums[i] - d.get(m - 1)) < w)
                return true;
            if (d.containsKey(m + 1) && Math.abs(nums[i] - d.get(m + 1)) < w)
                return true;
            // now bucket m is empty and no almost duplicate in neighbor buckets
            d.put(m, (long)nums[i]);
            if (i >= k) d.remove(getID(nums[i - k], w));
        }
        return false;
    }
}
```

#### 0226.invert-binary-tree [翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree) 

[这个梗](https://twitter.com/mxcl/status/608682016205344768)

#### 0229.majority-element-ii [求众数 II](https://leetcode-cn.com/problems/majority-element-ii) 

* Boyer-Moore，[代码](https://leetcode.com/problems/majority-element-ii/discuss/466876/Python-O(N)-time-O(1)-Space-Explanation-in-Comments )

#### 0240.search-a-2d-matrix-ii [搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii) 

* [《剑指offer》第4题](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)
* 关键在于起点的选取，从左下角或者右上角开始

#### 0295.find-median-from-data-stream [数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/)
* [《剑指offer》第41题](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)
* 思路1: AVL树的平衡因子改为左、右子树节点数目之差
* 思路2: 左边最大堆，右边最小堆
  * 书上代码：push_heap和pop_heap
  * 也可直接用priority_queue，注意小顶堆的定义：`priority_queue<int, vector<int>, greater<int>> hi;`

```c++
min.push_back(num);
push_heap(min.begin(),min.end(),greater<int>());
```

* 字节后端面试：变式题，在本题基础上增加erase功能，需要把堆改成BST（即set），保证删除性能

#### 0297.serialize-and-deserialize-binary-tree [二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)
* [《剑指offer》第37题](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)
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

#### 0343.integer-break [整数拆分](https://leetcode-cn.com/problems/integer-break) 

* 简单DP

#### 0426.convert-binary-search-tree-to-sorted-doubly-linked-list[将二叉搜索树转化为排序的双向链表](https://leetcode-cn.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/)
* [《剑指offer》第36题](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)
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

#### 0653.two-sum-iv-input-is-a-bst [两数之和 IV - 输入 BST](https://leetcode-cn.com/problems/two-sum-iv-input-is-a-bst) 

* 我用的方法比较奇怪：分治的思想，利用BST特性减少运算量，直接递归即可通过。
* 其它的常见方法：
  * 方法一：使用HashSet
  * 方法二：中序遍历BST树，转化为排序数组的两数之和问题
```c++
class Solution {
public:
    typedef TreeNode* Link;
    bool searchR(Link p,int x){ //判断子树内是否存在x
        if(p==NULL)return 0;
        int key=p->val;
        if(x==key)return 1;
        if(x<key)return searchR(p->left,x);
        else return searchR(p->right,x);
    }
    bool findTarget(TreeNode* root, int k) {
        int result=0;
        if(root==NULL)return 0;
        if(k>2*root->val){  //利用BST特性减少运算量
            if(findTarget(root->right,k))return 1;
        }
        else if(k<2*root->val){
            if(findTarget(root->left,k))return 1;
        }
        if(searchR(root->left,k-root->val))return 1;
        if(searchR(root->right,k-root->val))return 1;
        if(findinAB(root->left,root->right,k))
            return 1;
        return 0;
    }
    
    bool findinAB(TreeNode* A,TreeNode* B,int k){ //在A、B中各取一个节点
        int result=0;
        if(A==NULL||B==NULL)
            return 0;
        if(searchR(B,k-A->val))return 1;
        if(findinAB(A->left,B,k))return 1;
        if(findinAB(A->right,B,k))return 1;
        return 0;
    }
};
```

#### 0946.validate-stack-sequences [验证栈序列](https://leetcode-cn.com/problems/validate-stack-sequences) 

* 建一个辅助栈模拟这一过程

#### 0974.subarray-sums-divisible-by-k [和可被 K 整除的子数组](https://leetcode-cn.com/problems/subarray-sums-divisible-by-k) 

* 记录前缀和数组v[i]，

<img src="https://www.zhihu.com/equation?tex=%5Crm%7Bresult%7D%3D%5Csum_%7Bi%3D0%7D%5E%7Bi%3DK-1%7D%5Cbinom%7Bv%5Bi%5D%7D%7B2%7D%5Cnotag%0A" alt="\rm{result}=\sum_{i=0}^{i=K-1}\binom{v[i]}{2}\notag
" class="ee_img tr_noresize" eeimg="1">

#### 1209.remove-all-adjacent-duplicates-in-string-ii [删除字符串中的所有相邻重复项 II](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string-ii) 

* 利用pair存储当前连续字符数，建立栈模拟操作，符合条件则出栈

