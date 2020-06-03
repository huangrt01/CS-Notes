## Cryptography I, Stanford University, Dan Boneh

* [coursera课程](https://www.coursera.org/learn/crypto/home/welcome)
* [密码学资源推荐](https://blog.cryptographyengineering.com/useful-cryptography-resources/)
* [A Graduate Course In Applied Cryptography](https://toc.cryptobook.us/)
* [A Computational Introduction to Number Theory and Algebra](https://www.shoup.net/ntb/)


#### 密码学在计算机领域的应用

##### hash function原理

```c++
int sum=0; 
for(int i=0; i<v.size(); ++i) sum=sum*131+v[i]; 
return sum;
```

* 对`vector<int>`做hash的方法：

  I. 用上面的方法，选取质数131，可能需要再设另一个质数取模

  II. 两个或三个的简单情形，可以利用pair和map

  III. 对于每一个整数， 把0\~7、8\~15、 16\~23、 24\~31的位置取出来变成char，cat之后再hash

* 方法I中取模用质数更好的原因

  * “ 如果p是一个质数，n是任意非零整数（不是p的倍数）， 那么px+ny=z对于任意的x,y,z都有解”， 这样可以保证取模相对均匀一些， 避免所谓的 primary clustering， 要证明这个需要引理：“方程 ax+by=1 有整数解当且仅当 a 和 b 互质”

  * 哈希算法可能用到乘除法。模素数的剩余系除去 0 ，这个集合关于乘法构成群。只有群才能保证每个元素都有逆元，除法才能合法。假设要计算 (p / q) mod m，如果想让结果与 (p mod m) / (q mod m) 相等，必须令 m 为素数，否则逆元求不出来。

##### hash function应用
* Git中的id是由SHA-1 hash生成，40个16进制字符
  * SHA-1: 160bit 
  * SHA-2: 有不同位数，比如SHA-256