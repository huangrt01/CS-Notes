## Algorithms-Potpourri

* [格林公式在面积并问题中的应用](https://trinkle23897.github.io/posts/calc-circle-area-union) --- by n+e

* [Three Optimization Tips for C](https://www.slideshare.net/andreialexandrescu1/three-optimization-tips-for-c-15708507)

  * You can't improve what you can't measure

  * Reduce strength

  * Minimize array writes


```c++
uint32_t digits10(uint64_t v) {
  if (v < P01) return 1;
  if (v < P02) return 2;
  if (v < P03) return 3;
  if (v < P12) {
    if (v < P08) {
      if (v < P06) {
        if (v < P04) return 4;
        return 5 + (v < P05);
      }
      return 7 + (v >= P07);
    }
    if (v < P10) {
      return 9 + (v >= P09);
    }
    return 11 + (v >= P11);
  }
  return 12 + digits10(v/P12);
}

unsigned u64ToAsciiTable(uint64_t value, char* dst) {
  static const char digits[201] =
    "0001020304050607080910111213141516171819"
    "2021222324252627282930313233343536373839"
    "4041424344454647484950515253545556575859"
    "6061626364656667686970717273747576777879"
    "8081828384858687888990919293949596979899"
  uint32_t const length = digits10(value);
  uint32_t next = length - 1;
  while (value >= 100) {
    auto const i = (value % 100) * 2;
    value /= 100;
    dst[next] = digits[i + 1];
    dst[next - 1] = digits[i];
    next -= 2;
  }
  if (value < 10) {
    dst[next] = '0' + uint32_t(value);
  } else {
    auto i = uint32_t(value) * 2;
    dst[next] = digits[i + 1];
    dst[next - 1] = digits[i];
  }
  return length;
}
```

### 数据结构

#### Hash Table

* Open Hashing (Closed Addressing) v.s. Close Hashing (Open Addressing)
  * https://programming.guide/hash-tables-open-vs-closed-addressing.html
  * Open Hashing 的缺点：
    * 读放大
    * load key 有额外一次 memory read
    * rehash 时整个结构要重建
* [Hopscotch hashing](https://en.wikipedia.org/wiki/Hopscotch_hashing)
* Cuckoo Hashing: https://web.stanford.edu/class/archive/cs/cs166/cs166.1146/lectures/13/Small13.pdf
* GeoHash

#### LRU cache

* [linked-list based LRU cache](https://krishankantsinghal.medium.com/my-first-blog-on-medium-583159139237)
  * hashmap的value存双向链表节点的指针
* array-list based LRU cache
  * hashmap的value存array-list的index
    * Array-list的value存pre-index + post-index + entry
  * e.g. Persia

#### Radix Tree

[radix tree (Linux 内核实现)](https://lwn.net/Articles/175432/)：压缩前缀树，维护 kv 查找

前缀树的结构：

* 一棵子树的所有子节点都有相同前缀的 key 值

* 只有叶子节点有对应的 value 值

* key的最大长度固定

当 key 有以下特性的时候压缩前缀树比 hashmap 更具优势：

* 当存储的 key 值本身就有很好的 hash 特性，但是又非常稀疏时。比如说网段，地址空间，可以不用设计复杂 hash 函数。每次插入查找没有hash 计算的开销。
* 大量的 key 有着相同的前缀时，相比于 hashmap 每个节点都要存储完整的 key 值，更具有空间复杂度优势。

应用：[Trie](https://en.wikipedia.org/wiki/Trie)

#### Pairing Heap

https://en.wikipedia.org/wiki/Pairing_heap

实现简单、均摊复杂度优越，用于实现优先队列

定义：一个配对堆要么是一个空堆，要么由一个根元素与一个可能为空的配对堆子树列表所组成。所有子树的根元素都大于该堆的根元素。

```python
type PairingTree[Elem] = Heap(elem: Elem, subheaps: List[PairingTree[Elem]])
type PairingHeap[Elem] = Empty | PairingTree[Elem]
```

操作：

* 合并：一个空堆与另一个堆合并将会返回另一个堆；否则将会返回一个新堆，其将两个堆的根元素中较小的元素当作新堆的根元素，并将较大的元素所在的堆合并到新堆的子堆中。

```C++
function merge(heap1, heap2: PairingHeap[Elem]) -> PairingHeap[Elem]
  if heap1 is Empty
    return heap2
  elsif heap2 is Empty
    return heap1
  elsif heap1.elem < heap2.elem
    return Heap(heap1.elem, heap2 :: heap1.subheaps)
  else
    return Heap(heap2.elem, heap1 :: heap2.subheaps)
```

* 插入：将一个仅有该元素的新堆与需要被插入的堆合并。

```c++
function insert(elem: Elem, heap: PairingHeap[Elem]) -> PairingHeap[Elem]
  return merge(Heap(elem, []), heap)
```

* 删除最小：根元素即为最小元素。删除根元素，然后合并子树，合并方法为从左到右两两合并，然后再从左向右顺序合并。

```c++
function delete-min(heap: PairingHeap[Elem]) -> PairingHeap[Elem]
  if heap is Empty
    error
  else
    merge-pairs(heap.subheaps)
    return elem: Elem
function merge-pairs(list: List[PairingTree[Elem]]) -> PairingHeap[Elem]
  if length(list) == 0
    return Empty
  elsif length(list) == 1
    return list[0]
  else
    return merge(merge(list[0], list[1]), merge-pairs(list[2..]))
```

#### BloomFilter

https://en.wikipedia.org/wiki/Bloom_filter
