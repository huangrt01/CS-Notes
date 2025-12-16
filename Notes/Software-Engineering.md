# Software Engineering


[toc]

## Intro

> todo 《A Philosophy of Software Design》
>
> todo 《Software Design X-Rays》

### Intro

*  the three most impactful points are interfaces, stateful systems, and data models.

### Interfaces

* *Interfaces* are contracts between systems. Effective interfaces decouple clients from the encapsulated implementation. Durable interfaces **expose all the underlying essential complexity and none of the underlying accidental complexity**.
* Delightful interfaces are [Eagerly discerning, discerningly eager](https://increment.com/apis/api-design-for-eager-discering-developers/).

### State

* *State* is the hardest part of any system to change, and that resistance to change makes *stateful systems* another critical leverage point. State gets complex faster than other systems and has an inertia that makes it relatively expensive to improve later.
* 安全隐私合规：As you incorporate business obligations around security, privacy, and compliance, changing your stateful systems becomes even more challenging.

### Data models

* *Data models* are the intersection of the interfaces and state, constraining your stateful system’s capabilities down to what your application considers legal.
* A good data model is rigid: it only exposes what it genuinely supports and prevents invalid states’ expression.
* 兼容性：A good data model is tolerant of evolution over time.
* Effective data models are not even slightly clever.



### A/B Testing

#### 实验设计

* **核心原则**：确保实验组 (Experiment) 和对照组 (Control) 在统计学上的同质性 (Homogeneity)，唯一变量是实验策略。
* **流量分配**：通常基于 Hash(User_ID) % 1000 进行分桶。
* **分流模型**：
  * **正交分层**：不同层级的实验（如 UI 层 vs 算法层）相互正交，流量复用。
  * **互斥实验**：同一层级的不同策略实验，流量互斥。

#### 常见陷阱

##### 1. 辛普森悖论 (Simpson's Paradox)
* **现象**：在分组比较中占优势的一方，在总评中反而处于劣势。

##### 3. 幸存者偏差 (Survivorship Bias)
* **现象**：只统计了留存下来的用户，忽略了流失用户。
* **场景**：长周期实验中，实验组策略导致低活跃用户流失，剩下的高活跃用户拉高了平均指标，看似实验效果正向，实则总量下降。

#### 指标统计 SQL 模板

* [Snippet: 通用 A/B 实验指标统计 SQL](snippets/sql-abtest-metrics.sql)

## 研发效率和质量

### Intro

* If you have a development velocity problem, it might be optimizing test runtimes, moving your Docker compile step onto a RAM disk, or using the techniques described in Software Design X-Rays to find the specific files to improve.

### 代码质量 code quality



### 衡量 Measure technical quality

> [Building Evolutionary Architectures](https://www.amazon.com/Building-Evolutionary-Architectures-Support-Constant/dp/1491986360/) and [Reclaim unreasonable software](https://lethain.com/reclaim-unreasonable-software/).

* What percentage of the code is statically typed?
* How many files have associated tests?
* What is test coverage within your codebase?
* How narrow are the public interfaces across modules?
* What percentage of files use the preferred HTTP library?
* Do endpoints respond to requests within 500ms after a cold start?
* How many functions have dangerous read-after-write behavior? Or perform unnecessary reads against the primary database instance?
* How many endpoints perform all state mutation within a single transaction?
* How many functions acquire low-granularity locks?
* How many hot files exist which are changed in more than half of pull requests?

#### proxy measurement

* the number of files changed in each pull request on the understanding
  * smaller pull requests are generally higher quality.
* measure a codebase’s lines of code per file
  * on the assumption that very large files are generally hard to extend.

#### 埋点 instrumentation

* instrumentation is a requirement for useful metrics. Instrumentation complexity is the biggest friction point for adopting these techniques in practice, but if you can push through, you unlock something pretty phenomenal: a real, dynamic quality score that you can track over time and use to create a clarity of alignment in your approach that conceptual alignment cannot.

### 研发效率团队 Technical quality team

> https://staffeng.com/guides/manage-technical-quality/

* Intro	
  * maybe one engineer working on developer tooling for every fifteen product engineers, in addition to your infrastructure engineering investment.
* 人员配置：
  * Technical Program Manager, but typically that is after they cross into operating a Quality program
  * 1-N个P9兼管
* 要点：
  * **Trust metrics over intuition.** 
  * **Keep your intuition fresh**
    * team embedding、team rotation、1:1 discussion
  * **Listen to and learn from your users.**
  * **Do fewer things, but do them better**
  * **Don’t hoard impact.**

* 衡量产出：
  * discounted developer productivity (in the spirit of [discounted cash flow](https://en.wikipedia.org/wiki/Discounted_cash_flow))



## DevOps --> 「云原生-ToB.md」

> todo 《Accelerate: The Science of Lean Software and DevOps: Building and Scaling High Performing Technology Organizations》

### Intro

* DevOps的重点：
  * version control
  * trunk-based development
  * CI/CD
  * production observability (including developers on-call for the systems they write)
  * working in small, atomic changes.

### Monitoring 可观测性服务 —— 运维监控

> 经验中，云原生系统的可观测性开销，往往占到云开销的 15%-25%。
>
> 这么高吗？

#### Intro

* 阿里云有非常丰富的可观测性服务，包括日志服务 SLS，云监控 CloudMonitor， 应用实时监控服务 ARMS

#### 网络

* PingMesh https://cloud.tencent.com/developer/article/1780947

#### 通用

* [Grafana：SpaceX 的数据监测利器，云原生领域的 Tableau](https://mp.weixin.qq.com/s/zgd8KjpGoqwPGC6b1I9owg)
  * 本质是提升数据可观测性（Data Observability），打破数据边界，提供一个“统一的视窗”，实现对数据的全览和实时监控
  * 也有观点认为，可视化的重要性远大于指标、日志和链路追踪
  * 推动“数据民主化”

### Logging



### Tracing

### Alert



### CI/CD平台 DevOps

* 阿里云云效
  * https://www.aliyun.com/product/yunxiao
* [vivo自建](https://mp.weixin.qq.com/mp/wappoc_appmsgcaptcha?poc_token=HCbf5Gij-A3fYkuQVymNKQEvJbLry_qZo2HlMLcR&target_url=https%3A%2F%2Fmp.weixin.qq.com%2Fs%3F__biz%3DMzI4NjY4MTU5Nw%3D%3D%26mid%3D2247498843%26idx%3D1%26sn%3D314aff57db845b164d2e70d0d58ad12a%26token%3D789170263%26lang%3Dzh_CN%26scene%3D21#wechat_redirect)

#### Jenkins clusters



## 系统架构

### 系统迁移 (Migrations)

> 参考: [Migrations: the sole scalable fix to tech debt.](https://lethain.com/migrations/)

系统迁移是在公司和代码库增长过程中，唯一能够规模化解决技术债的有效机制。当公司快速发展时，任何工具或流程都将达到其规模上限，迁移因此成为必然。有效的迁移能力是维持组织高效迭代的关键，否则最终将陷入技术债的泥潭或被迫进行更具破坏性的完全重写。

#### 迁移执行三阶段

一次成功的迁移可以遵循一个标准化的三阶段手册：

1.  **去风险 (Derisk)**
    *   **目标**: 尽快、低成本地验证方案并建立信任。
    *   **执行**: 
        *   与最困难、最边缘的团队深入沟通，迭代设计文档。
        *   **不要从最简单的案例开始**。选择并嵌入1-2个最复杂的团队，与他们共同构建并完成迁移，这能真正暴露方案的弱点。
        *   成功完成早期迁移是为后续大规模推广建立信誉的关键。
2.  **赋能 (Enable)**
    *   **目标**: 规模化推广，降低整个组织的迁移成本。
    *   **执行**: 
        *   **构建自动化工具**: 投入时间开发能自动化处理90%简单场景的迁移工具，而不是急于分发任务。
        *   为剩下10%的复杂场景提供清晰的文档和支持。
3.  **完成 (Finish)**
    *   **目标**: 彻底终结项目，不留尾巴。
    *   **执行**: 
        *   **设定明确的截止日期**: 这是确保项目完成的最有效手段。
        *   **停止支持旧系统**: 在截止日期后，正式停止对旧系统的维护，推动剩余部分完成迁移。
        *   **清理旧代码**: 迁移完成后，务必将旧代码和基础设施彻底移除。




## 代码质量

### 《The Art of Readable Code》 by Dustin Boswell and Trevor Foucher. Copyright 2012 Dustin Boswell and Trevor Foucher, 978-0-596-80229-5

#### chpt 1 Code Should Be Easy to Understand

* Code should be written to minimize the time it would take for someone else to understand it.

#### Part I: Surface Level Improvements

#### chpt 2 Packing Information into Names

* Word Alternatives
  * send: deliver, dispatch, announce, distribute, route
  * find: search, extract, locate, recover
  * start: launch, create, begin, open
  * make: create, set up, build, generate, compose, add, new
* Avoid Generic Names Like tmp and retval
  * `sum_squares += v[i] * v[i];`
  * The name tmp should be used only in cases when being short-lived and temporary is the most important fact about that variable
    * `tmp_file`
  * loop iterators: ci, mi, ui
* Prefer Concrete Names over Abstract Names
  * ServerCanStart() -> CanListenOnPort()
  * `#define DISALLOW_COPY_AND_ASSIGN(ClassName) ...`
* Attaching Extra Information to a Name
  * delay_secs, size_mb, max_kbps, degrees_cw (cw means clockwise)
  * untrustedUrl, **plaintext_**password, **unescaped_**comment, html**_utf8**, data**_urlenc**
  * 拓展：Hungarian notation
    * pszbuffer, z(zero-terminated)
* How Long Should a Name Be?
  * Shorter Names Are Okay for Shorter Scope
  * `ConvertToString()->ToString()`

* Use Name Formatting to Convey Meaning
  * kMaxOpenFile 方便和宏区分
  * 私有成员加下划线后缀

```c++
static const int kMaxOpenFiles = 100;
class LogReader {
  public:
		void OpenFile(string local_file);
	private:
		int offset_;
  	DISALLOW_COPY_AND_ASSIGN(LogReader);
};
```

* about HTML/CSS
  * use underscores to separate words in IDs and dashes to separate words in classes
  * `<div id="middle_column" class="main-content">`

#### chpt 3 Names That Can’t Be Misconstrued

* `filter()` -> `select()` or `exclude()`
* `Clip(text, length)`  -> `truncate(text, max_chars)`
* The clearest way to name a limit is to put `max_` or `min_` in front of the thing being limited.
* when considering ranges
  * Prefer first and last for Inclusive Ranges
  * Prefer begin and end for Inclusive/Exclusive Ranges
* when using bool
  * `read_password` -> `need_password` or `user_is_authenticated`
  * avoid *negated* terms
  * `HasSpaceLeft()` , use `is` or `has`
* Matching Expectations of Users, users may expect `get()` or `size()` to be lightweight methods.
  * `get_mean` -> `compute_mean()`
  * `list::size()`不一定是O(1)
* Example: Evaluating Multiple Name Candidates
  * `inherit_from_experiment_id:` or `copy_experiment:`

#### chpt 4 Aesthetics

* principles
  * Use consistent layout, with patterns the reader can get used to.
  * Make similar code look similar.
  * Group related lines of code into blocks.

* Rearrange Line Breaks to Be Consistent and Compact

```java
public class PerformanceTester {
        // TcpConnectionSimulator(throughput, latency, jitter, packet_loss)
        //                            [Kbps]   [ms]    [ms]    [percent]
        public static final TcpConnectionSimulator wifi =
        		new TcpConnectionSimulator(500, 	80, 		200, 			1);
        public static final TcpConnectionSimulator t3_fiber =
        		new TcpConnectionSimulator(45000, 10, 			0, 			0);
        public static final TcpConnectionSimulator cell =
        		new TcpConnectionSimulator(100,  400, 		250, 			5);
}
```

* Use Methods to Clean Up Irregularity
  * If multiple blocks of code are doing similar things, try to give them the same silhouette.

```c++
void CheckFullName(string partial_name,
                   string expected_full_name,
									 string expected_error) {
  // database_connection is now a class member
  string error;
  string full_name = ExpandFullName(database_connection, partial_name, &error); 			assert(error == expected_error);
  assert(full_name == expected_full_name);
}
```

* Use Column Alignment When Helpful
* Pick a Meaningful Order, and Use It Consistently
  * Match the order of the variables to the order of the `input` fields on the corresponding HTML form.
  * Order them from “most important” to “least important.”
  * Order them alphabetically.
* Organize Declarations into Blocks
* Break Code into “Paragraphs”

```python
def suggest_new_friends(user, email_password):
  # Get the user's friends' email addresses.
  friends = user.friends()
  friend_emails = set(f.email for f in friends)

  # Import all email addresses from this user's email account.
  contacts = import_contacts(user.email, email_password)
  contact_emails = set(c.email for c in contacts)

  # Find matching users that they aren't already friends with.
  non_friend_emails = contact_emails - friend_emails
  suggested_friends = User.objects.select(email__in=non_friend_emails)
  
	# Display these lists on the page.
  display['user'] = user
	display['friends'] = friends
  display['suggested_friends'] = suggested_friends

	return render("suggested_friends.html", display)
```

* Personal Style versus Consistency
  * Consistent style is more important than the “right” style.

#### chpt 5 Knowing What to Comment

The purpose of commenting is to help the reader know as much as the writer did.

* What NOT to Comment
  * Don’t comment on facts that can be derived quickly from the code itself.
  * Don’t Comment Just for the Sake of Commenting
  * Don’t Comment Bad Names—Fix the Names Instead

```python
# remove everything after the second '*'
name = '*'.join(line.split('*')[:2])
```

```c++
// Find a Node with the given 'name' or return NULL.
// If depth <= 0, only 'subtree' is inspected.
// If depth == N, only 'subtree' and N levels below are inspected.
Node* FindNodeInSubtree(Node* subtree, string name, int depth);
```

```c++
// Make sure 'reply' meets the count/byte/etc. limits from the 'request'
void EnforceLimitsFromRequest(Request request, Reply reply);

void ReleaseRegistryHandle(RegistryKey* key);
```

* Recording Your Thoughts
  * Include “Director Commentary”
  * Comment the Flaws in Your Code
  * Comment on Your Constants

```c++
// Surprisingly, a binary tree was 40% faster than a hash table for this data.
// The cost of computing a hash was more than the left/right comparisons.

// This heuristic might miss a few words. That's OK; solving this 100% is hard.

// This class is getting messy. Maybe we should create a 'ResourceNode' subclass to
// help organize things.
```

```c++
// TODO: use a faster algorithm
// TODO(dustin): handle other image formats besides JPEG

// FIXME
// HACK
// XXX: Danger! Major problem here!

// todo: (lower case) or maybe-later:
```

```c++
NUM_THREADS = 8; // as long as it's >= 2 * num_processors, that's good enough.

// Impose a reasonable limit - no human can read that much anyway.
const int MAX_RSS_SUBSCRIPTIONS = 1000;

image_quality = 0.72; // users thought 0.72 gave the best size/quality tradeoff
```

* Put Yourself in the Reader’s Shoes
  * Anticipating Likely Questions
  * Advertising Likely Pitfalls
  * “Big Picture” Comments
  * Summary Comments

```c++
// Force vector to relinquish its memory (look up "STL swap trick")
vector<float>().swap(data);
```

```c++
// Calls an external service to deliver email.  (Times out after 1 minute.)
void SendEmail(string to, string subject, string body);

// Runtime is O(number_tags * average_tag_depth), so watch out for badly nested inputs.
def FixBrokenHtml(html): ...
```

```c++
// This file contains helper functions that provide a more convenient interface to
// our file system. It handles file permissions and other nitty-gritty details.
```

```python
def GenerateUserReport():
  # Acquire a lock for this user
  ...
  # Read user's info from the database
  ...
  # Write info to a file
  ...
  # Release the lock for this user
```

* Final Thoughts—Getting Over Writer’s Block

```c++
// Oh crap, this stuff will get tricky if there are ever duplicates in this list.
--->
// Careful: this code doesn't handle duplicates in the list (because that's hard to do)
```

#### chpt 6 Making Comments Precise and Compact

**Comments should have a high information-to-space ratio.**

* Keep Comments Compact

```c++
// CategoryType -> (score, weight)
typedef hash_map<int, pair<float, float> > ScoreMap;
```

* Avoid Ambiguous Pronouns

```c++
// Insert the data into the cache, but check if it's too big first.
--->
// Insert the data into the cache, but check if the data is too big first.
--->
// If the data is small enough, insert it into the cache.
```

* Polish Sloppy Sentences
  * e.g.  Give higher priority to URLs we've never crawled before.

* Describe Function Behavior Precisely
  * e.g. Count how many newline bytes ('\n') are in the file.
* Use Input/Output Examples That Illustrate Corner Cases

```c++
// ...
// Example: Strip("abba/a/ba", "ab") returns "/a/"
String Strip(String src, String chars) { ... }

// Rearrange 'v' so that elements < pivot come before those >= pivot;
// Then return the largest 'i' for which v[i] < pivot (or -1 if none are < pivot)
// Example: Partition([8 5 9 8 2], 8) might result in [5 2 | 8 9 8] and return 1
int Partition(vector<int>* v, int pivot);
```

* State the Intent of Your Code

```c++
void DisplayProducts(list<Product> products) {
  products.sort(CompareProductByPrice);
  // Display each price, from highest to lowest
  for (list<Product>::reverse_iterator it = products.rbegin(); it != products.rend(); ++it)
    DisplayPrice(it->price);
		... 
	}
```

* “Named Function Parameter” Comments

```c++
void Connect(int timeout, bool use_encryption) { ... }

// Call the function with commented parameters
Connect(/* timeout_ms = */ 10, /* use_encryption = */ false);
```

* Use Information-Dense Words
  * // This class acts as a **caching layer** to the database.
  * // **Canonicalize** the street address (remove extra spaces, "Avenue" -> "Ave.", etc.)

#### Part II: Simplifying Loops and Logic

#### chpt 7 Making Control Flow Easy to Read

* The Order of Arguments in Conditionals
  * `while (bytes_received < bytes_expected)`
* The Order of if/else Blocks
  * Prefer dealing with the *positive* case first instead of the negative—e.g., if (debug) instead of if (!debug).
  * Prefer dealing with the *simpler* case first to get it out of the way. This approach might also allow both the if and the else to be visible on the screen at the same time, which is nice.
  * Prefer dealing with the more *interesting* or conspicuous case first.
* The ?: Conditional Expression (a.k.a. “Ternary Operator”)
  * By default, use an if/else. The ternary ?: should be used only for the simplest cases.
* Avoid do/while Loops

```java
public boolean ListHasNode(Node node, String name, int max_length) {
  while (node != null && max_length-- > 0) {
    if (node.name().equals(name)) return true;
    node = node.next();
  }
  return false;
}
```

```c++
do {
  continue;
} while (false);
// loop just once
```

* Returning Early from a Function
  * cleanup code
    * C++: destructor
    * Java, Python: try finally
      * [Do it with a Python decorator](https://stackoverflow.com/questions/63954327/python-is-there-a-way-to-make-a-function-clean-up-gracefully-if-the-user-tries/63954413#63954413)
    * Python: with
    * C#: using

```c++
struct StateFreeHelper {
  state* a;
  StateFreeHelper(state* a) : a(a) {}
  ~StateFreeHelper() { free(a); }
};

void func(state* a) {
  StateFreeHelper(a);
  if (...) {
    return;
  } else {
    ...
  }
}
```

```python
def do_stuff(self):
  self.some_state = True
  try:
    # do stuff which may take some time - and user may quit here
  finally:
    self.some_state = False
```

* The Infamous goto
  * 问题在于滥用，比如多种goto混合、goto到前面的代码
* Minimize Nesting
  * Removing Nesting by Returning Early
  * Removing Nesting Inside Loops: use continue for independent iterations

* Can You Follow the Flow of Execution?

![flow](./Software-Engineering/flow_of_execution.png)

#### chpt 8 Breaking Down Giant Expressions

* Explaining Variables

```python
username = line.split(':')[0].strip()
if username == "root":
	...
```

* Summary Variables

```java
final boolean user_owns_document = (request.user.id == document.owner_id);
if (user_owns_document) {
}
...
if (!user_owns_document) {
  // document is read-only...
}
```

* Using De Morgan’s Laws
* Abusing Short-Circuit Logic
  * There is also a newer idiom worth mentioning: in languages like Python, JavaScript, and Ruby, the “or” operator returns one of its arguments (it doesn’t convert to a boolean), so code like: x = a || b || c, can be used to pick out **the first “truthy” value** from a, b, or c.

```c++
assert((!(bucket = FindBucket(key))) || !bucket->IsOccupied());
--->
bucket = FindBucket(key);
if (bucket != NULL) assert(!bucket->IsOccupied());
```

* Example: Wrestling with Complicated Logic

```c++
struct Range {
	int begin;
	int end;
  // For example, [0,5) overlaps with [3,8)
  bool OverlapsWith(Range other);
};

bool Range::OverlapsWith(Range other) {
  return (begin >= other.begin && begin < other.end) ||
         (end > other.begin && end <= other.end) ||
         (begin <= other.begin && end >= other.end);
}

bool Range::OverlapsWith(Range other) {
  if (other.end <= begin) return false;  // They end before we begin
  if (other.begin >= end) return false;  // They begin after we end
  return true;  // Only possibility left: they overlap
}
```

* Breaking Down Giant Statements

* Another Creative Way to Simplify Expressions

```c++
 void AddStats(const Stats& add_from, Stats* add_to) {
   #define ADD_FIELD(field) add_to->set_##field(add_from.field() + add_to->field())
   ADD_FIELD(total_memory);
   ADD_FIELD(free_memory);
   ADD_FIELD(swap_memory);
   ADD_FIELD(status_string);
   ADD_FIELD(num_processes);
   ...
   #undef ADD_FIELD
 }
```

#### chpt 9 Variables and Readability

* Eliminating Variables
  * Useless Temporary Variables
  * Eliminating Intermediate Results
  * Eliminating Control Flow Variables
* Shrink the Scope of Your Variables
  * Another way to restrict access to class members is to **make as many methods static as possible**. Static methods are a great way to let the reader know “these lines of code are isolated from those variables.”
  * break the large class into smaller classes
  * if Statement Scope in C++
  * Creating “Private” Variables in JavaScript
  * JavaScript Global Scope
    * always define variables using the var keyword (e.g., var x = 1)
  * No Nested Scope in Python and JavaScript
    * 在最近祖先手动定义 xxx = None
  * Moving Definitions Down

```c++
if (PaymentInfo* info = database.ReadPaymentInfo()) {
  cout << "User paid: " << info->amount() << endl;
}
```

```javascript
var submit_form = (function () {
	var submitted = false; // Note: can only be accessed by the function below
	return function (form_name) {
    if (submitted) {
      return;  // don't double-submit the form
    }
		...
		submitted = true;
  };
}());
```

* Prefer Write-Once Variables
  * The more places a variable is manipulated, the harder it is to reason about its current value.
* A Final Example

```javascript
var setFirstEmptyInput = function (new_value) {
  for (var i = 1; true; i++) {
    var elem = document.getElementById('input' + i);
    if (elem === null)
      return null;  // Search Failed. No empty input found.
    if (elem.value === '') {
      elem.value = new_value;
      return elem;
    }
  }
};
```

#### Part III: Reorganizing Your Code

#### chpt 10 Extracting Unrelated Subproblems

* Introductory Example: findClosestLocation()
* Pure Utility Code
  * read file to string
* Other General-Purpose Code

```javascript
var format_pretty = function (obj, indent) {
  // Handle null, undefined, strings, and non-objects.
  if (obj === null) return "null";
  if (obj === undefined) return "undefined";
  if (typeof obj === "string") return '"' + obj + '"';
  if (typeof obj !== "object") return String(obj);
  if (indent === undefined) indent = "";
  // Handle (non-null) objects.
  var str = "{\n";
  for (var key in obj) {
    str += indent + "  " + key + " = ";
    str += format_pretty(obj[key], indent + " ") + "\n";
  }
  return str + indent + "}";
};
```

* Create a Lot of General-Purpose Code

* Project-Specific Functionality

```python
CHARS_TO_REMOVE = re.compile(r"['\.]+")
CHARS_TO_DASH = re.compile(r"[^a-z0-9]+")

def make_url_friendly(text):
  text = text.lower()
  text = CHARS_TO_REMOVE.sub('', text)
  text = CHARS_TO_DASH.sub('-', text)
  return text.strip("-")

business = Business()
business.name = request.POST["name"]
business.url = "/biz/" + make_url_friendly(business.name)
business.date_created = datetime.datetime.utcnow()
business.save_to_database()
```

* Simplifying an Existing Interface
* Reshaping an Interface to Your Needs

```python
def url_safe_encrypt(obj):
  obj_str = json.dumps(obj)
  cipher = Cipher("aes_128_cbc", key=PRIVATE_KEY, init_vector=INIT_VECTOR, op=ENCODE)
  encrypted_bytes = cipher.update(obj_str)
  encrypted_bytes += cipher.final() # flush out the current 128 bit block
  return base64.urlsafe_b64encode(encrypted_bytes)
```

* Taking Things Too Far

#### chpt 11 One Task at a Time

* Tasks Can Be Small
  * e.g. 分解 old vote 和 new vote
* Extracting Values from an Object

```javascript
var first_half, second_half;

if (country === "USA") {
  first_half = town || city || "Middle-of-Nowhere";
  second_half = state || "USA";
} else {
  first_half = town || city || state || "Middle-of-Nowhere";
  second_half = country || "Planet Earth";
}

return first_half + ", " + second_half;
```

* A Larger Example

#### chpt 12 Turning Thoughts into Code

* Describing Logic Clearly
  *  “rubber ducking”
  *  You do not really understand something unless you can explain it to your grandmother. —Albert Einstein

```php
if (is_admin_request()) {
  // authorized
} elseif ($document && ($document['username'] == $_SESSION['username'])) {
  // authorized
} else {
  return not_authorized();
}
// continue rendering the page ...
```

* Knowing Your Libraries Helps
* Applying This Method to Larger Problems

```python
def PrintStockTransactions():
  stock_iter = ...
	price_iter = ...
  num_shares_iter = ...

  while True:
    time = AdvanceToMatchingTime(stock_iter, price_iter, num_shares_iter)
    if time is None:
      return

    # Print the aligned rows.
    print "@", time,
    print stock_iter.ticker_symbol,
    print price_iter.price,
    print num_shares_iter.number_of_shares

    stock_iter.NextRow()
    price_iter.NextRow()
    num_shares_iter.NextRow()
    
def AdvanceToMatchingTime(row_iter1, row_iter2, row_iter3):
  while row_iter1 and row_iter2 and row_iter3:
    t1 = row_iter1.time
    t2 = row_iter2.time
    t3 = row_iter3.time

    if t1 == t2 == t3:
      return t1

    tmax = max(t1, t2, t3)

    # If any row is "behind," advance it.
    # Eventually, this while loop will align them all.
    if t1 < tmax: row_iter1.NextRow()
    if t2 < tmax: row_iter2.NextRow()
    if t3 < tmax: row_iter3.NextRow()

  return None  # no alignment could be found
```

#### chpt 13 Writing Less Code

* Don’t Bother Implementing That Feature—You Won’t Need It
* Question and Break Down Your Requirements
  * Example: A Store Locator ---- For any given user’s latitude/longitude, find the store with the closest latitude/longitude.
    * When the locations are on either side of the International Date Line
    * When the locations are near the North or South Pole
    * Adjusting for the curvature of the Earth, as “longitudinal degrees per mile” changes
  * Example: Adding a Cache
* Keeping Your Codebase Small

* Be Familiar with the Libraries Around You
  * Example: Lists and Sets in Python
* Example: Using Unix Tools Instead of Coding
  * When a web server frequently returns 4xx or 5xx HTTP response codes, it’s a sign of a potential problem (4xx being a client error; 5xx being a server error). 

#### PART IV Selected Topics

#### chpt 14 Testing and Readability

* testing中的一些概念：
  * 单测：单元性和隔离性
  * property-based testing属于单测



* Make Tests Easy to Read and Maintain
* What’s Wrong with This Test?

```c++
void CheckScoresBeforeAfter(string input, string expected_output) {
  vector<ScoredDocument> docs = ScoredDocsFromString(input);
  SortAndFilterDocs(&docs);
  string output = ScoredDocsToString(docs);
  assert(output == expected_output);
}

vector<ScoredDocument> ScoredDocsFromString(string scores) {
  vector<ScoredDocument> docs;
  replace(scores.begin(), scores.end(), ',', ' ');
  // Populate 'docs' from a string of space-separated scores.
  istringstream stream(scores);
  double score;
  while (stream >> score) {
    AddScoredDoc(docs, score);
  }
  return docs;
}
string ScoredDocsToString(vector<ScoredDocument> docs) {
  ostringstream stream;
  for (int i = 0; i < docs.size(); i++) {
    if (i > 0) stream << ", ";
    stream << docs[i].score;
  }
  return stream.str();
}
```

* Making Error Messages Readable
  * Python `import unittest`

```c++
BOOST_REQUIRE_EQUAL(output, expected_output)
```

* Choosing Good Test Inputs
  * In general, you should pick the simplest set of inputs that completely exercise the code.
  * Simplifying the Input Values
    * -1e100、-1
    * it’s more effective to construct large inputs programmatically, constructing a large input of (say) 100,000 values
* Naming Test Functions
* What Was Wrong with That Test?

* Test-Friendly Development
  * Test-driven development (TDD)
  * Table 14.1: Characteristics of less testable code
    * Use of global variables ---> gtest set_up()
    * Code depends on a lot of external components
    * Code has nondeterministic behavior

* Going Too Far
  * Sacrificing the readability of your real code, for the sake of enabling tests.
  * Being obsessive about 100% test coverage.
  * Letting testing get in the way of product development.

#### chpt 15 Designing and Implementing a “Minute/Hour Counter”

* Defining the Class Interface

```c++
// Track the cumulative counts over the past minute and over the past hour.
// Useful, for example, to track recent bandwidth usage.
class MinuteHourCounter {
  // Add a new data point (count >= 0).
  // For the next minute, MinuteCount() will be larger by +count. 
  // For the next hour, HourCount() will be larger by +count.
  void Add(int count);

  // Return the accumulated count over the past 60 seconds.
  int MinuteCount();
  
  // Return the accumulated count over the past 3600 seconds.
  int HourCount();
};
```

* Attempt 1: A Naive Solution
  * list, reverse_iterator，效率低
* Attempt 2: Conveyor Belt Design
  * 两个传送带，内存消耗大，拓展成本高
* Attempt 3: A Time-Bucketed Design
  * 本质利用了统计精度可牺牲的特点，离散化实现

```c++
// A class that keeps counts for the past N buckets of time.
class TrailingBucketCounter {
  public:
    // Example: TrailingBucketCounter(30, 60) tracks the last 30 minute-buckets of time.
    TrailingBucketCounter(int num_buckets, int secs_per_bucket);
    void Add(int count, time_t now);
    // Return the total count over the last num_buckets worth of time
    int TrailingCount(time_t now);
};
class ConveyorQueue;
```

