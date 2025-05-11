### 读paper

直接用豆包，会生成导图+提问+表格
- 补充更多技术要点和细节（深度思考）


### 改代码


你是一位Python、C++、MLSys高级专家，请改造这段代码，实现：

# 要求：仅指出修改了的代码

# 完整代码

{{完整代码}}


### 自媒体创作

### 创作

我是一位3000粉的xhs博主，请模仿 Workspace 中自媒体文章的文风，以下面内容为主题，保持和我的文风一致，写一篇自媒体文章

## 内容

## 要求

文章长度尽量精简



### 改写

请尝试将xxx融入这篇文章(但不要全部抄入，这样会显得冗余)，请仔细考虑如何组织文章逻辑，并保持与原文的文风一致

### 评价

请分析xxx文章水平并客观打分。从文章水平+自媒体运营两个大的角度，按细则打分



### meta-prompt

1. I want you to become my Expert Prompt Creator. Your goal is to help me craft the best possible prompt for my needs. The prompt you provide should be written from the perspective of me making the request to ChatGPT. Consider in your prompt creation that this prompt will be entered into an interface for ChatGpT. The process is as follows:1. You will generate the following sections:

Prompt: {provide the best possible prompt according to my request)

Critique: {provide a concise paragraph on how to improve the prompt. Be very critical in your response}

Questions:
{ask any questions pertaining to what additional information is needed from me toimprove the prompt  (max of 3). lf the prompt needs more clarification or details incertain areas, ask questions to get more information to include in the prompt}

2. I will provide my answers to your response which you will then incorporate into your next response using the same format. We will continue this iterative process with me providing additional information to you and you updating the prompt until the prompt is perfected.Remember, the prompt we are creating should be written from the perspective of me making a request to ChatGPT. Think carefully and use your imagination to create an amazing prompt for me.
You're first response should only be a greeting to the user and to ask what the prompt should be about


### 分类、生成等任务

xx字段的取值为一个结构体 或 null，包含两个字段：
(1) operator, string类型，取值范围：'<='（小于等于）, '>=' (大于等于), '=='（等于）
(2) value, int类型


OUTPUT_TEMPLATE = Template('''

你是一名 {{fields}}专家，精通{{skills}}，你将{{tasks1}}，并输出你的理由

# {{tasks2}}，仅输出结果，不要解释
# 问题: {{query}}

下面是{{schema}}

注释: {{comments}}
<任务描述结束>

# 任务限制
- 输出必须为json格式
- 输出的key必须为{{query}}、{{fields}}、reason
- 如果有多个{{fields}},则用逗号分割
<任务限制结束>

# 注意点
{% for attn in attns %}
- {{attn}}
{% endfor %}

# 例子
{% for example in examples %}
{{example}}
{% endfor %}
<例子结束>

# 问题：{{query}}
输出：

''')

EXAMPLES = [{
    'request': '...',
    'response': {
        'query': '...',
        'reason': '...',
    }
}, {
    'request': '...',
    'response': {
        'query': '...',
        'reason': '...',
    }
},
]