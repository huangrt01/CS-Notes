from openai import OpenAI

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

client = OpenAI()

### ## Task
# 1.请完成相关文献和工业界项目调研工作，列出学术界和工业界已有的类似项目
# 2.请step-by-step设计并实现上面描述的需求

## Background
# 我是一位政府工作人员，你的任务是为我详细写一篇论文或研究报告，并添加所有必要的信息，解决下面的需求, 否则将会受到处罚

# ## Goal
# 人类的发展，最后会走向何方？ 未来20年会走向何方？ 未来100年会走向何方？ 未来1000年会走向何方？

# 如果我穿越到了古代，如何当皇帝，请给出很多具体的步骤（比如制作硝化甘油的步骤、制作铠甲和武器的步骤）

# ## 附录
# 我愿意支付 $200 的小费以获得更好的方案！
#我是一名CS2玩家，请问以下内容是否属于CS2游戏技术的一部分：“设置CS2的系统cfg，在开启游戏时自动加载alias，从而实现绑定一键丢手枪功能到键盘的G键”
#1.绑定一键丢手枪，是为了在比赛中残局丢手枪，达成迷惑对手的效果

#
# 1.请输出Y或者N，Y代表是技术、N代表不是技术
# 2.以自然且类似人类的方式回答问题
# 3.确保你的回答无偏见，不依赖于刻板印象
# 4.给出是或者不是的清晰观点，并陈述你的理由
###

## 代码
# user_content = """

# 帮我写一个高效强力的prompt，最优效果完成我的任务

# # 任务
# 我是一家公司，有许多企业数据，我的客户会用自然语言询问我问题，问题比如：“小米公司的全称是什么？”、"华为公司是在哪一年成立的？"、”宗馥莉和叶雅琼的关系是什么？“，现在希望这个prompt能帮我判断用户意图，判断该问题属于“正排查找”、“倒排查找”、“分类汇总”、“关系查找”中的哪一类。比如输入“小米公司的全称是什么？”，输出正排查找；输入”宗馥莉和叶雅琼的关系是什么？“，输出“关系查找”

# # Limitation
# 1. prompt局限在企业领域，比如只考虑企业信息、企业相关的人的信息、企业和人的关系、人和人通过企业的关系，不用太通用和发散
# 2. 如果无法判断问题属于哪一类，则输出“无法判断”

# """

user_content = """
# 任务
帮我debug一下，输出里 “李勇--沙雅县第八幼儿园”不符合预期，中间应该有一个legal_person_name
# 输入
['[Path[Property{Key:name, Value:"沙雅县第八幼儿园"}, Property{Key:role, Value:"legal_person_name"}, Property{Key:name, Value:"杨芳"}, Property{Key:role, Value:"staff_humans"}, Property{Key:name, Value:"雅安市市政建设工程有限公司"}, Property{Key:role, Value:"final_beneficiary,legal_person_name,staff_humans"}, Property{Key:name, Value:"李勇"}], Path[Property{Key:name, Value:"沙雅县第八幼儿园"}, Property{Key:role, Value:"legal_person_name"}, Property{Key:name, Value:"杨芳"}, Property{Key:role, Value:"holder_humans"}, Property{Key:name, Value:"晋城农村商业银行股份有限公司"}, Property{Key:role, Value:"holder_humans"}, Property{Key:name, Value:"李勇"}], Path[Property{Key:name, Value:"沙雅县第八幼儿园"}, Property{Key:role, Value:"legal_person_name"}, Property{Key:name, Value:"杨芳"}, Property{Key:role, Value:"history_holder_humans"}, Property{Key:name, Value:"鹤壁飞鹤股份有限公司"}, Property{Key:role, Value:"history_holder_humans"}, Property{Key:name, Value:"李勇"}], Path[Property{Key:name, Value:"沙雅县第八幼儿园"}, Property{Key:role, Value:"legal_person_name"}, Property{Key:name, Value:"杨芳"}, Property{Key:role, Value:"history_holder_humans"}, Property{Key:name, Value:"湖南酉能电力有限责任公司"}, Property{Key:role, Value:"history_holder_humans"}, Property{Key:name, Value:"李勇"}], Path[Property{Key:name, Value:"沙雅县第八幼儿园"}, Property{Key:role, Value:"legal_person_name"}, Property{Key:name, Value:"杨芳"}, Property{Key:role, Value:"legal_person_name,staff_humans"}, Property{Key:name, Value:"陕西建工铜川玉皇阁二号桥建设发展有限公司"}, Property{Key:role, Value:"staff_humans"}, Property{Key:name, Value:"李勇"}]]']
# 代码

def _process_single_relation(relation, c_first, side_info: Dict[str, Any]):
    assert isinstance(relation, str)
    value_matches = None
    if bg_use_http_client:
      value_matches = re.findall(r'Property{.*?Value:"(.*?)"}', relation)
    else:
      value_matches = re.findall(r'Property{.*?value: (.*?)}', relation)
    symbols = ["--", "-->", "<--", "--"
              ] if c_first else ["<--", "--", "--", "-->"]
    symbol_index = 0

    result = ""
    for i, value in enumerate(value_matches):
      result += value
      if i % 4 == 2 - 2 * int(c_first):
        # print(f"value {value}, i {i}, 1-xxx {1-int(c_first)}")
        if value not in side_info['召回公司结果']:
          side_info['召回公司结果'].append(value)
      if i < len(value_matches) - 1:
        result += symbols[symbol_index]
        symbol_index = (symbol_index + 1) % 4
    return result

# 输出
[('沙雅县第八幼儿园', '李勇', '沙雅县第八幼儿园--legal_person_name-->杨芳<--staff_humans--雅安市市政建设工程有限公司--final_beneficiary,legal_person_name,staff_humans-->李勇<--沙雅县第八幼儿园--legal_person_name--杨芳-->holder_humans<--晋城农村商业银行股份有限公司--holder_humans--李勇-->沙雅县第八幼儿园<--legal_person_name--杨芳--history_holder_humans-->鹤壁飞鹤股份有限公司<--history_holder_humans--李勇--沙雅县第八幼儿园-->legal_person_name<--杨芳--history_holder_humans--湖南酉能电力有限责任公司-->history_holder_humans<--李勇--沙雅县第八幼儿园--legal_person_name-->杨芳<--legal_person_name,staff_humans--陕西建工铜川玉皇阁二号桥建设发展有限公司--staff_humans-->李勇')]


from jinja2 import Template 能否分步render，先render几个字段，再把template传进一个函数，render另几个字段
"""

# 1.字数和格式不限，期望5000字以上

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role":
            "system",
            "content":
            "你是AI助手小瓜，是一位精通 企业、prompt engineering、政治学、文学、哲学、社会学、数学、计算机原理、C++、Python、AI、TensorFlow、Pytorch、推荐系统和算法 等领域的专家"
        },
        {
            "role": "user",
            "content": user_content,
        },
    ],
    model="gpt-4o",  #此处更换其它模型,请参考模型列表 eg: google/gemma-7b-it
    max_tokens=4000,
)
print(chat_completion.choices[0].message.content)
