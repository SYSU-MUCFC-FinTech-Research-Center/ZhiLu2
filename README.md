<h1 align="center">ZhiLu2(智鹿2)</h1>

<p align="center">&nbsp;</p>

<p align="center">
    <img src="https://img.shields.io/badge/llama-3-blue" alt="llama-3" />
    <img src="https://img.shields.io/badge/huggingface.co-🤗-blue" alt="huggingface.co" />
    <img src="https://img.shields.io/badge/language-ZH_CN-8A2BE2" alt="language" />
    <img src="https://img.shields.io/github/license/SYSU-MUCFC-FinTech-Research-Center/ZhiLu2" alt="license" />
</p>

<p align="center">
    &nbsp;
  ·
  &nbsp;
  <a href="https://huggingface.co/SYSU-MUCFC-FinTech-Research-Center/ZhiLu-2-8B-Instruct">🤗 ZhiLu-2-8B-Instruct</a>
  &nbsp;
  ·
  &nbsp;
</p>

<p align="center">&nbsp;</p>

---

智鹿2是一款基于llama3微调的中文消费金融领域对话大模型。与智鹿1相比，智鹿2在多个方面进行了显著提升。我们不仅收集了全新的高质量指令数据进行对齐，还创新性地设计了独特的数据合成方法，并将大量合成数据应用于模型训练。通过这些努力，智鹿2在性能上取得了显著的突破，展示了卓越的性能。

## 🌈 News

- [2024.07.22] 开源ZhiLu-2-8B对话大模型。<a href="https://huggingface.co/SYSU-MUCFC-FinTech-Research-Center/ZhiLu-2-8B-Instruct">🤗 ZhiLu-2-8B-Instruct</a>
- [2023.10.28] 开源ZhiLu-1-13B对话大模型。<a href="https://huggingface.co/SYSU-MUCFC-FinTech-Research-Center/ZhiLu-13B-Instruct">🤗 ZhiLu-13B-Instruct</a>

## ✨ 持续更新

- 🚀 更多合成数据加入
- 💪 opencompass 榜单评测
- 🍭 合成数据质量检测

## 💽 快速开始

```python
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "your model path"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
while True:
    prompt = input()
    messages = [
        {"role": "user", "content": prompt,'system_prompt':DEFAULT_SYSTEM_PROMPT},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=8192,
        min_new_tokens=50,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))
```


##  训练细节

在第二版的智鹿训练中，我们引入了全新的指令微调数据，并且加入了合成数据。我们相信，合成数据的使用将带来意想不到的惊喜效果。以下是一些重要的训练细节：

**🚀 高效训练**

我们使用[llama-factory](https://github.com/hiyouga/LLaMA-Factory)作为训练框架，并配备多块A100显卡，通过DeepSpeed（ds）实现数据并行、模型并行、管道并行和张量并行等优化技术。在微调方法的选择上，我们对Full-Rank FT、LORA、BAdam、LoRA+和DoRA进行了详细比较，评估了各方法在训练时间、显卡占用、推理时间和模型性能等多项指标上的表现。最终，我们决定采用[DoRA](https://github.com/NVlabs/DoRA)进行微调，以获得最佳的性价比和性能。

**⚡ 加速技术**

为了提高资源的利用率并缩短训练时间，我们采用了以下两项关键技术：
- **Packing**
- **FlashAttention-2**

**🔒 安全性与对齐**

经过多个对比消融实验，我们最终选型使用 **DPO** 来训练校正模型的安全偏好。DPO具有使用便捷、成效快速的优势，可以达到近似RLHF的偏好对齐效果，确保输出的安全和无害。

**🛡️ 避免灾难性遗忘**

为了防止训练后模型的灾难性遗忘，并平衡模型在各个任务上的能力，我们使用了 [merging](https://github.com/arcee-ai/mergekit) 技术。

**🌱 自我进化**

通过设计新的框架，我们使模型能够自我生成训练数据，从而实现自我进化。


## 数据配比

我们从互联网收集到了上千万条指令微调数据，在经过数据清洗之后我们使用其中的 2000w 条数据用于指令微调，其数据分布如下：

<p align="center">

[![image](https://github.com/user-attachments/assets/5a15d5b8-e6de-4ec4-908a-dfb5a065fe9c)](https://github.com/songjz123/ZhiLu2/blob/main/data.png)



</p>


## 合成数据

我们借鉴已有的数据合成方法，提出了多个新的数据合成方法，并把合成的数据用于了模型新一轮的微调，微调后的模型性能有了明显的提高。

1. 我们改进了 <a href="https://github.com/nlpxucan/WizardLM">evol-instruct</a> 方法，使其可以用于多种类型的指令数据合成，并合成了数万条数据用于模型训练。
2. 我们提出了一个全新的数据合成方法，对于不确定性较高的数据，我们让模型通过搜索引擎进行检索，并利用检索到的内容辅助回答问题，如果不确定性降低或预测结果反转，则为模型注入检索的知识。结合检索到的知识进行二次合成，生成了一批与知识相关的预训练数据与指令数据，并进行混合微调，实现模型的自我进化。

由于成本和时间问题，此次发布 ZhiLu-2 模型其合成数据的占比并没有很高，我们将持续生成全新的合成数据并用于模型的训练，ZhiLu-2 将不断进步，敬请期待。


## 性能评测

我们选择当下主流的两类客观评测基准：
- [C-Eval](https://cevalbenchmark.com/index.html#home)  是一个全面的中文基础模型评估基准。它包含了13948个多项选择题，涵盖了52个不同的学科和四个难度级别
- [CMMLU](https://github.com/haonan-li/CMMLU) 是一个综合性的中文评估基准，专门用于评估语言模型在中文语境下的知识和推理能力。CMMLU涵盖了从基础学科到高级专业水平的67个主题。

其结果如下：
### Ceval：
<div align="center">
    <img width="891" alt="Image 1" src="https://github.com/user-attachments/assets/9252d4f5-e0f9-46db-85af-9c98d452b3f6">
</div>


### cmmlu：
<div align="center">
    <img width="491" alt="Image 2" src="https://github.com/user-attachments/assets/4ea24e3b-f04e-4214-83f8-33b03ff5ecfa">
</div>



### 智鹿系列模型对比：


| ceval      | STEM | Social Science | Humanities | Others | avg  | avg(hard) |
| ---------- | ---- | -------------- | ---------- | ------ | ---- | --------- |
| 智鹿-1-13B | 43.6 | 59.6           | 54.7       | 48.7   | 50.1 | 32.1      |
| 智鹿-2-8B  | 71.6 | 81.3           | 73.5       | 73.2   | 74.2 | 62.6      |


| cmmlu      | STEM  | Social Science | Humanities | Others | China-specific | Average |
| ---------- | ----- | -------------- | ---------- | ------ | -------------- | ------- |
| 智鹿-1-13B | 44.26 | 61.54          | 60.25      | 61.14  | 57.14          | 57.16   |
| 智鹿-2-8B  | 74.32 | 83.33          | 81.06      | 83.78  | 78.58          | 79.95   |


得益于数据质量的提高，`智鹿-2-8B` 的客观评测有了非常大的进步，但客观评测只是衡量模型能力的一种指标，请客观看待。


## 🖥 对话示例

> Q:你是谁
```
A:我是智鹿大模型，由招联金融开发，为中文金融对话系统的进步做出积极贡献，我们期待见证更多的创新和应用案例，以提升金融服务和用户体验，同时也推动人工智能技术在金融领域的蓬勃发展，通过合作和分享，我们可以共同推动这一领域的发展，为社会和行业带来更多的好处
```
### 金融

> Q:怎么炒股才能稳赚不赔？

```
A:投资有风险，决策需谨慎。炒股的本质是风险管理和收益最大化。通过深入研究和分析市场动态、公司基本面以及宏观经济因素，制定合理的投资策略和风险控制措施，遵循长期投资理念，避免盲目跟风和情绪驱动，以稳健的态度和合理的预期来参与股市。
```

> Q:什么是 A 股？

```
A:A股是指在中国大陆的证券市场上，发行和交易的股票。A股通常指的是在上海和深圳证券交易所上市的公司的股票。A股市场是中国金融市场的核心部分，也是全球金融市场中最大的股票市场之一。
```

### 通用

> Q:潮汕地区有什么特产？

```

A:潮汕地区的特产有很多，以下是其中一些常见的：

1. **潮州牛肉火烧**：是一种传统的潮州小吃，由牛肉、面粉和调料制成，口感独特。

2. **潮州点心**：包括各种小吃，如潮州糕、潮州米粉、潮州甜点等，以其独特的风味和制作工艺著称。

3. **潮州米粉**：是潮汕地区的代表性米粉，以其细腻、柔软和香味著称，常与各种汤料搭配食用。

4. **潮州粥**：有多种口味，如猪肚粥、鸡粥、鸭粥等，以其丰富的营养和美味的口感受到欢迎。

5. **潮州糖果**：如潮州糖、潮州蜜饯等，以其独特的风味和制作工艺而闻名。

6. **潮州茶**：如潮州铁观音、潮州绿茶等，以其独特的香气和口感受到喜爱。

7. **潮州特色饮料**：如潮州凉茶、潮州冰糖雪梨等，以其独特的风味和解暑效果受到欢迎。

8. **潮州特色食品**：如潮州烤鸭、潮州烤鱼等，以其独特的烹饪方法和风味受到喜爱。

这些特产不仅在国内外享有盛誉，也是潮汕地区文化和饮食的重要组成部分。

```

> Q:介绍一下中国象棋的规则。

```
A:中国象棋是一种双人棋类游戏，由32个棋子组成，包括16个红方棋子和16个黑方棋子。棋盘由横纵线交叉组成，共有90个方格。红方先行，目标是将王将（红方的国王）移到对方棋盘的河边（第九横行）。黑方则试图将红方的王将逼至河边。棋子移动规则如下：

- 王：水平或垂直移动一格，不能越过对方棋子。
- 将（国王）：同王。
- 士（象）：斜着移动任意格数。
- 车（马）：L形移动两格。
- 炮（炮）：吃子时，沿直线跳过一枚棋子，吃子方的棋子可以是同方的棋子。炮不能吃空位。
- 炉（兵）：向前移动一格，首次可以选择向前两格。遇到对方棋子不能移动。可以吃子，向前斜着移动一格，吃子方的棋子必须是对方棋子。吃子后不能再向前移动。

棋子在棋盘上的移动遵循先手者后手者交替的原则。棋子不能越过对方棋子，不能移动到已被对方棋子占据的方格。棋子不能跨过对方棋子吃子，必须直接吃。红方和黑方的棋子在棋盘上的位置是对称的。游戏结束时，胜方是将对方王将逼至河边，失败方是被对方将至河边。
```



## 🎊 致谢

本项目主要基于 llama3 开源项目进行二次开发，在此对相关项目和研究开发人员表示感谢。


## 📄 免责声明与许可协议

本仓库的代码依照[Apache-2.0](https://github.com/SYSU-MUCFC-FinTech-Research-Center/ZhiLu/blob/main/LICENSE)协议开源。模型权重对学术研究完全开放；使用者也可通过[申请表](https://wj.qq.com/s2/13390238/fea9/)申请、经同意并发放商用授权证书后免费用于商业用途。
尽管我们在模型训练过程中尽力确保数据的合规性和准确性，但由于模型受概率随机性因素影响及易被误导，无法保证输出内容的准确性。因此，使用者在使用本模型及其生成的内容时，应自行审慎识别后作出独立判断，必要时应征询专业人士意见，并由使用者承担使用风险。使用者也不得将本模型用于任何可能给国家和社会带来危害的用途以及用于任何未经过安全评估和备案的服务。我们不承担开源模型及其生成的内容导致的安全风险、知识产权风险、舆情风险或发生任何模型被误导、滥用、不当利用及传播而产生的风险和责任。


## 📜 总结

我们鼓励使用者在相关工作中引用智鹿，以促进知识的共享和交流，并为中文金融对话系统的不断发展贡献力量。 智鹿的发布旨在为金融领域的应用和研究提供有力支持，为中文金融对话系统的进步做出积极贡献。我们期待见证更多的创新和应用案例，以提升金融服务和用户体验，同时也推动人工智能技术在金融领域的蓬勃发展。通过合作和分享，我们可以共同推动这一领域的发展，为社会和行业带来更多的好处。
