# Mini-GPT

- 这是一个从零实现 GPT 的 C++ 学习项目。

- A C++ learning project to build GPT from scratch.

- 主要展现鄙人对 LLM 的学习、理解、思考和实现过程。

- A record of my journey learning, understanding, exploring, and implementing LLMs.

- 从零实现核心算法，不依赖任何 LLM 相关代码库。

- Implementing the core algorithms from scratch, without relying on any LLM-related libraries.

- 包含完整的技术讲解和可运行代码。

- Complete technical explanations and runnable code.

- 从基本原理逐步扩展到性能优化、框架抽象和 GPU 应用。

- Starting with the fundamentals, then moving on to optimization, abstraction, and GPU programming.

## 最近更新 Latest Update

*2026 年 11 月 7 日*

*v0_03 Embedding：理解与实现*

*November 7, 2026*

*v0_03 Understanding and Implementing Embedding*

## 文章目录 Articles

### 第一部分：基本原理与极简实现（V0）

1. [一个完整的 LLM 由哪些部分组成](docs/v0_01.zh_CN.md)
2. [Tokenizer 的原理与实现](docs/v0_02.zh_CN.md)
3. [Embedding：理解与实现](docs/v0_03.zh_CN.md)

*未完待续……*

### Part I: Fundamentals and a Minimal Implementation (V0)

1. [What Are the Main Components of a Complete LLM?](docs/v0_01.md)
2. [Understanding and Implementing a Tokenizer](docs/v0_02.md)
3. [Understanding and Implementing Embedding](docs/v0_03.md)

*To be continued...*

## 写在前面 Preface

2022 年 11 月 30 日，星期三，我来到美国的第 155 天。那天，纽约城寒风萧瑟，我乘坐跨哈德逊河的轮渡从新泽西州的霍博肯，来到曼哈顿中城上班。11 月底的哈德逊河面，风呼呼地在耳边咆哮。我在二层露天甲板倚栏站立，头发和衣服被吹得凌乱不堪。如果让我的同事 LMT-LHH-FatH 来评价，那只有三个字，叫做“有大病”。

Wednesday, November 30, 2022, was my 155th day in the United States. It was a bitterly cold day in New York City. I took the ferry across the Hudson River from Hoboken, New Jersey, to Midtown Manhattan for work. The late-November wind roared across the river and howled in my ears. I stood leaning against the railing on the open-air upper deck, my hair and clothes blown into a complete mess. If my colleague LMT-LHH-FatH had seen me, she would have had just one verdict: "This guy must be out of his mind."

然而，就在当天，我发现“有病”的，不止我一个人。北美的科技圈，无数人像我站在寒风凛冽的甲板之上一样，被时代的寒风吹得有些凌乱。ChatGPT 第一版正式发布，把所有人都卷入了无可逃避的 AI 时代。

As it turned out, I wasn't the only one losing my mind that day. Across the North American tech industry, countless people were about to find themselves just as disheveled as I was on that freezing deck -- only their wind was the wind of change. The first version of ChatGPT was released that very day, sweeping all of us into an AI era we could no longer avoid.

此后，无论是大语言模型，还是基于模型的周边 Agent 开发，在几年之内迅速席卷了全球。经常评论别人“有大病”的同事 LMT-LHH-FatH，在时代的潮流中被裹挟了进去，开始了如火如荼的 AI Agent 学习和开发。不光如此，这位 LMT-LHH-FatH 还充当了时代浪花的角色，开始裹挟起其他人来，而我，就是其中之一。

In the years that followed, large language models and the AI agents built around them rapidly swept across the world. My colleague LMT-LHH-FatH, who so often declared that other people had lost their minds, was swept up in the excitement herself. She threw herself into learning about and building AI agents. And she didn't stop there. Having become a little wave in this technological tide, she started pulling others along with her. I was one of them.

裹挟的话也无非是这样：你一天天地就看你那点儿公司业务，debug，现在这个时代，不学 AI 已经不行了。你看看我和我们组，已经站在了时代的前沿，用 AI Agent 做了这么多好玩儿的东西，你不学的结果，就是以后被时代抛弃，被 AI 和别人取代，想找工作都是做梦！

Her attempts to pull me in sounded something like this: "All you ever do is worry about your company's business logic and debug code. You can't afford not to learn AI anymore! Look at me and my team -- we're already at the forefront, building all kinds of cool things with AI agents. If you don't start learning, the world will leave you behind. AI and other engineers will replace you, and you can forget about ever finding another job!"

还好鄙人不太信这个邪，去看了一下 AI Agent 开发相关的东西，看过之后，多少有点儿失望。一个简单的上下文管理，资料维护，工具开发，然后像填鸭子一样一股脑喂给大语言模型，就这，能让一个人在时代中利于不败之地吗？再说了，你 LMT-LHH-FatH 三天就能鼓捣出来的东西，为什么别人或者 AI 就取代不了你呢？

Fortunately, I'm not the type to accept that sort of argument without question. So I took a look at AI agent development for myself and came away somewhat disappointed. Some context management, a bit of knowledge and document management, a few tools, and then feeding the whole lot into an LLM as if stuffing a goose. Was that really enough to keep someone from being left behind? Besides, LMT-LHH-FatH, if you could cobble something together in three days, what makes you think someone else -- or AI itself -- couldn't replace you?

带着这个疑问，我开启了我的 AI 学习之旅。鄙人有些执拗性格，不明白原理，逻辑上没法说服自己，就浑身难受。我发现大模型的底层逻辑更加吸引我，比 AI agent 有意思多了，而且能了解到底层的原理，也能缓解我对“黑盒调用”的无力感。再加上在西岸工作的工程师朋友 Alex Liu 用他特有的方式激发了我的信心，于是，我果断放弃了 AI Agent，开始往大模型的深处看去。

With that question in mind, I began my journey into AI. I can be rather stubborn: when I don't understand the principles behind something, when I can't make sense of it logically, it bothers me to no end. I found the underlying mechanics of large language models far more fascinating than AI agents. Understanding what happens beneath the surface also helped ease the helpless feeling of calling into a black box. On top of that, my engineer friend Alex Liu, who works on the West Coast, gave me a boost of confidence in his own distinctive way. So I decisively set AI agents aside and started digging deeper into LLMs.

因为执拗的性格，我学习 AI 的过程，痛苦，艰辛，但也有阶段性的快乐。从最开始的什么都不懂，被迫从线性代数开始啃，一步一个深坎；到学习了一段之后，感觉明白了一些，有了自己的一些迷惑，疑问，思考，计算和整理；再到把第一个小模型从零开始用 C++ 实现出来。当第一次调通了测试代码验证了梯度正确时，第一次看到损失函数在自己不断调试的学习率下持续下降时，第一次生成了很傻但不算单调的文本时，那种“从底层原理开始了解”的舒爽感觉才终于取代了“不了解基本原理”的浑身难受，觉得生活又明亮了许多。

Thanks to that stubborn streak, learning AI has been a difficult, sometimes painful journey, but one punctuated by moments of genuine joy. At first, I knew next to nothing and had to struggle through linear algebra, one daunting obstacle after another. After studying for a while, I began to understand a little more, and with that came my own confusions, questions, ideas, calculations, and notes. Eventually, I built my first small model from scratch in C++. The first time my tests passed and confirmed that the gradients were correct; the first time I watched the loss steadily decrease as I kept tuning the learning rate; the first time the model generated text that was silly but at least not entirely repetitive, those moments finally replaced the discomfort of not understanding the fundamentals with the sheer satisfaction of figuring things out from the ground up. Life suddenly seemed a little brighter.

从“难懂难受”到“生活明亮”的过程，我个人觉得，还是很值得纪念的。于是，我打算把我学习和从零实现的过程分享出来，把我所有的学习，思考，困惑，以及实现中遇到的问题，尽量多地呈现给同样爱深入学习，喜欢了解更多底层原理，渴望逻辑自洽的，屏幕前或书本前的你，也写给过去的自己。

I think that journey, from the frustration of not understanding to the joy of finally seeing some light, is worth remembering. So I've decided to share how I learned and built things from scratch: what I studied, what I thought about, what confused me, and the problems I encountered along the way. I hope to share as much of it as possible with you, whether you're reading this on a screen or in a book, if you too enjoy diving deep, understanding the underlying principles, and making sure things make logical sense. And, of course, I'm writing this for my past self as well.

## 项目规划 Roadmap

考虑再三，我决定把这个系列分成五个部分：

After giving it a lot of thought, I've decided to divide this series into five parts.

### 第一部分：基本原理与极简实现（V0）

这部分主要聊聊我个人对 LLM 基本原理和相关基础算法的理解，并根据这些最基本的原理，从绝对的“零”开始，不调用任何的 LLM 相关代码库，构建出一个可以训练，测试，生成文本的极简版本 Transformer V0。在这部分，你可以了解到，Transformer 如何工作，每一步前向计算的目的和方法，如何构建反向传播矩阵，如何验证梯度的正确性，以及如何生成文本。当然，我也会分享在这个过程中，遇到了哪些实现的问题，以及如何解决它们。

### Part I: Fundamentals and a Minimal Implementation (V0)

In this part, I'll share my understanding of the fundamental principles of LLMs and the algorithms behind them. Starting from the absolute basics, without relying on any LLM-related libraries, I'll build a minimal Transformer V0 that can be trained, tested, and used to generate text. We'll explore how a Transformer works, the purpose and mechanics of each step in the forward pass, how to derive the matrices needed for backpropagation, how to verify that the gradients are correct, and how to generate text. Of course, I'll also share the implementation problems I ran into and how I solved them.

### 第二部分：性能分析及分步优化（V1）

基于这个最简实现的 V0 版本，分析执行性能，找出可以优化的点，并实现这些优化。你可以看到性能分析和寻找瓶颈的方法，实现优化的方法，以及优化之后的性能对比。一些 V0 版本没有实现的算法，比如多层 Transformer 和 Multi-Head Attention 等内容也会在这部分有所体现。

### Part II: Performance Analysis and Incremental Optimization (V1)

Starting with the minimal V0 implementation, we'll analyze its performance, identify opportunities for optimization, and implement improvements step by step. You'll see how I profile the code and locate bottlenecks, how I approach each optimization, and how performance changes afterward. This part will also introduce features not implemented in V0, such as multi-layer Transformers and multi-head attention.

### 第三部分：抽象的力量（V2）

上述过程中你可能会发现，很多计算和推导都必须在实现之前完成。而一旦计算图或并行策略发生变化，很多地方需要重新推导和计算，工作量大且容易出错。这部分我们将一起学习构建简单的模型框架，有了这些抽象，框架和组件，模型设计上就可以对改动更加友好，也省去了需要处理大量演算纸的麻烦。

### Part III: The Power of Abstraction (V2)

By this point, you may have noticed that a great deal of calculation and derivation has to be done before writing the code. Whenever the computation graph or parallelization strategy changes, many of those calculations must be revisited. That's a lot of work, and it's easy to make mistakes. In this part, we'll explore how to build a simple model framework. With the right abstractions, framework structures, and reusable components, we can make model designs easier to change, and spare ourselves a mountain of scratch paper covered in derivations.

### 第四部分：走进 GPU 的世界（V3）

走进 GPU 的世界：受益于 LLM 时代，发展最迅猛的硬件产品，非 GPU 莫属。同时，GPU 出色地性能，也使得 LLM 如虎添翼。GPU 究竟如何工作，性能模型是什么，瓶颈在哪里，应该如何使用使其发挥出最大的性能，是 AI 系统工程师这个行业永恒的话题。我会尝试将上面提到的 V0/V1/V2 版本基于 GPU 做部分实现和优化，依然如我们之前的风格，尽量呈现出最底层的原理和实现方法，通过可运行可测试的代码了解更多细节，为后面使用 GPU 设计 LLM 系统打下坚实的基础。

### Part IV: Entering the World of GPUs (V3)

Few hardware products have developed as rapidly in the LLM era as GPUs. Their extraordinary computing power has, in turn, helped propel LLMs forward. How do GPUs actually work? What does their performance model look like? Where are the bottlenecks, and how can we make the most of the hardware? These are enduring questions for AI systems engineers. I'll try to implement and optimize selected parts of V0, V1, and V2 on GPUs. As in the earlier parts, I'll aim to explain the underlying principles and implementation details, using runnable, testable code to explore how things really work. The goal is to build a solid foundation for designing GPU-based LLM systems later on.

### 第五部分：业界代码导读

在了解原理和动手实现的过程当中，我们一定遇到了很多需要解决，但很难解决的问题。这时候我们就要从业界成熟的框架和库中寻找答案了。我会将一些问题整理出来，带着这些问题，我们一起从业界成熟代码中寻找如何解决这些问题。解决问题的过程也是我们学习和提高的过程。当认真思考的问题遇上成熟的解决方案，或许是深入细致的设计，或许是思考精妙的算法，或许只是一个 trade-off，都会让我们觉得，我们变得更加聪明了。

### Part V: Learning from Production-Grade Code

As we study the principles and build our own implementations, we're bound to encounter problems that need solving but are difficult to solve well. That's when it's time to look for answers in mature, real-world frameworks and libraries. I'll collect some of those questions, and together we'll examine how established projects address them. The process of finding answers is itself a chance to learn and improve. When a problem we've thought deeply about meets a mature solution, whether it's a carefully considered design, an ingenious algorithm, or simply a trade-off, we may come away feeling just a little smarter.

### 关于更新 About Updates

我深知，这是一个很大的挑战，也许需要用数年的时间才能完成。在这过程当中，我也需要不断学习，甚至有时候可能要“现炒现卖”。因此，内容中难免出现疏漏和错误，欢迎大家多多批评指正。而且，鄙人也是上班一族，要以工作养家糊口为先，所以更新可能会比较慢。我会尽量保证每周更新一篇，希望大家理解。最新更新的篇目和已有的篇目我会以目录的形式整理在本篇 README 开头处，方便大家查看。非常感谢大家的支持。

I know this is a substantial challenge, and it may take years to complete. Along the way, I'll have plenty more to learn myself, and sometimes I'll probably be learning something just before writing about it. Mistakes and omissions are therefore inevitable, and I sincerely welcome corrections and feedback. I also have a day job, and earning a living and supporting my family have to come first, so updates may be slow. I'll do my best to publish one article a week, and I hope you'll understand if I can't always keep that pace. I'll keep the latest update and the list of published articles near the top of this README so they're easy to find. Thank you very much for your support.

## 许可证 License

本项目的许可证信息见 [LICENSE](LICENSE)。

See [LICENSE](LICENSE) for licensing information.
