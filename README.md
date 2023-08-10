# DeepLearningRoadMap

[TOC]

## 前言—研究工作流：问题驱动、协作学习

### 1. 参考博士论文研究的方法定制工作流

深度学习有着数不清的研究方向，如何找到适合自己的的方向并做出创新，我们可以参考网上博士们分享的经验[^ref2][^ref3][^ref4][^ref5]，总结出这两点：

1. 研究选题的”点子“很难凭空想象，还是需要基于大量文献，总结出适合做研究的主题。
2. 读文献要广度和深度相结合，广而泛的读大量文献寻找主题，找到主题后再深入阅读这一主题下的重要论文。

具体到我们的学习过程，我们还会遇到几个问题，对这几个问题的解答构成了我们自己定制的工作流：

> 1. 需要学什么？需要拿一本教科书从头读到尾吗？

对于这个这个问题[^ref5]给出了一个很好的思路，我们可以从文献数据库导出大量围绕同一主题的相关文献，分析它们的引用关系，找出被引最多、最有影响力的文献作为种子文献，针对种子文献进行完全的学习，制作自己的讲义，这样可以很自然的按照学术发展路线进行学习，逻辑更加合理。而教科书基本是模型的平铺展开，通篇阅读会花很多时间在主题无关的课题上，可以作为参考书，在遇到问题时翻到指定章节查阅即可。

> 2. **如何串联起学习的过程，让从简单到难的过程平滑过渡？**

我们可以用问题/目标的派生关系形式将学习内容串联起来，例如以下问题集合就表示了一个主题循序渐进的学习过程：

> #1 找到“深度学习用于时间序列预测”这一主题的种子文献
>
> #1.1 基础准备：文献数据库准备，分析工具安装下载
>
> #1.2 搜索主题并分析绘制引用关联图
>
> #1.3 从引用关联图找到种子文献
>
> #2 种子文献详细解读
>
> #2.1 为了解决什么问题
>
> #2.2 使用到的模型
>
> #2.3 模型的运用方法
>
> #2.4 结论的有效性
>
> #2.5 文章的限制条件
>
> #3 基础增强：深入学习文章中提到的几类模型
>
> #3.1 模型的数学推导过程：矩阵微积分如何求偏导？
>
> ...

用issue关联的好处是，基于上述issue的派生关系，我们可以确保当前学习的内容和整体目标方向是一致的，高效的将精力集中于最终目标上。我们可以把学习过程中遇到的任何思路和问题提为issue，难以一次性解决的issue拆分为更小的、更简单的issue，逐个解决，当解决完所有相关issue后，我们可以将issue按照思维导图的形式整理在一起，这就是关于当前目标的一个全面的知识网络。

> 3. **为什么要协作学习，如何协作？**

这里的协作学习是这样一种学习模式：在同一个项目中，多人提交issue，我们可以**随机分配**issue的研究人，同时在一个人解决issue后，需要让所有参与人审阅，全部审阅通过后才可以正式确认issue已经完全解决。

随机分配issue、共同审阅解决方案是协作学习的核心。这个方法是为了解决一个心理倾向：我们倾向于省力，解决问题的成本很高，往往会简化目标或者简化回答；提问的成本很低，可以简单的对他人的回答提出各个角度的疑问。为了得到完善的解决方案，我们自己提交的方案能做到经受提问的考验；为了扩展综合思维的能力，我们可以尝试多角度观察提问其他人的方案。解决问题和提问都能加深对项目的理解，随机分配issue和共同审阅机制可以实现让每个协作人以同等的机会参与提问和解决问题。


综合以上内容，我们得到了下图1所示的工作流，协作部分见第2节的图2：[【金山文档】 研究工作流](https://kdocs.cn/l/cabEAZPhwsyu)


![研究工作流](https://img.tucang.cc/api/image/show/13b7a9247ee334922f787598dce9de0a)

<center style="color:#9e9d9d;text-decoration:underline">图1 研究工作流程图</center>

### 2. 使用Github的功能实现工作流

#### 2.1 GitHub工作流程图

根据图1，我们可以得到更加具体的GitHub工作流程图，见下图2:：[【金山文档】 研究工作流](https://kdocs.cn/l/cabEAZPhwsyu)

![研究工作流](https://img.tucang.cc/api/image/show/0151754979d2d362c9dcdae544ce3097)

<center style="color:#9e9d9d;text-decoration:underline">图2 GitHub工作流</center>

图3是两个流程的横向对比：

![研究工作流](https://img.tucang.cc/api/image/show/98860050478a64d85ebdce695f6ca495)

<center style="color:#9e9d9d;text-decoration:underline">图3 流程比照</center>

下面我们更具体的学习GitHub的使用方式，来实现需要的提问、提交解决方案、共同审阅的动作。

#### 2.2 GitHub是什么

这个github的官方视频[^ref7]展示了GitHub的核心理念：协同工作。更加具体的介绍可以参考这个通识课[^ref8]的第6讲-开源协作，视频解释了git和github的区别，以及为什么开源模式是能有效解放生产力的，这套通识课讲的很全面，感兴趣话可以学习完整的课程。

#### 2.3 提问/提出目标：使用Github issue

具体如何使用GitHub的issue功能可以参考这个视频[^ref9]， 这是一个全英文教学视频，可以打开YouTube自动字幕作为听力辅助，未来遇到的研究资料相当一部分会是英文的，我们通过逐渐接触英文资料为今后的研究打好基础。中文版视频也有[^ref10]，讲的比较简略。

issue研究人的随机分配，我们会通过引入一个GitHub action来实现，在创建issue时系统会自动分配，无需人工干预。如果对issue感兴趣或者有不想参与的issue，也可以手动调整研究人(Assignee)。

#### 2.4 提交解决方案：git工作流及其可视化

git在[^ref8]中已经有介绍过它的内容，git的教程非常多，比较贴合本项目的一个教程可以参考[^ref11]，这个教程也提到了GitHub上issue和pull request的基本操作。在本项目中我们不会用到fork、fetch、rebase等略复杂的操作，主要使用以下几个命令即可实现日常需求：

1. clone - 从现有Git仓库拷贝项目代码及版本历史到本地,实现项目的本地化。

2. checkout - 切换分支或恢复工作区文件到某个提交版本。
3. pull - 从远程仓库获取更新并合并到本地分支。
4. commit - 提交对工作区的修改,将改动记录到本地仓库。
5. revert - 撤销指定的提交,将项目恢复到之前的状态。
6. push - 将本地仓库的更新推送到远程仓库,与远程同步。
7. merge - 合并分支,将一个分支上的修改合并到当前分支。

相应的工作流如下：

![git工作流](https://img.tucang.cc/api/image/show/c7a728919c900846e817c209415f466e)

<center style="color:#9e9d9d;text-decoration:underline">图4 git工作流</center>

实际在使用时，我们不需要用到一些教程提到的命令行模式，而是使用界面上的菜单进行操作。如何配置pycharm和GitHub之间的连接可以参考[^ref13]，连接成功后git命令的使用可以参考[^ref12]。

#### 2.5 共同评审：pull request

pull request是GitHub协同工作流的关键一步，在这里我们可以汇聚所有协同者的工作，推进主分支不断完善。具体操作可参考这个教程[^ref14]。为了实现上面的GitHub工作流，我们还需要做几个约定：

 1. issue与pull request做关联。可以在issue界面的右下角development设置中搜索pull request名称，实现关联，关联后方便我们后续迅速的搜索解决方案。

    ![issue与pull request做关联](https://img.tucang.cc/api/image/show/49f4ea39d3a63400eb65bf614956ad5a	)

 2. Pull reqeust统一从develop，或自定义分支向main分支合并。

 3. pull request的通过需要所有项目参与者的审阅，提交人自己也需参与，目的是增加检查的准确性。

#### 2.6 主分支管理

issue的解决方案全部关联主分支对应的文档或代码，确保所有close的issue的解决方案都是经过审阅的，然后我们可以基于主分支的代码进行知识整理、复习。

### 3. 文档写作流程

#### 3.1 markdown基本用法

markdown是网站博客演化而来的一种纯文本标记语言，目的在于简化排版，让使用者专注写作。它比word功能少很多，例如不能任意调整字体、大小和颜色、没有页的概念、没有批注等，但优点也很明显，一是打开方便，可以在任意平台如电脑、手机等使用浏览器或者编辑器阅读；二是可以使用git进行版本管理，任何对文档的修改都可以记录下来，方便多人协作编辑和查漏补缺。

markdown的教程非常多，可以参考视频[^ref6]作为入门，编辑器推荐视频中使用的**Typora**。在Typora的帮助菜单下有一个Markdown Reference功能，打开是一个markdown文件，包含了Typora支持的各种语法，可以作为词典查阅。

#### 3.2 图片和附件

在markdown中图片是一个链接，链接既可以是本地地址也可以是网络地址。如果将图片直接存储在项目路径下，会导致仓库体积快速膨胀，影响git性能，最好选择网络图床[^注1]。在密钥管理中我们会实时更新图床状态。

附件我们也使用外部下载链接管理。附件包括文档的pdf、视频，以及代码用到的数据集。

#### 3.3 文献引用

文献引用主要使用markdown的脚注功能。为了查找方便，本项目的文献不只是论文、书籍，一些网页资料和视频资料也会放在其中，如果这类非权威资料逻辑是正确、完整、自洽的话，也可以作为我们参考的依据。在最后一部分参考文献里，我们会把文献的来源和网盘下载地址附上，方便立刻查看。此外，由于学习的顺序可能是循环往复的，后读到文献编号虽然线性增加，但在文中被引用的位置可能更靠前，办法是定期做一次编号重排，让内容相似的文献编号靠近，方便我们做关联查询。相关功能开发见Issue #1.

#### 3.4 思维导图和流程图

在读文献综述时思维导图很有用，在写代码时流程图很有用。比较有名的本地软件有XMind和Visio，对于轻度使用，一般的在线文档编辑软件如wps文档、腾讯文档等都有这两个功能，可以根据自己的喜好选择工具。

#### 3.5 标题锚点定位

markdown可以自动将标题识别为锚点，可以生成一个指向该标题的网络链接，打开链接时不仅会打开markdown文件，还会直接跳至指定标题的位置。这样我们可以在issue的评论中附带解答的锚点链接，方便查询。

### 4. 代码写作流程

#### 4.1 代码组织形式

代码在不同阶段的组织形式非常不同。在初期研究阶段，我们需要的是jupyter notebook，在这里我们可以使用markdown和代码混合编写，记录每一步操作的理论基础和运行结果，这样的文档方便初学。在中期对代码熟悉以后，我们往往希望直接调试代码，去掉大量markdown解释甚至部分代码注释，让代码看起来更清晰，此外也会适当调整代码的组织形式，例如将不同功能的代码块包装到不同的类里方便管理，这时我们需要的是一个从notebook“脱水”的单个py文件。在后期对写代码非常熟悉后，我们需要的是将模型抽象为多个模块，每个模块按不同文件夹、不同py文件的形式组织在一起，需要的是对模块划分的全局介绍。下图表示了代码分阶段的组织形式：[【金山文档】 代码组织形式](https://kdocs.cn/l/cbqN6aq8Bm01)


![代码组织形式](https://img.tucang.cc/api/image/show/255c88eaa69a584ffbd6fd30f6276ab7)

初期的Jupyter Notebook在进行编辑和git管理时，我们按照这样的方式管理：

1. 每一个notebook同时保存一份md文件和ipynb文件(notebook的原生格式)。

2. md文件包含所有的Markdown解释和代码，不包括代码运行效果；ipynb只包含代码，不包含markdown解释和代码运行效果。

3. 确保在每次更新代码时，同步更新md文件和ipynb文件。

这样做主要是为了将大量解释从代码中剥离，我们在练习测试代码时可以专注使用ipynb做调试，另外代码运行效果，即ipynb中的output，可以通过运行input代码生成，可以不用管理。

#### 4.2 运行环境配置

**Python：**Python 3.10或更高版本，官网安装

**IDE：**推荐Pycharm专业版（淘宝购买）或VSCode

**IDE配置Python环境：**可以参考[^ref16]的第二章-调试与运行

**Python第三方依赖：**下面的依赖我们可以使用pip安装，建议都使用最新版本。

1. pandas：可以理解为python版excel，数据批量处理、格式转换的核心工具

2. numpy：常用数据计算库，大部分时候不需要，因其大部分功能已经被集成在pandas和pycharm中

3. pytorch：目前使用最广泛的深度学习库

4. matplotlib：常用的绘图库

5. Jupiter Notebook：读写项目中的notebook

**GPU支持：**学习阶段暂时用不到GPU加速，后期遇到相关Issue我们再补充环境搭建

#### 4.3 IDE使用与Debug方法

现代IDE有非常多的功能，包括自动补全、debug、代码重构、版本管理等等，需要在实践中学习才能取得最好的效果。这个资料[^ref16]详细的讲解了几乎所有Pycharm的功能，需要时可以当作词典查阅。

### 5. 强大的AI技术辅助

#### 5.1 AI功能与可用AI汇总

AI在帮助我们搜索、翻译、解释概念、分析文件等许多方面已经取得了惊人成绩，灵活使用能极大的帮助我们提升学习效率。需要注意的是AI的解答很有可能是错误的，特别是数学类的问题，使用时需要仔细甄别。目前比较活跃的AI模型有以下几个(在密钥管理中有地址和使用方式的介绍，及时更新)，为了降低错误概率，我们可以一个问题同时向多个AI提问，根据不同的回答综合得出合理答案。一个综合的使用案例可以参考视频[^ref15]，已经可以做到辅助写代码和论文正文了。

| 网站           | 介绍                                  |
| -------------- | ------------------------------------- |
| Claude         | ChatGPT变种，更擅长对话，可以分析文件 |
| Chat-GPT4      | 擅长生成长篇文字                      |
| New Bing       | 会给出回答引用的原文，可靠性高        |
| POE            | AI对话网站聚合                        |
| Github Copilot | 专注代码生成                          |

#### 5.2 使用AI分析论文示例

Claude目前已经支持上传pdf并按照自定义指令总结文章，效果相当出色，我们选择一篇论文[^ref19]进行分析的结果如下图所示：

![image-20230809010629964](https://img.tucang.cc/api/image/show/fa95538bc530f817b50b2779fa780bed)

作为对照，这篇论文的摘要是：

> We propose a deep learning method for event-driven stock market prediction. First, events are extracted from news text, and represented as dense vectors, trained using a novel neural tensor network. Second, a deep convolutional neural network is used to model both short-term and long-term influences of events on stock price movements. Experimental results show that our model can achieve nearly 6% improvements on S&P 500 index prediction and individual stock prediction, respectively, compared to state-of-the-art baseline methods. In addition, market simulation results show that our system is more capable of making profits than previously reported systems trained on S&P 500 stock historical data.

就这篇文章而言，可以认为AI的总结与作者的摘要几乎没有分别，信息一致、语言流畅，甚至事件嵌入更优这一结论是原文摘要没有提到的一个优点。我们也可以据此引入一个AI分析论文的正确性校验机制，在分析全文之前，**先让AI写摘要，人工与作者的摘要比对，核对一致后再让AI分析更具体的内容。**

### 章节注释

[^注1]: 图床（Image Hosting Service）是指用于存储和托管图片的在线服务。它允许用户将自己的图片上传到互联网上的服务器，并生成图片的访问链接，以便在需要的时候在网页、论坛、博客等地方引用这些图片。



## Part 1—搜索种子文献

### 1. 搜索种子文献

我们使用[^ref5]的方式尝试搜索种子文献。Web Of Science的登录信息在密钥管理中，可用的HistCite软件在[这里](https://pan.baidu.com/s/1VzwZMMFq2-o8urWK8juVqA?pwd=ab12)下载。

#### 1.1 确定搜索关键词

我们希望将深度学习技术运用在时间序列，特别是股价的预测上，关键词可以向这方面考靠拢，我们整理出了以下排列组合：

| 编号 | 主题                                |
| ---- | ----------------------------------- |
| 1    | deep learning price                 |
| 2    | deep learning stock price           |
| 3    | deep learning future price          |
| 4    | deep learning price prediction      |
| 5    | deep learning financial             |
| 6    | deep learning financial market      |
| 7    | deep learning time series           |
| 8    | deep learning finantial time series |

#### 1.2 搜索与分析的操作流程

我们以deep learning price为例记录操作流程。首先登录Web Of Science，在搜索前选择数据库Web Of Science核心合集。

![修改数据库](https://img.tucang.cc/api/image/show/933f7ba0212a5629ac655bc22377306f)

其他搜索条件默认，选择搜索文献，得到的查询结果如下：![WoS查询结果](https://p0.meituan.net/csc/92259bb59225eb5a0cdf83a90a752a3b421301.png)

有2748条查询结果，相当多的数量。现在我们使用导出功能将文献本身它的参考文献导出，用来做关联分析。点击导出-纯文本文件，记录内容选择全记录与引用的参考文献，注意到提示一次不能超过500条记录，我们选择用记录n到m的方式分段导出，2748条查询结果需要导出6次，分别命名。

![image-20230808110808343](https://img.tucang.cc/api/image/show/ddbfb53d0629be010385f2ea5019f781)

6个文件全部导出后，我们把所有文件放到HistCite Pro2.1的TXT目录中，效果如下：

![image-20230808111954537](https://img.tucang.cc/api/image/show/779a3f457c2e9bf6d0a5ca88f6097042)然后回到上一级目录，双击main.exe，在弹出的命令行中输入1，按回车，会看到文件解析的日志。如果看到如下的日志信息，说明文件格式不对，需要手工修复，我们可以使用带有正则表达式功能的编辑器进行批量替换，例如Pycharm，或者Sublime Text，NotePad++等。开启正则表达式替换，将`PD \d{4}`全部替换为`PD`。

![image-20230808112412069](https://img.tucang.cc/api/image/show/2f0b86b625096355d4d82d6aaa11510e)

替换后再次运行main.exe，应该不会有任何报错了，如下图所示：

![image-20230808113720131](https://img.tucang.cc/api/image/show/ceb381d58d0f04e58da3cc0fb37d46a5)

#### 1.3 读懂HistCite分析结果

分析完成后HistCite会弹出一个网页，效果如下图所示：

![image-20230811000413028](https://img.tucang.cc/api/image/show/46402e389f1b3340798c28077efc76f1)

HistCite主要提供了3个角度的分析结果，见图中标号。分别解释如下：

##### 1.3.1 LCS被引次数

图中标号1处的四个字段的直接解释如下[^18]：

| 列名 | 全称                       | 直接解释                                 |
| ---- | -------------------------- | ---------------------------------------- |
| LCS  | Local Citatioin Score      | 在当前文献集合里被引用的次数             |
| GCS  | Global Citation Score      | 文章被引总次数，WoS提供数据              |
| LCR  | Local Cited References     | 本文引用的文献属于在当前文献集合中的数量 |
| CR   | Number Of Cited References | 本文的所有文献引用数量                   |

更加详细的解释参考[^ref17]，其中提到：

> 1. GCS是global citation score，即引用次数，也就是你在web of science网站上看到的引用次数。
> 2. CR是cited references，即文章引用的参考文献数量。如果某篇文献引用了50篇参考文献，则CR为50。这个数据通常能帮我们初步判断一下某篇文献是一般论文还是综述。
> 3. LCS和LCR是histcite里比较重要的两个参数。LCS是local citation score的简写，即本地引用次数。与gcs相对应，gcs是总被引次数。lcs是某篇文章在当前数据库中被应用的次数。所以LCS一定是小于或等于GCS的。一篇文章GCS很高，说明被全球科学家关注较多。但是如果一篇GCS很高，而LCS很小，说明这种关注主要来自与你不是同一领域的科学家。此时，这篇文献对你的参考意义可能不大。举个例子，2003年发表在nature上的两篇文章P1 （GCS:580,LCS:12) 和 P2(GCS:36,LCS：24)。第一篇文章gcs很高，lcs很低，说明关注这篇文章的绝大部分作者与你关注的方向不同。而第二篇文章gcs较低，但LCS比第一批要高，即很多引用p2的文章都在当前数据库，也即与你的研究方向相关。所以，p1 p2相比，p2应该更贴近你的研究方向，参考价值更大。
> 4. LCR与CR对应是local cited references，是指某篇文献引用的所有文献中，有多少篇文献在当前数据库中。如果最近有两篇文章，p1 p2,都引用了30篇参考文献，其中p1引用的30篇文献中有20篇在当前数据库，p2只有2篇文献在当前数据库。此时，p1相对更有参考价值，因为它引用了大量和你的研究相关的文献。根据LCS可以快速定位一个领域的经典文献， LCR可以快速找出最新的文献中哪些是和自己研究方向最相关的文章。

因此，LCS被引次数是一个重要指标，我们可以点击LCS，HistCite会自动根据LCS从大到小的顺序排列，方便我们找到高引文献。

##### 1.3.2 Cited References共引次数

我们研究的主题：使用深度学习方法预测时间序列是一个衍生/交叉学科，其基础方法基本来自深度学习/金融时间序列两个相对更加基础的学科。Cited References的统计方法是，将当前文献集合的所有参考文献放到一起，按照文献名计数并排序，称之为共引文献。这里的高引文献一般都是基础学科的理论奠基文章，是研究当前交叉学科所必须的基础知识。不过，基础知识的学习不一定要直接读奠基论文，各类视频、教科书已经从各个角度以更清晰的方式进行讲解，我们可以按照这些论文查找更合适的学习资料。

##### 1.3.3 引用关系图

此外，HistCite的另外一个重要功能是绘制引用关系图，在引用关系图中，圆圈越大、被指向的箭头越多，说明这篇文章在当前领域的重要性越高，以它作为灵感源头激活了领域内的大量更深入的研究，我们称之为种子文献，适合作为重点学习。以deep learning financial time series关键词查得得论文数据集为例，按照LCS降序、Limit 100的条件绘制关系图，得到如下结果：

![deep learning financial time series LCS前100节点关系图](https://img.tucang.cc/api/image/show/28bd19435ea3a671b76d6faa81167dbb)

可以在单独标签页中打开大图查看。这张图的主要节点相对较多，较为明显的是其中的第17号和第87号，分别如下：

> 17号：Cavalcante R C, Brasileiro R C, Souza V L F, et al. Computational intelligence and financial markets: A survey and future directions[J]. Expert Systems with Applications, 2016, 55: 194-211.
>
> 87号：Fischer T, Krauss C. Deep learning with long short-term memory networks for financial market predictions[J]. European journal of operational research, 2018, 270(2): 654-669.

由于节点圈面积大、被指向多，可以认为这两篇文章是改领域的种子文献。从标题可以看出，第17号是一篇综述，第87号是经典的使用RNN网络预测股价的模型，综述类论文主要被引用来概述之前的研究，模型类论文主要是引用其研究方法，都对我们自己写论文和做模型有很大参考价值。在图上还有其他较为明显的节点，我们可以在深入学习种子文献后，通过学习其他主要节点的摘要和结论获取新的灵感。

#### 1.4 其他文献搜索方式

上述WoS搜索总体看还有几个问题：1. 对于快速发展的学科，其论文数据更新不够及时，例如图上的高被引论文主要在2019年或更早，而深度学习算法近年来发展非常迅速，重要的新论文可能没有被收录；2. 其核心数据库不包含arXiv文章以及其他公司网站等形式发表的、非正式期刊登载的论文，虽然这些平台的论文的准确性确实没有官方背书，但已经有大量有效、高引用、影响力大的论文发表在上面，我们也需要参考这些平台的知识。3. 金融行业由于存在套利机制，预测效果最好的模型不会提前发表，一般会现在业界实践较长时间后才会逐步变成公开模型。我们可以关注对冲基金的最新动态，以及尝试使用不同于学术主流的新模型来做出创新。

*文献搜索方式待补充*

### 2 WebOfScience和HistCite的种子文献搜索结果分析

为了找到深度学习预测股价等金融时间序列的论文，我们使用了8个近似的主题进行搜索，如下所示：

| 检索主题                            | 总论文数量 | 采用数(相关性从高到低) | 引用关系图(前100节点)                                        |
| ----------------------------------- | ---------- | ---------------------- | ------------------------------------------------------------ |
| deep learning financial             | 10137      | 3000                   | ![img](http://www.kdocs.cn/api/v3/office/copy/OWNkZ2VSSDNiRzR3V0cyTVgrR2c5TStxbzRzclpIeU0zNGxiM3FGVG5BV0VpSTJMakZRN2RtMEYwQXBPYzdsNkZPNERId2UvK2Q5dGQ1eVYwamN4NEFVWWU0ekZwdzVqZVZsWHZGMGdvSjBwbTdCNHZIZ05aMHVacEh3NXhBSkxPaGJ1YlU5V1d0anBpY0lGaDhNL1JkUE1Jd0tiTzhLYjVBbExXMnpkUTg5TC84THArL1RZTC9DRTNTbzd4US9ETUlUN0lJZnVEekc0YnFGdlZiUXRCSmxQMFM3Q3lXY1FTdGozUlNhR1lkSU14V2hjWFcwQkdLUjNrVzVEbGI2M0owUmVsaXVDY3pnPQ==/attach/object/D4Y6IBAABQ?) |
| deep learning financial market      | 1026       | 1026                   | ![img](http://www.kdocs.cn/api/v3/office/copy/OWNkZ2VSSDNiRzR3V0cyTVgrR2c5TStxbzRzclpIeU0zNGxiM3FGVG5BV0VpSTJMakZRN2RtMEYwQXBPYzdsNkZPNERId2UvK2Q5dGQ1eVYwamN4NEFVWWU0ekZwdzVqZVZsWHZGMGdvSjBwbTdCNHZIZ05aMHVacEh3NXhBSkxPaGJ1YlU5V1d0anBpY0lGaDhNL1JkUE1Jd0tiTzhLYjVBbExXMnpkUTg5TC84THArL1RZTC9DRTNTbzd4US9ETUlUN0lJZnVEekc0YnFGdlZiUXRCSmxQMFM3Q3lXY1FTdGozUlNhR1lkSU14V2hjWFcwQkdLUjNrVzVEbGI2M0owUmVsaXVDY3pnPQ==/attach/object/SIZOIBAAK4?) |
| deep learning financial time series | 896        | 896                    | ![img](http://www.kdocs.cn/api/v3/office/copy/OWNkZ2VSSDNiRzR3V0cyTVgrR2c5TStxbzRzclpIeU0zNGxiM3FGVG5BV0VpSTJMakZRN2RtMEYwQXBPYzdsNkZPNERId2UvK2Q5dGQ1eVYwamN4NEFVWWU0ekZwdzVqZVZsWHZGMGdvSjBwbTdCNHZIZ05aMHVacEh3NXhBSkxPaGJ1YlU5V1d0anBpY0lGaDhNL1JkUE1Jd0tiTzhLYjVBbExXMnpkUTg5TC84THArL1RZTC9DRTNTbzd4US9ETUlUN0lJZnVEekc0YnFGdlZiUXRCSmxQMFM3Q3lXY1FTdGozUlNhR1lkSU14V2hjWFcwQkdLUjNrVzVEbGI2M0owUmVsaXVDY3pnPQ==/attach/object/JQ2OIBAAWA?) |
| deep learning future price          | 534        | 534                    | ![img](http://www.kdocs.cn/api/v3/office/copy/OWNkZ2VSSDNiRzR3V0cyTVgrR2c5TStxbzRzclpIeU0zNGxiM3FGVG5BV0VpSTJMakZRN2RtMEYwQXBPYzdsNkZPNERId2UvK2Q5dGQ1eVYwamN4NEFVWWU0ekZwdzVqZVZsWHZGMGdvSjBwbTdCNHZIZ05aMHVacEh3NXhBSkxPaGJ1YlU5V1d0anBpY0lGaDhNL1JkUE1Jd0tiTzhLYjVBbExXMnpkUTg5TC84THArL1RZTC9DRTNTbzd4US9ETUlUN0lJZnVEekc0YnFGdlZiUXRCSmxQMFM3Q3lXY1FTdGozUlNhR1lkSU14V2hjWFcwQkdLUjNrVzVEbGI2M0owUmVsaXVDY3pnPQ==/attach/object/Q33OIBAAIM?) |
| deep learning price                 | 2748       | 2748                   | ![img](http://www.kdocs.cn/api/v3/office/copy/OWNkZ2VSSDNiRzR3V0cyTVgrR2c5TStxbzRzclpIeU0zNGxiM3FGVG5BV0VpSTJMakZRN2RtMEYwQXBPYzdsNkZPNERId2UvK2Q5dGQ1eVYwamN4NEFVWWU0ekZwdzVqZVZsWHZGMGdvSjBwbTdCNHZIZ05aMHVacEh3NXhBSkxPaGJ1YlU5V1d0anBpY0lGaDhNL1JkUE1Jd0tiTzhLYjVBbExXMnpkUTg5TC84THArL1RZTC9DRTNTbzd4US9ETUlUN0lJZnVEekc0YnFGdlZiUXRCSmxQMFM3Q3lXY1FTdGozUlNhR1lkSU14V2hjWFcwQkdLUjNrVzVEbGI2M0owUmVsaXVDY3pnPQ==/attach/object/MUTOIBAA2U?) |
| deep learning price prediction      | 1207       | 1207                   | ![img](http://www.kdocs.cn/api/v3/office/copy/OWNkZ2VSSDNiRzR3V0cyTVgrR2c5TStxbzRzclpIeU0zNGxiM3FGVG5BV0VpSTJMakZRN2RtMEYwQXBPYzdsNkZPNERId2UvK2Q5dGQ1eVYwamN4NEFVWWU0ekZwdzVqZVZsWHZGMGdvSjBwbTdCNHZIZ05aMHVacEh3NXhBSkxPaGJ1YlU5V1d0anBpY0lGaDhNL1JkUE1Jd0tiTzhLYjVBbExXMnpkUTg5TC84THArL1RZTC9DRTNTbzd4US9ETUlUN0lJZnVEekc0YnFGdlZiUXRCSmxQMFM3Q3lXY1FTdGozUlNhR1lkSU14V2hjWFcwQkdLUjNrVzVEbGI2M0owUmVsaXVDY3pnPQ==/attach/object/H356IBAAIA?) |
| deep learning stock price           | 733        | 733                    | ![img](http://www.kdocs.cn/api/v3/office/copy/OWNkZ2VSSDNiRzR3V0cyTVgrR2c5TStxbzRzclpIeU0zNGxiM3FGVG5BV0VpSTJMakZRN2RtMEYwQXBPYzdsNkZPNERId2UvK2Q5dGQ1eVYwamN4NEFVWWU0ekZwdzVqZVZsWHZGMGdvSjBwbTdCNHZIZ05aMHVacEh3NXhBSkxPaGJ1YlU5V1d0anBpY0lGaDhNL1JkUE1Jd0tiTzhLYjVBbExXMnpkUTg5TC84THArL1RZTC9DRTNTbzd4US9ETUlUN0lJZnVEekc0YnFGdlZiUXRCSmxQMFM3Q3lXY1FTdGozUlNhR1lkSU14V2hjWFcwQkdLUjNrVzVEbGI2M0owUmVsaXVDY3pnPQ==/attach/object/XM3OIBAAFE?) |
| deep learning time series           | 10794      | 3000                   | ![img](http://www.kdocs.cn/api/v3/office/copy/OWNkZ2VSSDNiRzR3V0cyTVgrR2c5TStxbzRzclpIeU0zNGxiM3FGVG5BV0VpSTJMakZRN2RtMEYwQXBPYzdsNkZPNERId2UvK2Q5dGQ1eVYwamN4NEFVWWU0ekZwdzVqZVZsWHZGMGdvSjBwbTdCNHZIZ05aMHVacEh3NXhBSkxPaGJ1YlU5V1d0anBpY0lGaDhNL1JkUE1Jd0tiTzhLYjVBbExXMnpkUTg5TC84THArL1RZTC9DRTNTbzd4US9ETUlUN0lJZnVEekc0YnFGdlZiUXRCSmxQMFM3Q3lXY1FTdGozUlNhR1lkSU14V2hjWFcwQkdLUjNrVzVEbGI2M0owUmVsaXVDY3pnPQ==/attach/object/2M36IBAA2U?) |

具体的文献搜索结果由于数据量太大，我们使用外链，见[【金山文档】 WebOfScience种子文献搜索结果](https://kdocs.cn/l/co5BvfJv5LxZ)。下面从1.3中的3个角度分析哪些是重点论文、重点论文的特征以及我们的学习方式。

#### 2.1 高被引部分

如上表所示，不同主题的搜索结果数量差别很大，为了精炼论文选取并保留各主题的差异，我们尝试使用**被引数>主题前10被引平均数**这一规则筛选文献，得到结果如下：

| 检索主题                            | LCS-被引顺序 | LCS-被引次数 | LCS-论文全名                                                 |
| ----------------------------------- | ------------ | ------------ | ------------------------------------------------------------ |
| deep learning financial             | 1            | 87           | Deep learning with long short-term memory networks for financial market predictions |
| deep learning financial             | 2            | 71           | Deep Direct Reinforcement Learning for Financial Signal Representation and Trading |
| deep learning financial             | 3            | 60           | Deep learning networks for stock market analysis and prediction: Methodology, data representations, and case studies |
| deep learning financial market      | 1            | 112          | Deep learning with long short-term memory networks for financial market predictions |
| deep learning financial market      | 2            | 73           | Deep learning networks for stock market analysis and prediction: Methodology, data representations, and case studies |
| deep learning financial market      | 3            | 63           | Deep Direct Reinforcement Learning for Financial Signal Representation and Trading |
| deep learning finantial time series | 1            | 79           | Deep learning with long short-term memory networks for financial market predictions |
| deep learning finantial time series | 2            | 40           | Algorithmic financial trading with deep convolutional neural networks: Time series to image conversion approach |
| deep learning finantial time series | 3            | 34           | Forecasting Stock Prices from the Limit Order Book using Convolutional Neural Networks |
| deep learning future price          | 1            | 18           | Computational Intelligence and Financial Markets: A Survey and Future Directions |
| deep learning future price          | 2            | 10           | Forecasting Stock Prices from the Limit Order Book using Convolutional Neural Networks |
| deep learning price                 | 1            | 84           | Deep Learning for Event-Driven Stock Prediction              |
| deep learning price                 | 2            | 71           | ModAugNet: A new forecasting framework for stock market index value with an overfitting prevention LSTM module and a prediction LSTM module |
| deep learning price                 | 3            | 68           | Forecasting spot electricity prices: Deep learning approaches and empirical comparison of traditional algorithms |
| deep learning price                 | 4            | 58           | Forecasting Stock Prices from the Limit Order Book using Convolutional Neural Networks |
| deep learning price prediction      | 1            | 71           | Deep Learning for Event-Driven Stock Prediction              |
| deep learning price prediction      | 2            | 59           | ModAugNet: A new forecasting framework for stock market index value with an overfitting prevention LSTM module and a prediction LSTM module |
| deep learning price prediction      | 3            | 45           | Deep learning-based feature engineering for stock price movement prediction |
| deep learning price prediction      | 4            | 42           | Stock prediction using deep learning                         |
| deep learning stock price           | 1            | 79           | Deep Learning for Event-Driven Stock Prediction              |
| deep learning stock price           | 2            | 51           | Deep learning-based feature engineering for stock price movement prediction |
| deep learning stock price           | 3            | 49           | ModAugNet: A new forecasting framework for stock market index value with an overfitting prevention LSTM module and a prediction LSTM module |
| deep learning time series           | 1            | 214          | Deep learning for time series classification: a review       |
| deep learning time series           | 2            | 183          | Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline |

在选定范围后，我们尝试根据文献题目和摘要进行进一步筛选。*Deep Learning for Event-Driven Stock Prediction*这篇文章是基于新闻事件驱动的股价预测，没有使用时间序列，可以排除，其他文章全部和我们研究的主题有关，在进一步去重合和排序后，得到的结果如下：

| 检索主题                                                     | LCS-被引顺序 | LCS-被引次数 | LCS-论文全名                                                 |
| ------------------------------------------------------------ | ------------ | ------------ | ------------------------------------------------------------ |
| deep learning time series                                    | 1            | 214          | Deep learning for time series classification: a review       |
| deep learning time series                                    | 2            | 183          | Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline |
| deep learning financial market,deep learning financial,deep learning financial time series | 1            | 112          | Deep learning with long short-term memory networks for financial market predictions |
| deep learning financial market,deep learning financial       | 2            | 73           | Deep learning networks for stock market analysis and prediction: Methodology, data representations, and case studies |
| deep learning price,deep learning price prediction,deep learning stock price | 2            | 71           | ModAugNet: A new forecasting framework for stock market index value with an overfitting prevention LSTM module and a prediction LSTM module |
| deep learning price                                          | 3            | 68           | Forecasting spot electricity prices: Deep learning approaches and empirical comparison of traditional algorithms |
| deep learning financial market,deep learning financial       | 3            | 63           | Deep Direct Reinforcement Learning for Financial Signal Representation and Trading |
| deep learning price,deep learning financial time series,deep learning future price | 4            | 58           | Forecasting Stock Prices from the Limit Order Book using Convolutional Neural Networks |
| deep learning stock price,deep learning price prediction     | 2            | 51           | Deep learning-based feature engineering for stock price movement prediction |
| deep learning price prediction                               | 4            | 42           | Stock prediction using deep learning                         |
| deep learning financial time series                          | 2            | 40           | Algorithmic financial trading with deep convolutional neural networks: Time series to image conversion approach |
| deep learning future price                                   | 1            | 18           | Computational Intelligence and Financial Markets: A Survey and Future Directions |

这些论文可以认为是我们研究主题的核心论文，值得深入阅读，可以按照被引次数的顺序进行学习。除去综述文章，这里的关键模型只有两个：

- CNN
- LSTM RNN

以及一篇涉及强化学习的论文。作为预测的通用模型，我们可以将其作为基础学习，后续可以在此基础上引入更现代的模型，例如transformer做深入研究。

#### 2.2 高共引部分

我们使用2.1中的方式筛选论文，第一步的结果如下：

| 检索主题                            | 共同引用-被引顺序 | 共同引用-被引次数 | 共同引用-论文全名                                            |
| ----------------------------------- | ----------------- | ----------------- | ------------------------------------------------------------ |
| deep learning financial             | 1                 | 381               | ImageNet Classification with Deep Convolutional Neural Networks |
| deep learning financial             | 2                 | 297               | Adam: A method for stochastic optimization                   |
| deep learning financial             | 3                 | 287               | Long short-term memory                                       |
| deep learning financial             | 4                 | 242               | Deep learning (book)                                         |
| deep learning financial market      | 1                 | 171               | Long short-term memory                                       |
| deep learning financial market      | 2                 | 112               | Deep learning with long short-term memory networks for financial market predictions |
| deep learning financial market      | 3                 | 96                | Adam: A method for stochastic optimization                   |
| deep learning financial time series | 1                 | 218               | Long short-term memory                                       |
| deep learning financial time series | 2                 | 95                | Adam: A method for stochastic optimization                   |
| deep learning financial time series | 3                 | 85                | ImageNet Classification with Deep Convolutional Neural Networks |
| deep learning future price          | 1                 | 96                | Long short-term memory                                       |
| deep learning future price          | 2                 | 37                | Deep learning with long short-term memory networks for financial market predictions |
| deep learning price prediction      | 1                 | 287               | Long short-term memory                                       |
| deep learning price prediction      | 2                 | 123               | Deep learning with long short-term memory networks for financial market predictions |
| deep learning stock price           | 1                 | 193               | Long short-term memory                                       |
| deep learning stock price           | 2                 | 109               | Deep learning with long short-term memory networks for financial market predictions |
| deep learning stock price           | 3                 | 91                | A deep learning framework for financial time series using stacked autoencoders and long-short term memory |
| deep learning time series           | 1                 | 695               | Long short-term memory                                       |
| deep learning time series           | 2                 | 374               | Adam: A method for stochastic optimization                   |
| deep learning time series           | 3                 | 346               | ImageNet Classification with Deep Convolutional Neural Networks |
| deep learning price                 | 1                 | 416               | Supervised Sequence Labelling (book)                         |
| deep learning price                 | 2                 | 253               | Adam: A method for stochastic optimization                   |
| deep learning price                 | 3                 | 203               | Human-level control through deep reinforcement learning      |
| deep learning price                 | 4                 | 190               | ImageNet Classification with Deep Convolutional Neural Networks |

进一步汇总排序的结果如下：

| 检索主题                                                     | 共同引用-被引顺序 | 共同引用-被引次数 | 共同引用-论文全名                                            |
| ------------------------------------------------------------ | ----------------- | ----------------- | ------------------------------------------------------------ |
| deep learning time series,deep learning financial,deep learning price prediction,deep learning financial time series,deep learning stock price,deep learning financial market,deep learning future price | 1                 | 695               | Long short-term memory                                       |
| deep learning price                                          | 1                 | 416               | Supervised Sequence Labelling with Recurrent Neural Networks (book) |
| deep learning financial,deep learning time series,deep learning price,deep learning financial time series | 1                 | 381               | ImageNet Classification with Deep Convolutional Neural Networks |
| deep learning time series,deep learning financial,deep learning price,deep learning financial market,deep learning financial time series | 2                 | 297               | Adam: A method for stochastic optimization                   |
| deep learning financial                                      | 4                 | 242               | Deep learning (book)                                         |
| deep learning price                                          | 3                 | 203               | Human-level control through deep reinforcement learning      |
| deep learning price prediction,deep learning financial market,deep learning stock price,deep learning future price | 2                 | 123               | Deep learning with long short-term memory networks for financial market predictions |
| deep learning stock price                                    | 3                 | 91                | A deep learning framework for financial time series using stacked autoencoders and long-short term memory |

从被引量足以看出这些论文做为理论基础的重要性，模型还是CNN和RNN。这些文章我们可以按顺序学习，因为都是经典论文，我们容易从教科书上找到更简单清晰的解释，但要注意的是，学科仍处于快速发展阶段，同样的模型有多种理解方式，再加上没有标准教材，作者水平良莠不齐详略不一，要多加对比挑选更适合个人的教材。

#### 2.3 种子文献部分

由于种子节点由目测得到，结果相对较少，经过筛选排序后的结果如下所示：

| 检索主题                                                     | 引用关系图-被引次数 | 引用关系图-论文全名                                          |
| ------------------------------------------------------------ | ------------------- | ------------------------------------------------------------ |
| deep learning time series                                    | 79                  | A review of unsupervised feature learning and deep learning for time-series modeling |
| deep learning financial time series                          | 79                  | Deep learning with long short-term memory networks for financial market predictions |
| deep learning time series                                    | 72                  | InceptionTime: Finding AlexNet for time series classification |
| deep learning financial,deep learning financial market       | 71                  | Deep Direct Reinforcement Learning for Financial Signal Representation and Trading |
| deep learning price prediction                               | 59                  | ModAugNet: A new forecasting framework for stock market index value with an overfitting prevention LSTM module and a prediction LSTM module |
| deep learning financial,deep learning financial market,deep learning financial time series,deep learning future price,deep learning stock price | 36                  | Computational Intelligence and Financial Markets: A Survey and Future Directions |
| deep learning future price                                   | 10                  | Forecasting Stock Prices from the Limit Order Book using Convolutional Neural Networks |

可以看出大部分是高被引文献，*A review of unsupervised feature learning and deep learning for time-series modeling*和*InceptionTime: Finding AlexNet for time series classification*比较特殊，走出了自己独特的发展路线。这里的论文作为领域的重要线索，也值得我们深入学习。

#### 2.4 学习路径设计

我们可以按照共引文献-高被引文献-种子文献的顺序学习，这样应该能得到一条平滑上升的学习曲线，并在学习完毕之后具备深度学习预测时间序列这一主题所必备的大部分知识。每一篇论文我们尽量做到完全理解，重点是代码部分，理论部分如果涉及太多技术细节可以跳过，我们的目标是使用模型做出合理的预测，掌握模型的使用方式最为重要。在学习完几个模型之后，最好画出思维导图进行总结，包括模型之间的关系、模型的难点及其突破方式等等。上述文献的学习我们可以都归类为基础增强，在学习完成之后再开始进入选题和扩展文献搜索路径、深入研究阶段。

## Part 2—基础增强

### 1. 高共引文献学习

### 2. 高被引文献学习

### 3. 种子文献学习

### 4. 知识网络

## 参考文献

[^ref1]: 梁宏涛,刘硕,杜军威等.深度学习应用于时序预测研究综述[J].计算机科学与探索,2023,17(06):1285-1300. |**获取方式**|知网|doi:10.3778/j.issn.1673-9418.2211108|[网盘下载](https://pan.baidu.com/s/1Vqf1g3X7JbHnhqplPflkZQ?pwd=ab12)

[^ref2]: 博士应该采取什么策略读文献？ - 快乐不包邮的回答|**获取方式**|[知乎](https://www.zhihu.com/question/37781628/answer/82246218)|[网盘下载](https://pan.baidu.com/s/1CDCepTlBhk_i8aSmLAEcLg?pwd=ab12)

[^ref3]: 顶会论文写作建议（上）：宏观布局，避免“hard to follow” - 丁霄汉的文章|**获取方式**|[知乎](https://zhuanlan.zhihu.com/p/593195527)|~~[网盘下载]()~~

[^ref4]: 博士应该采取什么策略读文献？ - Nintendoyes的回答|**获取方式**|[知乎](https://www.zhihu.com/question/37781628/answer/1986296184)|~~[网盘下载]()~~

[^ref5]: 博士应该采取什么策略读文献？ - Bless Wu的回答|**获取方式**|[知乎](https://www.zhihu.com/question/37781628/answer/664793289)|~~[网盘下载]()~~

[^ref6]: 【[Markdown + Typora/VSCode 超全教程] 给大一新生安利的文本神器 !】 |**获取方式**|[B站](https://www.bilibili.com/video/BV1hG411p7fX)|~~[网盘下载]()~~

[^ref7]: 【什么是GitHub？】 |**获取方式**|[B站](https://www.bilibili.com/video/BV16W41137gP)|~~[网盘下载]()~~

[^ref8]: 合集·2023开源通识课|**获取方式**|[B站](https://space.bilibili.com/510793367/channel/collectiondetail?sid=1166727)|~~[网盘下载]()~~

[^ref9]: Git and GitHub Tutorials #5 - Understanding GitHub Issues|**获取方式**|[youtube](https://youtu.be/TKJ4RdhyB5Y)|[网盘下载](https://pan.baidu.com/s/1FNRvRYoUHR0RuD2P13csnQ?pwd=ab12)

[^ref10]: 【如何参与开源项目？0基础入门：怎么打开GitHub？什么是issue？什么是PR？】 |**获取方式**|[B站](https://www.bilibili.com/video/BV1EP411d7Np)|~~[网盘下载]()~~

[^ref11]:【Git | 从Github flow入门Git工作流程】|**获取方式**|[B站](https://www.bilibili.com/video/BV1pR4y1s79B)|~~[网盘下载]()~~

[^ref12]:【pycharm 多人合作使用 github/gitee 仓库】|**获取方式**|[B站](https://www.bilibili.com/video/BV1r34y1t7vN)|~~[网盘下载]()~~

[^ref13]:Pycharm 配置 Git 和 GitHub 及 clone 项目全流程 - POPO的文章 - 知乎|**获取方式**|[知乎](https://zhuanlan.zhihu.com/p/475107124)|~~[网盘下载]()~~

[^ref14]:合集·GitHub Pull Request Mini Series|**获取方式**|[B站](https://space.bilibili.com/472463946/channel/collectiondetail?sid=917876)|~~[网盘下载]()~~

[^ref15]:【如何用GPT全阶段辅助论文写作】 |**获取方式**|[B站](https://www.bilibili.com/video/BV13X4y187rJ)|~~[网盘下载]()~~

[^ref16]:PyCharm 中文指南|**获取方式**|[个人网站,免费但需微信扫码关注](https://pycharm.iswbm.com/index.html)|~~[网盘下载]()~~

[^ref17]:HistCite | 快速找到研究领域的关键文献 - imagine的文章|**获取方式**|[知乎](https://zhuanlan.zhihu.com/p/113685114)|~~[网盘下载]()~~|注：对CR的解释有误

[^ref18]: HistCite Help|**获取方式**|[网络](https://edisciplinas.usp.br/pluginfile.php/4428395/mod_page/content/118/Manual_histCIte_bom.pdf)|[网盘下载](https://pan.baidu.com/s/1X9-WGi8IL0n7Mk8kF5PD0A?pwd=ab12)

[^ref19]: Ding X, Zhang Y, Liu T, et al. Deep learning for event-driven stock prediction[C]//Twenty-fourth international joint conference on artificial intelligence. 2015.|**获取方式**|熊猫学术|[网盘下载](https://pan.baidu.com/s/1Y84f6eI7M_bXwr2aLX5yyg?pwd=ab12)