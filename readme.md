练习monte carlo tree search


写一个ttt游戏,井字棋

对手有三种等级
1   随机对手
2   自弈
3   高端局(更强的ai,或者更强的智人选手)


实验关注这些目的,
不仅要解决问题,还要研究成本.
算法不仅要收敛,还要更多地关注收敛的时长,你花费了多少时间成本?


state 世界的状态
action 动作


动作是state的函数.

每个动作有不同的价值,一般选择最高价值的动作.

动作完成后,环境会被主体改变.
之后,世界还会发生变化(对手走棋)

之后 才轮到主体继续行动.

但直到游戏结束 ,主体都得不到奖励.也就是延迟奖励.



不同的环境,主体应该作出不同的决策.
练习次数增加,主体也要更新价值向量.


环境就是状态,就是世界,在本游戏中，就是棋盘.

p 探索概率,
lr 学习率

探索算法:
1   k阶贪心,每次在前k个最优解中随机取一个.
2   随机.以p的概率无视价值表,随机选择一个行为.


从树根走到树叶只需要n步,所以要快速地走一遍,走完,才能知道奖励是多少,只有叶子上有奖励.

因为世界中不止一个变化的主体.还有rival对手,甚至世界会自发地发生变化 ,我们的主角是难以预料这种客观变化的.


遍历可以找出所有可能发生的客观变化 ,但这就不是ai算法的事了.
算法要的是更快地收敛.要在有限的时间成本内比遍历成绩更好.

遍历 是说，我遍历 出所有的对局情况,就能指导出怎么走棋.
但遍历 需要的时间太长了,我们需要在有限次的模拟后就实现高分数 的机器 人.

这也是mc的思想,用更短的时间,获得一个不那么完美的解,但练习时长非常短,速度很快.

确实还是像学生的应试教育.

学生经过少量的模拟考试,然后参加测试.
我们没有时间让你去吃掉世界上所有的试题.

你需要用限定的时间把成绩提高到差强人意的程度.

我们没有时间让你磨出一把完美的匕首，你必须在限定的时间内差强人意。这不仅是蒙特卡洛的要求，也是老板对工人的要求啊。


我实在是晕得不行

这样,你随机地去探索,然后一直走到叶子,拿到奖励,再回来更新最一开始的价值.

阅读当前棋盘,就应该得出一个价值表,我觉得这一定是神经网络该干的活.

因为我们无法把所有棋盘全记住啊.

这样,所谓的收敛价值表,就变成了收敛这个 神经网络.

有了价值表,我们要选择一个行为来执行.
有限次的执行之后 ,我们会游戏结束 ,获得奖励.

每次执行后,世界也会发生不可预知的变化 .

确实,我们可以把这个 世界,也想象成一个主体在操作,但它一般没有所谓的价值了.或者说它是更高维度的主体,它代表全局价值.集体的利益.


开发人工智能玩狼人杀.


如果你的算法不收敛,就tm像猴子敲打字机.


应该开辟两块内存,存储两个主体，它们可以用同样的算法.然后对弈,而不是自己与自己对弈.

还可以一个主体学一会再让另一个主体学一会.

如果对手太强,则它不容易找到奖励,算法就一直无法收敛.
如果对手太弱,同理.所以要控制对手的强度.

最好 就是五五开的胜率,始终让它不卑不亢,就能稳定地进步了.


如何用先验知识来训练神经网络 ?
把先验棋谱也想象成一个神经网络,它不学习,不收敛.用它的输出作为标准答案来训练已有的神经网络.




# 本项目暂停。

算法是设计好了。tmd 游戏写着好费劲 ，先别急了吧。

算法：
https://www.bilibili.com/video/BV1uv4y1j7ZW/