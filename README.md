# RL-Sekiro
D3QN 强化学习打只狼

## 理论基础
### Q-Learning
更新公式：

![image](https://user-images.githubusercontent.com/62683546/153753386-af7c79d8-16be-46f0-baa5-47b220734ba5.png)

特点：off-policy

### DQN
更新公式：

![image](https://user-images.githubusercontent.com/62683546/153753433-fbe00fa7-0a1d-431d-9698-aab8a302a5f5.png)

特点：off-policy

伪代码：

![image](https://user-images.githubusercontent.com/62683546/153753736-ce87f887-7206-4027-ab48-400f76101290.png)

#### 目标网络（Target Network）
首先，目标网络解决的是一个「回归问题」（与分类问题中网络产生一个分布不同），其输入是环境的状态s，输出是多个动作产生的不同值Q(s,a)。（实际过程中，我们需要通过索引来获取这个Q值。

其思路就是基于贝尔曼方程，并利用时间差分(TD)的方法，让target network和用于训练的网络(这里就简记为agent网络)的差值尽可能近似于收益值，该收益值指的是从当前状态经过决策之后到达下一个状态所获取的收益。需要注明的是，DQN中的target network的参数就是直接拷贝agent网络的参数，使用的是一样的网络结构。但是在实际训练中，只能通过固定target network的输出来训练agent，而固定该网络的输出的方法就是延迟更新target network的参数，使其在固定步骤内输出不变，这样能够有效化agent网络的参数更新过程。

#### 经验重放（Experience Replay Buffer）
如果agent每次更新参数的时候都要与环境互动，这就大大降低了模型参数更新的效率，所以经验重放机制被提出。该机制就类似于一个有固定空间大小的存储器，把agent与环境互动所产生的部分结果进行存储（每次只能选取一个动作，得到的收益值也是一个标量）。等到了训练阶段的时候，每一次训练过程都会从该存储器中「均匀采样」出一批 (batch) 数量的样本（总量远小于存储器的最大容量），用于agent网络模型参数的更新。

####  ε-greedy（策略的选择）
DQN中策略的选择（假设这里是确定性策略）就是选取能够使动作值达到最大的那个动作，用数学形式表示就是：

![image](https://user-images.githubusercontent.com/62683546/153754009-2b2a9c74-29ec-4f21-bd58-56cd04c44fde.png)

而ε-greedy方法是贪心算法的一个变体。具体实现的方法就是先让程序由均匀分布生成一个[0,1]区间内的随机数，如果该数值小于预设的ε，则选取能够最大化动作值的动作，否则随机选取动作。

由于这样的策略选择方式，使得DQN为Off-policy的强化学习方法
