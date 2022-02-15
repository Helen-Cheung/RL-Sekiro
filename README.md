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

### DQN的变体
* Double DQN
* Dueling DQN
#### Double DQN
DQN的实践过程中会出现一些问题，比如高估了动作值（overestimation），这时候研究人员就提出了Double DQN的技术。从下图可以看出，原先的DQN选用的target值其实还是由同一个网络生成的值，只是说这个网络所选用的参数是之前的参数。而Double DQN中将target的值做了小的改变，能够达到它是由“两个网络”生成的效果。从第二行的表达式可以看出，尽管这里依旧用的是agent含有旧参数的网络，但是这里的动作索引是通过agent当前参数网络得到的，取得该值的方法就是最大化agent当前参数的网络所输出的动作值（其输入值是环境返回的下一个状态），显然这样就解耦了动作的选取和动作值的计算，动作的选取（产生的是一系列大小为（batch_size, 1）的索引）是由新参数的agent网络获取，动作值的估算是由旧参数的agent网络所得到。

DQN的更新方式：

![image](https://user-images.githubusercontent.com/62683546/153756142-e1a43947-396c-44d0-86e4-2857c8560088.png)

Double-DQN的更新方式：

![image](https://user-images.githubusercontent.com/62683546/153756236-e8ff9f42-546e-494f-b329-238d1117d72a.png)

#### Dueling DQN
Dueling DQN最重要的一点就是改进了DQN中的网络结构，将Q值拆分成状态值V和优势函数（Advantage Function）。该方法能够更有效率地对Q值进行更新，因为每一次V值更新之后，都要加在A函数的所有维度上（相当于一个bias），相当于其他动作的值也同时被更新了。

![image](https://user-images.githubusercontent.com/62683546/153756279-2eb31794-e0c0-4165-b2c9-fd0fd8f7194d.png)

## 只狼的强化学习环境设置
* 状态空间
* 动作空间
* 奖励函数

### 状态空间
直接截取游戏画面作为状态描述，如图所示。

![0d5cce59191949dcb4acab94e562cc8](https://user-images.githubusercontent.com/62683546/154037390-b2444642-0706-4be0-85bc-67b42c1c547a.png)


### 动作空间
利用Python程序模拟按键输出，将只狼游戏的不同技能操作预设为动作函数。包含共8种动作(根据训练需要调整动作数以改变训练难度)。

* 普通攻击
* 格挡
* 跳跃
* 向前瞬步
* 向后瞬步
* 重攻击（突刺）
* 义手忍具攻击
* 技能攻击

### 奖励函数
用Boss血量、Boss架势条、己方血量、己方架势条综合计算奖励，其中血量和架势条通过截取图片计算特定像素值的数量来衡量。

当Boss血量减少或Boss架势条增加时获得基础奖励值10点，己方血量减少或己方架势条增加时获得基础奖励值-10点，并乘以相应的奖励系数。

Boss血量、Boss架势条、己方血量、己方架势条的奖励系数分别为：0.4，0.5，0.6，0.5（优先保证自身血量的保守倾向，可通过修改奖励系数改变agent的策略倾向）

