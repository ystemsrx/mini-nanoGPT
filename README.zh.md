[English](https://github.com/ystemsrx/mini-nanoGPT) | [简体中文](README.zh.md)

# Mini NanoGPT 🚀

#### 训练一个GPT原来可以这么简单？

> 让 GPT 模型训练变得简单有趣！一个基于 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) 的可视化训练平台。

## 📖 这是什么？

Mini-NanoGPT 是一个帮助你轻松入门 GPT 模型的工具。无论你是：
- 🎓 深度学习初学者
- 👨‍🔬 研究人员
- 🛠️ 开发者

亦或只是对这个感兴趣，想体验一下大模型的魅力，

都能通过简单的图形界面完成模型训练！

> 初始版本的 Mini NanoGPT（不再更新）请查看 [**old** 分支](https://github.com/ystemsrx/mini-nanoGPT/tree/old)

## ✨ 主要特点

### 1. 简单易用
- 📱 **可视化界面**：告别命令行，用鼠标点点就能完成训练
- 🌍 **中英双语**：完整的中英文界面支持
- 🎯 **一键操作**：数据处理、训练、生成文本都能一键完成

### 2. 功能强大
- 🔤 **灵活的分词**：支持字符级和 GPT-2 或 Qwen 的分词器，支持多语言
- 🚄 **高效训练**：支持多进程加速和分布式训练
- 📊 **实时反馈**：训练过程实时显示进度和效果
- ⚙️ **参数可视化**：所有训练参数都能在界面上直接调整
- 🧩 **数据库管理**：更简单地管理模型，随时保存训练参数下次使用

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆仓库
git clone --depth 1 https://github.com/ystemsrx/mini-nanoGPT.git
cd mini-nanogpt

# 安装依赖（Python 3.7+）
pip install -r requirements.txt
```

### 2. 启动项目
```bash
python main.py
```
打开浏览器访问显示的链接，就能看到训练界面了！（一般是 http://localhost:7860）

## 🎮 使用指南

### 第一步：准备数据
- 打开"数据处理"页面，选择或粘贴你的训练文本并选择分词方式。若要追求更好的效果，可以勾选使用分词器，会自动根据你的文本内容构建词汇表。
- 如果你暂时不想使用验证集，可以勾选“暂不使用验证集”。
- 完成后点击"开始处理"。
这里我用一小段文本来举例：

![image](https://github.com/user-attachments/assets/ec8db0d6-5673-43ae-a4cb-ac064f7209ae)


### 第二步：训练模型
- 切换到"训练"页面，根据需要调整参数（如果只是想体验，可以保持默认值）。
- 程序支持实时显示训练集和验证集的损失曲线。如果在第一步中你生成了验证集，理论上下方损失曲线处会出现两条，蓝色为训练集损失曲线，橙色为验证集损失曲线。
- 如果只显示了1条曲线，请检查终端输出，如果有类似这样的输出
  ```
  Error while evaluating val loss: Dataset too small: minimum dataset(val) size is 147, but block size is 512. Either reduce block size or add more data.
  ```
  说明你设置的block size比你的验证集大，请将它的大小调小，例如128。
- 这样你应当能够正常的看到两条动态变化的曲线。
- 点击"开始训练"，等待模型训练完成

![image](https://github.com/user-attachments/assets/c43ca548-fd6b-4f0a-98c5-55586bec42db)

#### 仅评估模式？
- 这个模式能够让你评估模型在验证集上的损失。请将`评估种子数量 (Number of Evaluation Seeds)`调为大于0的任意值，将开启仅评估模式。你能看到模型在使用不同种子上的损失。

### 第三步：生成文本
1. 进入"推理"页面
2. 输入一段开头文字
3. 点击"生成"，看看模型会写出什么！

![image](https://github.com/user-attachments/assets/5f985e89-d7c2-4f3a-9500-5713497148cd)

## 📁 项目结构
```
mini-nanogpt/
├── main.py          # 启动程序
├── src/             # 配置文件以及其他模块
├── data/            # 数据存储
├── out/             # 模型权重
└── assets/          # Tokenizer 词表文件等
```

## ❓ 常见问题

### 运行得太慢怎么办？
- 💡 可以减小 batch_size 或模型大小
- 💡 使用 GPU 会大大加快速度
- 💡 将评估间隔调大

### 生成的文本不够好？
- 💡 试试增加训练数据量
- 💡 适当调整模型参数
- 💡 改变生成时的温度参数

### 想继续之前的训练？
- 💡 在“训练页面”中的“初始化方式”中选择"resume"
- 💡 指定之前的输出目录即可

## 🤝 参与贡献
欢迎提出建议和改进！可以通过以下方式参与：
- 提交 Issue
- 提交 Pull Request
- 分享你的使用经验

## 📝 许可证
本项目采用 [MIT License](LICENSE) 协议开源。

---

🎉 **开始你的 GPT 之旅吧！**
