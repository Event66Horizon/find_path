import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据加载与预处理
data = load_iris()
X = data.data
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 转为张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.long).to(device)

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 2. 定义模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 3层CNN结构，调整padding和输出特征计算
        self.conv1 = nn.Conv1d(1, 16, kernel_size=2, padding=0)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=2, padding=0)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=2, padding=0)
        self.bn3 = nn.BatchNorm1d(64)

        # 计算卷积后的特征尺寸
        # 对于kernel_size=2, padding=0的卷积，每层减少1个特征
        cnn_output_dim = max(1, input_dim - 3)  # 3层卷积后
        self.fc1 = nn.Linear(64 * cnn_output_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # 输入形状: [batch_size, features]
        x = x.unsqueeze(1)  # 变为 [batch_size, 1, features]

        # 第一层卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        # 第二层卷积
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        # 第三层卷积
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)

        # 全连接层前先展平
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class SimpleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 手动实现GRU层
        self.hidden_dim = hidden_dim

        # 更新门参数
        self.w_z = nn.Linear(input_dim, hidden_dim)
        self.u_z = nn.Linear(hidden_dim, hidden_dim)

        # 重置门参数
        self.w_r = nn.Linear(input_dim, hidden_dim)
        self.u_r = nn.Linear(hidden_dim, hidden_dim)

        # 候选隐藏状态参数
        self.w_h = nn.Linear(input_dim, hidden_dim)
        self.u_h = nn.Linear(hidden_dim, hidden_dim)

        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len=1, input_dim)
        x = x.unsqueeze(1) if x.dim() == 2 else x
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 初始化隐藏状态
        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        # 处理序列
        for t in range(seq_len):
            x_t = x[:, t, :]  # 当前时间步的输入

            # 更新门
            z_t = torch.sigmoid(self.w_z(x_t) + self.u_z(h_t))

            # 重置门
            r_t = torch.sigmoid(self.w_r(x_t) + self.u_r(h_t))

            # 候选隐藏状态
            h_tilde = torch.tanh(self.w_h(x_t) + self.u_h(r_t * h_t))

            # 最终隐藏状态
            h_t = (1 - z_t) * h_t + z_t * h_tilde

        # 使用最后的隐藏状态进行预测
        return self.fc(h_t)


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 简化自注意力机制，避免维度问题
        self.hidden_dim = 32

        # 单头自注意力，减少维度转换带来的复杂性
        self.query = nn.Linear(input_dim, self.hidden_dim)
        self.key = nn.Linear(input_dim, self.hidden_dim)
        self.value = nn.Linear(input_dim, self.hidden_dim)

        # 输出层
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        # x: [batch_size, input_dim]
        # 创建一个序列长度为1的输入
        x_seq = x.unsqueeze(1)  # [batch_size, 1, input_dim]

        # 线性变换
        q = self.query(x_seq)  # [batch_size, 1, hidden_dim]
        k = self.key(x_seq)  # [batch_size, 1, hidden_dim]
        v = self.value(x_seq)  # [batch_size, 1, hidden_dim]

        # 计算注意力分数 (点积注意力)
        # q: [batch_size, 1, hidden_dim]
        # k: [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim, 1]
        # scores: [batch_size, 1, 1]
        scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.hidden_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        # attn_weights: [batch_size, 1, 1]
        # v: [batch_size, 1, hidden_dim]
        # context: [batch_size, 1, hidden_dim]
        context = torch.bmm(attn_weights, v)

        # 残差连接和层归一化
        output = self.out_proj(context)
        output = self.dropout(output)
        output = self.layer_norm(output + context)

        # 输出层
        return self.fc(output.squeeze(1))


# 3. 训练与评估函数
def train_and_eval(model, X_train, y_train, X_test, y_test, epochs=50):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 记录每个epoch的损失和准确率
    losses = []
    accuracies = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        # 记录损失值
        losses.append(loss.item())

        # 评估当前epoch准确率
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).argmax(dim=1).cpu().numpy()
            y_true = y_test.cpu().numpy()
            acc = accuracy_score(y_true, y_pred)
            accuracies.append(acc)

    # 最终评估模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).argmax(dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()

        # 计算多个评价指标
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro")
        rec = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "losses": losses,
        "accuracy_history": accuracies,
    }


# 4. 运行四种模型
input_dim = X.shape[1]
output_dim = len(np.unique(y.cpu().numpy()))
print(f"输入维度: {input_dim}, 输出维度: {output_dim}")

# 打印数据集形状
print(f"X_train形状: {X_train.shape}, y_train形状: {y_train.shape}")
print(f"X_test形状: {X_test.shape}, y_test形状: {y_test.shape}")

try:
    print("开始训练MLP模型...")
    mlp = MLP(input_dim, 16, output_dim)
    mlp_results = train_and_eval(
        mlp, X_train, y_train, X_test, y_test, epochs=100
    )
    print("MLP模型训练完成")
except Exception as e:
    print(f"MLP模型训练失败: {e}")
    mlp_results = None

try:
    print("开始训练CNN模型...")
    cnn = SimpleCNN(input_dim, output_dim)
    cnn_results = train_and_eval(
        cnn, X_train, y_train, X_test, y_test, epochs=100
    )
    print("CNN模型训练完成")
except Exception as e:
    print(f"CNN模型训练失败: {e}")
    cnn_results = None

try:
    print("开始训练GRU模型...")
    gru = SimpleGRU(input_dim, 16, output_dim)
    gru_results = train_and_eval(
        gru, X_train, y_train, X_test, y_test, epochs=100
    )
    print("GRU模型训练完成")
except Exception as e:
    print(f"GRU模型训练失败: {e}")
    gru_results = None

try:
    print("开始训练自注意力模型...")
    attn = SelfAttention(input_dim, output_dim)
    attn_results = train_and_eval(
        attn, X_train, y_train, X_test, y_test, epochs=100
    )
    print("自注意力模型训练完成")
except Exception as e:
    print(f"自注意力模型训练失败: {e}")
    attn_results = None

# 收集结果
results = {}
if mlp_results:
    results["MLP"] = mlp_results
if cnn_results:
    results["CNN"] = cnn_results
if gru_results:
    results["GRU"] = gru_results
if attn_results:
    results["Self-Attention"] = attn_results

# 打印评价指标
for model_name, metrics in results.items():
    print(
        f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, "
        f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
    )

# 只有当至少有一个模型成功训练时才进行可视化
if results:
    # 5. 可视化结果
    plt.figure(figsize=(15, 10))

    # 获取模型名称列表
    model_names = list(results.keys())

    # 5.1 损失函数下降曲线
    plt.subplot(2, 2, 1)
    for model_name, metrics in results.items():
        plt.plot(metrics["losses"], label=model_name)
    plt.title("Training Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 5.2 准确率增长曲线
    plt.subplot(2, 2, 2)
    for model_name, metrics in results.items():
        plt.plot(metrics["accuracy_history"], label=model_name)
    plt.title("Accuracy Growth Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    if len(model_names) > 1:  # 只有多个模型时才进行对比
        # 5.3 F1 Score对比
        plt.subplot(2, 2, 3)
        metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
        x = np.arange(len(model_names))
        width = 0.2
        for i, metric in enumerate(["accuracy", "precision", "recall", "f1"]):
            values = [metrics[metric] for metrics in results.values()]
            plt.bar(x + i * width - 0.3, values, width, label=metrics_names[i])
        plt.title("Performance Metrics Comparison")
        plt.xticks(x, model_names)
        plt.ylim(0, 1.1)
        plt.legend()

    # 5.4 混淆矩阵 (只展示最好的模型)
    plt.subplot(2, 2, 4)
    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])[0]
    cm = results[best_model]["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {best_model}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig("iris_network_comparison.png")
    plt.show()
else:
    print("没有模型成功训练，无法生成可视化结果")
