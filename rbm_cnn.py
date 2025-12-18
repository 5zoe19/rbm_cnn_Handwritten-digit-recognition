import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps
import tkinter as tk
from tkinter import ttk, messagebox
import os
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== RBM模型定义 ====================
class RBM(nn.Module):
    def __init__(self, n_visible=784, n_hidden=512, k=1):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k

        # 权重和偏置
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.1)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_from_p(self, p):
        return F.relu(torch.sign(p - torch.rand_like(p)))

    def v_to_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h, self.sample_from_p(p_h)

    def h_to_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v, self.sample_from_p(p_v)

    def forward(self, v):
        # 对比散度算法

        h0, h_sample0 = self.v_to_h(v)

        # Gibbs采样
        v_k = v
        for _ in range(self.k):
            _, h_k = self.v_to_h(v_k)
            p_v_k, v_k = self.h_to_v(h_k)

        return v, p_v_k

    def free_energy(self, v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()


# ==================== CNN模型定义 ====================
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10, use_rbm_init=False, rbm_weights=None):
        super(CNNClassifier, self).__init__()
        self.use_rbm_init = use_rbm_init

        # 第一层卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)

        # 如果使用RBM初始化，设置第一层权重
        if use_rbm_init and rbm_weights is not None:
            self.initialize_with_rbm(rbm_weights)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def initialize_with_rbm(self, rbm_weights):
        """使用RBM权重初始化第一层卷积层"""
        print("使用RBM权重初始化CNN第一层...")

        # RBM权重形状: (n_hidden, n_visible) = (512, 784)
        rbm_weights = rbm_weights.cpu().detach()

        # 将RBM权重重塑为卷积核形状
        # 目标形状: (32, 1, 5, 5) - 32个滤波器，每个是1x5x5
        with torch.no_grad():
            # 方法1: 从RBM权重中选择前32*25=800个权重，重塑为(32, 1, 5, 5)
            if rbm_weights.numel() >= 800:
                selected_weights = rbm_weights.flatten()[:800].view(32, 1, 5, 5)
                self.conv1.weight.data = selected_weights
            else:
                # 方法2: 如果RBM隐藏层太小，使用所有权重并填充
                available_weights = rbm_weights.flatten()
                if available_weights.numel() < 800:
                    # 重复权重直到填满
                    repeated_weights = available_weights.repeat(800 // available_weights.numel() + 1)
                    selected_weights = repeated_weights[:800].view(32, 1, 5, 5)
                else:
                    selected_weights = available_weights[:800].view(32, 1, 5, 5)
                self.conv1.weight.data = selected_weights

            print(f"RBM权重初始化完成: {self.conv1.weight.data.shape}")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ==================== 模型训练类 ====================
class ModelTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rbm = None
        self.cnn = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_loaded = False

    def check_data_exists(self):
        """检查MNIST数据是否已经存在"""
        data_path = './data/MNIST/raw'
        if not os.path.exists(data_path):
            return False

        required_files = [
            'train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
            't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'
        ]

        for file in required_files:
            file_path = os.path.join(data_path, file)
            if not os.path.exists(file_path):
                return False
        return True

    def load_data(self):
        """加载MNIST数据集"""
        if self.data_loaded:
            print("数据已加载，跳过重复加载")
            return

        if self.check_data_exists():
            print("发现已存在的MNIST数据集，直接加载...")
        else:
            print("正在下载MNIST数据集...")

        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=self.transform)
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=self.transform)

        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        self.data_loaded = True
        print("MNIST数据集加载完成")

    def train_rbm(self, n_epochs=10, n_hidden=512):
        """训练RBM模型"""
        print("开始训练RBM模型...")
        self.rbm = RBM(n_visible=784, n_hidden=n_hidden).to(self.device)
        optimizer = optim.Adam(self.rbm.parameters(), lr=0.001)

        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_idx, (data, _) in enumerate(self.train_loader):
                data = data.view(-1, 784).to(self.device)
                data = (data > 0.5).float()  # 二值化

                v0, v_k = self.rbm(data)
                loss = self.rbm.free_energy(v0) - self.rbm.free_energy(v_k)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f'RBM Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / len(self.train_loader):.4f}')

        # 保存RBM模型
        torch.save(self.rbm.state_dict(), 'rbm_model.pth')
        print("RBM模型训练完成并保存")

    def train_cnn_with_rbm_init(self, n_epochs=10):
        """使用RBM初始化的CNN模型训练"""
        print("开始训练使用RBM初始化的CNN模型...")

        # 使用RBM权重初始化CNN第一层
        rbm_weights = self.rbm.W if self.rbm is not None else None
        self.cnn = CNNClassifier(use_rbm_init=True, rbm_weights=rbm_weights).to(self.device)

        optimizer = optim.Adam(self.cnn.parameters(), lr=0.001)

        train_losses = []
        train_accuracies = []

        for epoch in range(n_epochs):
            self.cnn.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.cnn(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            print(f'CNN Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        # 保存CNN模型
        torch.save(self.cnn.state_dict(), 'cnn_rbm_init_model.pth')
        print("使用RBM初始化的CNN模型训练完成并保存")

        return train_losses, train_accuracies

    def train_cnn_normal(self, n_epochs=10):
        """训练普通CNN模型（不使用RBM初始化）"""
        print("开始训练普通CNN模型...")

        self.cnn = CNNClassifier(use_rbm_init=False).to(self.device)
        optimizer = optim.Adam(self.cnn.parameters(), lr=0.001)

        train_losses = []
        train_accuracies = []

        for epoch in range(n_epochs):
            self.cnn.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.cnn(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            print(f'CNN Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        # 保存CNN模型
        torch.save(self.cnn.state_dict(), 'cnn_normal_model.pth')
        print("普通CNN模型训练完成并保存")

        return train_losses, train_accuracies

    def evaluate_models(self):
        """评估模型性能"""
        if self.cnn is None:
            print("CNN模型未训练")
            return 0

        # 确保数据已加载
        if not self.data_loaded:
            print("数据未加载，正在加载数据...")
            self.load_data()

        # 评估CNN
        self.cnn.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.cnn(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        test_loss /= total
        accuracy = 100. * correct / total

        print(f'\n模型评估结果:')
        print(f'测试集损失: {test_loss:.4f}')
        print(f'测试集准确率: {accuracy:.2f}%')

        return accuracy

    def load_models(self, use_rbm_init=True):
        """加载预训练模型"""
        try:
            # 首先确保数据已加载
            if not self.data_loaded:
                self.load_data()

            model_path = 'cnn_rbm_init_model.pth' if use_rbm_init else 'cnn_normal_model.pth'

            if os.path.exists(model_path):
                # 加载RBM模型用于初始化（如果需要）
                rbm_weights = None
                if use_rbm_init and os.path.exists('rbm_model.pth'):
                    self.rbm = RBM(n_visible=784, n_hidden=512).to(self.device)
                    self.rbm.load_state_dict(torch.load('rbm_model.pth', map_location=self.device))
                    rbm_weights = self.rbm.W
                    print("RBM模型加载成功")

                self.cnn = CNNClassifier(use_rbm_init=use_rbm_init, rbm_weights=rbm_weights).to(self.device)
                self.cnn.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"CNN模型加载成功 ({'RBM初始化' if use_rbm_init else '普通'})")
                return True
            return False
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def train_all_models(self, use_rbm_init=True):
        """训练所有模型"""
        print("开始训练所有模型...")

        # 加载数据
        self.load_data()

        if use_rbm_init:
            # 训练RBM模型
            self.train_rbm(n_epochs=10, n_hidden=512)
            # 使用RBM初始化训练CNN模型
            self.train_cnn_with_rbm_init(n_epochs=10)
        else:
            # 训练普通CNN模型
            self.train_cnn_normal(n_epochs=10)

        # 评估模型
        accuracy = self.evaluate_models()

        print("所有模型训练完成！")
        return accuracy

    def compare_models(self):
        """比较RBM初始化模型和普通模型的性能"""
        print("开始比较模型性能...")

        # 训练普通CNN模型
        print("\n=== 训练普通CNN模型 ===")
        self.train_cnn_normal(n_epochs=5)
        normal_accuracy = self.evaluate_models()

        # 训练RBM初始化CNN模型
        print("\n=== 训练RBM初始化CNN模型 ===")
        self.train_rbm(n_epochs=5, n_hidden=512)
        self.train_cnn_with_rbm_init(n_epochs=5)
        rbm_init_accuracy = self.evaluate_models()

        print("\n=== 模型比较结果 ===")
        print(f"普通CNN模型准确率: {normal_accuracy:.2f}%")
        print(f"RBM初始化CNN模型准确率: {rbm_init_accuracy:.2f}%")

        if rbm_init_accuracy > normal_accuracy:
            print("✅ RBM初始化提高了模型性能")
        else:
            print("❌ RBM初始化未提高模型性能")

        return normal_accuracy, rbm_init_accuracy


# ==================== 图形化界面 ====================
class HandwritingApp:
    def __init__(self, root, trainer):
        self.root = root
        self.root.title("手写数字识别系统 - RBM初始化CNN")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)

        # 居中显示
        self.center_window()

        # 初始化模型训练器
        self.trainer = trainer
        self.models_trained = True

        # 初始化变量
        self.current_color = "black"
        self.setup_ui()
        self.setup_bindings()

        # 更新模型状态
        self.update_model_display()

    def center_window(self):
        """窗口居中显示"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def setup_ui(self):
        """设置用户界面"""
        # 标题区域
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        title_frame.pack_propagate(False)

        title_label = tk.Label(title_frame, text="手写数字识别系统 - RBM初始化CNN",
                               font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=20)

        # 模型信息框架
        model_frame = tk.Frame(self.root, bg='#ecf0f1', relief=tk.RAISED, bd=2)
        model_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.model_status = tk.Label(model_frame, text="模型状态: 加载中...",
                                     font=('Arial', 10), bg='#ecf0f1', fg='#e74c3c')
        self.model_status.pack(pady=5)

        # 主内容区域
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧绘图区域 (300px)
        left_frame = tk.Frame(main_frame, width=300, relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        # 右侧功能区域 (650px)
        right_frame = tk.Frame(main_frame, width=650)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right_frame.pack_propagate(False)

        self.setup_left_frame(left_frame)
        self.setup_right_frame(right_frame)

    def setup_left_frame(self, parent):
        """设置左侧绘图区域"""
        # 提示文字
        hint_label = tk.Label(parent, text="请在下方区域书写数字 (0-9)，尽量写大且居中",
                              font=('Arial', 10), bg='white')
        hint_label.pack(pady=10)

        # 画布
        self.canvas = tk.Canvas(parent, width=280, height=280, bg='white',
                                relief=tk.SUNKEN, bd=2)
        self.canvas.pack(pady=10)

        # 颜色选择区域
        color_frame = tk.Frame(parent)
        color_frame.pack(pady=10)

        tk.Label(color_frame, text="画笔颜色:", font=('Arial', 10)).pack(side=tk.LEFT)

        self.color_var = tk.StringVar(value="black")
        colors = [("黑色", "black"), ("蓝色", "blue"), ("红色", "red")]

        for text, color in colors:
            tk.Radiobutton(color_frame, text=text, variable=self.color_var,
                           value=color, command=self.change_color).pack(side=tk.LEFT, padx=5)

        # 当前颜色显示
        self.color_display = tk.Label(parent, text="●", font=('Arial', 20),
                                      fg=self.current_color)
        self.color_display.pack()

        # PIL图像用于存储绘图数据
        self.pil_image = Image.new('RGB', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.pil_image)

    def setup_right_frame(self, parent):
        """设置右侧功能区域"""
        # 图像显示区域
        image_frame = tk.LabelFrame(parent, text="预处理图像", font=('Arial', 12, 'bold'),
                                    relief=tk.RAISED, bd=2)
        image_frame.pack(fill=tk.X, pady=(0, 10))

        image_inner = tk.Frame(image_frame)
        image_inner.pack(pady=10)

        # 预处理后图像
        tk.Label(image_inner, text="预处理后图像 (28x28)", font=('Arial', 10)).pack()
        self.original_canvas = tk.Canvas(image_inner, width=150, height=150,
                                         bg='white', relief=tk.SUNKEN, bd=1)
        self.original_canvas.pack(pady=5)

        # 控制按钮区域
        control_frame = tk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=10)

        self.clear_btn = tk.Button(control_frame, text="清除画布", font=('Arial', 12),
                                   bg='#e74c3c', fg='white', width=15, height=2,
                                   command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=20)

        self.recognize_btn = tk.Button(control_frame, text="识别数字", font=('Arial', 12),
                                       bg='#27ae60', fg='white', width=15, height=2,
                                       command=self.recognize_digit)
        self.recognize_btn.pack(side=tk.RIGHT, padx=20)

        # 结果显示区域
        result_frame = tk.LabelFrame(parent, text="识别结果", font=('Arial', 12, 'bold'),
                                     relief=tk.RAISED, bd=2)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # 识别结果
        self.result_label = tk.Label(result_frame, text="?", font=('Arial', 48, 'bold'),
                                     fg='#34495e')
        self.result_label.pack(pady=10)

        # 置信度
        self.confidence_label = tk.Label(result_frame, text="置信度: 0%",
                                         font=('Arial', 14))
        self.confidence_label.pack()

        # 状态信息
        self.status_label = tk.Label(result_frame, text="系统就绪",
                                     font=('Arial', 10), fg='#7f8c8d')
        self.status_label.pack(pady=10)

    def setup_bindings(self):
        """设置事件绑定"""
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.last_x = None
        self.last_y = None

    def change_color(self):
        """改变画笔颜色"""
        self.current_color = self.color_var.get()
        self.color_display.config(fg=self.current_color)
        self.update_status(f"已选择{self.current_color}画笔")

    def paint(self, event):
        """绘图函数"""
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # 在Tkinter画布上绘制
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=15, fill=self.current_color,
                                    capstyle=tk.ROUND, smooth=tk.TRUE)

            # 在PIL图像上绘制
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill=self.current_color, width=15)

        self.last_x = x
        self.last_y = y

    def reset(self, event):
        """重置坐标"""
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        """清除画布"""
        self.canvas.delete("all")
        self.pil_image = Image.new('RGB', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.pil_image)
        self.original_canvas.delete("all")
        self.result_label.config(text="?", fg='#34495e')
        self.confidence_label.config(text="置信度: 0%")
        self.update_status("画布已清除")

    def preprocess_image(self, image):
        """图像预处理"""
        try:
            # 转换为灰度图
            if image.mode != 'L':
                image = image.convert('L')

            # 反色处理（黑底白字 -> 白底黑字）
            image = ImageOps.invert(image)

            # 找到数字的边界框
            bbox = image.getbbox()
            if not bbox:
                return None

            # 裁剪数字区域
            image = image.crop(bbox)

            # 计算新的尺寸，保持宽高比，最大边长为20
            width, height = image.size
            max_size = 20
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)

            image = image.resize((new_width, new_height), Image.LANCZOS)

            # 创建28x28的图像并将数字居中
            new_image = Image.new('L', (28, 28), 0)  # 黑色背景
            x_offset = (28 - new_width) // 2
            y_offset = (28 - new_height) // 2
            new_image.paste(image, (x_offset, y_offset))

            # 转换为numpy数组并归一化
            image_array = np.array(new_image, dtype=np.float32) / 255.0

            return image_array

        except Exception as e:
            self.update_status(f"图像预处理错误: {e}")
            return None

    def recognize_digit(self):
        """识别数字"""
        if not self.models_trained:
            messagebox.showwarning("警告", "模型未训练！")
            return

        # 检查画布是否为空
        if self.canvas.find_all() == ():
            messagebox.showwarning("警告", "请先在画布上书写数字！")
            return

        try:
            self.update_status("正在处理图像...")

            # 预处理图像
            processed_array = self.preprocess_image(self.pil_image)
            if processed_array is None:
                messagebox.showwarning("警告", "未检测到有效数字，请重新书写！")
                return

            # 显示预处理后的图像
            self.display_processed_image(processed_array)

            # 转换为PyTorch张量
            input_tensor = torch.FloatTensor(processed_array).unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.to(self.trainer.device)

            self.update_status("正在使用CNN进行分类...")

            # CNN分类
            with torch.no_grad():
                output = self.trainer.cnn(input_tensor)
                probabilities = torch.exp(output)
                confidence, prediction = torch.max(probabilities, 1)
                confidence = confidence.item()
                prediction = prediction.item()

            # 显示结果
            self.display_result(prediction, confidence)
            self.update_status("识别完成")

        except Exception as e:
            self.update_status(f"识别错误: {e}")
            messagebox.showerror("错误", f"识别过程中发生错误: {e}")

    def display_processed_image(self, image_array):
        """显示预处理后的图像"""
        # 转换为PIL图像
        image = Image.fromarray((image_array * 255).astype(np.uint8))
        image = image.resize((150, 150), Image.NEAREST)

        # 在画布上显示
        self.original_canvas.delete("all")
        photo = self.pil_to_photo(image)
        self.original_canvas.create_image(75, 75, image=photo)
        self.original_canvas.image = photo

    def display_result(self, prediction, confidence):
        """显示识别结果"""
        self.result_label.config(text=str(prediction))

        conf_percent = confidence * 100
        self.confidence_label.config(text=f"置信度: {conf_percent:.1f}%")

        # 根据置信度调整颜色
        if conf_percent > 70:
            color = '#27ae60'  # 绿色
        elif conf_percent > 50:
            color = '#f39c12'  # 橙色
        else:
            color = '#e74c3c'  # 红色

        self.result_label.config(fg=color)
        self.confidence_label.config(fg=color)

    def pil_to_photo(self, pil_image):
        """将PIL图像转换为Tkinter PhotoImage"""
        from PIL import ImageTk
        return ImageTk.PhotoImage(pil_image)

    def update_status(self, message):
        """更新状态信息"""
        self.status_label.config(text=message)
        self.root.update()

    def update_model_display(self):
        """更新模型状态显示"""
        accuracy = self.trainer.evaluate_models()
        if accuracy > 90:
            color = '#27ae60'  # 绿色
        elif accuracy > 80:
            color = '#f39c12'  # 橙色
        else:
            color = '#e74c3c'  # 红色

        self.model_status.config(
            text=f"模型状态: RBM初始化CNN (准确率: {accuracy:.2f}%)",
            fg=color
        )
        self.update_status(f"模型加载完成，准确率: {accuracy:.2f}%")


# ==================== 主程序 ====================
def check_and_train_models():
    """检查模型是否存在，如果不存在则训练模型"""
    trainer = ModelTrainer()

    # 检查模型文件是否存在（优先加载RBM初始化模型）
    model_paths = ['cnn_rbm_init_model.pth', 'cnn_normal_model.pth']
    model_loaded = False

    for model_path in model_paths:
        if os.path.exists(model_path):
            use_rbm_init = 'rbm_init' in model_path
            print(f"发现预训练模型: {model_path}，正在加载...")
            if trainer.load_models(use_rbm_init=use_rbm_init):
                accuracy = trainer.evaluate_models()
                print(f"模型加载成功，准确率: {accuracy:.2f}%")
                model_loaded = True
                break

    if not model_loaded:
        print("未找到预训练模型，开始训练新模型...")
        # 默认训练RBM初始化模型
        accuracy = trainer.train_all_models(use_rbm_init=True)
        print(f"模型训练完成，准确率: {accuracy:.2f}%")

    return trainer


def main():
    # 先检查并训练模型
    print("=== RBM初始化CNN手写数字识别系统 ===")
    trainer = check_and_train_models()

    # 模型准备完成后，启动图形化界面
    print("启动图形化界面...")
    root = tk.Tk()
    app = HandwritingApp(root, trainer)
    root.mainloop()


if __name__ == "__main__":
    main()