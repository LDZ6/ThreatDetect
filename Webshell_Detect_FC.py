import hashlib
import os

import numpy as np
from skimage.transform import resize
from torch import nn


def detect_Webshell(file_path):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def script_to_image(file_path, image_size=(128, 128)):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            script = f.read()
        file_MD5 = hashlib.md5(script.encode('utf-8')).hexdigest()
        # 使用 ASCII 值将脚本转换为图像数组
        image_array = np.array([[ord(char) for char in line] for line in script])

        # 将值归一化为 [0， 1] 范围
        image_array = image_array / 255.0

        # 调整图像数组的大小
        image_array = resize(image_array, image_size, mode='constant', anti_aliasing=True)

        # # 添加通道维度（假设你期望的输入是灰度图像）
        image_array = np.expand_dims(image_array, axis=0)  # 添加 channel 维度
        image_array = np.expand_dims(image_array, axis=0)  # 添加 batch 维度

        return image_array, file_MD5

    class CNN128_Network(nn.Module):
        def __init__(self):
            super(CNN128_Network, self).__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            self.batchnorm1 = nn.BatchNorm2d(64)
            self.relu1 = nn.LeakyReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.batchnorm2 = nn.BatchNorm2d(128)
            self.relu2 = nn.LeakyReLU()
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.batchnorm3 = nn.BatchNorm2d(256)
            self.relu3 = nn.LeakyReLU()
            self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.batchnorm4 = nn.BatchNorm2d(512)
            self.relu4 = nn.LeakyReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
            self.batchnorm5 = nn.BatchNorm2d(1024)
            self.relu5 = nn.LeakyReLU()
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            # 全连接层
            self.fc1 = nn.Linear(1024 * 8 * 8, 1024)
            self.relu6 = nn.LeakyReLU()
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(1024, 512)
            self.relu7 = nn.LeakyReLU()
            self.dropout2 = nn.Dropout(0.5)
            self.fc3 = nn.Linear(512, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = self.relu2(x)
            x = self.pool1(x)
            x = self.conv3(x)
            x = self.batchnorm3(x)
            x = self.relu3(x)
            x = self.pool1(x)
            x = self.conv4(x)
            x = self.batchnorm4(x)
            x = self.relu4(x)
            x = self.pool2(x)
            x = self.conv5(x)
            x = self.batchnorm5(x)
            x = self.relu5(x)
            x = x.view(-1, 1024 * 8 * 8)
            x = self.fc1(x)
            x = self.relu6(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu7(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            x = self.sigmoid(x)
            return x

    script, file_MD5 = script_to_image(file_path)
    CNN_model = CNN128_Network().to(device)
    CNN_model.load_state_dict(torch.load('Webshell_detect_models/CNN_128_models/new_37model.pth'))

    from gensim.models import Word2Vec
    from Webshell_detect_models.GCN.Network import detect_process_single_graph, GCNClassifier_fc3
    w2v_model = Word2Vec.load('Webshell_detect_models/GCN/word2vec.model')
    GCN_model = GCNClassifier_fc3(w2v_model.vector_size, 48).to(device)
    GCN_model.load_state_dict(torch.load('Webshell_detect_models/GCN/81model.pth'))

    import torch
    from transformers import BertTokenizer, BertModel

    class Classifier(nn.Module):
        def __init__(self, input_dim, hidden_dim1, output_dim):
            super(Classifier, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim1)
            self.relu1 = nn.LeakyReLU(0.1)  # 使用LeakyReLU，你可以调整参数alpha的值
            self.fc2 = nn.Linear(hidden_dim1, 64)
            self.relu2 = nn.LeakyReLU(0.1)
            self.fc3 = nn.Linear(64, output_dim)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.fc3(out)
            out = self.sigmoid(out)
            return out

    # 初始化BERT模型和tokenizer
    # 分词器和模型所在的路径
    tokenizer_path = "Webshell_detect_models/BERT/bert-base_tokenizer"
    model_path = "Webshell_detect_models/BERT/bert-base_model"

    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    Bert_model = BertModel.from_pretrained(model_path)
    Bert_model.to(device)  # 将BERT模型移动到GPU上

    def encode_and_extract_features(text):
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        inputs = inputs.to(device)  # 将输入也移到GPU上
        with torch.no_grad():
            outputs = Bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    # 读取代码文件内容
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        code_content = f.read()

    # 对代码进行编码和特征提取
    code_embedding = encode_and_extract_features(code_content)

    # 将特征转移到 GPU 上（如果在 GPU 上训练的话）
    code_embedding = code_embedding.to(device)

    input_dim = 768
    hidden_dim = 128
    output_dim = 1

    # 创建模型实例
    model = Classifier(input_dim, hidden_dim, output_dim).to(device)

    # 加载之前保存的权重
    model.load_state_dict(torch.load('Webshell_detect_models/BERT/15model.pth'))
    model.eval()

    # 使用模型进行预测
    with torch.no_grad():
        output = model(code_embedding)
        Bert_prediction = (output > 0.5).float().cpu().numpy()
    # print(Bert_prediction.item())
    # print(output)
    # print('BERT检测'+f'文件类型为webshell' if Bert_prediction else 'BERT检测'+f'文件类型为正常文件')

    c = Bert_prediction.item()
    if c == 1:
        result = 1
        print('Bert检测为Webshell')
        return result, file_MD5
    elif c == 0:
        CNN_model.eval()
        with torch.no_grad():
            input_data = torch.tensor(script).to(device)
            input_data = input_data.float()  # 将输入数据类型转换为 float
            outputs = CNN_model(input_data)
            CNN_predicted = (outputs > 0.55).float()
            b = CNN_predicted.item()
        if b == 0:
            result = 0
            print('CNN检测为正常')
            try:
                a = 999
                GCN_model.eval()
                with torch.no_grad():
                    graph = detect_process_single_graph(file_path, w2v_model)
                    graph = graph.to(device)
                    outputs = GCN_model(graph).to(device)
                    graph.y.view_as(outputs).to(device)
                    GCN_predictions = (outputs.squeeze() > 0.7).int()
                    a = GCN_predictions.item()
                    print('GCN检测为恶意' if a == 1 else 'GCN检测为正常')
            except TypeError:
                a = 999
                pass
            if a == 999:
                return result, file_MD5
            elif a == 0:
                return result, file_MD5
            elif a == 1:
                result = 0.5
                return result, file_MD5
        elif b == 1:
            result = 1
            print('CNN检测为Webshell')
            return result, file_MD5


if __name__ == '__main__':
    # 正常文件地址为文件上传目录中的那个文件，这里为测试数据
    file_path = 'uploads/new.php'
    file_type, MD5 = detect_Webshell(file_path)
    print(f'文件为Webshell，文件MD5值为{MD5}' if file_type == 1 else f'文件为正常文件，文件MD5值为{MD5}')
