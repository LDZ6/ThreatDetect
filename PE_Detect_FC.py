import hashlib
import json
import random

import numpy as np
import pefile
import torch
import torch.nn.functional as F
from torch import nn
from transformers import RobertaTokenizer, RobertaModel

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def calculate_md5(file_path):
    try:
        with open(file_path, 'rb') as file:
            # 创建MD5散列对象
            md5_hash = hashlib.md5()
            # 读取文件块并更新散列
            while chunk := file.read(8192):
                md5_hash.update(chunk)
            # 计算MD5散列值
            md5_value = md5_hash.hexdigest()
            return md5_value
    except IOError as e:
        print(f"无法读取文件 {file_path}: {e}")
        return None


def compress_embeddings(embeddings, attention_weights):
    seq_length = embeddings.size(0)
    if attention_weights.dim() > 1:
        attention_weights = attention_weights.squeeze(0)
    # print(f"原始注意力权重: {attention_weights}")  # 打印原始注意力权重
    _, sorted_indices = torch.sort(attention_weights, descending=True)
    # print(f"排序后的索引: {sorted_indices}")  # 打印排序后的索引
    # 保留权重最大的N/2个字
    preserve_indices = sorted_indices[:int(seq_length // 30)]

    # print(f"原始序列长度: {seq_length}")  # 打印原始序列长度
    # print(f"保留字的数量: {len(preserve_indices)}")  # 打印保留字的数量

    # 创建一个保留标记列表，1代表保留，0代表池化
    preserve_mask = torch.zeros(seq_length, dtype=torch.bool)
    preserve_mask[preserve_indices] = True

    # 初始化压缩后的序列为空Tensor
    compressed_embeddings = torch.tensor([], device=embeddings.device)
    # 用于暂存连续未被选中的字
    buffer = []

    for idx in range(seq_length):
        if preserve_mask[idx]:  # 当前字被选中保留
            # 处理buffer中的字，如果有超过一个字则进行池化
            if len(buffer) > 1:
                # 对buffer中的字进行平均池化
                pooled_embedding = torch.stack(buffer).mean(dim=0, keepdim=True)
                # 将池化后的结果拼接到压缩序列中
                compressed_embeddings = torch.cat((compressed_embeddings, pooled_embedding), dim=0)
                buffer = []  # 重置buffer
            elif len(buffer) == 1:
                buffer = []  # 如果只有一个字，则忽略不进行池化
            # 将当前被选中的字拼接到压缩序列中
            compressed_embeddings = torch.cat((compressed_embeddings, embeddings[idx].unsqueeze(0)), dim=0)
        else:  # 当前字未被选中，加入buffer等待处理
            buffer.append(embeddings[idx])

    # 序列末尾处理：如果buffer中还有未处理的字
    if len(buffer) > 1:
        # 对最后的连续未被选中的字进行平均池化
        pooled_embedding = torch.stack(buffer).mean(dim=0, keepdim=True)
        # 将池化后的结果拼接到压缩序列中
        compressed_embeddings = torch.cat((compressed_embeddings, pooled_embedding), dim=0)

    # compressed_seq_length = compressed_embeddings.size(0)
    # print(f"压缩后序列长度: {compressed_seq_length}")  # 打印压缩后序列长度

    return compressed_embeddings


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.score_linear = nn.Linear(input_dim, 1)

    def forward(self, batch_embeddings, mask):
        # 使用线性层计算注意力得分
        attention_scores = self.score_linear(batch_embeddings).squeeze(-1)

        # 使用掩码来忽略填充的部分（将它们的得分设置为负无穷）
        attention_scores.masked_fill_(mask == 0, float('-inf'))

        # 应用softmax得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=1)

        # 根据注意力权重重新计算嵌入表示
        weighted_embeddings = batch_embeddings * attention_weights.unsqueeze(-1)

        return weighted_embeddings, attention_weights


def save_embeddings(texts, attention_model, tokenizer, bert_model, device):
    for i in range(len(texts)):
        batch_texts = [texts[i]]

        # 处理文本长度大于512的情况
        if len(texts[i]) > 512:
            # 直接对原始文本进行切块
            chunks = [texts[i][j:j + 512] for j in range(0, len(texts[i]), 511)]
            all_embeddings = []
            # 在save_embeddings函数中对长度大于512的文本处理部分的调整
            index = 0
            for chunk in chunks:
                inputs = tokenizer(chunk, truncation=True, padding=True, return_tensors="pt", max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                attention_mask = inputs['attention_mask']
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    batch_embeddings = outputs.last_hidden_state
                    if attention_model is not None:
                        # 注意：这里直接使用Attention模型返回的注意力权重
                        weighted_embeddings, attention_weights = attention_model(batch_embeddings, attention_mask)
                        if len(texts[i]) > 512 and index != 0:
                            compressed_embedding = compress_embeddings(batch_embeddings.cpu().squeeze(0),
                                                                       attention_weights.cpu())
                            index += 1
                        else:
                            compressed_embedding = weighted_embeddings.cpu().squeeze(0)
                            index += 1
                    else:
                        compressed_embedding = batch_embeddings.cpu().squeeze(0)
                        index += 1
                all_embeddings.append(compressed_embedding)

            # 将分块后的压缩嵌入向量按序列长度进行拼接
            torch.cat(all_embeddings, dim=0)

            # 将分块后的嵌入按序列长度进行拼接
            final_embedding = torch.cat(all_embeddings, dim=0)
        else:
            # 对于长度小于或等于512的文本，按原来的流程处理
            inputs = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt", max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = bert_model(**inputs)
                batch_embeddings = outputs.last_hidden_state

                final_embedding = batch_embeddings.cpu().squeeze(0)

        return final_embedding


def extract_pe_features(pe_file):
    pe_features = {
        'imports': [],
        'sections': [],
    }
    pe = pefile.PE(pe_file)

    # 提取导入表信息
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        dll_functions = {}
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll_name = entry.dll.decode('utf-8')
            if dll_name not in dll_functions:
                dll_functions[dll_name] = []
            for imp in entry.imports:
                if imp.name:
                    func_name = imp.name.decode('utf-8')
                    dll_functions[dll_name].append(func_name)

        # 将收集到的函数合并为一个字符串，并添加到特征列表中
        for dll, functions in dll_functions.items():
            functions_str = ' , '.join(functions)  # 使用中文逗号作为分隔符
            pe_features['imports'].append({'dll': dll, 'function': functions_str})

    # 提取节信息
    for section in pe.sections:
        section_name = section.Name.decode().rstrip('\x00')
        pe_features['sections'].append(section_name)

    json_str = json.dumps(pe_features, default=str, indent=4)

    return json_str


def PE_detect(file_path):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 保存模型和tokenizer到当前目录
    model_path = "./PE/model_directory"
    tokenizer_path = "./PE/tokenizer_directory"
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    bert_model = RobertaModel.from_pretrained(model_path).to(device)

    attention_model = Attention(input_dim=768).to(device)
    MD5 = calculate_md5(file_path)

    try:
        pe_features = extract_pe_features(file_path)
        embeddings = save_embeddings([pe_features], attention_model, tokenizer, bert_model, device)

        # 此处开始模型结构定义
        class Attention_LSTM(nn.Module):
            def __init__(self, hidden_dim):
                super(Attention_LSTM, self).__init__()
                self.attention = nn.Linear(hidden_dim * 2, 1)

            def forward(self, x):
                # x 形状： [batch_size， seq_length， hidden_dim * 2]
                attention_weights = torch.softmax(self.attention(x), dim=1)
                # attention_weights形状： [batch_size， seq_length， 1]
                attention_output = torch.sum(x * attention_weights, dim=1)
                # attention_output形状：[batch_size，hidden_dim*2]
                return attention_output

        class TextCNN(nn.Module):
            def __init__(self, input_dim, num_filters, filter_sizes):
                super(TextCNN, self).__init__()
                self.convs = nn.ModuleList([
                    nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=fs)
                    for fs in filter_sizes
                ])

            def forward(self, x):
                # 转置以将“input_dim”放入通道维度
                x = x.transpose(1, 2)
                x = [F.leaky_relu(conv(x)) for conv in self.convs]
                x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
                x = torch.cat(x, 1)
                return x

        class BiLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers):
                super(BiLSTM, self).__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

            def forward(self, x):
                x, _ = self.lstm(x)
                return x

        class CombinedModel(nn.Module):
            def __init__(self, input_dim, num_filters, filter_sizes, hidden_dim, num_layers, output_dim):
                super(CombinedModel, self).__init__()
                self.text_cnn = TextCNN(input_dim, num_filters, filter_sizes)
                self.bilstm = BiLSTM(input_dim, hidden_dim, num_layers)
                self.attention = Attention_LSTM(hidden_dim)  # 添加注意力层
                self.direct_fc = nn.Linear(input_dim, 256)
                fc_input_dim = num_filters * len(filter_sizes) + hidden_dim * 2 + 256
                self.fc1 = nn.Linear(fc_input_dim, 128)
                self.bn1 = nn.BatchNorm1d(128)
                self.fc2 = nn.Linear(128, output_dim)
                self.sigmoid = nn.Sigmoid()  # 使用 Sigmoid 替代 Softmax

            def forward(self, x):
                cnn_out = self.text_cnn(x)
                lstm_out = self.bilstm(x)
                attn_out = self.attention(lstm_out)  # 使用注意力层处理Bi-LSTM的输出
                direct_out = F.leaky_relu(self.direct_fc(x))
                direct_out = torch.mean(direct_out, dim=1)
                combined_out = torch.cat((cnn_out, attn_out, direct_out), 1)
                x = F.leaky_relu(self.bn1(self.fc1(combined_out)))
                x = F.dropout(x, p=0.5)
                x = self.fc2(x)
                output = self.sigmoid(x)  # 使用 Sigmoid 代替 Softmax
                return output

        model = CombinedModel(input_dim=768, num_filters=128, filter_sizes=[2, 3, 4], hidden_dim=128, num_layers=2,
                              output_dim=1).to(device)

        embeddings = embeddings.to(device)

        model.load_state_dict(torch.load('PE/jingtai_0.934.pth'))

        # 推理预测
        model.eval()
        with torch.no_grad():
            embeddings = torch.unsqueeze(embeddings, dim=0)  # 在第一维度添加维度
            outputs = model(embeddings)
            prediction = torch.sigmoid(outputs).squeeze()  # 使用sigmoid函数获取概率
            print(f'静态检测置信度为{prediction.item():.4f}')
            jingtai_label = (prediction > 0.55).item()  # 根据概率进行分类

        # 暂定静态预测置信度结果直接作为result
        if prediction.item()>0.55:
            result = jingtai_label
        elif prediction.item()>0.525:
            result = 0.5
        else:
            result = 0
        
        torch.cuda.empty_cache()  # 清除 GPU 缓存

        return result, MD5
    except:
        print('PEfile库静态解析失败')
        return -1, MD5


if __name__ == '__main__':
    file_path = './uploads/7zG.exe'
    result, MD5 = PE_detect(file_path)
    print(f'恶意软件,MD5值为{MD5}' if result == 1 else f'正常软件,MD5值为{MD5}')

