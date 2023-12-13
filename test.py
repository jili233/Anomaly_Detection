import torch
import argparse
from train import Backbone, DataModule  # 从train.py导入Backbone和DataModule类
from pytorch_lightning import Trainer
from sklearn.metrics import f1_score
import numpy as np

def load_model(weight_path, num_classes=2):
    model = Backbone(num_classes)
    model.load_state_dict(torch.load(weight_path))
    return model

def test_model(model, data_module):
    trainer = Trainer()
    results = trainer.test(model, datamodule=data_module)
    return results

def calculate_f1(results):
    all_preds = np.concatenate([result['preds'] for result in results])
    all_targets = np.concatenate([result['targets'] for result in results])
    f1 = f1_score(all_targets, all_preds, average='weighted')
    return f1

def main():
    parser = argparse.ArgumentParser(description='测试模型并获取F1分数')
    parser.add_argument("--data_path", type=str, required=True, help="数据集路径")
    parser.add_argument("--weight_path", type=str, required=True, help="模型权重文件路径")
    args = parser.parse_args()

    data_module = DataModule(data_dir=args.data_path)

    model = load_model(args.weight_path)
    results = test_model(model, data_module)
    f1 = calculate_f1(results)
    
    print(f"F1 Score: {f1}")

if __name__ == '__main__':
    main()

