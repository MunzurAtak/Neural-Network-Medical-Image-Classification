import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from dataset import load_metadata, HAM10000Dataset, get_transforms
from model import build_model, get_device


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate skin lesion classifier on test set')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='models')
    return parser.parse_args()


def run_inference(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds, class_names, output_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Confusion matrix saved to {output_path}')


def plot_f1_per_class(report_dict, class_names, output_path):
    f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(class_names, f1_scores, color='steelblue', edgecolor='black')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score per Class')
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'F1 per class chart saved to {output_path}')


def main():
    args = get_args()
    device = get_device()

    test_df = pd.read_csv(os.path.join(args.data_dir, 'test_split.csv'))
    image_dir = os.path.join(args.data_dir, 'images')

    full_df = load_metadata(os.path.join(args.data_dir, 'GroundTruth.csv'))
    label_encoder = LabelEncoder()
    label_encoder.fit(full_df['dx'])
    class_names = list(label_encoder.classes_)

    test_dataset = HAM10000Dataset(test_df, [image_dir], get_transforms('test'), label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = build_model(num_classes=7)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    labels, preds, probs = run_inference(model, test_loader, device)

    accuracy = (labels == preds).mean()
    print(f'\nOverall accuracy: {accuracy * 100:.2f}%\n')

    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    print(classification_report(labels, preds, target_names=class_names))
    print(f'Macro F1:    {report["macro avg"]["f1-score"]:.4f}')
    print(f'Weighted F1: {report["weighted avg"]["f1-score"]:.4f}')

    os.makedirs(args.output_dir, exist_ok=True)
    plot_confusion_matrix(labels, preds, class_names,
                          os.path.join(args.output_dir, 'confusion_matrix.png'))
    plot_f1_per_class(report, class_names,
                      os.path.join(args.output_dir, 'f1_per_class.png'))


if __name__ == '__main__':
    main()
