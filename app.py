import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import gradio as gr
from PIL import Image
from predict import load_model, predict

MODEL_PATH = 'models/best_model.pth'

try:
    model, device = load_model(MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    print(f'Model file not found at {MODEL_PATH}. Train the model first.')


def predict_image(image):
    if not model_loaded:
        return 'Model not loaded. Run train.py first.', {}

    if image is None:
        return 'No image provided.', {}

    results = predict(image, model, device)
    top_name, top_conf = next(iter(results.items()))
    summary = f'{top_name} ({top_conf:.1f}% confidence)'
    return summary, results


with gr.Blocks(title='Skin Lesion Classifier') as demo:
    gr.Markdown('# Skin Lesion Classifier')
    gr.Markdown(
        'Upload a dermoscopy image to classify it across 7 skin lesion types. '
        'Trained on the HAM10000 / ISIC 2018 dataset using EfficientNet-B0.'
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type='pil', label='Upload skin lesion image')
            classify_btn = gr.Button('Classify', variant='primary')
        with gr.Column():
            top_prediction = gr.Textbox(label='Top prediction')
            all_predictions = gr.Label(label='Confidence scores', num_top_classes=7)

    classify_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=[top_prediction, all_predictions],
    )

    gr.Markdown(
        '<p style="color: red; font-weight: bold;">'
        'This is a research demo only. Not for medical diagnosis.'
        '</p>'
    )

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0')
