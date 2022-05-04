import torch
import numpy as np
from sklearn.metrics import classification_report
def validate_epoch(model, valid_loader,criterion,test_loader, input_type='bow'):
    model.eval()
    test_accuracy, n_examples = 0, 0
    y_true, y_pred = [], []
    input_type = 'bow'

    with torch.no_grad():
        for seq, bow, tfidf, target, text in test_loader:
            inputs = bow
            probs = model(inputs)
            if input_type == 'tdidf':
                inputs = tfidf
                probs = model(inputs)
            
            probs = probs.detach().cpu().numpy()
            predictions = np.argmax(probs, axis=1)
            target = target.cpu().numpy()
            
            y_true.extend(predictions)
            y_pred.extend(target)
            
    print(classification_report(y_true, y_pred))