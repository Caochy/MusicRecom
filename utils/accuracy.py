import torch

from utils.util import check_multi


def top1(outputs, label, config, result=None):
    if check_multi(config):
        if len(label[0]) != len(outputs[0]):
            raise ValueError('Input dimensions of labels and outputs must match.')

        outputs = outputs.data
        labels = label.data

        if result is None:
            result = []

        total = 0
        nr_classes = outputs.size(1)

        while len(result) < nr_classes:
            result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

        for i in range(nr_classes):
            outputs1 = (outputs[:, i] >= 0.5).long()
            labels1 = (labels[:, i].float() >= 0.5).long()
            total += int((labels1 * outputs1).sum())
            total += int(((1 - labels1) * (1 - outputs1)).sum())

            if result is None:
                continue

            # if len(result) < i:
            #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

            result[i]["TP"] += int((labels1 * outputs1).sum())
            result[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
            result[i]["FP"] += int(((1 - labels1) * outputs1).sum())
            result[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())

        return torch.Tensor([1.0 * total / len(outputs) / len(outputs[0])]), result
    else:

        if not (result is None):
            # print(label)
            id1 = torch.max(outputs, dim=1)[1]
            # id2 = torch.max(label, dim=1)[1]
            id2 = label
            nr_classes = outputs.size(1)
            while len(result) < nr_classes:
                result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})
            for a in range(0, len(id1)):
                # if len(result) < a:
                #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

                it_is = int(id1[a])
                should_be = int(id2[a])
                if it_is == should_be:
                    result[it_is]["TP"] += 1
                else:
                    result[it_is]["FP"] += 1
                    result[should_be]["FN"] += 1
        pre, prediction = torch.max(outputs, 1)
        prediction = prediction.view(-1)

        return torch.mean(torch.eq(prediction, label).float()), result


def top2(outputs, label, config, result=None):
    if not (result is None):
        # print(label)
        id1 = torch.max(outputs, dim=1)[1]
        # id2 = torch.max(label, dim=1)[1]
        id2 = label
        nr_classes = outputs.size(1)
        while len(result) < nr_classes:
            result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})
        for a in range(0, len(id1)):
            # if len(result) < a:
            #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

            it_is = int(id1[a])
            should_be = int(id2[a])
            if it_is == should_be:
                result[it_is]["TP"] += 1
            else:
                result[it_is]["FP"] += 1
                result[should_be]["FN"] += 1

    _, prediction = torch.topk(outputs, 2, 1, largest=True)
    prediction1 = prediction[:, 0:1]
    prediction2 = prediction[:, 1:]

    prediction1 = prediction1.view(-1)
    prediction2 = prediction2.view(-1)

    return torch.mean(torch.eq(prediction1, label).float()) + torch.mean(torch.eq(prediction2, label).float()), result


from sklearn import metrics
import numpy as np
def auc(outputs, label, config, result = None):
    if result is None:
        result = {'pred': outputs[:,1], 'label': label}
    else:
        result['pred'] = torch.cat([result['pred'], outputs[:,1]], dim = 0)
        result['label'] = torch.cat([result['label'], label], dim = 0)
        
    pred = np.array(result['pred'].cpu().tolist())
    y = result['label'].cpu().tolist()
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)

    return torch.tensor(metrics.auc(fpr, tpr)), result
    
