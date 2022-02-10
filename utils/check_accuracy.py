from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import torch.nn.functional as F
import torch


def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    model.eval()

    preds_list = []
    truth_list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            prob_y = F.softmax(model(x), dim=1)

            preds = prob_y.max(1, keepdims=True)[1]

            num_correct += preds.eq(y.view_as(preds)).sum().item()
            num_pixels += torch.numel(preds)

            preds_list.append(preds)
            truth_list.append(y)

    preds_np = torch.cat(preds_list).cpu().detach().numpy().flatten()
    truth_np = torch.cat(truth_list).cpu().detach().numpy().flatten()

    f1 = f1_score(preds_np, truth_np, average="weighted")
    kappa = cohen_kappa_score(preds_np, truth_np)

    acc = num_correct / num_pixels

    print(
        f"Got {num_correct}/{num_pixels} with acc {acc*100:.2f}, f1: {f1:.2f}, kappa {kappa:.2f}"
    )
    model.train()
    return acc, f1, kappa
