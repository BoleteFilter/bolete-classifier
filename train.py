import torch
import torch.nn.functional as F


def check_accuracy(
    X_val, y_val, model, device=torch.device("cpu"), dtype=torch.float32
):
    if loader.dataset.train:
        print("Checking accuracy on validation set")
    else:
        print("Checking accuracy on test set")
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for i in range(X_val.shape[0]):
            x = X_val[i].to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y_val[i].to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples
        print("Got %d / %d correct (%.2f)" % (num_correct, num_samples, 100 * acc))
        return acc


def train_model(
    model,
    optimizer,
    X_train,
    y_train,
    X_val,
    y_val,
    loss_fcn=F.cross_entropy,
    batch_size=32,
    epochs=1,
    device=torch.device("cpu"),
    dtype=torch.float32,
):
    """
    Train a model using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    history = []
    for e in range(epochs):
        for i in range(X_train.shape[0], batch_size):
            model.train()  # put model to training mode

            lim = i + batch_size
            if lim > X_train.shape[0]:
                lim = X_train.shape[0]

            print(i, lim)

            x = X_train[i:lim].to(
                device=device, dtype=dtype
            )  # move to device, e.g. GPU
            y = y_train[i:lim].to(device=device, dtype=torch.long)

            scores = model(x)
            loss = loss_fcn(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print("Iteration %d, loss = %.4f" % (t, loss.item()))
                history.append([loss.item(), check_accuracy(model, X_val, y_val)])
                print()
    return history
