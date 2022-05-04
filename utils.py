import matplotlib.pyplot as plt

def plot(train_losses,valid_losses,n_epochs):
    epoch_ticks = range(1, n_epochs + 1)
    plt.plot(epoch_ticks, train_losses)
    plt.plot(epoch_ticks, valid_losses)
    plt.style.use('ggplot')
    plt.legend(['Train Loss', 'Valid Loss'])
    plt.title('Losses') 
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    #plt.xticks(epoch_ticks)
    #plt.figure(figsize=(15,7))
    plt.savefig("./figures/train_res.png")
    plt.show()