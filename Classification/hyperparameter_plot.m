%% FCN hyperparameter plot
hidden_size = [32, 64, 128, 256, 512]
validation_acc = [90.4545, 90.6818, 89.0909, 88.1818, 88.8636]

plot(hidden_size, validation_acc)
title('Validation accuracy versus hidden size with Learning rate = 1e-4')
xlabel('Hidden size')
ylabel('Accuracy in percent')
xticks([32, 64, 128, 256, 512])
legend('Hidden size')