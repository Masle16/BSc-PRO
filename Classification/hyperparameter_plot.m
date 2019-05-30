%% FCN hyperparameter plot
hidden_size = [32, 64, 128, 256, 512]
validation_acc = [90.4545, 90.6818, 89.0909, 88.1818, 88.8636]

plot(hidden_size, validation_acc,'-s','MarkerSize',12, 'MarkerFaceColor','blue', 'LineWidth', 2)
xlabel('Hidden size', 'FontSize', 24)
ylabel('Accuracy [%]', 'FontSize', 24)
xlim([0 520])
xticks([32, 64, 128, 256, 512])
set(gca,'FontSize',24)
legend('Hidden size', 'FontSize', 24)