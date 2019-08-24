from svm_manual.svm import SVM
from svm_manual import utils


x, y = utils.load_data('data/random_data.txt')
model = SVM(c=1.0, kernel='linear')
support_vectors, iterations = model.fit(x, y)
y_hat = model.predict(x)
score = model.accuracy(y, y_hat)
print(score)
utils.plot_hyperplane(model, x, y)
