import numpy as np
import cvxopt
import cvxopt.solvers

cvxopt.solvers.options['show_progress'] = False

class SVMClassifier():
	"""
	SVM classifier using RBF (guassian) kernel.
	"""
	def __init__(self, gamma=3.0):
		self.gamma = gamma


	def rbf(self, x, y):
		diff = np.subtract(x, y)
		return np.exp(-1.0 * self.gamma * (diff.T @ diff))


	def transform(self, X):
		K = np.zeros([X.shape[0], X.shape[0]])
		for i in range(X.shape[0]):
			for j in range(X.shape[0]):
				K[i, j] = self.rbf(X[i], X[j])
		return K


	def fit(self, X, y):
		X = np.array(X)
		y = np.array(y, dtype=np.double)

		num_points = X.shape[0]

		K = self.transform(X)

		P = cvxopt.matrix(np.outer(y,y) * K)
		q = cvxopt.matrix(np.ones(num_points) * -1)
		A = cvxopt.matrix(y, (1, num_points))
		b = cvxopt.matrix(0.0)
		G = cvxopt.matrix(np.diag(np.ones(num_points) * -1))
		h = cvxopt.matrix(np.zeros(num_points))

		alphas = np.ravel(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])

		support_idxs = alphas > 1e-5

		self._support_vectors = X[support_idxs]
		self._n_support = np.sum(support_idxs)
		self._alphas = alphas[support_idxs]
		self._support_labels = y[support_idxs]
		indices = np.arange(num_points)[support_idxs]

		self.intercept = 0

		for i in range(self._alphas.shape[0]):
			self.intercept += self._support_labels[i]
			self.intercept -= np.sum(self._alphas * self._support_labels * K[indices[i], support_idxs])
		self.intercept /= self._alphas.shape[0]


	def project(self, X):
		score = np.zeros(X.shape[0])
		for i in range(X.shape[0]):
			s = 0
			for (alpha, label, sv) in zip(self._alphas, self._support_labels, self._support_vectors):
				s += alpha * label * self.rbf(X[i], sv)
			score[i] = s
		score += self.intercept
		return score


	def predict(self,X):
		X = np.array(X)
		return np.where(self.project(X) > 0, 1, -1)