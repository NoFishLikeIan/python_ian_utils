from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


class EmbeddedData:
    def __init__(self, X_train, X_test, y_train, y_test, perplexity=40):
        self.raw_data = {
            'X_train': X_train,
            'y_train': X_test,
            'X_test':  y_train,
            'y_test': y_test,
        }

        self.perplexity = perplexity
        self.pca_function(n=2)
        self.pca_function(n=.9)
        self.umap_function()
        self.umap_function(semi_super=False)
        self.tsne_function()

    def pca_function(self, n=.9):
        pca = PCA(n_components=n)
        X_train = pca.fit_transform(self.raw_data["X_train"])
        X_test = pca.transform(self.raw_data["X_test"])
        d_data = {
            'X_train': X_train,
            'y_train': self.raw_data["y_train"],
            'X_test': X_test,
            'y_test': self.raw_data["y_test"],
        }

        if isinstance(n, int):
            self.pca2 = pca
            self.pca2_data = d_data
        else:
            self.pca = pca
            self.pca_data = d_data

    def umap_function(self, semi_super=True):
        umap = UMAP()
        X_train = umap.fit_transform(
            self.raw_data["X_train"], y=self.raw_data['y_test']) if semi_super else umap.fit_transform(self.raw_data["X_train"])
        X_test = umap.transform(self.raw_data["X_test"])

        d_data = {
            'X_train': X_train,
            'y_train': self.raw_data["y_train"],
            'X_test': X_test,
            'y_test': self.raw_data["y_test"],
        }

        if semi_super:
            self.umap_sup = umap
            self.umap_sup_data = d_data

        else:
            self.umap = umap
            self.umap_data = d_data

    def tsne_function(self):
        tsne = TSNE(perplexity=self.perplexity)
        X_train = tsne.fit_transform(self.raw_data["X_train"])
        X_test = tsne.fit_transform(self.raw_data["X_test"])

        d_data = {
            'X_train': X_train,
            'y_train': self.raw_data["y_train"],
            'X_test': X_test,
            'y_test': self.raw_data["y_test"],
        }

        self.tsne_data = d_data

    def __str__(self):
        dim_sum = {
            "pca_dim": (self.pca_data["X_train"].shape, self.pca_data["X_test"].shape),
            "pca2_dim": (self.pca2_data["X_train"].shape, self.pca2_data["X_test"].shape),
            "umap_dim": (self.umap_data["X_train"].shape, self.umap_data["X_test"].shape),
            "umap_sup_dim": (self.umap_sup_data["X_train"].shape, self.umap_sup_data["X_test"].shape),
            "tsne_dim": (self.tsne_data["X_train"].shape, self.tsne_data["X_test"].shape),
        }
        s = ''
        for k, dim in dim_sum:
            s += f'For {k} the new dimension are, train: {dim[0]}, test: {dim[1]}\n'

        return s
