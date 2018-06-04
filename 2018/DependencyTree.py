class DependencyTree:
    properties = ["address", "ctag", "feats", "head", "lemma", "rel", "tag", "word"]
    def __init__(self,property):
        for p in self.properties:
            return