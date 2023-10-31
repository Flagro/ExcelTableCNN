from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import f1_score, accuracy_score, average_precision_score, recall_score
import enum
from typing import Optional
from copy import deepcopy


class ClassifierMode(enum.Enum):
    NearestNeighbors = "NearestNeighbors"
    LinearSVM = "LinearSVM"
    # RBF_SVM = "RBF_SVM"
    # GaussianProcess = "GaussianProcess"
    DecisionTree = "DecisionTree"
    RandomForest = "RandomForest"
    NeuralNet = "NeuralNet"
    AdaBoost = "AdaBoost"
    # NaiveBayes = "NaiveBayes"
    # QDA = "QDA"


class ScoreMode(enum.Enum):
    f1 = "f1"
    accuracy_score = "accuracy_score"
    average_precision_score = "average_precision_score"
    recall_score = "recall_score"


class MLClassifier:
    def __init__(self, classifier_type: Optional[ClassifierMode] = ClassifierMode.RandomForest):
        classifiers = {
            ClassifierMode.NearestNeighbors: KNeighborsClassifier(3),
            ClassifierMode.LinearSVM: SVC(kernel="linear", C=0.025),
            # ClassifierMode.RBF_SVM: SVC(gamma=2, C=1),
            # ClassifierMode.GaussianProcess: GaussianProcessClassifier(),
            ClassifierMode.DecisionTree: DecisionTreeClassifier(max_depth=5),
            ClassifierMode.RandomForest: RandomForestClassifier(),
            ClassifierMode.NeuralNet: MLPClassifier(alpha=1, max_iter=1000),
            ClassifierMode.AdaBoost: AdaBoostClassifier(),
            # ClassifierMode.NaiveBayes: GaussianNB(),
            # ClassifierMode.QDA: QuadraticDiscriminantAnalysis(),
        }
        self.classifier = classifiers[classifier_type]

    def fit_predict(self, train_df, test_df):
        try:
            train_df = train_df.copy().drop(columns=["file_path", "sheet_name"])
            test_df = test_df.copy().drop(columns=["file_path", "sheet_name"])
        except:
            pass

        local_classifier = deepcopy(self.classifier)
        # clf = make_pipeline(StandardScaler(), local_classifier)
        clf = local_classifier
        local_train_df = shuffle(train_df)
        clf.fit(local_train_df.drop(columns=["is_header", "coordinate"]), local_train_df["is_header"])
        local_test_df = test_df.copy()
        if "is_header" in local_test_df.columns:
            local_test_df = local_test_df.drop(columns=["is_header"])
        predictions = local_classifier.predict(local_test_df.drop(columns=["coordinate"]))
        return predictions

    def get_score(self, train_df, score_type: Optional[ScoreMode] = ScoreMode.f1):
        try:
            train_df = train_df.copy().drop(columns=["file_path", "sheet_name"])
        except:
            pass
        scores = {
            ScoreMode.f1: lambda y, p: f1_score(y, p),
            ScoreMode.accuracy_score: lambda y, p: accuracy_score(y, p),
            ScoreMode.average_precision_score: lambda y, p: average_precision_score(y, p),
            ScoreMode.recall_score: lambda y, p: recall_score(y, p),
        }

        train_split_df, test_split_df = train_test_split(train_df, test_size=0.2, stratify=train_df["is_header"])

        predictions = self.fit_predict(train_split_df, test_split_df)

        test_split_df.insert(len(test_split_df.columns), 'predictions', predictions)

        score = scores[score_type](test_split_df["is_header"], predictions)

        fault_df = test_split_df.loc[lambda df: df['predictions'] != df["is_header"]]

        print("{}/{}".format(len(fault_df), len(test_split_df)))

        return score


def predict_header_coordinates(train_df, test_df):
    try:
        train_df = train_df.copy().drop(columns=["file_path", "sheet_name"])
        test_df = test_df.copy().drop(columns=["file_path", "sheet_name"])
    except:
        pass

    # for c in ClassifierMode:
    #     cur_clf = MLClassifier(c)
    #     print(c)
    #     for s in ScoreMode:
    #         print("{} score: {}".format(s, cur_clf.get_score(train_df, s)))
    #     print()

    clf = MLClassifier(ClassifierMode.RandomForest)

    predictions = clf.fit_predict(train_df, test_df)

    result = test_df.copy()[["coordinate"]]
    result["is_header"] = predictions

    header_coordinates = set(result[result["is_header"] == 1]["coordinate"])

    return header_coordinates
