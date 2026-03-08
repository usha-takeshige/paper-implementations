"""ParameterClassifier のテスト (test_cases.md §3, テスト 8-1〜8-7)"""

import pytest
import torch

from mpde_bo.gp_model_manager import GPConfig, GPModelManager
from mpde_bo.importance_analyzer import ImportanceAnalyzer
from mpde_bo.parameter_classifier import ClassificationConfig, ParameterClassification, ParameterClassifier


# ── フィクスチャ ────────────────────────────────────────────────────────────────

@pytest.fixture
def manager() -> GPModelManager:
    return GPModelManager(GPConfig(kernel="matern52"))


@pytest.fixture
def analyzer() -> ImportanceAnalyzer:
    return ImportanceAnalyzer()


@pytest.fixture
def classifier(analyzer) -> ParameterClassifier:
    # important_unimportant_data での実測: dims 0,1 の length scale ≈ 2.5
    # eps_l=5.0 は重要次元を通過させ、非重要次元 (ls >> 100) を除外する
    config = ClassificationConfig(eps_l=5.0, eps_e=0.1)
    return ParameterClassifier(analyzer=analyzer, config=config)


@pytest.fixture
def model_and_data(manager, important_unimportant_data):
    """次元 0,1 が重要 / 次元 2,3 が非重要な学習済みモデルとデータ"""
    X, Y = important_unimportant_data
    model = manager.build(X, Y)
    return model, X


# ── classify ──────────────────────────────────────────────────────────────────

class TestClassify:
    def test_returns_parameter_classification(self, classifier, model_and_data):
        """8-1: 返り値が ParameterClassification インスタンスである"""
        model, X = model_and_data
        result = classifier.classify(model, X)
        assert isinstance(result, ParameterClassification)

    def test_no_overlap(self, classifier, model_and_data):
        """8-2 [Property]: important と unimportant に重複がない"""
        model, X = model_and_data
        result = classifier.classify(model, X)
        overlap = set(result.important) & set(result.unimportant)
        assert len(overlap) == 0

    def test_all_dims_classified(self, classifier, model_and_data):
        """8-3 [Property]: 全次元が important または unimportant に分類される"""
        model, X = model_and_data
        N = X.shape[-1]
        result = classifier.classify(model, X)
        assert len(result.important) + len(result.unimportant) == N

    @pytest.mark.integration
    def test_important_dims_detected(self, classifier, model_and_data):
        """8-4 [Integration]: 次元 0,1 が important に分類される"""
        model, X = model_and_data
        result = classifier.classify(model, X)
        assert 0 in result.important
        assert 1 in result.important

    @pytest.mark.integration
    def test_unimportant_dims_detected(self, classifier, model_and_data):
        """8-5 [Integration]: 次元 2,3 が unimportant に分類される"""
        model, X = model_and_data
        result = classifier.classify(model, X)
        assert 2 in result.unimportant
        assert 3 in result.unimportant

    def test_very_small_eps_l_makes_all_unimportant(self, analyzer, model_and_data):
        """8-6: eps_l が非常に小さい (1e-10) と全次元が unimportant になる
        (全長さスケールが eps_l を上回るため ℓ_i < eps_l が常に False)
        """
        model, X = model_and_data
        N = X.shape[-1]
        config = ClassificationConfig(eps_l=1e-10, eps_e=0.0)
        clf = ParameterClassifier(analyzer=analyzer, config=config)
        result = clf.classify(model, X)
        assert len(result.unimportant) == N

    def test_very_large_eps_e_makes_all_unimportant(self, analyzer, model_and_data):
        """8-7: eps_e が非常に大きいと全次元が unimportant になる"""
        model, X = model_and_data
        N = X.shape[-1]
        config = ClassificationConfig(eps_l=0.0, eps_e=1e9)
        clf = ParameterClassifier(analyzer=analyzer, config=config)
        result = clf.classify(model, X)
        assert len(result.unimportant) == N
