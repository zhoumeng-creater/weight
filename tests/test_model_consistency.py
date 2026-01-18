import unittest
from unittest import mock

import numpy as np

from config import ConfigManager
from de_algorithm import DifferentialEvolution
from experiment_runner import EnhancedExperimentRunner
from metabolic_model import PersonProfile, MetabolicModel, AdvancedMetabolicModel
from solution_generator import SolutionGenerator, Solution


def _make_subject():
    return PersonProfile(
        age=30,
        gender="male",
        height=175,
        weight=85,
        body_fat_percentage=25,
        activity_level=1.4,
        weeks_on_diet=0,
    )


class DummyDE:
    instances = []

    def __init__(self, person, config, metabolic_model=None):
        self.person = person
        self.config = config
        self.metabolic_model = metabolic_model
        DummyDE.instances.append(self)

    def optimize(self):
        if self.metabolic_model is None:
            raise AssertionError("metabolic_model is required for this test")
        generator = SolutionGenerator(config=self.config)
        bmr = self.metabolic_model.calculate_bmr(self.person)
        tdee = bmr * self.person.activity_level
        vector = generator.generate_from_template("balanced", tdee)
        solution = Solution(vector)
        results = {
            "final_person_state": self.person,
            "best_solutions_history": [solution],
            "initial_weight": self.person.initial_weight,
            "total_iterations": 1,
        }
        return solution, results


class TestModelSelection(unittest.TestCase):
    def test_de_selects_advanced_model_from_config(self):
        config = ConfigManager()
        config.metabolic.use_advanced_model = True
        de = DifferentialEvolution(_make_subject(), config)
        self.assertIs(type(de.metabolic_model), AdvancedMetabolicModel)

    def test_de_selects_basic_model_from_config(self):
        config = ConfigManager()
        config.metabolic.use_advanced_model = False
        de = DifferentialEvolution(_make_subject(), config)
        self.assertIs(type(de.metabolic_model), MetabolicModel)

    def test_runner_passes_primary_model_to_de(self):
        config = ConfigManager()
        config.experiment.de_reoptimize_interval_weeks = 0
        runner = EnhancedExperimentRunner(config)
        subject = _make_subject()

        DummyDE.instances = []
        with mock.patch("experiment_runner.DifferentialEvolution", DummyDE):
            runner._de_optimized_method(subject, weeks=1)

        self.assertTrue(DummyDE.instances)
        self.assertIs(DummyDE.instances[0].metabolic_model, runner.primary_model)

    def test_runner_passes_primary_model_to_de_with_reoptimize(self):
        config = ConfigManager()
        config.experiment.de_reoptimize_interval_weeks = 1
        runner = EnhancedExperimentRunner(config)
        subject = _make_subject()

        DummyDE.instances = []
        with mock.patch("experiment_runner.DifferentialEvolution", DummyDE):
            runner._de_optimized_method(subject, weeks=2)

        self.assertTrue(DummyDE.instances)
        for instance in DummyDE.instances:
            self.assertIs(instance.metabolic_model, runner.primary_model)


class TestAblationBounds(unittest.TestCase):
    def test_ablated_apply_bounds_clips(self):
        config = ConfigManager()
        runner = EnhancedExperimentRunner(config)
        subject = _make_subject()
        optimizer = runner._create_ablated_optimizer(
            subject, config, "sleep_optimization"
        )
        constraints = optimizer.solution_generator.constraints

        vector = np.array([99999, -1.0, 2.0, -0.5, 99, 999, -5, 0], dtype=float)
        bounded = optimizer._apply_bounds(vector)

        self.assertGreaterEqual(bounded[0], constraints.min_calories)
        self.assertLessEqual(bounded[0], constraints.max_calories)
        self.assertGreaterEqual(bounded[1], constraints.min_protein_ratio)
        self.assertLessEqual(bounded[1], constraints.max_protein_ratio)
        self.assertGreaterEqual(bounded[2], constraints.min_carb_ratio)
        self.assertLessEqual(bounded[2], constraints.max_carb_ratio)
        self.assertGreaterEqual(bounded[3], constraints.min_fat_ratio)
        self.assertLessEqual(bounded[3], constraints.max_fat_ratio)
        self.assertGreaterEqual(bounded[4], constraints.min_cardio_freq)
        self.assertLessEqual(bounded[4], constraints.max_cardio_freq)
        self.assertIn(int(bounded[5]), constraints.cardio_duration_options)
        self.assertGreaterEqual(bounded[6], constraints.min_strength_freq)
        self.assertLessEqual(bounded[6], constraints.max_strength_freq)
        self.assertGreaterEqual(bounded[7], constraints.min_sleep_hours)
        self.assertLessEqual(bounded[7], constraints.max_sleep_hours)
