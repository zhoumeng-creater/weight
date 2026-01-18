import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from config import ConfigManager
from main import check_plateau_and_suggest, _extract_weight_series
from solution_generator import SolutionGenerator, Solution
from visualization import DataTracker


def _make_tracker(weights):
    tracker = DataTracker()
    for week, weight in enumerate(weights):
        tracker.add_record(week, weight=weight)
    return tracker


def _make_results():
    generator = SolutionGenerator()
    vector = generator.generate_random_solution(2000)
    solution = Solution(vector)
    return {
        "best_solution": solution,
        "best_solutions_history": [solution],
    }


class TestPlateauDetection(unittest.TestCase):
    def setUp(self):
        self.config = ConfigManager()
        self.config.experiment.plateau_detection_weeks = 2
        self.config.experiment.plateau_detection_threshold = 0.2
        self.config.experiment.breakthrough_threshold = 0.5

    def test_extract_weight_series(self):
        tracker = _make_tracker([100.0, 99.5, 99.0])
        weights = _extract_weight_series(tracker)
        self.assertEqual(weights, [100.0, 99.5, 99.0])

    @mock.patch("main.SolutionGenerator.generate_plateau_breaking_solutions", return_value=[])
    def test_plateau_triggers_suggestions(self, mock_generate):
        tracker = _make_tracker([100.0, 99.9, 99.85, 99.8])
        results = _make_results()
        with redirect_stdout(io.StringIO()):
            check_plateau_and_suggest(results, self.config, tracker)
        mock_generate.assert_called_once()

    @mock.patch("main.SolutionGenerator.generate_plateau_breaking_solutions", return_value=[])
    def test_no_plateau_no_suggestions(self, mock_generate):
        tracker = _make_tracker([100.0, 99.0, 98.0])
        results = _make_results()
        with redirect_stdout(io.StringIO()):
            check_plateau_and_suggest(results, self.config, tracker)
        mock_generate.assert_not_called()

    @mock.patch("main.SolutionGenerator.generate_plateau_breaking_solutions", return_value=[])
    def test_breakthrough_detected_no_suggestions(self, mock_generate):
        tracker = _make_tracker([100.0, 99.9, 99.85, 99.8, 99.1])
        results = _make_results()
        with redirect_stdout(io.StringIO()):
            check_plateau_and_suggest(results, self.config, tracker)
        mock_generate.assert_not_called()
