import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from src.data_loaders.bus_bra_loader import BUSLoader
from src.data_loaders.ddti_loader import DDTILoader
from src.image_data.image_data import Image
from src.image_processing.collecting_features.feature_extractor import FeatureExtractor
from src.image_processing.preprocessing.cropper import Cropper
from src.image_processing.segmentation.algorithms.wave_algorithm.points_finder import WavePointFinder
from src.image_processing.segmentation.algorithms.wave_algorithm.wave_segmenter import WaveSegmenter
from src.image_processing.segmentation.image_segmenter import ImageSegmenter
from src.image_processing.segmentation.mask_selector import MaskSelector
from src.training_module import root_loger
from src.training_module.data_model_bridge.data_adapters import BaseDataAdapter, DataAdapterRegistry
from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.transform_pipeline import TransformPipeline
from src.training_module.model_core.base_models import BaseModel, SklearnModel
from src.training_module.model_core.model_registry import ModelRegistry
from src.training_module.model_core.tunner_module.hyperparameter_selection_system import (
    HyperParameterManger,
    TuningOrchestrator,
)
from src.utils.logger import CustomLogger


@dataclass
class Config:
    xml_path: str = "/home/kush/machine_learning/ThyroidScan/src/data_processing/xml_files"
    images_path: str = "/home/kush/machine_learning/ThyroidScan/src/data_processing/images"
    models: Optional[list[str]] = None
    transform_plugins: list[tuple[str, dict[str, Any]]] = field(
        default_factory=lambda: [
            ("fill_nan", {}),
            ("standard_scaler", {}),
            ("select_k_best", {"k": 100}),
            ("pca", {"n_components": 0.95}),
            ("k_means", {"n_clusters": 5}),
        ]
    )
    experiment_dir: str = "experiments"
    log_level: str = "INFO"
    target_metric: str = "f1"
    tune_params: bool = False
    is_update: bool = False


class Observer(ABC):
    @abstractmethod
    def update(self, event_type: str, data: dict[str, Any]) -> None:
        pass


class Subject:
    def __init__(self) -> None:
        self._observers: list[Observer] = []

    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def notify(self, event_type: str, data: dict[str, Any]) -> None:
        for observer in self._observers:
            observer.update(event_type, data)


class MetricsCollector(Observer):
    def __init__(self) -> None:
        self.metrics: list[Any] = []
        self.training_times: dict[str, float] = {}
        self.model_scores: dict[str, float] = {}

    def update(self, event_type: str, data: dict[str, Any]) -> None:
        timestamp = datetime.now().isoformat()
        metric_entry = {"timestamp": timestamp, "event_type": event_type, "data": data}
        self.metrics.append(metric_entry)

        if event_type == "training_completed":
            self.training_times[data["model_name"]] = data["training_time"]
            self.model_scores[data["model_name"]] = data.get("scores", {})

    def get_training_summary(self) -> dict[str, Any]:
        return {
            "training_times": self.training_times,
            "model_scores": self.model_scores,
            "total_events": len(self.metrics),
        }


class LoggingMonitor(Observer):
    def __init__(self, logger_name: str = "MLTraining", log_to_file: bool = True):
        from src.utils.logger import CustomLogger

        self.logger = CustomLogger(name=logger_name, log_to_file=log_to_file, log_dir="logs/ml_training").get_logger()

    def update(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type == "training_completed":
            self.logger.info(f"Model '{data['model_name']}' trained in {data['training_time']:.2f}s")
        elif event_type == "model_training_started":
            self.logger.info(f"Starting training for model: {data['model_name']}")
        elif event_type == "hyperparameter_tuning_started":
            self.logger.info(f"Starting hyperparameter tuning for: {data['model_name']}")
        elif event_type == "hyperparameter_tuning_completed":
            self.logger.info(f"Hyperparameter tuning completed for: {data['model_name']}")
        elif event_type == "batch_training_started":
            self.logger.info(f"Starting batch training for models: {data['models']}")
        elif event_type == "batch_training_completed":
            self.logger.info(f"Batch training completed for all models")
        elif event_type == "system_initialized":
            self.logger.info(f"Training system initialized - Train: {data['train_size']}, Test: {data['test_size']}")
        else:
            self.logger.info(f"Event: {event_type}, Data: {data}")


class ExperimentStrategy(ABC):
    @abstractmethod
    def run_experiment(self, training_system: "TrainingSystem", config: Config) -> dict[str, Any]:
        pass


class ModelComparisonExperiment(ExperimentStrategy):
    def run_experiment(self, training_system: "TrainingSystem", config: Config) -> dict[str, Any]:
        results = {}
        if config.models is None:
            config.models = list(ModelRegistry._models.keys())
        for model_name in config.models:
            start_time = time.time()
            training_system.fit_model(model_name, tune_params=config.tune_params, is_update=config.is_update)
            end_time = time.time()

            model = training_system.models[model_name]

            from src.training_module.data_model_bridge.data_adapters import DataAdapterRegistry

            data_adapter_type = model._data_adapter_type
            test_adapter = DataAdapterRegistry.create(
                data_adapter_type, images=training_system.test_images, is_bin_classification=training_system.is_binary
            )
            test_adapter.prepare()

            test_metrics = model.evaluate(test_adapter)

            results[model_name] = {
                "training_time": end_time - start_time,
                "model_params": getattr(model, "get_params", lambda: {})(),
                "test_metrics": test_metrics,
            }
        return results


class HyperparameterExperiment(ExperimentStrategy):
    def run_experiment(self, training_system: "TrainingSystem", config: Config) -> dict[str, Any]:
        results = {}
        param_variations = [
            [("fill_nan", {}), ("standard_scaler", {}), ("pca", {"n_components": 0.9})],
            [("fill_nan", {}), ("standard_scaler", {}), ("pca", {"n_components": 0.95})],
            [("fill_nan", {}), ("standard_scaler", {}), ("pca", {"n_components": 0.99})],
        ]

        base_model = config.models[0] if config.models else "svm"

        for i, params in enumerate(param_variations):
            exp_name = f"param_set_{i}"
            training_system_copy = TrainingSystem(
                training_system.train_images + training_system.test_images, params, training_system.is_binary
            )

            start_time = time.time()
            training_system_copy.fit_model(base_model, tune_params=True)
            end_time = time.time()

            model = training_system_copy.models[base_model]

            from src.training_module.data_model_bridge.data_adapters import DataAdapterRegistry

            data_adapter_type = model._data_adapter_type
            test_adapter = DataAdapterRegistry.create(
                data_adapter_type,
                images=training_system_copy.test_images,
                is_bin_classification=training_system_copy.is_binary,
            )
            test_adapter.prepare()

            test_metrics = model.evaluate(test_adapter)

            results[exp_name] = {
                "params": params,
                "training_time": end_time - start_time,
                "test_metrics": test_metrics,
                "model": model,
            }
        return results


class ExperimentManager:
    def __init__(self, config: Config):
        self.config = config
        self.experiments_dir = Path(config.experiment_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        self.experiment_history: list[Any] = []

        self.logger = CustomLogger(name="ExperimentManager", log_to_file=True, log_dir="logs/experiments").get_logger()

    def run_experiment(
        self, strategy: ExperimentStrategy, training_system: "TrainingSystem", experiment_name: str = None
    ) -> str:
        experiment_id = experiment_name or str(uuid.uuid4())
        experiment_dir = self.experiments_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)

        self.logger.info(f"Starting experiment: {experiment_id}")

        start_time = time.time()
        results = strategy.run_experiment(training_system, self.config)
        end_time = time.time()

        duration = end_time - start_time
        self.logger.info(f"Experiment {experiment_id} completed in {duration:.2f}s")

        experiment_data = {
            "id": experiment_id,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration": duration,
            "results": results,
            "config": self.config.__dict__,
        }

        results_file = experiment_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(experiment_data, f, indent=2, default=str)

        self.logger.info(f"Experiment results saved to: {results_file}")

        self.experiment_history.append(experiment_data)
        return experiment_id

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        experiment_file = self.experiments_dir / experiment_id / "results.json"
        if experiment_file.exists():
            with open(experiment_file, "r") as f:
                return json.load(f)
        return {}


class Visualizer:
    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager
        self.logger = CustomLogger(name="Visualizer", log_to_file=True, log_dir="logs/experiments").get_logger()
        plt.style.use("classic")

    def plot_model_comparison(self, experiment_id: str, target_metric: str = "f1", save_path: str = None) -> None:
        experiment = self.experiment_manager.get_experiment(experiment_id)
        if not experiment:
            self.logger.info(f"Experiment {experiment_id} not found")
            return

        models = []
        times = []
        metrics = []

        for model_name, data in experiment["results"].items():
            if "training_time" in data and "test_metrics" in data:
                models.append(model_name)
                times.append(data["training_time"])
                metric_value = data["test_metrics"].get(target_metric, 0)
                metrics.append(metric_value)

        if not models:
            self.logger.info("No model data found for visualization")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        bars1 = ax1.bar(models, times, color="skyblue", alpha=0.7)
        ax1.set_title("Training Time Comparison")
        ax1.set_xlabel("Models")
        ax1.set_ylabel("Time (seconds)")
        ax1.tick_params(axis="x", rotation=45)

        for bar, time_val in zip(bars1, times):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(times) * 0.01,
                f"{time_val:.2f}s",
                ha="center",
                va="bottom",
            )

        bars2 = ax2.bar(models, metrics, color="lightgreen", alpha=0.7)
        ax2.set_title(f"{target_metric.upper()} Score Comparison")
        ax2.set_xlabel("Models")
        ax2.set_ylabel(f"{target_metric.upper()} Score")
        ax2.tick_params(axis="x", rotation=45)

        for bar, metric_val in zip(bars2, metrics):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(metrics) * 0.01,
                f"{metric_val:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        if save_path:
            experiment_dir = self.experiment_manager.experiments_dir / experiment_id
            full_save_path = experiment_dir / f"{save_path}_comparison.png"
            plt.savefig(full_save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Plots saved to: {full_save_path}")

        plt.show()

    def plot_hyperparameter_comparison(
        self, experiment_id: str, target_metric: str = "f1", save_path: str = None
    ) -> None:
        experiment = self.experiment_manager.get_experiment(experiment_id)
        if not experiment:
            print(f"Experiment {experiment_id} not found")
            return

        param_sets = []
        times = []
        metrics = []

        for param_name, data in experiment["results"].items():
            if "training_time" in data and "test_metrics" in data:
                param_sets.append(param_name)
                times.append(data["training_time"])
                metric_value = data["test_metrics"].get(target_metric, 0)
                metrics.append(metric_value)

        if not param_sets:
            self.logger.info("No parameter data found for visualization")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        bars1 = ax1.bar(param_sets, times, color="coral", alpha=0.7)
        ax1.set_title("Training Time by Parameter Set")
        ax1.set_xlabel("Parameter Sets")
        ax1.set_ylabel("Time (seconds)")
        ax1.tick_params(axis="x", rotation=45)

        for bar, time_val in zip(bars1, times):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(times) * 0.01,
                f"{time_val:.2f}s",
                ha="center",
                va="bottom",
            )

        bars2 = ax2.bar(param_sets, metrics, color="gold", alpha=0.7)
        ax2.set_title(f"{target_metric.upper()} Score by Parameter Set")
        ax2.set_xlabel("Parameter Sets")
        ax2.set_ylabel(f"{target_metric.upper()} Score")
        ax2.tick_params(axis="x", rotation=45)

        for bar, metric_val in zip(bars2, metrics):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(metrics) * 0.01,
                f"{metric_val:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        if save_path:
            experiment_dir = self.experiment_manager.experiments_dir / experiment_id
            full_save_path = experiment_dir / f"{save_path}_hyperparams.png"
            plt.savefig(full_save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Plots saved to: {full_save_path}")

        plt.show()


class ImageProcessor:
    def __init__(self, xml_path: str, images_path: str):
        self.xml_path = xml_path
        self.images_path = images_path

    def get_images(self) -> list[Image]:
        loader = DDTILoader(self.xml_path, self.images_path)
        image_list = loader.load_dataset()

        pipeline = Pipeline(
            [
                ("cropper", Cropper()),
                ("segmenter", ImageSegmenter(WaveSegmenter(18, 4), WavePointFinder(32, 83, 36))),
                ("selector", MaskSelector(10)),
                ("extractor", FeatureExtractor()),
            ]
        )

        return pipeline.fit_transform(image_list)


class TrainingSystem(Subject):
    def __init__(self, images: list[Image], plugins: list[tuple[str, dict[str, Any]]], is_binary: bool) -> None:
        super().__init__()
        self.DataAdapterRegistry = DataAdapterRegistry
        self.HyperParameterManger = HyperParameterManger
        self.TuningOrchestrator = TuningOrchestrator

        self.transform_pipeline = TransformPipeline(plugins)
        self.is_binary = is_binary
        self.train_images, self.test_images = self.split_image(images)
        self.transform_image()
        self.models: dict[str, Any] = {}

        self.notify(
            "system_initialized",
            {"train_size": len(self.train_images), "test_size": len(self.test_images), "is_binary": is_binary},
        )

    @staticmethod
    def split_image(images: list[Image]) -> tuple[list[Image], list[Image]]:
        from src.training_module.feature_engineering_layer.splitter import Splitter

        splitter = Splitter(images)
        return splitter.get_split_data()

    def transform_image(self, is_update=True) -> None:
        self.notify("transformation_started", {})
        self.transform_pipeline.fit_transform(self.train_images)
        if is_update:
            self.transform_pipeline.save_config()
            self.transform_pipeline.save_state()
        self.transform_pipeline.transform(self.test_images)
        self.notify("transformation_completed", {})

    def fit_model(self, model_name: str, tune_params: bool = True, is_update: bool = True) -> None:
        start_time = time.time()

        self.notify("model_training_started", {"model_name": model_name})

        model = BaseModel.get_model(model_name=model_name, is_binary=True)

        data_adapter_type = model._data_adapter_type
        train_adapter = self.DataAdapterRegistry.create(
            data_adapter_type, images=self.train_images, is_bin_classification=self.is_binary
        )
        train_adapter.prepare()
        test_adapter = self.DataAdapterRegistry.create(
            data_adapter_type, images=self.test_images, is_bin_classification=self.is_binary
        )
        test_adapter.prepare()

        if tune_params:
            tuner_type = "sklearn_gridsearch" if isinstance(model, SklearnModel) else "pytorch_optuna"
            self.tune_hyperparameters(model, train_adapter, tuner_type, is_update)

        model.fit(train_adapter, test_adapter)
        self.models[model_name] = model

        end_time = time.time()
        training_time = end_time - start_time

        self.notify(
            "training_completed",
            {"model_name": model_name, "training_time": training_time, "model_type": type(model).__name__},
        )

    def fit_all_models(self, models: list[str] = None, tune_params: bool = True, is_update: bool = True) -> None:
        from src.training_module.model_core.model_registry import ModelRegistry

        if models is None:
            models = list(ModelRegistry._models.keys())

        self.notify("batch_training_started", {"models": models})

        for name in models:
            self.fit_model(name, tune_params, is_update)

        self.notify("batch_training_completed", {"models": models})

    def tune_hyperparameters(
        self, model: BaseModel, data_adapter: BaseDataAdapter, tuner_type: str, is_update: bool
    ) -> None:
        hp_manager = self.HyperParameterManger("../config/hyperparameters")
        tune_orch = self.TuningOrchestrator(hp_manager)

        self.notify("hyperparameter_tuning_started", {"model_name": model.name})

        best_par = tune_orch.tune_model(model.name, tuner_type, data_adapter, is_update)
        print(best_par)

        if is_update:
            model.save()

        self.notify("hyperparameter_tuning_completed", {"model_name": model.name, "best_params": best_par})


class MLSystemFacade:
    def __init__(self, config: Config):
        TransformRegistry.discover_plugins()
        ModelRegistry.discover_models()
        DataAdapterRegistry.discover_adapters()
        if config.models is None:
            config.models = list(ModelRegistry._models.keys())
        self.config = config
        self.image_processor = ImageProcessor(config.xml_path, config.images_path)
        self.experiment_manager = ExperimentManager(config)
        self.visualizer = Visualizer(self.experiment_manager)

        from src.utils.logger import CustomLogger

        self.logger = CustomLogger(name="MLSystemFacade", log_to_file=True, log_dir="logs/ml_system").get_logger()

        self.metrics_collector = MetricsCollector()
        self.logging_monitor = LoggingMonitor("MLTraining", log_to_file=True)

        self.logger.info("ML System Facade initialized")

    def setup_training_system(self, images: list[Image] = None) -> TrainingSystem:
        if images is None:
            self.logger.info("Loading and processing images...")
            images = self.image_processor.get_images()
            self.logger.info(f"Loaded {len(images)} images")

        training_system = TrainingSystem(images=images, plugins=self.config.transform_plugins, is_binary=True)

        training_system.attach(self.metrics_collector)
        training_system.attach(self.logging_monitor)

        return training_system

    def train_single_model(self, model_name: str) -> str:
        self.logger.info(f"Training single model: {model_name}")
        training_system = self.setup_training_system()
        training_system.fit_model(model_name, tune_params=self.config.tune_params, is_update=self.config.is_update)
        self.logger.info(f"Single model training completed: {model_name}")
        return model_name

    def train_all_models(self) -> list[str]:
        self.logger.info("Training all models")
        training_system = self.setup_training_system()
        training_system.fit_all_models(
            self.config.models, tune_params=self.config.tune_params, is_update=self.config.is_update
        )
        self.logger.info("All models training completed")
        return self.config.models if self.config.models is not None else []

    def run_model_comparison_experiment(self, experiment_name: str = None, save_plots: bool = True) -> str:
        self.logger.info("Starting model comparison experiment")
        training_system = self.setup_training_system()
        strategy = ModelComparisonExperiment()
        exp_id = self.experiment_manager.run_experiment(strategy, training_system, experiment_name)

        if save_plots:
            save_path = "model_comparison" if not experiment_name else experiment_name
            self.visualizer.plot_model_comparison(exp_id, self.config.target_metric, save_path)

        self.logger.info(f"Model comparison experiment completed: {exp_id}")
        return exp_id

    def get_training_summary(self) -> dict[str, Any]:
        return self.metrics_collector.get_training_summary()


def main() -> None:
    config = Config(
        xml_path="/home/kush/machine_learning/ThyroidScan/src/data_processing/xml_files",
        images_path="/home/kush/machine_learning/ThyroidScan/src/data_processing/images",
        models=["dense_net_image"],
        experiment_dir="experiments",
        target_metric="f1",
        tune_params=False,
        is_update=True,
    )

    ml_system = MLSystemFacade(config)

    # Use-case 1: train single model
    model_name = ml_system.train_single_model("svm")
    root_loger.info(f"Model is trained: {model_name}")

    # Use-case 2: train all models
    models = ml_system.train_all_models()
    root_loger.info(f"Models are trained: {models}")

    # Use-case 3: model comparison
    exp_id = ml_system.run_model_comparison_experiment("model_comparison_v2", save_plots=True)
    root_loger.info(f"The model comparison experiment is completed: {exp_id}")

    summary = ml_system.get_training_summary()
    root_loger.info("\nTraining Summary:", summary)
