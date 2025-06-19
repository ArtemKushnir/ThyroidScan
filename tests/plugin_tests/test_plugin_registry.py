import importlib
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin


class MockTransformPlugin(TransformPlugin):
    """Mock plugin for testing"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def _fit(self, df):
        pass

    def _transform(self, df):
        return df

    def _get_state(self):
        return {}

    def _set_state(self, state):
        pass


class AnotherMockPlugin(TransformPlugin):
    """Another mock plugin for testing"""

    def __init__(self, param1=None, param2=None):
        self.param1 = param1
        self.param2 = param2

    def _fit(self, df):
        pass

    def _transform(self, df):
        return df

    def _get_state(self):
        return {}

    def _set_state(self, state):
        pass


class TestTransformRegistry:

    def setup_method(self):
        """Reset registry before each test"""
        TransformRegistry._plugins = {}
        TransformRegistry._initialized = False

    def test_register_plugin_decorator(self):
        """Test register_plugin decorator functionality"""

        @TransformRegistry.register_plugin("test_plugin")
        class TestPlugin(TransformPlugin):
            pass

        assert "test_plugin" in TransformRegistry._plugins
        assert TransformRegistry._plugins["test_plugin"] == TestPlugin

    def test_register_plugin_duplicate_name_raises_error(self):
        """Test that registering duplicate plugin name raises error"""

        @TransformRegistry.register_plugin("duplicate_plugin")
        class FirstPlugin(TransformPlugin):
            pass

        with pytest.raises(ValueError, match="Plugin 'duplicate_plugin' already registered"):

            @TransformRegistry.register_plugin("duplicate_plugin")
            class SecondPlugin(TransformPlugin):
                pass

    def test_get_plugin_success(self):
        """Test successful plugin retrieval"""
        TransformRegistry._plugins["mock_plugin"] = MockTransformPlugin

        plugin = TransformRegistry.get_plugin("mock_plugin", "arg1", "arg2", param="value")

        assert isinstance(plugin, MockTransformPlugin)
        assert plugin.args == ("arg1", "arg2")
        assert plugin.kwargs == {"param": "value"}

    def test_get_plugin_not_registered_raises_error(self):
        """Test get_plugin raises error for unregistered plugin"""
        with pytest.raises(ValueError, match="Plugin nonexistent_plugin is not registered"):
            TransformRegistry.get_plugin("nonexistent_plugin")

    @given(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc"))))
    def test_register_and_get_plugin_hypothesis(self, plugin_name):
        """Test register and get plugin with hypothesis-generated names"""
        # Skip if plugin already exists (due to hypothesis generating same name)
        if plugin_name in TransformRegistry._plugins:
            return

        @TransformRegistry.register_plugin(plugin_name)
        class HypothesisPlugin(MockTransformPlugin):
            pass

        plugin = TransformRegistry.get_plugin(plugin_name)
        assert isinstance(plugin, HypothesisPlugin)

    @patch("importlib.import_module")
    @patch("os.path.dirname")
    @patch("inspect.getfile")
    @patch("pkgutil.iter_modules")
    def test_discover_plugins_success(self, mock_iter_modules, mock_getfile, mock_dirname, mock_import):
        """Test successful plugin discovery"""
        mock_package = Mock()
        mock_import.return_value = mock_package
        mock_getfile.return_value = "/fake/path/plugins/__init__.py"
        mock_dirname.return_value = "/fake/path/plugins"
        mock_iter_modules.return_value = [
            (None, "plugin1", False),
            (None, "plugin2", True),
            (None, "__init__", False),
            (None, "transform_plugin", False),
        ]

        TransformRegistry.discover_plugins()

        assert TransformRegistry._initialized
        expected_calls = [
            "src.training_module.feature_engineering_layer.plugins",
            "src.training_module.feature_engineering_layer.plugins.plugin1",
            "src.training_module.feature_engineering_layer.plugins.plugin2",
        ]

        actual_calls = [call[0][0] for call in mock_import.call_args_list]
        for expected_call in expected_calls:
            assert expected_call in actual_calls

    @patch("importlib.import_module")
    def test_discover_plugins_import_error(self, mock_import):
        """Test discover_plugins handles ImportError gracefully"""
        mock_import.side_effect = ImportError("Module not found")

        TransformRegistry.discover_plugins()

        assert not TransformRegistry._initialized

    def test_discover_plugins_already_initialized(self):
        """Test that discover_plugins does nothing if already initialized"""
        TransformRegistry._initialized = True

        with patch("importlib.import_module") as mock_import:
            TransformRegistry.discover_plugins()
            mock_import.assert_not_called()

    @patch("importlib.import_module")
    def test_import_submodules_no_path(self, mock_import):
        """Test _import_submodules with package that has no __path__"""
        mock_package = Mock()
        del mock_package.__path__  # Remove __path__ attribute
        mock_import.return_value = mock_package

        TransformRegistry._import_submodules("test.package")

        assert mock_import.call_count == 1

    def test_multiple_plugins_registration(self):
        """Test registering multiple plugins"""

        @TransformRegistry.register_plugin("plugin1")
        class Plugin1(MockTransformPlugin):
            pass

        @TransformRegistry.register_plugin("plugin2")
        class Plugin2(AnotherMockPlugin):
            pass

        assert len(TransformRegistry._plugins) == 2
        assert TransformRegistry._plugins["plugin1"] == Plugin1
        assert TransformRegistry._plugins["plugin2"] == Plugin2

        p1 = TransformRegistry.get_plugin("plugin1")
        p2 = TransformRegistry.get_plugin("plugin2", param1="test", param2=42)

        assert isinstance(p1, Plugin1)
        assert isinstance(p2, Plugin2)
        assert p2.param1 == "test"
        assert p2.param2 == 42

    @given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5, unique=True))
    def test_register_multiple_plugins_hypothesis(self, plugin_names):
        """Test registering multiple plugins with hypothesis-generated names"""
        # Clean names to be valid Python identifiers
        clean_names = []
        for name in plugin_names:
            clean_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
            if clean_name and not clean_name[0].isdigit():
                clean_names.append(clean_name)

        if not clean_names:
            return

        for name in clean_names:
            if name not in TransformRegistry._plugins:

                @TransformRegistry.register_plugin(name)
                class DynamicPlugin(MockTransformPlugin):
                    pass

        for name in clean_names:
            if name in TransformRegistry._plugins:
                plugin = TransformRegistry.get_plugin(name)
                assert isinstance(plugin, MockTransformPlugin)

    def test_plugin_args_and_kwargs_passing(self):
        """Test that plugin initialization args and kwargs are passed correctly"""
        TransformRegistry._plugins["test_plugin"] = AnotherMockPlugin

        plugin = TransformRegistry.get_plugin("test_plugin", param1="hello", param2=123)

        assert plugin.param1 == "hello"
        assert plugin.param2 == 123

    @given(st.integers(min_value=1, max_value=100), st.text(min_size=1, max_size=20))
    def test_plugin_with_hypothesis_params(self, int_param, str_param):
        """Test plugin creation with hypothesis-generated parameters"""
        TransformRegistry._plugins["param_plugin"] = AnotherMockPlugin

        plugin = TransformRegistry.get_plugin("param_plugin", param1=str_param, param2=int_param)

        assert plugin.param1 == str_param
        assert plugin.param2 == int_param
