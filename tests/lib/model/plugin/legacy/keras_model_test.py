#!/usr/bin/env python3
# pylint:disable=protected-access
""" Pytest unit tests for :mod:`lib.model.plugin.legacy.keras_model`"""

from __future__ import annotations

import io
import json
import pathlib
import typing as T
import zipfile

import h5py
import numpy as np
import pytest

from lib.model.plugin.legacy.keras_model import (  # pylint:disable=import-error
    LayerInfo,
    KerasConfigParser,
    KerasModel,
    LayerSorter,
)
_ENC_PREFIX = "layers.functional.layers.functional.layers."


# =============================================================================
# Helpers
# =============================================================================


def _empty_zip(path: str) -> None:
    """ Write an empty .keras zip so loading it raises ValueError (missing config) """
    with zipfile.ZipFile(path, "w"):
        pass


def _zip_without_weights(config: dict[str, T.Any], path: str) -> None:
    """ Write a .keras zip that has config.json but no weights file """
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("config.json", json.dumps(config))


def _layer_info(layer_name: str) -> LayerInfo:
    """ Construct a LayerInfo whose name satisfies the arch regexes (Conv2D / BN) """
    return LayerInfo(layer_name, layer_name, "Conv2D")


def _build_keras_fixture(tmp_path: pathlib.Path) -> str:
    """ Build a minimal ``.keras`` zip for end-to-end loading tests

    Creates a Sequential model with 3 Dense layers (24 flattened entries via
    :class:`KerasConfigParser.flatten`) and a matching ``model.weights.h5`` so
    :class:`KerasModel` can be loaded end-to-end without depending on external
    fixture files
    """
    keras_path = str(tmp_path / "test_model.keras")

    config = {"class_name": "Sequential",
              "name": "test_model",
              "inbound_nodes": [[{}]],
              "config": {"layers": [{"class_name": "Dense",
                                     "name": f"dense_{i}",
                                     "config": {"units": 64}}
                                    for i in range(3)]}}

    with zipfile.ZipFile(keras_path, "w") as zf:
        zf.writestr("config.json", json.dumps(config))

        # Build matching h5 weights: each Dense layer produces 2 entries
        # (dense, dense_1, dense_2, dense_3, dense_4, dense_5)
        buf = io.BytesIO()
        with h5py.File(buf, "w") as hf:
            grp = hf.create_group("layers")
            for i in range(6):
                name = f"dense_{i}" if i > 0 else "dense"
                layer_grp = grp.create_group(name)
                layer_grp.create_dataset("vars/0", data=np.random.randn(64, 64).astype(np.float32))
                layer_grp.create_dataset("vars/1", data=np.random.randn(64).astype(np.float32))
        zf.writestr("model.weights.h5", buf.getvalue())

    # State file with a simple architecture (no special reordering needed)
    state_path = keras_path.replace(".keras", "_state.json")
    with open(state_path, "w", encoding="utf-8") as sf:
        json.dump({"name": "iae", "config": {}}, sf)

    return keras_path


# =============================================================================
# LayerInfo
# =============================================================================


def test_layer_info_uses_defaults() -> None:
    """ LayerInfo applies default values for input_layers and input_shapes """
    info = LayerInfo("name", "weights", "InputLayer")
    assert info.layer_name == "name"
    assert info.weights_name == "weights"
    assert info.layer_type == "InputLayer"
    assert info.input_layers == []  # pylint:disable=use-implicit-booleaness-not-comparison
    assert info.input_shapes == []  # pylint:disable=use-implicit-booleaness-not-comparison


def test_layer_info_stores_all_fields() -> None:
    """ LayerInfo stores all provided fields including input_layers and input_shapes """
    info = LayerInfo("n", "w", "Conv2D", input_layers=["a"], input_shapes=[(10, 10)])
    assert info.input_layers == ["a"]
    assert info.input_shapes == [(10, 10)]


# =============================================================================
# LayerSorter
# =============================================================================


def test_sort_unknown_model_is_identity() -> None:
    """ LayerSorter returns layers unchanged for unknown architectures """
    layers = {"x": _layer_info("x"), "y": _layer_info("y")}
    state = {"name": "not_a_known_architecture", "config": {}}
    result = LayerSorter(state).sort(layers)
    assert list(result.keys()) == ["x", "y"]
    assert all(result[k] is layers[k] for k in layers)  # pylint:disable=consider-using-dict-items


def test_iae_reorders_graph_to_build_order() -> None:
    """ LayerSorter reorders IAE layers into build order (functional_2 before functional_1) """
    state = {"name": "iae", "config": {}}
    inp = {"layers.functional_2.b": _layer_info("b"),
           "layers.functional_1.a": _layer_info("a"),
           "z": _layer_info("both")}
    result = LayerSorter(state).sort(inp)
    assert list(result.keys()) == ["layers.functional_2.b", "layers.functional_1.a", "z"]


def test_inception_resnet_v2_keeps_unsuffixed_layers_in_place() -> None:
    """ InceptionResNetV2 leaves unsuffixed layers in their original order """
    state = {"name": "inception_resnet_v2", "config": {}}
    inp = {f"{_ENC_PREFIX}a": _layer_info("a"),
           f"{_ENC_PREFIX}b": _layer_info("b"),
           f"{_ENC_PREFIX}c": _layer_info("c")}
    result = LayerSorter(state).sort(inp)
    assert list(result.keys()) == [f"{_ENC_PREFIX}a", f"{_ENC_PREFIX}b", f"{_ENC_PREFIX}c"]


def test_inception_v3_remembers_mixed9_rename() -> None:
    """ InceptionV3 renames mixed9_0 to mixed9 and sorts accordingly """
    state = {"name": "inception_v3", "config": {}}
    inp = {f"{_ENC_PREFIX}mixed9_0": _layer_info("mixed9_0"),
           f"{_ENC_PREFIX}mixed9": _layer_info("mixed9"),
           f"{_ENC_PREFIX}mixed9_1": _layer_info("mixed9_1"),
           f"{_ENC_PREFIX}mixed9_2": _layer_info("mixed9_2")}
    result = LayerSorter(state).sort(inp)
    assert list(result.keys()) == [f"{_ENC_PREFIX}mixed9",
                                   f"{_ENC_PREFIX}mixed9_0",
                                   f"{_ENC_PREFIX}mixed9_1",
                                   f"{_ENC_PREFIX}mixed9_2"]


def test_nasnet_orders_blocks_then_type() -> None:
    """ NASNet sorts by block index first, then layer type within each block """
    state = {"name": "nasnet_large", "config": {}}
    inp = {f"{_ENC_PREFIX}reduction_bn_4_right1_6": _layer_info("reduction_bn_4_right1_6"),
           f"{_ENC_PREFIX}adjust_conv_5_left1_3": _layer_info("adjust_conv_5_left1_3"),
           f"{_ENC_PREFIX}separable_conv_2_left1_2": _layer_info("separable_conv_2_left1_2")}
    result = LayerSorter(state).sort(inp)
    assert list(result.keys()) == [f"{_ENC_PREFIX}separable_conv_2_left1_2",
                                   f"{_ENC_PREFIX}adjust_conv_5_left1_3",
                                   f"{_ENC_PREFIX}reduction_bn_4_right1_6"]


def test_xception_groups_skips_before_blocks() -> None:
    """ Xception sorts by group index, skipping layers before the first block """
    state = {"name": "xception", "config": {}}
    inp = {f"{_ENC_PREFIX}adjust": _layer_info("adjust"),
           f"{_ENC_PREFIX}block3": _layer_info("block3"),
           f"{_ENC_PREFIX}reduction": _layer_info("reduction"),
           f"{_ENC_PREFIX}block4": _layer_info("block4")}
    result = LayerSorter(state).sort(inp)
    assert list(result.keys()) == [f"{_ENC_PREFIX}adjust",
                                   f"{_ENC_PREFIX}reduction",
                                   f"{_ENC_PREFIX}block3",
                                   f"{_ENC_PREFIX}block4"]


def test_phaze_a_shares_intermediate_layers() -> None:
    """ Phaze-A with shared FC layers shares intermediate encoder layers """
    state = {"name": "phaze_a",
             "config": {"enc_architecture": "none", "shared_fc": "full", "split_fc": True}}
    inp = {"layers.functional_2.b": _layer_info("b"),
           "layers.functional_3.s": _layer_info("s"),
           "layers.functional_1.a": _layer_info("a")}
    result = LayerSorter(state).sort(inp)
    assert list(result.keys()) == ["layers.functional_1.a",
                                   "layers.functional_3.s",
                                   "layers.functional_2.b"]


def test_phaze_a_dispatches_encoder_architecture() -> None:
    """ Phaze-A dispatches to the correct encoder-specific reordering function """
    state = {
        "name": "phaze_a",
        "config": {"enc_architecture": "inception_v3", "shared_fc": "none", "split_fc": False},
    }
    inp = {f"{_ENC_PREFIX}mixed9_0": _layer_info("mixed9_0"),
           f"{_ENC_PREFIX}mixed9": _layer_info("mixed9"),
           f"{_ENC_PREFIX}mixed9_1": _layer_info("mixed9_1")}
    result = LayerSorter(state).sort(inp)
    assert list(result.keys()) == [f"{_ENC_PREFIX}mixed9_0",
                                   f"{_ENC_PREFIX}mixed9",
                                   f"{_ENC_PREFIX}mixed9_1"]


def test_phaze_a_noop_when_neither_branch_triggered() -> None:
    """ Phaze-A with unknown encoder and non-full shared FC returns layers unchanged """
    state = {
        "name": "phaze_a",
        "config": {"enc_architecture": "unknown_arch", "shared_fc": "none", "split_fc": False},
    }
    inp = {"layers.functional_2.b": _layer_info("b"),
           "layers.functional_1.a": _layer_info("a")}
    result = LayerSorter(state).sort(inp)
    assert list(result.keys()) == ["layers.functional_2.b", "layers.functional_1.a"]


def test_phaze_a_combined_encoder_and_shared_reordered() -> None:
    """ Phaze-A with encoder + shared FC applies both reorderings in sequence """
    state = {"name": "phaze_a",
             "config": {"enc_architecture": "inception_resnet_v2",
                        "shared_fc": "full",
                        "split_fc": True}}
    inp = {
        # Encoder layers (inception_resnet_v2: no rename, sorted by numeric suffix)
        f"{_ENC_PREFIX}mixed9_1": _layer_info("mixed9_1"),
        f"{_ENC_PREFIX}mixed9": _layer_info("mixed9"),
        f"{_ENC_PREFIX}mixed9_0": _layer_info("mixed9_0"),
        # Shared FC layers (reordered to functional_1, functional_3, functional_2)
        "layers.functional_2.b": _layer_info("b"),
        "layers.functional_3.s": _layer_info("s"),
        "layers.functional_1.a": _layer_info("a"),
    }
    result = LayerSorter(state).sort(inp)
    keys = list(result.keys())
    # Encoder: sorted by numeric suffix (0, 1) since no v3 rename
    assert keys[:3] == [f"{_ENC_PREFIX}mixed9", f"{_ENC_PREFIX}mixed9_0", f"{_ENC_PREFIX}mixed9_1"]
    # Shared: functional_1, functional_3, functional_2 order
    assert keys[3:] == ["layers.functional_1.a", "layers.functional_3.s", "layers.functional_2.b"]


# =============================================================================
# KerasConfigParser
# =============================================================================


def test_functional_maps_inputs_to_standardized_names() -> None:
    """ Functional layers map their input tensors to standardized names """
    config = {
        "class_name": "Model",
        "name": "m",
        "inbound_nodes": [],
        "config": {"input_layers": [["layers.input_layer"]],
                   "output_layers": [["layers.dense"]],
                   "layers": [{"class_name": "InputLayer", "name": "dense", "config": {}},
                              {"class_name": "Dense", "name": "dense", "config": {"units": 10}}]},
    }
    result = KerasConfigParser.flatten(config)
    assert list(result.keys()) == ["layers.input_layer", "layers.dense"]
    assert result["layers.input_layer"].weights_name == "layers.input_layer"
    assert result["layers.dense"].weights_name == "layers.dense"


def test_repeated_layers_get_numeric_suffix() -> None:
    """ Repeated layer names receive numeric suffixes to ensure uniqueness """
    config = {
        "class_name": "Model",
        "name": "m",
        "inbound_nodes": [],
        "config": {"input_layers": [["i1"]],
                   "output_layers": [["o1"]],
                   "layers": [{"class_name": "InputLayer", "name": "i1", "config": {}},
                              {"class_name": "Dense", "name": "dense", "config": {"units": 5}},
                              {"class_name": "Dense", "name": "dense", "config": {"units": 7}}]},
    }
    result = KerasConfigParser.flatten(config)
    assert list(result.keys()) == ["layers.input_layer", "layers.dense", "layers.dense_1"]


def test_nested_sub_model_flattens_under_model() -> None:
    """ Nested sub-models are flattened under their parent model with dot-separated names """
    config = {
        "class_name": "Model",
        "name": "outer",
        "inbound_nodes": [],
        "config": {
            "input_layers": [["ext_in"]],
            "output_layers": [["o1"]],
            "layers": [
                {"class_name": "InputLayer", "name": "ext_in", "config": {}},
                {
                    "class_name": "Model",
                    "name": "inner",
                    "inbound_nodes": [],
                    "config": {
                        "input_layers": [["ext_in"]],
                        "output_layers": [["inner_out"]],
                        "layers": [
                            {"class_name": "InputLayer", "name": "inner_in", "config": {}},
                            {"class_name": "Dense", "name": "dense", "config": {"units": 3}},
                        ],
                    },
                },
            ],
        },
    }
    result = KerasConfigParser.flatten(config)
    assert list(result.keys()) == ["layers.input_layer",
                                   "layers.model.layers.input_layer",
                                   "layers.model.layers.dense"]


def test_sequential_flatten_names_layers() -> None:
    """ Sequential models have their layers named sequentially with suffixes for duplicates """
    config = {"class_name": "Sequential",
              "name": "s",
              "inbound_nodes": [{"args": []}],
              "config": {
                  "layers": [{"class_name": "InputLayer", "name": "dense", "config": {}},
                             {"class_name": "Dense", "name": "dense", "config": {"units": 10}},
                             {"class_name": "Dense", "name": "dense", "config": {"units": 5}}]
              }}
    result = KerasConfigParser.flatten(config)
    assert "layers.dense" in result
    assert "layers.dense_1" in result


# =============================================================================
# KerasModel
# =============================================================================


def test_loads_keras_model_end_to_end(tmp_path: pathlib.Path) -> None:
    """ KerasModel loads a generated .keras file end-to-end and validates weight key structure """
    model_path = _build_keras_fixture(tmp_path)
    model = KerasModel(model_path)
    assert all(key.startswith("layers.") for key in model.weights)
    assert any(k.endswith(".vars.0") for k in model.weights)


def test_missing_config_file_raises_valueerror(tmp_path: pathlib.Path) -> None:
    """ KerasModel raises ValueError when the .keras zip lacks a config.json """
    model_path = str(tmp_path / "_empty.keras")
    _empty_zip(model_path)
    with pytest.raises(ValueError):
        KerasModel(model_path)


def test_missing_weights_file_raises_valueerror(tmp_path: pathlib.Path) -> None:
    """ KerasModel raises ValueError when the .keras zip has config but no weights """
    config = {"name": "phaze_a", "config": {}}
    model_path = str(tmp_path / "_no_weights.keras")
    _zip_without_weights(config, model_path)
    with pytest.raises(ValueError):
        KerasModel(model_path)


def test_absent_state_file_warns_and_empties_state(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """ KerasModel warns and returns empty dict when the state file is absent """
    model = KerasModel.__new__(KerasModel)
    model._model_path = str(tmp_path / "phaze_a.keras")  # no _state.json alongside
    with caplog.at_level("WARNING"):
        result = model._load_state_file()
    assert result == {}
    assert any("not found" in rec.message for rec in caplog.records)


def test_sort_weights_raises_on_unmapped_weights() -> None:
    """ _sort_weights asserts when weights remain after mapping all layers """
    model = KerasModel.__new__(KerasModel)
    model.state = {"name": "iae", "config": {}}
    model.layers = {"layers.dense": _layer_info("dense")}
    model.weights = {"layers.dense.weight": np.array([1.0]), "layers.orphan.bias": np.array([0.0])}
    with pytest.raises(AssertionError, match="Not all weights mapped"):
        model._sort_weights(model.weights)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
