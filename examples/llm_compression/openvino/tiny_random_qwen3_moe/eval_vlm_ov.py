"""Run lm-eval on an OpenVINO IR exported with OVModelForVisualCausalLM.

`lm_eval --model openvino` instantiates OVModelForCausalLM, which feeds `input_ids`.
A VLM export splits the token embedding into its own IR, so the language graph takes
`inputs_embeds` and the CLI fails with:

    Port for tensor name input_ids was not found.

OVModelForVisualCausalLM.forward() does take `input_ids` (it runs the embedding IR
itself), and lm_eval's HFLM accepts a pre-instantiated model, so we build the model
here and hand the object to OptimumLM instead of going through the CLI.

Note: `dtype` is a no-op for the OpenVINO backend (the CLI drops it too -- an IR's
precision is fixed at export time).
"""

from lm_eval import simple_evaluate
from lm_eval.models.optimum_lm import OptimumLM
from optimum.intel.openvino import OVModelForVisualCausalLM


MODEL_PATH = "/data/anazir/models/qwen3.5-moe-with-groupedmm"
# MODEL_PATH = "/data/anazir/models/qwen3.5-moe-with-bmm"

TASKS = ["gsm8k"]
DEVICE = "cpu"  # the OV GPU plugin has no kernels for the A100s on this box
BATCH_SIZE = 1
LIMIT = None  # samples per task; None runs all of them
NUM_FEWSHOT = None  # None keeps the task default (gsm8k = 5)
APPLY_CHAT_TEMPLATE = True
FEWSHOT_AS_MULTITURN = True
TRUST_REMOTE_CODE = True
ENABLE_THINKING = False


def load_model(model_path: str) -> OVModelForVisualCausalLM:
    """
    Load a VLM-exported OpenVINO IR and make it usable as an lm-eval model.

    :param model_path: Path to the exported OV IR directory.
    :return: The loaded model, patched so lm-eval can introspect it.
    """
    model = OVModelForVisualCausalLM.from_pretrained(
        model_path,
        export=False,
        trust_remote_code=TRUST_REMOTE_CODE,
        device=DEVICE.upper(),
    )

    # The OV wrapper isinstance-checks as an nn.Module but never runs nn.Module.__init__,
    # so anything HFLM does that walks the module tree (tie_weights, num_parameters, ...)
    # dies on a missing attribute. Give it empty internals: the traversals then succeed
    # and yield nothing, which is correct -- the weights live in the compiled OV graph.
    for attr, empty in (
        ("_parameters", {}),
        ("_buffers", {}),
        ("_modules", {}),
        ("_non_persistent_buffers_set", set()),
    ):
        if not hasattr(model, attr):
            object.__setattr__(model, attr, empty)

    return model


def main() -> None:
    lm = OptimumLM(
        pretrained=load_model(MODEL_PATH),  # object, not a path -> HFLM skips _create_model
        # required: the OV wrapper has no .name_or_path for tokenizer autodetection
        tokenizer=MODEL_PATH,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        trust_remote_code=TRUST_REMOTE_CODE,
        enable_thinking=ENABLE_THINKING,
    )

    results = simple_evaluate(
        lm,
        tasks=TASKS,
        num_fewshot=NUM_FEWSHOT,
        limit=LIMIT,
        apply_chat_template=APPLY_CHAT_TEMPLATE,
        fewshot_as_multiturn=FEWSHOT_AS_MULTITURN,
        log_samples=False,
    )

    for task, metrics in results["results"].items():
        print(f"\n=== {task} ===")
        for name, value in metrics.items():
            if name != "alias":
                print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
