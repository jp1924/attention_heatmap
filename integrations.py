from transformers.integrations import WandbCallback
from transformers.utils import logging, is_torch_tpu_available
from pathlib import Path
import os
from typing import List, Dict, Optional, Union
from tqdm import tqdm

logger = logging.get_logger(__name__)

class CustomWansbCallBack(WandbCallback):
    def __init__(self):
        super().__init__()
        self.default_setting = {}

    def setup(self, args, state, model, **kwargs):
        if self._wandb is None:
            return

        if state.is_world_process_zero:
            self._initialized = True

            logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}

            init_args: dict = self.__get_init_argument()
            trial_name = state.trial_name  # hyperparameter_searching을 진행할 떄 사용한다.
            if trial_name is not None:
                run_name = trial_name
                init_args["group"] = args.run_name
            else:
                run_name = args.run_name

            if self._wandb.run is None:
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    name=run_name,
                    **init_args,
                )
                code_dir = os.getenv("CODE_DIR", None)
                if code_dir and os.path.isdir(code_dir):
                    self._wandb.log_code(code_dir)
                # inti은 wandb처음 시작할 때 사용하는 것이기 때문에 일부러 if문을 사용함.
            # add config parameters (run may have been created manually)
            self._wandb.config.update(combined_dict, allow_val_change=True)
            # 재 시작 할 때도 있기 때문에 config는 외부에 빠져나와 있는것.

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model,
                    log=os.getenv("WANDB_WATCH", "gradients"),
                    log_freq=max(100, args.logging_steps),
                )

    def __get_init_argument(self) -> dict:
        init_args = dict()

        init_args["entity"]: Optional[str] = os.getenv("INIT_ENTITY", None)
        init_args["reinit"]: bool = os.getenv("INIT_REINIT", None)
        init_args["tags"]: Optional["sequence"] = os.getenv("INIT_", None)
        init_args["group"]: Optional[str] = os.getenv("INIT_GROUPS", None)
        init_args["notes"]: Optional[str] = os.getenv("INIT_NOTES", None)
        init_args["magic"]: Union[dict, str, bool] = os.getenv("INIT_MAGICS", None)
        init_args["anonymous"]: Optional[str] = os.getenv("INIT_", None)
        init_args["mode"]: Optional[str] = os.getenv("INIT_", None)
        init_args["resume"]: Optional[Union[bool, str]] = os.getenv("INIT_RESUME", None)
        init_args["force"]: Optional[bool] = os.getenv("INIT_FORCE", None)
        init_args["sync_tensorboard"] = os.getenv("INIT_TENSORBOARD", None)
        init_args["monitor_gym"]: bool = True if os.getenv("INIT_MONITORGYM", None) == "true" else False
        init_args["save_code"]: bool = True if os.getenv("INIT_SAVECODE", None) == "true" else False
        init_args["id"]: str = os.getenv("INIT_ID", None)
        init_args["settings"]: Union["settings", Dict[str, any], None] = os.getenv("INIT_SETTINGS", None)

        return init_args

def load_project_files(dir_path:os.PathLike) -> List[os.PathLike]:
    """
        wandb에 업로드할 파일을 pathlib의 Path를 이용해 읽어오는 함수 입니다.
    """
    project_path = Path(dir_path)
    ignore = [".git", ".pack", ".jpg", ".png"]
    file_list = list()
    for project_file in project_path.rglob("*"):
        ext_check = project_file.suffix not in ignore
        name_check = project_file.stem not in ignore

        if ext_check and name_check:
            file_list.append(project_file.as_posix())

    return file_list

def code_upload_wandb(dir_path: os.PathLike, run) -> None:
    """
            불러온 파일을 wandb에 업로드하는 함수입니다.
    """
    project_files = load_project_files(dir_path)

    for file_path in tqdm(project_files):
        run.upload_file(file_path)
