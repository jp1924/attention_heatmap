from transformers import Trainer
import torch
import wandb
from typing import Dict, Union, Any
import torch.nn as nn
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available
import os

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

if is_apex_available():
    from apex import amp


class HeatmapTrainer(Trainer):
    heatmap_print: bool = False

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """Heatmap을 표시하기 위한 Trainer 입니다.
            heatmap을 print하기 위해선 heatmap_print을 True로 설정해야 합니다.
            heatmap은 deprecated됩니다! 그렇기 때문에 구 버전의 wandb를 사용해야 합니다.
        Args:
            model (nn.Module): Trainer로 부터 model을 전달받습니다.
            inputs (Dict[str, Union[torch.Tensor, Any]]): collator에 처리가 끝난 데이터를 전달받습니다.

        Returns:
            torch.Tensor: loss값을 Trainer로 반환합니다.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

            if self.heatmap_print and os.getenv("WANDB_DISABLED") != "true":
                self.print_heatmap(loss)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def print_heatmap(self, loss: torch.Tensor) -> None:
        """Heatmap을 wandb에 출력합니다.

        Args:
            loss (torch.Tensor): 모델의 device를 설정하기 위해 loss를 입력합니다.
        """

        step_caculate = self.state.global_step % self.args.eval_steps == 0
        processer_check = self.state.is_world_process_zero

        if step_caculate and processer_check:

            label_sentence = "발연기 도저히 못보겠다 진짜 이렇게 연기를 못할거라곤 상상도 못했네"
            cls_token = self.tokenizer._cls_token
            sep_token = self.tokenizer._sep_token

            with torch.no_grad():
                test_sentence = self.tokenizer(label_sentence, return_attention_mask=True)
                label_sentence = self.tokenizer.tokenize(label_sentence)

                output_for_map = self.model.bert(
                    torch.tensor([test_sentence["input_ids"]], device=loss.device),
                    attention_mask=torch.tensor([test_sentence["attention_mask"]], device=loss.device),
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                attention_probs = output_for_map.attentions[11]
                attention_first_head = attention_probs[0][0]

                sentence = f"{cls_token} {label_sentence} {sep_token}".split(" ")
                heatmap = wandb.plots.HeatMap(
                    sentence,
                    sentence,
                    attention_first_head.cpu(),
                    show_text=False,
                )

                wandb.log({f"attention_heatmap_{self.state.global_step}": heatmap})
