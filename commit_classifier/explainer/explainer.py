from typing import Union, Tuple, List, Optional

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers_interpret import LIGAttributions
from transformers_interpret.errors import InputIdsNotCalculatedError, \
    AttributionTypeNotSupportedError
from transformers_interpret.explainers.text import \
    PairwiseSequenceClassificationExplainer
import torch
from transformers_interpret.explainers.text.sequence_classification import \
    SUPPORTED_ATTRIBUTION_TYPES


class JavaBERTCommitClassificationExplainer(PairwiseSequenceClassificationExplainer):

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 attribution_type: str = "lig",
                 custom_labels: Optional[List[str]] = None):
        """
        Args:
            model (PreTrainedModel): Pretrained huggingface Sequence Classification model.
            tokenizer (PreTrainedTokenizer): Pretrained huggingface tokenizer
            attribution_type (str, optional): The attribution method to calculate on. Defaults to "lig".
            custom_labels (List[str], optional): Applies custom labels to label2id and id2label configs.
                                                 Labels must be same length as the base model configs' labels.
                                                 Labels and ids are applied index-wise. Defaults to None.

        Raises:
            AttributionTypeNotSupportedError:
        """

        self.pad_token_id = tokenizer.pad_token_id

        super().__init__(model, tokenizer)
        if attribution_type not in SUPPORTED_ATTRIBUTION_TYPES:
            raise AttributionTypeNotSupportedError(
                f"""Attribution type '{attribution_type}' is not supported.
                        Supported types are {SUPPORTED_ATTRIBUTION_TYPES}"""
            )
        self.attribution_type = attribution_type

        if custom_labels is not None:
            # if len(custom_labels) != len(model.config.label2id):
            #     raise ValueError(
            #         f"""`custom_labels` size '{len(custom_labels)}' should match pretrained model's label2id size
            #                 '{len(model.config.label2id)}'"""
            #     )

            self.id2label, self.label2id = self._get_id2label_and_label2id_dict(
                custom_labels)
        else:
            self.label2id = model.config.label2id
            self.id2label = model.config.id2label

        self.attributions: Union[None, LIGAttributions] = None
        self.input_ids: torch.Tensor = torch.Tensor()

        self._single_node_output = False

        self.internal_batch_size = None
        self.n_steps = 50

    @property
    def predicted_class_index(self) -> int:
        "Returns predicted class index (int) for model with last calculated `input_ids`"
        if len(self.input_ids) > 0:
            # we call this before _forward() so it has to be calculated twice
            preds = self.model(self.input_ids,
                               attention_mask=self.input_ids.ne(0))[0]
            preds = torch.sigmoid(preds[0])[0]
            self.pred_class = (preds > 0.5).int()
            return self.pred_class.cpu().detach().numpy()

        else:
            raise InputIdsNotCalculatedError(
                "input_ids have not been created yet.`")

    def _make_input_reference_pair(
        self, text1: Union[List, str], text2: Union[List, str]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:

        input_ids = self.tokenizer.encode(text1, text2,
                                          add_special_tokens=True)

        t1_ids = self.tokenizer.encode(text1, add_special_tokens=False)
        t2_ids = self.tokenizer.encode(text2, add_special_tokens=False)
        while len(t1_ids) + len(t2_ids) > len(input_ids) - 3:
            if len(t1_ids) + 1 > len(t2_ids):
                t1_ids = t1_ids[:-1]
            else:
                t2_ids = t2_ids[:-1]

        ref_input_ids = (
            [self.cls_token_id]
            + [self.ref_token_id] * len(t1_ids)
            + [self.sep_token_id]
            + [self.ref_token_id] * len(t2_ids)
            + [self.sep_token_id]
        )

        ref_input_ids += [self.pad_token_id] * (
                len(input_ids) - len(ref_input_ids))

        return (
            torch.tensor([input_ids], device=self.device),
            torch.tensor([ref_input_ids], device=self.device),
            len(t1_ids) + len(t2_ids),
        )

    def _make_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.ne(0)

    def _get_preds(
        self,
        input_ids: torch.Tensor,
        token_type_ids=None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
