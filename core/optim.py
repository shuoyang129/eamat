from transformers import AdamW, get_linear_schedule_with_warmup

from .config import config


def build_optimizer_and_scheduler(model, lr, num_train_steps,
                                  warmup_proportion):
    no_decay = ['bias', 'layer_norm',
                'LayerNorm']  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        config.TRAIN.WEIGHT_DECAY
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.0
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_train_steps * warmup_proportion, num_train_steps)
    return optimizer, scheduler