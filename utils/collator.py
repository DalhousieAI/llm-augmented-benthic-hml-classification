import inspect


class QwenCollator:

    def __init__(self, processor, model, report_shape=True):
        self.processor = processor
        self.model = model
        try:
            params = inspect.signature(processor.apply_chat_template).parameters
            self.nest_kwargs = "processor_kwargs" in params
        except (TypeError, ValueError):
            self.nest_kwargs = False
        self._report_shape = report_shape

    def __call__(self, batch):
        image_paths, messages = zip(*batch)

        template_kwargs = dict(
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        processor_kwargs = dict(
            padding=True,
            return_tensors="pt",
        )

        if self.nest_kwargs:
            inputs = self.processor.apply_chat_template(
                list(messages),
                **template_kwargs,
                processor_kwargs=processor_kwargs,
            )
        else:
            inputs = self.processor.apply_chat_template(
                list(messages),
                **template_kwargs,
                **processor_kwargs,
            )

        if self._report_shape:
            self._report_shape = False
            ids = inputs.get("input_ids", None) if hasattr(inputs, "get") else None
            if ids is None or not hasattr(ids, "shape"):
                print("Warning: input_ids is not a tensor, check transformers version.", flush=True)
            else:
                print(
                    f"[QwenCollator] first batch input_ids shape "
                    f"{tuple(ids.shape)} (nest_kwargs={self.nest_kwargs})",
                    flush=True,
                )

        inputs = inputs.to(self.model.device)

        return list(image_paths), inputs
