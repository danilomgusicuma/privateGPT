from typing import Optional, List
from functools import partial

from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain.llms import GPT4All
from langchain.llms.utils import enforce_stop_tokens


class AsyncGPT4All(GPT4All):
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> str:
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)
        text = ""
        for token in self.client.generate(prompt, **self._default_params()):
            if text_callback:
                await text_callback(token)
            text += token
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text
