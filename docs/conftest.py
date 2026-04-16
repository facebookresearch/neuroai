import typing as tp

import exca
import numpy as np
import pydantic


class TutorialTask(pydantic.BaseModel):
    param: int = 12
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def process(self) -> float:
        return self.param * np.random.rand()


class TutorialMap(pydantic.BaseModel):
    param: int = 12
    infra: exca.MapInfra = exca.MapInfra(version="1")

    @infra.apply(item_uid=str)
    def process(self, items: tp.Iterable[int]) -> tp.Iterator[np.ndarray]:
        for item in items:
            yield np.random.rand(item, self.param)


def pytest_markdown_docs_globals() -> tp.Dict[str, tp.Any]:
    return {
        "TutorialTask": TutorialTask,
        "TutorialMap": TutorialMap,
        "MapInfra": exca.MapInfra,
        "TaskInfra": exca.TaskInfra,
        "pydantic": pydantic,
    }
