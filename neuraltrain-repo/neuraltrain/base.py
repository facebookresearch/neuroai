# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pprint

import pydantic


class BaseModel(pydantic.BaseModel):
    """Base pydantic model with extra=forbid and nicer print"""

    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")

    def __repr__(self) -> str:
        data = self.model_dump()
        return f"{self.__class__.__name__}(**\n{pprint.pformat(data, indent=2)}\n)"
