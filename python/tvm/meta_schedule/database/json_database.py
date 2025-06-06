# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The default database that uses a JSON File to store tuning records"""
import os.path as osp
from typing import Optional

from tvm.ffi import register_object

from .. import _ffi_api
from .database import Database


@register_object("meta_schedule.JSONDatabase")
class JSONDatabase(Database):
    """Database class backed by JSON.

    Parameters
    ----------
    path_workload : str
        The path to the workload table.
    path_tuning_record : str
        The path to the tuning record table.
    module_equality : Optional[str]
        A string to specify the module equality testing and hashing method.
        It must be one of the followings:
          - "structural": Use StructuralEqual/Hash
          - "ignore-ndarray": Same as "structural", but ignore ndarray raw data during
                              equality testing and hashing.
          - "anchor-block": Apply equality testing and hashing on the anchor block extracted from a
                            given module. The "ignore-ndarray" varint is used for the extracted
                            blocks or in case no anchor block is found.
                            For the definition of the anchor block, see tir/analysis/analysis.py.
    """

    path_workload: str
    path_tuning_record: str

    def __init__(
        self,
        path_workload: Optional[str] = None,
        path_tuning_record: Optional[str] = None,
        *,
        work_dir: Optional[str] = None,
        allow_missing: bool = True,
        module_equality: str = "structural",
    ) -> None:
        """Constructor.

        Parameters
        ----------
        path_workload : Optional[str] = None
            The path to the workload table. If not specified,
            will be generated from `work_dir` as `$work_dir/database_workload.json`.
        path_tuning_record : Optional[str] = None
            The path to the tuning record table. If not specified,
            will be generated from `work_dir` as `$work_dir/database_tuning_record.json`.
        work_dir : Optional[str] = None
            The work directory, if specified, will be used to generate `path_tuning_record`
            and `path_workload`.
        allow_missing : bool
            Whether to create new file when the given path is not found.
        """
        if work_dir is not None:
            if path_workload is None:
                path_workload = osp.join(work_dir, "database_workload.json")
            if path_tuning_record is None:
                path_tuning_record = osp.join(work_dir, "database_tuning_record.json")
        if path_workload is None:
            raise ValueError("`path_workload` is not specified.")
        if path_tuning_record is None:
            raise ValueError("`path_tuning_record` is not specified.")
        self.__init_handle_by_constructor__(
            _ffi_api.DatabaseJSONDatabase,  # type: ignore # pylint: disable=no-member
            path_workload,
            path_tuning_record,
            allow_missing,
            module_equality,
        )
