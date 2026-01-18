#=============================================================================
# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================


# CPM Search for raft
#
# Make sure we search for a build-dir config module for the CPM project
set(possible_package_dir "/inspire/hdd/global_user/liuzhigeng-253108120105/miniconda/envs/quest/lib/cmake/raft")
if(possible_package_dir AND NOT DEFINED raft_DIR)
  set(raft_DIR "${possible_package_dir}")
endif()

CPMFindPackage(
  "NAME;raft;VERSION;24.02.00;GIT_REPOSITORY;https://github.com/rapidsai/raft.git;GIT_TAG;branch-24.02;SOURCE_SUBDIR;cpp;OPTIONS;BUILD_TESTS OFF;BUILD_PRIMS_BENCH OFF;BUILD_ANN_BENCH OFF;RAFT_NVTX   ;RAFT_COMPILE_LIBRARY ON;FIND_PACKAGE_ARGUMENTS;COMPONENTS  compiled"
  )

if(possible_package_dir)
  unset(possible_package_dir)
endif()
#=============================================================================
