# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-quest/quest/quest/ops/build_sm89_sm90/_deps/rapids-cmake-src")
  file(MAKE_DIRECTORY "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-quest/quest/quest/ops/build_sm89_sm90/_deps/rapids-cmake-src")
endif()
file(MAKE_DIRECTORY
  "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-quest/quest/quest/ops/build_sm89_sm90/_deps/rapids-cmake-build"
  "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-quest/quest/quest/ops/build_sm89_sm90/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix"
  "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-quest/quest/quest/ops/build_sm89_sm90/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/tmp"
  "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-quest/quest/quest/ops/build_sm89_sm90/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp"
  "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-quest/quest/quest/ops/build_sm89_sm90/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src"
  "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-quest/quest/quest/ops/build_sm89_sm90/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-quest/quest/quest/ops/build_sm89_sm90/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-quest/quest/quest/ops/build_sm89_sm90/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
