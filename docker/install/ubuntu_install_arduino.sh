#!/bin/bash
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

set -e
set -u
set -o pipefail

apt-install-and-clear -y ca-certificates

ARDUINO_CLI_VERSION="0.21.1"
# Install arduino-cli
wget -O - https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh -s ${ARDUINO_CLI_VERSION}

# Install the cores we want to test on
arduino-cli core install arduino:mbed_nano
arduino-cli core install arduino:sam

# ARDUINO_DIRECTORIES_USER wouldn't normally be created until we
# install a package, which would cause chmod to fail
mkdir -p "${ARDUINO_DIRECTORIES_DATA}" "${ARDUINO_DIRECTORIES_USER}" "${ARDUINO_DIRECTORIES_DOWNLOADS}"
chmod -R o+rw "${ARDUINO_DIRECTORIES_DATA}" "${ARDUINO_DIRECTORIES_USER}" "${ARDUINO_DIRECTORIES_DOWNLOADS}"
