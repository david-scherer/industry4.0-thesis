# Copyright 2023 The Nuclio Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# uses vader lib (will be installed automatically via build commands) to identify sentiments in the body string
# return score result in the form of: {'neg': 0.0, 'neu': 0.323, 'pos': 0.677, 'compound': 0.6369}


def handler(context, event):
    return str(event.body) + "\n TEST\n"
