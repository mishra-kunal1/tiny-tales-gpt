# Copyright 2019 Google LLC All Rights Reserved.
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

# Echo commands and fail on error
set -ev

# [START getting_started_gce_startup_script]
# Install or update needed software
apt-get update
#apt-get install -yq git
#apt-get install -yq git supervisor python python-pip python3-distutils
#pip install --upgrade pip virtualenv

# Fetch source code
export HOME=/root
#git clone ssh://kunal.mishra.1096@gmail.com@source.developers.google.com:2022/p/deep-learning-project-419801/r/github_mishra-kunal1_tiny-tales-gpt
gcloud source repos clone github_mishra-kunal1_tiny-tales-gpt --project=deep-learning-project-419801
# Install Cloud Ops Agent
sudo bash /opt/app/add-google-cloud-ops-agent-repo.sh --also-install

# Account to own server process
useradd -m -d /home/pythonapp pythonapp

# Python environment setup
virtualenv -p python3 /opt/app/env
/bin/bash -c "source /opt/app/env/bin/activate"
/opt/app/env/bin/pip install -r /opt/app/requirements.txt

# Set ownership to newly created account
chown -R pythonapp:pythonapp /opt/app

# Put supervisor configuration in proper place
cp /opt/app/python-app.conf /etc/supervisor/conf.d/python-app.conf

# Start service via supervisorctl
supervisorctl reread
supervisorctl update
# [END getting_started_gce_startup_script]
